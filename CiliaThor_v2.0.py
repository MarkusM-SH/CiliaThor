"""
Cilia Analysis Tool - GUI Application
Automated 3D cilia segmentation and measurement from TIFF images
"""

import numpy as np
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from pathlib import Path
import tifffile
from skan import Skeleton, summarize
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import sys
import warnings
from datetime import datetime
import traceback
import pandas as pd
from PIL import Image, ImageTk

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class LogCapture:
    """Captures stdout and stderr to a file and string buffer"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log_buffer = []

    def write(self, message):
        self.terminal.write(message)
        self.log_buffer.append(message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message)

    def flush(self):
        self.terminal.flush()

    def get_log(self):
        return ''.join(self.log_buffer)

class CiliaAnalyzer:
    """Core analysis functions"""

    @staticmethod
    def smooth_3d(img, sigma=2.0):
        """Apply 3D Gaussian smoothing"""
        return ndi.gaussian_filter(img, sigma=sigma)

    @staticmethod
    def ridge_detection_3d(img, sigmas=range(1, 4), black_ridges=False):
        """Ridge detection using Frangi filter"""
        from skimage.filters import frangi
        ridges = frangi(img, sigmas=sigmas, black_ridges=black_ridges)
        threshold = filters.threshold_otsu(ridges)
        segmentation = ridges > threshold
        return segmentation, ridges

    @staticmethod
    def create_3d_objects(segmentation, fill_holes=True):
        """Create 3D labeled objects from binary segmentation"""
        if fill_holes:
            filled = segmentation.copy()
            for z in range(filled.shape[0]):
                filled[z] = ndi.binary_fill_holes(filled[z])
            for y in range(filled.shape[1]):
                filled[:, y, :] = ndi.binary_fill_holes(filled[:, y, :])
            for x in range(filled.shape[2]):
                filled[:, :, x] = ndi.binary_fill_holes(filled[:, :, x])
            segmentation = filled

        labeled = measure.label(segmentation, connectivity=2)
        return labeled

    @staticmethod
    def calculate_circularity_3d(region):
        """Calculate 3D sphericity"""
        volume = region.area
        padded = np.pad(region.image, 1, mode='constant', constant_values=0)
        try:
            verts, faces, normals, values = measure.marching_cubes(padded, level=0.5)
            surface_area = measure.mesh_surface_area(verts, faces)
            if surface_area == 0:
                return 0
            circularity = (36 * np.pi * volume**2) / (surface_area**3)
            return circularity
        except:
            return 1.0

    @staticmethod
    def filter_objects_by_properties(labeled, min_volume=30, max_circularity=0.6, log_func=print):
        """Filter 3D objects based on volume and circularity"""
        regions = measure.regionprops(labeled)
        filtered_labels = np.zeros_like(labeled)
        new_label = 1
        filtered_properties = []

        log_func(f"\n  Filtering {len(regions)} objects...")
        log_func(f"  Min volume: {min_volume} voxels, Max circularity: {max_circularity}")

        for region in regions:
            volume = region.area
            circularity = CiliaAnalyzer.calculate_circularity_3d(region)
            keep = (volume >= min_volume) and (circularity <= max_circularity)

            if keep:
                coords = region.coords
                filtered_labels[coords[:, 0], coords[:, 1], coords[:, 2]] = new_label
                filtered_properties.append({
                    'label': new_label,
                    'original_label': region.label,
                    'volume': volume,
                    'circularity': circularity,
                    'centroid': region.centroid,
                    'bbox': region.bbox
                })
                new_label += 1

        log_func(f"  Kept {new_label - 1} objects after filtering")
        return filtered_labels, filtered_properties

    @staticmethod
    def find_longest_path_in_skeleton(skel_coords):
        """Find longest path through skeleton"""
        if len(skel_coords) < 2:
            return 0, skel_coords
        distances = cdist(skel_coords, skel_coords)
        i, j = np.unravel_index(distances.argmax(), distances.shape)
        max_distance = distances[i, j]
        return max_distance, skel_coords

    @staticmethod
    def skeletonize_and_measure_length(labeled, properties_list, log_func=print):
        """Skeletonize and measure length"""
        skeleton_data = {}

        for props in properties_list:
            label = props['label']
            obj_mask = (labeled == label)
            skeleton = morphology.skeletonize(obj_mask)
            skel_coords = np.argwhere(skeleton)

            if len(skel_coords) < 2:
                continue

            try:
                skel_obj = Skeleton(skeleton)
                branch_data = summarize(skel_obj)
                if len(branch_data) > 0:
                    longest_idx = branch_data['euclidean-distance'].idxmax()
                    longest_branch = branch_data.loc[longest_idx]
                    length = longest_branch['euclidean-distance']
                    path_coords = skel_obj.path_coordinates(longest_idx)
                else:
                    length, path_coords = CiliaAnalyzer.find_longest_path_in_skeleton(skel_coords)
            except:
                length, path_coords = CiliaAnalyzer.find_longest_path_in_skeleton(skel_coords)

            skeleton_data[label] = {
                'skeleton_coords': skel_coords,
                'path_coords': path_coords,
                'length': length,
                'volume': props['volume'],
                'circularity': props['circularity'],
                'centroid': props['centroid']
            }

            log_func(f"  Object {label}: Length = {length:.2f} voxels, Volume = {props['volume']} voxels")

        return skeleton_data

    @staticmethod
    def visualize_results(img_background, labeled, skeleton_data, output_path):
        """Create visualization"""
        bg_proj = np.max(img_background, axis=0)
        max_label = np.max(labeled) if labeled.size > 0 else 0
        num_objects = len(skeleton_data)
        num_colors_needed = max(num_objects, int(max_label), 1)
        num_colors_needed = max(num_colors_needed, 20)

        if num_colors_needed <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        elif num_colors_needed <= 100:
            tab20b_colors = plt.cm.tab20b(np.linspace(0, 1, 20))
            tab20c_colors = plt.cm.tab20c(np.linspace(0, 1, 20))
            hsv_count = max(num_colors_needed - 40, 1)
            hsv_colors = plt.cm.hsv(np.linspace(0, 1, hsv_count))
            colors = np.vstack([tab20b_colors, tab20c_colors, hsv_colors])
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, num_colors_needed))

        colors = colors[:num_colors_needed]
        np.random.seed(42)
        np.random.shuffle(colors)

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        axes[0, 0].imshow(bg_proj, cmap='gray')
        axes[0, 0].set_title('Raw Cilia Channel (MAX Projection)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        colored_objects = np.zeros((*bg_proj.shape, 4))
        for label, data in skeleton_data.items():
            if label - 1 < len(colors):
                obj_mask_proj = np.max(labeled == label, axis=0)
                colored_objects[obj_mask_proj > 0] = colors[label - 1]
        axes[0, 1].imshow(colored_objects)
        axes[0, 1].set_title('Segmented Objects', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(np.ones_like(bg_proj), cmap='gray', vmin=0, vmax=1)
        for label, data in skeleton_data.items():
            centroid_2d = (data['centroid'][2], data['centroid'][1])
            axes[1, 0].text(centroid_2d[0], centroid_2d[1], str(label),
                          color=colors[label-1][:3], fontsize=10, fontweight='bold',
                          ha='center', va='center')
        axes[1, 0].set_title('Object Labels', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(bg_proj, cmap='gray')
        for label, data in skeleton_data.items():
            if label - 1 < len(colors):
                obj_mask_proj = np.max(labeled == label, axis=0)
                colored_mask = np.zeros((*obj_mask_proj.shape, 4))
                colored_mask[obj_mask_proj > 0] = colors[label - 1]
                axes[1, 1].imshow(colored_mask, alpha=0.5)
                skel_coords = data['skeleton_coords']
                axes[1, 1].scatter(skel_coords[:, 2], skel_coords[:, 1], c='red', s=0.2, alpha=0.8)
        axes[1, 1].set_title('Raw + Objects + Skeletons', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        plt.suptitle('Cilia Analysis Results', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def process_single_image(image_path, output_dir, parameters, output_options, progress_callback=None):
        """Process a single image with all steps"""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = image_path.stem
        log_file = output_dir / f"log_{base_name}.txt"

        log_capture = LogCapture(log_file)
        original_stdout = sys.stdout
        sys.stdout = log_capture

        try:
            print(f"\n{'='*80}")
            print(f"CILIA ANALYSIS - {base_name}")
            print(f"{'='*80}")
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Image: {image_path}")
            print(f"Output: {output_dir}")
            print(f"\nParameters:")
            print(f"  Cilia Channel: {parameters['cilia_channel']}")
            print(f"  Smooth Sigma: {parameters['smooth_sigma']}")
            print(f"  Min Volume: {parameters['min_volume']} voxels")
            print(f"  Max Circularity: {parameters['max_circularity']}")

            if progress_callback:
                progress_callback("Loading image data...")
            print(f"\n[1/7] Loading TIFF image...")
            raw_image = tifffile.imread(str(image_path))

            with tifffile.TiffFile(str(image_path)) as tif:
                first_page = tif.pages[0]
                tags = first_page.tags
                x_resolution_tag = tags.get('XResolution')
                y_resolution_tag = tags.get('YResolution')

                if x_resolution_tag and y_resolution_tag:
                    x_resolution = 1/(x_resolution_tag.value[0] / x_resolution_tag.value[1])
                    y_resolution = 1/(y_resolution_tag.value[0] / y_resolution_tag.value[1])
                else:
                    x_resolution = y_resolution = 1.0

                try:
                    img_desc = tags.get('ImageDescription').value
                    z_match = re.search(r'spacing=([\d.]+)', img_desc)
                    z_resolution = float(z_match.group(1)) if z_match else 1.0
                except:
                    z_resolution = 1.0

            print(f"  Image shape: {raw_image.shape}")
            print(f"  X Resolution: {x_resolution:.4f} µm/pixel")
            print(f"  Y Resolution: {y_resolution:.4f} µm/pixel")
            print(f"  Z Resolution: {z_resolution:.4f} µm/pixel")

            if progress_callback:
                progress_callback("Extracting cilia channel...")
            print(f"\n[2/7] Extracting cilia channel {parameters['cilia_channel']}...")
            if len(raw_image.shape) == 4:
                cilia_raw = raw_image[:, parameters['cilia_channel'], :, :].astype(np.float32)
            else:
                cilia_raw = raw_image.astype(np.float32)

            if progress_callback:
                progress_callback("Applying Gaussian smoothing...")
            print(f"\n[3/7] Applying 3D Gaussian smoothing (sigma={parameters['smooth_sigma']})...")
            cilia_smooth = CiliaAnalyzer.smooth_3d(cilia_raw, sigma=parameters['smooth_sigma'])

            if progress_callback:
                progress_callback("Ridge detection (Frangi filter)...")
            print(f"\n[4/7] Ridge Detection (Frangi filter)...")
            seg_ridges, ridge_response = CiliaAnalyzer.ridge_detection_3d(cilia_smooth, sigmas=range(1, 4))

            if progress_callback:
                progress_callback("Creating 3D objects...")
            print(f"\n[5/7] Creating 3D objects...")
            labeled = CiliaAnalyzer.create_3d_objects(seg_ridges, fill_holes=True)
            print(f"  Found {labeled.max()} objects")

            if progress_callback:
                progress_callback("Filtering objects...")
            print(f"\n[6/7] Filtering objects by properties...")
            labeled_filtered, properties_list = CiliaAnalyzer.filter_objects_by_properties(
                labeled,
                min_volume=parameters['min_volume'],
                max_circularity=parameters['max_circularity'],
                log_func=print
            )

            if progress_callback:
                progress_callback("Skeleton analysis...")
            print(f"\n[7/7] Skeletonize and measure length...")
            skeleton_data = CiliaAnalyzer.skeletonize_and_measure_length(
                labeled_filtered,
                properties_list,
                log_func=print
            )

            print(f"\n{'='*80}")
            print(f"SAVING RESULTS")
            print(f"{'='*80}")

            viz_path = output_dir / f"{base_name}_cilia_analysis_results.png"
            print(f"\nCreating visualization: {viz_path.name}")
            CiliaAnalyzer.visualize_results(cilia_smooth, labeled_filtered, skeleton_data, viz_path)

            if output_options.get('csv', True):
                csv_path = output_dir / f"{base_name}_cilia_analysis.csv"
                print(f"Exporting measurements: {csv_path.name}")
                if len(skeleton_data) > 0:
                    data_rows = []
                    for label, data in skeleton_data.items():
                        data_rows.append({
                            'Object_ID': label,
                            'Length_voxels': data['length'],
                            'Volume_voxels': data['volume'],
                            'Length_um': data['length'] * x_resolution,
                            'Volume_um3': data['volume'] * x_resolution * y_resolution * z_resolution,
                            'Circularity': data['circularity'],
                            'Centroid_Z': data['centroid'][0],
                            'Centroid_Y': data['centroid'][1],
                            'Centroid_X': data['centroid'][2]
                        })
                    df = pd.DataFrame(data_rows)
                    df.to_csv(csv_path, index=False)
                    print(f"  Exported {len(data_rows)} objects")

            if output_options.get('preprocessed', True):
                preprocessed_path = output_dir / f"{base_name}_preprocessed_cilia.tif"
                print(f"Saving preprocessed TIFF: {preprocessed_path.name}")
                tifffile.imwrite(str(preprocessed_path), cilia_smooth.astype(np.float32))

            if output_options.get('segmentation', True):
                seg_path = output_dir / f"{base_name}_binary_mask.tif"
                print(f"Saving segmentation TIFF: {seg_path.name}")
                tifffile.imwrite(str(seg_path), seg_ridges.astype(np.uint8))

            if output_options.get('labeled', True):
                labeled_path = output_dir / f"{base_name}_annotation_mask.tif"
                print(f"Saving labeled TIFF: {labeled_path.name}")
                tifffile.imwrite(str(labeled_path), labeled_filtered.astype(np.uint16))

            print(f"\n{'='*80}")
            print(f"ANALYSIS COMPLETE")
            print(f"{'='*80}")
            print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Objects found: {len(skeleton_data)}")
            print(f"Log saved: {log_file.name}")

            sys.stdout = original_stdout
            return True, f"Success: {len(skeleton_data)} objects analyzed"

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR OCCURRED")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            print(f"\nTraceback:")
            traceback.print_exc()
            print(f"\nAnalysis failed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            sys.stdout = original_stdout
            return False, f"Error: {str(e)}"

class CiliaAnalyzerGUI:
    """GUI Application for Cilia Analysis"""

    def __init__(self, root):
        self.root = root
        self.root.title("Cilia Analysis Tool v2.0")
        self.root.geometry("900x850")

        self.files = []
        self.output_dir = None
        self.processing = False
        self.progress_queue = queue.Queue()

        self.params = {
            'cilia_channel': tk.IntVar(value=2),
            'smooth_sigma': tk.DoubleVar(value=2.0),
            'min_volume': tk.IntVar(value=30),
            'max_circularity': tk.DoubleVar(value=0.6)
        }

        self.output_options = {
            'csv': tk.BooleanVar(value=True),
            'preprocessed': tk.BooleanVar(value=True),
            'labeled': tk.BooleanVar(value=True),
            'segmentation': tk.BooleanVar(value=True)
        }

        self.setup_ui()
        self.check_queue()

    def setup_ui(self):
        """Create the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)

        logo_img = Image.open("./CiliaThor_logo.png").resize((100, 100))
        self.logo = ImageTk.PhotoImage(logo_img)

        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        ttk.Label(title_frame, image=self.logo).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="CiliaThor", font=('Arial', 32, 'bold')).pack()
        ttk.Label(title_frame, text="Automated 3D cilia segmentation and measurement", font=('Arial', 18)).pack()

        file_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="10")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(file_frame, text="📁 Select TIFF Files", command=self.select_files).pack(fill=tk.X, pady=(0, 5))
        self.file_label = ttk.Label(file_frame, text="No files selected", foreground="gray")
        self.file_label.pack(fill=tk.X)

        output_frame = ttk.LabelFrame(main_frame, text="Output Directory", padding="10")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(output_frame, text="📂 Select Output Folder", command=self.select_output_dir).pack(fill=tk.X, pady=(0, 5))
        self.output_label = ttk.Label(output_frame, text="No output folder selected", foreground="gray")
        self.output_label.pack(fill=tk.X)

        param_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding="10")
        param_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        params_grid = ttk.Frame(param_frame)
        params_grid.pack(fill=tk.X)

        row = 0
        ttk.Label(params_grid, text="Cilia Channel:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_grid, from_=0, to=10, textvariable=self.params['cilia_channel'], width=15).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(params_grid, text="(Channel index, 0-based)", foreground="gray", font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, pady=5)

        row += 1
        ttk.Label(params_grid, text="Smooth Sigma:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_grid, from_=0.1, to=10.0, increment=0.1, textvariable=self.params['smooth_sigma'], width=15).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(params_grid, text="(Gaussian smoothing strength)", foreground="gray", font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, pady=5)

        row += 1
        ttk.Label(params_grid, text="Min Volume:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_grid, from_=1, to=1000, textvariable=self.params['min_volume'], width=15).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(params_grid, text="(Minimum object size in voxels)", foreground="gray", font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, pady=5)

        row += 1
        ttk.Label(params_grid, text="Max Circularity:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_grid, from_=0.0, to=1.0, increment=0.01, textvariable=self.params['max_circularity'], width=15).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(params_grid, text="(Max sphericity, lower = elongated)", foreground="gray", font=('Arial', 8)).grid(row=row, column=2, sticky=tk.W, pady=5)

        params_grid.columnconfigure(1, weight=1)

        output_options_frame = ttk.LabelFrame(main_frame, text="Output Files", padding="10")
        output_options_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        output_grid = ttk.Frame(output_options_frame)
        output_grid.pack(fill=tk.X)

        ttk.Checkbutton(output_grid, text="CSV (measurements)", variable=self.output_options['csv']).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(output_grid, text="Preprocessed TIFF", variable=self.output_options['preprocessed']).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(output_grid, text="Annotation Masek", variable=self.output_options['labeled']).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(output_grid, text="Binary Segmentation Mask", variable=self.output_options['segmentation']).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        self.run_button = ttk.Button(main_frame, text="▶ Start Analysis", command=self.start_processing, style='Accent.TButton')
        self.run_button.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        progress_frame = ttk.LabelFrame(main_frame, text="Processing Status", padding="10")
        progress_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        progress_frame.rowconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)

        progress_scroll = ttk.Scrollbar(progress_frame)
        progress_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.progress_text = tk.Text(progress_frame, height=20, width=80, yscrollcommand=progress_scroll.set, state='disabled', wrap='word')
        self.progress_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        progress_scroll.config(command=self.progress_text.yview)

        self.progress_text.tag_config('success', foreground='green')
        self.progress_text.tag_config('error', foreground='red')
        self.progress_text.tag_config('processing', foreground='blue')
        self.progress_text.tag_config('info', foreground='black')

        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

    def select_files(self):
        """Select input TIFF files"""
        files = filedialog.askopenfilenames(title="Select TIFF Files", filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")])
        if files:
            self.files = list(files)
            self.file_label.config(text=f"{len(self.files)} file(s) selected", foreground="black")

    def select_output_dir(self):
        """Select output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_label.config(text=directory, foreground="black")

    def log_message(self, message, tag='info'):
        """Add message to progress text"""
        self.progress_text.config(state='normal')
        self.progress_text.insert(tk.END, message + '\n', tag)
        self.progress_text.see(tk.END)
        self.progress_text.config(state='disabled')

    def start_processing(self):
        """Start processing in a separate thread"""
        if not self.files:
            messagebox.showwarning("No Files", "Please select input files")
            return

        if not self.output_dir:
            messagebox.showwarning("No Output", "Please select output directory")
            return

        if self.processing:
            return

        self.processing = True
        self.run_button.config(state='disabled', text="⏸ Processing...")
        self.progress_bar.start()

        self.progress_text.config(state='normal')
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state='disabled')

        params = {
            'cilia_channel': self.params['cilia_channel'].get(),
            'smooth_sigma': self.params['smooth_sigma'].get(),
            'min_volume': self.params['min_volume'].get(),
            'max_circularity': self.params['max_circularity'].get()
        }

        output_options = {
            'csv': self.output_options['csv'].get(),
            'preprocessed': self.output_options['preprocessed'].get(),
            'labeled': self.output_options['labeled'].get(),
            'segmentation': self.output_options['segmentation'].get()
        }

        thread = threading.Thread(target=self.process_files, args=(params, output_options), daemon=True)
        thread.start()

    def process_files(self, params, output_options):
        """Process all files"""
        total_files = len(self.files)
        successful = 0
        failed = 0

        self.progress_queue.put(('info', f"Starting batch processing of {total_files} files...\n"))

        for idx, file_path in enumerate(self.files, 1):
            file_name = Path(file_path).name
            self.progress_queue.put(('processing', f"\n[{idx}/{total_files}] Processing: {file_name}"))

            def progress_callback(msg):
                self.progress_queue.put(('info', f"  → {msg}"))

            try:
                success, message = CiliaAnalyzer.process_single_image(file_path, self.output_dir, params, output_options, progress_callback)

                if success:
                    successful += 1
                    self.progress_queue.put(('success', f"  ✓ {message}"))
                else:
                    failed += 1
                    self.progress_queue.put(('error', f"  ✗ {message}"))

            except Exception as e:
                failed += 1
                self.progress_queue.put(('error', f"  ✗ Error: {str(e)}"))

        self.progress_queue.put(('info', f"\n{'='*60}"))
        self.progress_queue.put(('info', f"BATCH PROCESSING COMPLETE"))
        self.progress_queue.put(('info', f"{'='*60}"))
        self.progress_queue.put(('success', f"✓ Successful: {successful}/{total_files}"))
        if failed > 0:
            self.progress_queue.put(('error', f"✗ Failed: {failed}/{total_files}"))
        self.progress_queue.put(('info', f"Output directory: {self.output_dir}\n"))

        self.progress_queue.put(('DONE', None))

    def check_queue(self):
        """Check for messages from processing thread"""
        try:
            while True:
                item = self.progress_queue.get_nowait()
                if item[0] == 'DONE':
                    self.processing = False
                    self.run_button.config(state='normal', text="▶ Start Analysis")
                    self.progress_bar.stop()
                    messagebox.showinfo("Complete", "Batch processing complete!")
                else:
                    tag, message = item
                    self.log_message(message, tag)
        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

def main():
    """Main entry point"""
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = CiliaAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
