<img src="CiliaThor_logo.png" alt="Logo" width="200" />
#CiliaThor
3D Cilia Segmentation tool.

> [!CAUTION]
> The current version uses embedded metadata for size calculation.
> This GUI is still in development, so measurements and calculations need to be double-checked.

# Building and using as an executable
To convert multiple images without Python experience, you can create a small executable. 
This allows batch analysis of multiple **tiff** files.

## Build
**Download the Python script**
```
git clone https://github.com/MarkusM-SH/CiliaThor.git
```
**Create an executable:**
```
pip install pyinstaller
cd path\to\the\CiliaThor_v2.0.py
pyinstaller --onefile --windowed --add-data "CiliaThor_logo.png:." CiliaThor_v2.0.py
```

## Usage
- Run the Python executable from the dist folder ('dist/CiliaThor_v2.0.exe')
- **Select TIFF Files** -> Multiple tiff files can be loaded simultaneously
- **Select Output Folder** -> Select the folder to save output files
- **Parameters** -> Adapt the Parameters
> [!CAUTION] The Cilia Channel is the position starting at index 0.
> For example, the first channel is DAPI and the second channel is Cilia, so the Cilia Channel is 1.
- **Output Files** -> Choose which file types should be saved.
- Click **Start Analysis**

> [!NOTE]
> Large files may not work depending on available RAM.
















