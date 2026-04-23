# Capstone_CubeSat

# Authors: Mathieu Rioux, Anthony Lamontagne, David Bouchard and Frédérick Routhier. (Laval university)

This repository contains Python tools for image processing and acquisition around three main areas:

- image alignment in `Alignment_Scripts`
- image deconvolution in `Deconvolution`
- Raspberry Pi camera interface and NDWI visualization in `Interface`

The project mainly relies on `OpenCV`, `NumPy`, `Matplotlib`, and `SciPy`, with additional hardware-specific dependencies for the Raspberry Pi interface.

## Project Structure

```text
Capstone_CubeSat/
|-- Alignment_Scripts/
|   |-- Akaze_alignment.py
|   |-- Sift_Ransac_alignment.py
|   |-- ECC_alignment.py
|   |-- utils.py
|   |-- Image_examples/
|   `-- result/
|-- Deconvolution/
|   |-- deconvolution.py
|   |-- tile_local_deconvolution.py
|   |-- tile_blind_deconvolution.py
|   |-- plot_psf.py
|   |-- Image_examples/
|   |-- outputs_tile_local_deconvolution/
|   `-- outputs_tile_blind_deconvolution/
`-- Interface/
    |-- interface_pi.py
    |-- alignment_akaze.py
    `-- requirements.txt
```

## Requirements

- Python 3.10 or newer is recommended
- A virtual environment is recommended

Quick setup for the image-processing scripts:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install opencv-python numpy matplotlib scipy pillow
```

Note:
the `requirements.txt` files found in some subfolders may include many more packages than are needed for basic script execution.

## Module 1: Image Alignment

The [`Alignment_Scripts`](./Alignment_Scripts) folder contains several approaches for registering one image to another:

- `Akaze_alignment.py`: AKAZE-based feature detection and matching, homography estimation, and quality metrics.
- `Sift_Ransac_alignment.py`: SIFT-based alignment using FLANN matching and affine transformation, with optional RANSAC.
- `ECC_alignment.py`: alignment using ECC optimization through `findTransformECC` with an affine transform.
- `utils.py`: helper functions for preprocessing, matching, transform estimation, and metric computation.

### Example Usage

From the project root:

```powershell
python Alignment_Scripts\Akaze_alignment.py
python Alignment_Scripts\Sift_Ransac_alignment.py
python Alignment_Scripts\ECC_alignment.py
```

### Inputs

The scripts currently use sample images stored in:

- [`Alignment_Scripts/Image_examples`](./Alignment_Scripts/Image_examples)

Image paths are generally defined directly in the `if __name__ == "__main__":` section of each script.

### Outputs

Results are saved in:

- [`Alignment_Scripts/result`](./Alignment_Scripts/result)

Examples of generated files:

- `Matches.jpg`
- `I2_corrected.jpg`
- `joint_histogram.jpg`
- `target_new.jpg`
- `source_new.jpg`

### Metrics

Depending on the script, the following metrics are used:

- cross-correlation
- mutual information
- number of keypoints and matches

## Module 2: Deconvolution

The [`Deconvolution`](./Deconvolution) folder contains several scripts for restoring blurred images.

### Main Scripts

- `deconvolution.py`: standard deconvolution with either a single PSF or multiple local PSFs. Available methods include Wiener and Richardson-Lucy.
- `tile_local_deconvolution.py`: tile-based local deconvolution using an empirically estimated Gaussian blur that varies across the image.
- `tile_blind_deconvolution.py`: tile-based blind deconvolution with joint estimation of the latent image and the PSF.
- `plot_psf.py`: extraction, estimation, and visualization of a PSF from an image.

### Example Usage

From the project root:

```powershell
python Deconvolution\tile_local_deconvolution.py
python Deconvolution\tile_blind_deconvolution.py
```

Example with command-line arguments:

```powershell
python Deconvolution\deconvolution.py --input Deconvolution\Image_examples\irtest.jpg --aberrated-input Deconvolution\Image_examples\irtest.jpg --psf path\to\psf.npy --method both
```

### Outputs

Examples of output folders already present:

- [`Deconvolution/outputs_tile_local_deconvolution`](./Deconvolution/outputs_tile_local_deconvolution)
- [`Deconvolution/outputs_tile_blind_deconvolution`](./Deconvolution/outputs_tile_blind_deconvolution)

Examples of generated files:

- `restored_tile_wiener.png`
- `restored_tile_blind.png`
- `tile_sigma_map.png`
- `tile_psf_grid.png`
- `useful_region_mask.png`
- `comparison_strip.png`

## Module 3: Interface

The [`Interface`](./Interface) folder contains Raspberry Pi-oriented code for dual-camera acquisition, live preview, image saving, and NDWI visualization.

### Main Scripts

- `interface_pi.py`: Tkinter-based graphical interface for a Raspberry Pi 5 with two cameras. It provides:
  - a live dual-camera preview
  - manual FPS and exposure control
  - snapshot capture
  - NDWI computation and display
  - saving of NDWI plots and individual camera images
- `alignment_akaze.py`: preprocessing and utility functions used by the interface for AKAZE-based multispectral alignment.
- `requirements.txt`: environment-specific package list used on the Raspberry Pi side.

### Interface Notes

- `interface_pi.py` depends on Raspberry Pi hardware and libraries such as `picamera2`.
- The default save paths in the script target a Linux environment on the Raspberry Pi:
  - `/home/capstone/Desktop/NDWI_graphs_saved`
  - `/home/capstone/Desktop/Individual_camera_images_saved`
- The interface uses two camera feeds, labeled `860 nm` and `560 nm`, to compute an NDWI map after image alignment.

### Typical Usage

From the `Interface` folder on the Raspberry Pi:

```powershell
python interface_pi.py
```

## Usage Notes

- Several scripts open OpenCV windows to display intermediate or final results.
- Some parameters are still hard-coded directly in the scripts.
- To adapt the project to your own data, you will usually only need to change the input paths in the scripts or use CLI arguments when available.
- The interface module is hardware-dependent and is intended to run on the Raspberry Pi setup used for the project.

## Possible Improvements

- standardize file paths and command-line arguments across scripts
- simplify the dependency lists in `requirements.txt`
- centralize input and output configuration
- document the hardware setup and deployment steps for the Raspberry Pi interface
