# Shape from Heat Conduction

This repository contains the official implementation of the paper "Shape from Heat Conduction" (ECCV 2024 Oral). The project presents a novel approach to 3D shape reconstruction by leveraging heat transport properties on object surfaces.

[Project Website](https://www.cs.cmu.edu/~ILIM/shape_from_heat/)

## Overview

Shape from Heat Conduction is a novel shape recovery approach that leverages the properties of heat transport, specifically heat conduction, induced on objects through a heating or cooling process. The method can reconstruct shapes of objects with diverse visible reflectance properties, including those that are transparent or translucent to visible light.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/nnsriram97/shape-from-heat.git
cd shape-from-heat
```

2. Create a conda environment and install dependencies:
```bash
conda create -n shape-from-heat python=3.10
conda activate shape-from-heat
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
pip install gdown  # Required for downloading data
```

## Data Download

The dataset can be downloaded using the provided script:

```bash
# Make the script executable (if not already)
chmod +x scripts/download_data.sh

# Run the download script
./scripts/download_data.sh
```

Alternatively, you can manually download the data from [Google Drive](https://drive.google.com/drive/folders/1PBby4Sja-j1e8alLoX8eFvUw0BH-LCcd?usp=sharing) and place it in the `DATA/object_study_raw` directory.

## Project Structure

```
shape-from-heat/
├── config/          # Configuration files
├── DATA/            # Dataset and input data
├── lib/             # Core library code
├── largesteps/      # Large steps optimization code
├── meshplot/        # Mesh visualization utilities
├── notebooks/       # Jupyter notebooks frontend for optimization
├── results/         # Output results
├── scripts/         # Utility scripts for visualization
```

## Usage

1. Data Preparation:
   - Place your thermal video data in the `DATA/object_study_raw` directory
2. Run the optimization:
    - Use the notebook `notebooks/real_shape_from_heat.ipynb` to run the optimization by using the right videos

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{10.1007/978-3-031-72920-1_24,
  author = {Narayanan, Sriram and Ramanagopal, Mani and Sheinin, Mark and Sankaranarayanan, Aswin C. and Narasimhan, Srinivasa G.},
  booktitle = {Computer Vision -- ECCV 2024},
  pages = {426--444},
  publisher = {Springer Nature Switzerland},
  title = {Shape from Heat Conduction},
  year = {2025}
  doi = {10.1007/978-3-031-72920-1_24}
}
```

## Acknowledgments

This work was partly supported by NSF grants IIS-2107236, CCF-1730147, and NSF-NIFA AI Institute for Resilient Agriculture. 