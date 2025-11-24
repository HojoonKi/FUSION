# Fast-DDPM with Fourier Feature Extraction

This repository implements Fast-DDPM (Fast Denoising Diffusion Probabilistic Models) with Fourier feature extraction for medical image processing tasks including image translation, CT denoising, and super-resolution.

## Overview

The project extends the Fast-DDPM framework by incorporating Fourier-based feature extraction into the diffusion model architecture. This enhancement enables better frequency-domain processing of medical images.

## Project Docs

- [AI604 CT Denoising Roadmap](docs/AI604_CT_Denoising_Plan.md): 제안서와 데이터 전처리 노트북을 기반으로 한 향후 개발 계획과 즉시 실행 항목을 정리했습니다.

### Key Features

- **Fast-DDPM Training**: Accelerated DDPM training with uniform and non-uniform scheduler sampling
- **Fourier Feature Extraction**: Integrated Fourier transform-based feature extraction in the model forward pass
- **Multiple Tasks Support**:
  - `sg` (Segmentation): Image translation and CT denoising with single condition input
  - `sr` (Super-Resolution): Multi-frame image super-resolution with dual condition input
- **Flexible Scheduler**: Support for uniform and non-uniform timestep sampling strategies
- **Test Mode**: Dummy data forward pass for model testing and debugging

## Architecture

### Model Types

- **sg (Segmentation)**: `in_channels=2` (x_img + x_noisy)
  - Used for: BRATS (brain image translation), LDFDCT (CT denoising)
- **sr (Super-Resolution)**: `in_channels=3` (x_bw + x_fw + x_noisy)
  - Used for: PMUB (multi-frame super-resolution)
  - Frame notation: BW (backward), MD (middle), FW (forward)

### Fourier Integration

The `models/fourier.py` module provides two implementations:

1. **Fourier Class**: Traditional lowpass/highpass filtering
   - `lowpass_filter()`: Applies frequency-domain lowpass filtering
   - `extract_features()`: Extracts frequency-domain features via FFT

2. **LearnableFourier Module**: Learnable frequency-domain transformations
   - Block-diagonal weight matrices for efficient computation
   - Adaptive frequency weighting with learnable parameters
   - Sparsity control via soft-thresholding

The Fourier features are extracted in `models/diffusion.py` during the forward pass:

```python
x_orig = tmp[:, :1].squeeze(1)  # Extract condition image (B, H, W)
ff = self.Fourier.forward(x_orig)  # Apply Fourier transform (B, H, W//2+1)
```

## Installation

### Requirements

- Python 3.10
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/HojoonKi/FUSION.git
cd FUSION
```

2. Create a conda environment (recommended):
```bash
conda create -n fusion python=3.10
conda activate fusion
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: The requirements include:
- PyTorch 2.0.1 with torchvision 0.15.2
- NumPy < 2.0 (for compatibility)
- OpenCV 4.8.1.78
- Additional packages: tensorboard, scikit-image, medpy, PyYAML, lmdb

## Usage

### Training

Train Fast-DDPM on different datasets:

**LDFDCT (CT Denoising)**:
```bash
python fast_ddpm_main.py --config configs/ldfd_linear.yml --dataset LDFDCT --scheduler_type uniform --timesteps 10
```

**BRATS (Brain Image Translation)**:
```bash
python fast_ddpm_main.py --config configs/brats_linear.yml --dataset BRATS --scheduler_type non-uniform --timesteps 10
```

**PMUB (Super-Resolution)**:
```bash
python fast_ddpm_main.py --config configs/pmub_linear.yml --dataset PMUB --scheduler_type uniform --timesteps 10
```

### Testing with Dummy Data

Test the model forward pass with random dummy data (256×256):

```bash
python fast_ddpm_main.py --test --config configs/ldfd_linear.yml
```

This mode:
- Skips checkpoint loading
- Creates random input tensors (B=1, C=channels, H=256, W=256)
- Performs a single forward pass through the model
- Prints tensor shapes for debugging

### Sampling

Generate samples from trained checkpoints:

```bash
python fast_ddpm_main.py --sample --fid --config configs/ldfd_linear.yml --dataset LDFDCT
```

## Configuration

Configuration files are located in `configs/`:

- `ldfd_linear.yml`: LDFDCT dataset with linear beta schedule
- `brats_linear.yml`: BRATS dataset with linear beta schedule  
- `pmub_linear.yml`: PMUB dataset with linear beta schedule

Key parameters:
- `model.type`: `sg` or `sr`
- `model.in_channels`: 2 for sg, 3 for sr
- `diffusion.beta_schedule`: `linear`, `quad`, `sigmoid`, `alpha_cosine`, etc.
- `training.batch_size`: Training batch size
- `sampling.ckpt_id`: List of checkpoint steps for inference

## Project Structure

```
FUSION/
├── configs/                # Configuration files
├── fusion_datasets/       # Dataset loaders (BRATS, LDFDCT, PMUB)
├── functions/             # Loss functions and utilities
├── models/
│   ├── diffusion.py      # Main U-Net diffusion model with Fourier integration
│   ├── fourier.py        # Fourier feature extraction modules
│   └── ema.py            # Exponential Moving Average helper
├── runners/
│   └── diffusion.py      # Training and sampling runners
├── fast_ddpm_main.py     # Main entry point
├── ddpm_main.py          # Original DDPM implementation
└── requirements.txt      # Python dependencies
```

## Training Methods

- `sg_train()`: Fast-DDPM training for single-condition tasks
- `sg_ddpm_train()`: Original DDPM training for single-condition tasks
- `sr_train()`: Fast-DDPM training for dual-condition super-resolution
- `sr_ddpm_train()`: Original DDPM training for dual-condition super-resolution

### Scheduler Types

- **uniform**: Evenly spaced timesteps across [0, T]
- **non-uniform**: Quadratically spaced timesteps for better sampling quality

## Sampling Strategies

- **generalized**: Generalized DDIM sampling with eta parameter
- **ddpm_noisy**: Standard DDPM sampling with all noise steps

## License

See `LICENSE` file for details.

## Acknowledgements

This implementation builds upon the DDPM and DDIM frameworks for diffusion models, adapted for medical imaging applications with Fourier domain enhancements.
