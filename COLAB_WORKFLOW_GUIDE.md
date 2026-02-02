# PINN Training: Local + Colab Workflow

## Overview
- **Step 1 (Local)**: Run FEniCS to generate lead field matrix (~1-2 min)
- **Step 2 (Colab)**: Use pre-calculated lead field to train PINN (~20-30 min)

## Why Split Setup?
- FEniCS doesn't work on Colab (C++ dependencies, complex build)
- Lead field calculation is the expensive part (~1-2 min)
- PINN training is pure PyTorch (works great on Colab GPU)
- Solution: Pre-calculate once, reuse many times

---

## Step 1: Generate Lead Field Locally

### Requirements
```bash
# Activate FEniCS environment
conda activate fenics

# Verify FEniCS
python -c "from dolfin import *; print('FEniCS OK')"
```

### Run Generation Script
```bash
cd /path/to/p4_meg
python generate_leadfield_local.py
```

### Output Files (upload to Google Drive)
```
leadfield_matrix.npy        (~1.3 GB for 10k sources)
source_coordinates.npy      (~400 MB)
sensor_positions.npy        (~20 KB)
config_info.npy            (~1 KB)
```

**Runtime**: 
- 5,000 sources: ~30 seconds
- 10,000 sources: ~1-2 minutes  
- 20,000 sources: ~5-10 minutes

---

## Step 2: Train PINN on Colab

### 1. Upload Files to Google Drive
```
My Drive/
├── MEG_PINN/
│   ├── leadfield_matrix.npy
│   ├── source_coordinates.npy
│   ├── sensor_positions.npy
│   └── config_info.npy
```

### 2. Create Colab Cell 0: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Set path
GDRIVE_PATH = '/content/drive/My Drive/MEG_PINN/'
```

### 3. Open & Run Notebook
1. Upload `train_pinn_leadfield.ipynb` to Colab
2. The notebook automatically detects Colab mode
3. Run cells in order - it will load pre-calculated files
4. Training should start automatically

### 4. Expected Colab Performance
- **Data loading**: 5-10 seconds
- **PINN training (100 epochs)**: 15-25 minutes on GPU
- **Evaluation & visualization**: 2-3 minutes
- **Total**: ~30-40 minutes

---

## File Sizes (Example)

For 10,000 training samples:

| File | Size | Notes |
|------|------|-------|
| leadfield_matrix.npy | 1.3 GB | Main computational bottleneck |
| source_coordinates.npy | 400 MB | Training data locations |
| sensor_positions.npy | ~20 KB | Negligible |
| config_info.npy | ~1 KB | Negligible |
| **Total** | **~1.7 GB** | One-time upload to Drive |

> Note: If 1.7 GB is too large for your Drive, reduce to 5,000 samples (~850 MB)

---

## Troubleshooting

### FEniCS Installation Issues Locally
```bash
# Option 1: Conda (recommended)
conda create -n fenics -c conda-forge fenics
conda activate fenics

# Option 2: Docker (if conda fails)
docker pull dolfinx/dolfinx
docker run -it dolfinx/dolfinx
```

### Upload Stuck on Colab
```python
# Option 1: Use smaller dataset
# Run locally: python generate_leadfield_local.py  (with n_training_samples=5000)

# Option 2: Split upload into chunks
# Upload one file at a time to Drive manually

# Option 3: Use Colab's upload widget
from google.colab import files
files.upload()
```

### Out of Memory on Colab
```python
# Reduce batch size in notebook
batch_size = 16  # (was 32)

# Or reduce dataset size locally
# Edit generate_leadfield_local.py: n_training_samples = 5000  # (was 10000)
```

### Load Time Issues
```python
# If loading is slow, try:
import numpy as np
lead_field_matrix = np.load('/content/drive/My Drive/MEG_PINN/leadfield_matrix.npy', mmap_mode='r')
# (memory-mapped reading, slower but uses less RAM)
```

---

## Scaling Up to 20,000 Samples

### Local Generation
```bash
# Edit generate_leadfield_local.py (line ~60)
n_training_samples = 20000  # (was 10000)

python generate_leadfield_local.py
# ~5-10 minutes runtime
```

### Colab Considerations
- File sizes: ~3.4 GB total
- Requires Colab Pro or split into multiple runs
- Training time: ~45-60 minutes on GPU

---

## Quick Start Command

```bash
# All-in-one local run:
conda activate fenics
cd /path/to/p4_meg
python generate_leadfield_local.py

# Then upload outputs to Google Drive and run Colab notebook
```

