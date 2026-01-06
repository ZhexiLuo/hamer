# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HaMeR (Hand Mesh Recovery) is a transformer-based 3D hand reconstruction system from the paper "Reconstructing Hands in 3D with Transformers" (CVPR 2024). It predicts MANO hand model parameters from single images.

## Common Commands

### Installation
```bash
# Create environment (Python 3.10 required)
python3.10 -m venv .hamer && source .hamer/bin/activate

# Install dependencies (CUDA 11.7 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose

# Download pretrained models
bash fetch_demo_data.sh
```

### Demo Inference
```bash
python demo.py --img_folder example_data --out_folder demo_out --batch_size=48 --side_view --save_mesh --full_frame
```

### Training
```bash
# Download training data first
bash fetch_training_data.sh

# Run training with Hydra config
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```

### Evaluation
```bash
python eval.py --dataset 'FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL'
```

## Architecture

### Inference Pipeline (demo.py)
1. **Person Detection**: Detectron2 ViTDet or RegNetY detects humans in image
2. **Keypoint Detection**: ViTPose extracts hand keypoints from detected persons
3. **Hand Bbox Extraction**: Left/right hand bboxes derived from keypoint confidence
4. **Hand Reconstruction**: HaMeR model predicts MANO parameters from cropped hand images
5. **Rendering**: PyRender visualizes 3D hand mesh overlaid on original image

### Core Model (hamer/models/hamer.py)
- `HAMER` class extends PyTorch Lightning `LightningModule`
- **Backbone**: ViT feature extractor (`hamer/models/backbones/vit.py`)
- **Head**: MANO parameter regressor (`hamer/models/heads/mano_head.py`) using transformer decoder
- **MANO**: Hand model wrapper (`hamer/models/mano_wrapper.py`) converts parameters to mesh vertices
- **Discriminator**: Adversarial training for realistic hand poses

### Configuration System
- **Hydra configs**: `hamer/configs_hydra/` - training orchestration (experiment, trainer, launcher)
- **YACS configs**: `hamer/configs/` - model and dataset parameters
- Key config files:
  - `hamer/configs_hydra/experiment/hamer_vit_transformer.yaml` - main experiment config
  - `hamer/configs/datasets_tar.yaml` - training dataset paths
  - `hamer/configs/datasets_eval.yaml` - evaluation dataset paths

### Data Pipeline (hamer/datasets/)
- `HAMERDataModule`: Lightning DataModule managing train/val loaders
- `MixedWebDataset`: Weighted sampling from multiple WebDataset tar files
- `ViTDetDataset`: Inference-time dataset for processing detected hand crops
- `ImageDataset` / `MoCapDataset`: Training data loaders

### Key Paths
- `_DATA/`: Model checkpoints and MANO files (created by fetch_demo_data.sh)
- `_DATA/data/mano/MANO_RIGHT.pkl`: Required MANO model (manual download from mano.is.tue.mpg.de)
- `hamer_training_data/`: Training data (created by fetch_training_data.sh)
- `logs/`: Training checkpoints and tensorboard logs
- `results/`: Evaluation output CSV files

## Dependencies

- PyTorch Lightning for training
- Detectron2 for person detection
- ViTPose (third-party/ViTPose) for keypoint detection
- SMPLX for MANO hand model
- PyRender for visualization
