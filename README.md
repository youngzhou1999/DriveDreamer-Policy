# DriveDreamer-Policy

**DriveDreamer-Policy: A Geometry-Grounded World–Action Model for Unified Generation and Planning**

[![arXiv](https://img.shields.io/badge/arXiv-2604.01765-b31b1b.svg)](https://arxiv.org/abs/2604.01765)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

![Pipeline](assets/pipe.png)

## Updates

- **[10/2026]** Our paper was accepted to NeurIPS 2026!
- **[05/2026]** Code and model weights released!
- **[04/2026]** Paper draft released on [arXiv](https://arxiv.org/abs/2604.01765).

---

## Table of Contents

1. [Setup](#1-setup)
2. [Data Preparation](#2-data-preparation)
3. [Model Weights](#3-model-weights)
4. [Add World & Action Tokens](#4-add-world--action-tokens)
5. [Inference](#5-inference)
6. [Evaluation](#6-evaluation)
7. [Training](#7-training)
8. [Project Structure](#8-project-structure)
9. [Citation](#9-citation)
10. [Acknowledgements](#10-acknowledgements)
11. [License](#11-license)

---

## 1. Setup

### 1.1 Python Environment

We use **Python 3.10**, **PyTorch 2.5.1**, and **CUDA 12.4**.

```bash
conda create -n vla python=3.10
conda activate vla

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 1.2 NAVSIM Devkit

The repository bundles two NAVSIM versions:

| Directory | Version | Used for | Conda env |
|-----------|---------|----------|-----------|
| `navsim/` | v2 | Training data + v2 evaluation | `vla` |
| `navsim_v1.1/navsim/` | v1.1 | NAVSIM v1.1 leaderboard evaluation only | `navsim_v1.1` |

**Install NAVSIM v2** (required for training and v2 evaluation):

```bash
cd navsim
pip install -e .
cd ..
```

**Install NAVSIM v1.1** (only needed for NAVSIM v1.1 leaderboard evaluation):

NAVSIM v1.1 has different dependencies from v2, so it requires a **separate conda environment**:

```bash
conda create --name navsim_v1.1 --clone vla
conda activate navsim_v1.1

pip uninstall navsim -y
cd navsim_v1.1/navsim
pip install -e .
cd ../..
```

For full installation details and dataset download instructions, refer to the [official NAVSIM install guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md).

### 1.3 Depth Environment (optional, required for 3D head)

Depth map generation requires a separate conda environment with [Depth-Anything-3](depth_process/Depth-Anything-3/) installed:

```bash
conda create -n <your-da3-env> python=3.10
conda activate <your-da3-env>
cd depth_process/Depth-Anything-3
pip install -e .
cd ../..
```

### 1.4 Core Dependencies

Our training framework is built on top of **[starVLA v1.0.1](https://github.com/starVLA/starVLA)**. Install the additional packages:

```bash
pip install -r requirements_vla.txt
```

### 1.5 Environment Variables

Fill in your paths in `env.sh` and source it before running any pipeline script:

```bash
vim env.sh
source env.sh
```

Key variables:

| Variable | Description |
|----------|-------------|
| `CUDA_HOME` | Path to your CUDA 12.4 installation |
| `HF_HOME` | HuggingFace model cache directory |
| `NUPLAN_MAPS_ROOT` | Path to nuPlan map files |
| `OPENSCENE_DATA_ROOT` | Root of the NAVSIM/OpenScene dataset |
| `NAVSIM_EXP_ROOT` | Where training experiments are saved |
| `BASE_VLM` | Path to the base Qwen3-VL-2B-WorldAction checkpoint |
| `WANDB_API_KEY` | Your Weights & Biases API key (optional) |

---

## 2. Data Preparation

We convert raw NAVSIM sensor logs into our own unified training format. The complete processing logic is in [`navsim_data_process/make_data.py`](navsim_data_process/make_data.py).

**Dataset notes:**
- For **planning only**, the standard `navtrain` split is sufficient.
- For **video generation** (2D world head), you additionally need the `trainval` RGB sensor data (lidar is **not** required). See the [NAVSIM splits guide](https://github.com/autonomousvision/navsim/blob/main/docs/splits.md) for download instructions.

Run the numbered scripts in order after sourcing your env file:

### Step 0 — Process raw NAVSIM data

```bash
bash 0-process_data.sh
```

Edit `SPLIT` and `DATA_ROOT` at the top of the script to match your setup. Add `--make_video` to the python call to also generate video clips; omit it to skip and significantly speed up processing (recommended for planning-only runs).

Writes one pickle file per scene:

```
navsim_dataset/
└── meta/
    └── {split}/
        └── {token}.pkl        # one file per scene
```

Each pickle contains ego poses, velocities, accelerations, driving commands, and per-camera image paths / calibration for all 8 views over T=13 frames.

When `--make_video` is passed, MP4 clips for the 3 front-facing cameras are also written to `navsim_dataset/navsim_video/{split}/`.

### Step 1 — Generate the data meta-list

```bash
bash 1-gen_data_meta_list.sh
```

Scans `navsim_dataset/meta/{split}/` and writes a shuffled token list:

```
{split}_meta.json   →   ["token_a", "token_b", ...]
```

This file is passed to the dataloader via `--datasets.vla_data.datalist_path`.

### Step 2 — Generate depth maps (optional, required for 3D head)

Requires a conda environment with [Depth-Anything-3](depth_process/Depth-Anything-3/) installed (see [Section 1.3](#13-depth-environment-optional-required-for-3d-head)):

```bash
conda activate <your-da3-env>
bash 2-gen_depth.sh
```

Uses the local `Depth-Anything-3` copy under `depth_process/`, loaded offline. Generates metric depth for the 3 front-facing cameras and writes results alongside scene pickles as `{token}.pkl-depth.pkl`.

### Step 3 — Dataset statistics (optional)

```bash
bash 3-stat_data.sh
```

Computes per-channel trajectory statistics (mean, std) over the training split. Re-run this step and update the normalisation constants in `starVLA/dataloader/navsim_dataset.py` if you change the training split or mix in other datasets.

---

## 3. Model Weights

### Base models (required for training)

| Model | Role | Link |
|-------|------|------|
| `Qwen3-VL-2B-Instruct` | Vision-language backbone | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| `Wan2.1-Fun-V1.1-1.3B-InP` | 2D video generation head | [HuggingFace](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP) |
| `Pixel-Perfect Depth (PPD)` | 3D depth head | [GitHub](https://github.com/gangweix/pixel-perfect-depth) |
| `Depth-Anything-V2 ViT-L` | Semantic encoder for PPD | [HuggingFace](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) |

After downloading, place the depth model checkpoints in the `depth_model_ckpts/` directory at the project root:

```
depth_model_ckpts/
├── ppd.pth                     # Pixel-Perfect Depth checkpoint
└── depth_anything_v2_vitl.pth  # Depth-Anything-V2 ViT-L weights
```

### Our model

| Checkpoint | Description | Link |
|------------|-------------|------|
| `DriveDreamer-Policy` | Full trained model (action + video + depth heads) | [HuggingFace](https://huggingface.co/yangzhou99/DriveDreamer-Policy) |

---

## 4. Add World & Action Tokens

The base `Qwen3-VL-2B-Instruct` vocabulary must be extended with special tokens for world generation and action prediction (e.g. `<2d_world_*>`, `<robot_action_*>`). Run **once** before inference or training:

```bash
source env.sh
bash 7-add_token.sh
```

After it finishes, update `BASE_VLM` in `env.sh` to point to the new extended model:

```bash
export BASE_VLM=/path/to/Qwen3-VL-2B-WorldAction   # TARGET_VLM from 7-add_token.sh
```

Also set `VIDEO_MODEL` in `8-train.sh` / `debug.sh` to your local Wan2.1 model root directory.

---

## 5. Inference

Set `MODEL_DIR` at the top of `4-infer.sh` to your checkpoint directory, then run:

```bash
source env.sh
bash 4-infer.sh
```

The script writes one `.npy` trajectory file per scene token under `navsim_planning_results/<run_id>/<split>/`.

You can also call `infer.py` directly:

```bash
python infer.py \
  --ckpt_dir /path/to/checkpoint \
  --datalist_path {split}_meta.json \
  --out_dir navsim_planning_results/ \
  --split test \
  --batch_size 8 \
  --num_workers 7
```

---

## 6. Evaluation

Evaluation requires a pre-computed metric cache. If you do not have one, run the caching script first (commented out at the top of each eval script).

### NAVSIM v2

```bash
source env.sh
bash 6-eval_v2.sh
```

Uses the PDM-Score evaluator from the NAVSIM v2 devkit (`navsim/`). Set `PRED_DIR` and `METRIC_CACHE_PATH` at the top of `6-eval_v2.sh`.

### NAVSIM v1.1

```bash
conda activate navsim_v1.1
source env.sh
bash 5-eval_v1.sh
```

Uses the PDM-Score evaluator from the NAVSIM v1.1 devkit (`navsim_v1.1/navsim/`). Requires the separate `navsim_v1.1` conda environment (see [Section 1.2](#12-navsim-devkit)). Set `PRED_DIR` and `METRIC_CACHE_PATH` at the top of `5-eval_v1.sh`.

---

## 7. Training

### Full training (8×GPU, navtrain split)

```bash
source env.sh
bash 8-train.sh
```

Launches DeepSpeed ZeRO-2 training across 8 GPUs with all three heads active (1D action + 2D video + 3D depth). Set `VIDEO_MODEL`, `VIDEO_CONFIG`, `VIDEO_DATA_DIR`, and `BASE_VLM` (extended model from [Section 4](#4-add-world--action-tokens)) at the top of `8-train.sh`. The run ID is auto-timestamped; checkpoints are saved to `$NAVSIM_EXP_ROOT/<run_id>/`.

### Debug / sanity check (1×GPU, mini split)

```bash
source env.sh
bash debug.sh
```

Set `GPU` and `PORT` at the top of `debug.sh` to match an available GPU. Runs a single forward + backward pass on the mini split to verify model loading, data loading, and gradient flow before a full training run.

### Key hyper-parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bz` | 4 | Per-device batch size |
| `num_processes` | 8 | Number of GPUs |
| `act_fm_size` | 1536 | Action DiT hidden size |
| `act_fm_layer` | 24 | Action DiT depth |
| `fm_repeat` | 8 | Repeated diffusion steps |
| `trainer.learning_rate.base` | 1e-5 | Base learning rate |
| `trainer.max_train_steps` | 100 000 | Total optimisation steps |

Default values live in `starVLA/config/training/cfg_yaw_1225.yaml`; variables declared at the top of each shell script override specific values. You generally only need to change the path variables and `SPLIT`.

---

## 8. Project Structure

```
DriveDreamer-Policy/
├── starVLA/                    # Core model package
│   ├── model/
│   │   ├── framework/          # Model wrappers (QwenOFT, …)
│   │   └── modules/
│   │       ├── action_model/   # Flow-matching DiT action head
│   │       ├── video_model/    # WAN-based 2D video generation head
│   │       ├── depth_model/    # Pixel-Perfect Depth adapter (3D head)
│   │       └── vlm/            # Qwen3-VL backbone utilities
│   ├── dataloader/
│   │   └── navsim_dataset.py   # NAVSIM dataset class
│   ├── training/
│   │   └── train_starvla.py    # Training entry point
│   └── config/
│       ├── deepseeds/          # DeepSpeed ZeRO configs
│       └── training/           # YAML training configs
├── navsim_data_process/        # Data processing scripts
│   ├── make_data.py            # Step 0: process raw NAVSIM data
│   ├── data_list.py            # Step 1: generate meta-list JSON
│   └── data_stat.py            # Step 3: dataset statistics
├── depth_process/              # Depth generation (Depth-Anything-3)
├── navsim/                     # NAVSIM v2 devkit
├── navsim_v1.1/                # NAVSIM v1.1 devkit
├── infer.py                    # Inference entry point
├── env.sh                      # Environment variable template
├── 0-process_data.sh           # Pipeline step 0: process NAVSIM data
├── 1-gen_data_meta_list.sh     # Pipeline step 1: generate meta-list JSON
├── 2-gen_depth.sh              # Pipeline step 2: generate depth maps
├── 3-stat_data.sh              # Pipeline step 3: dataset statistics
├── 4-infer.sh                  # Inference
├── 5-eval_v1.sh                # NAVSIM v1.1 evaluation  (conda: navsim_v1.1)
├── 6-eval_v2.sh                # NAVSIM v2 evaluation    (conda: vla)
├── 7-add_token.sh              # Extend VLM vocabulary (run once before training)
├── 8-train.sh                  # Full training (8 GPUs)
└── debug.sh                    # Single-GPU debug run
```

---

## 9. Citation

If you use DriveDreamer-Policy in your research, please cite:

```bibtex
@misc{zhou2026drivedreamerpolicy,
      title={DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning}, 
      author={Yang Zhou and Xiaofeng Wang and Hao Shao and Letian Wang and Guosheng Zhao and Jiangnan Shao and Jiagang Zhu and Tingdong Yu and Zheng Zhu and Guan Huang and Steven L. Waslander},
      year={2026},
      eprint={2604.01765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.01765}, 
}
```

---

## 10. Acknowledgements

This repository is built upon the following open-source projects. We sincerely thank their authors for making their work publicly available:

- [starVLA](https://github.com/starVLA/starVLA) — training framework and model architecture
- [Qwen3-VL](https://github.com/qwenlm/qwen3-vl) — vision-language backbone
- [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) — video generation modules
- [Pixel-Perfect Depth (PPD)](https://github.com/gangweix/pixel-perfect-depth) — metric depth estimation head
- [Depth-Anything](https://github.com/DepthAnything/Depth-Anything-V2) — monocular depth foundation model
- [NAVSIM](https://github.com/autonomousvision/navsim) — autonomous driving simulation and evaluation

---

## 11. License

All code in this repository is released under the [Apache License 2.0](LICENSE).

The bundled NAVSIM devkit retains its original licence; please refer to `navsim/LICENSE`.
