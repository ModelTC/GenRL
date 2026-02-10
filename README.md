<div align="center">

<table>
  <tr>
    <td>
      <img src="assets/logo.webp" alt="GenRL Logo" width="100">
    </td>
    <td style="padding-left: 12px; text-align: left;">
      <h1 style="margin-bottom: 4px;">GenRL</h1>
      <h3 style="margin-top: 0;">Reinforcement Learning Framework for Visual Generation</h3>
    </td>
  </tr>
</table>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-orange.svg)](LICENSE.txt)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx) -->

<!-- TODO: Add a teaser image / GIF here -->
<!-- <img src="assets/teaser.png" width="800"> -->

**GenRL** is a scalable, modular reinforcement learning framework for optimizing visual generation models — from images to videos — with plug-and-play reward functions, multi-GPU distributed training, and first-class support for diffusion & flow-based generators.

[🚀 Getting Started](#-getting-started) · [📖 Algorithms](#-supported-algorithms) · [📊 Performance](#-performance) · [🏗️ Architecture](#️-architecture)

</div>

---

## ✨ Highlights

- 🎯 **Unified RL for Visual Generation** — A single framework covering text-to-image (T2I), text-to-video (T2V), and image-to-video (I2V) generation
- 🔄 **Multi-Paradigm Support** — Native support for both **Diffusion** and **Rectified Flow** generation paradigms via unified SDE formulation
- 🧩 **Modular Reward System** — Plug-and-play reward functions: aesthetic scores, text-alignment, motion quality, OCR accuracy, and custom user-defined rewards
- ⚡ **Scalable & Efficient** — Multi-node FSDP training with activation checkpointing, LoRA / full fine-tune, EMA, 8-bit Adam, and memory-efficient reward model offloading
- 🎛️ **YAML-Driven Configuration** — Everything from model choice, reward weights, training schedule to FSDP sharding strategy is controlled via a single YAML config
- 🔬 **Reproducible by Design** — Deterministic seeding across sampling, training, and logging for bit-exact experiment reproduction

---

## 📖 Supported Algorithms

<!-- TODO: Add / update algorithm entries as they are implemented -->

| Algorithm | Type | Status | Description |
|-----------|------|--------|-------------|
| **[FlowGRPO](https://arxiv.org/abs/2505.05470)** | Policy Gradient | ✅ Supported | Group Relative Policy Optimization — compute advantages per-group with optional per-prompt stat tracking |
| **[MixGRPO](https://arxiv.org/abs/2507.21802)** | Policy Gradient | ✅ Supported | SDE sampling and GRPO-guided optimization only within the window  |
| **[CPS](https://arxiv.org/abs/2509.05952)** | Policy Gradient | ✅ Supported | A novel sampling formulation that adheres to the Coefficient-Preserving property  |
| **[LongCat-Video](https://arxiv.org/abs/2510.22200)** | Policy Gradient | ✅ Supported |  **Strong performance with multi-reward RLHF** |
| **[DiffusionNFT](https://arxiv.org/abs/2509.16117)** | Reward-conditioned Fine-tuning | 🚧 Coming Soon | Online RL paradigm that optimizes diffusion models directly on the forward process via flow matching |
| **[ReFL](https://arxiv.org/abs/2304.05977)** | Differentiable Reward Optimization | 🚧 Coming Soon | A direct tuning algorithm to optimize diffusion models against a scorer |
| **[DiffusionDPO](https://arxiv.org/abs/2311.12908)** | DPO | 🚧 Coming Soon | Direct Preference Optimization (DPO), a simpler alternative to RLHF which directly optimizes a policy under a classification objective. |

> 💡 *GenRL is designed to be algorithm-agnostic. Adding a new RL algorithm only requires implementing a new trainer — everything else (rewards, data, logging) is reusable. For GRPO-based algorithms, most implementations only need to modify a small amount of code in the trainer.*

---

## 🤖 Supported Models

<!-- TODO: Add / update model entries -->

| Model | Modality | Parameters | Status |
|-------|----------|------------|--------|
| [Wan2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | Text → Video | 1.3B | ✅ Supported |
| [Wan2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) | Text → Video | 14B | ✅ Supported |
| [Wan2.2-T2V](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | Text → Video | 14B | 🚧 Coming Soon |
| [Wan2.2-I2V](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) | Image → Video | 14B | 🚧 Coming Soon |
| [HunyuanImage-3.0-Instruct](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) | Image → Image | 80B | 🚧 Coming Soon |

---

## 🎁 Supported Reward Functions

| Reward | Domain | Source | Description |
|--------|--------|--------|-------------|
| `video_ocr` | 📝 Text | Built-in | OCR accuracy reward — measures text rendering quality via PaddleOCR |
| `hpsv3_general` | 🖼️ Aesthetics | [HPSv3](https://github.com/tgxs002/HPSv3) | Human Preference Score v3 — general aesthetic quality |
| `hpsv3_percentile` | 🖼️ Aesthetics | [HPSv3](https://github.com/tgxs002/HPSv3) | HPSv3 percentile-based reward normalization |
| `videoalign_mq` | 🎬 Motion | [VideoAlign](https://github.com/KwaiVGI/VideoAlign) | Video motion quality assessment |
| `videoalign_ta` | 🎬 Alignment | [VideoAlign](https://github.com/KwaiVGI/VideoAlign) | Video text-alignment score |
| **Custom** | 🔧 Any | User-defined | Bring your own reward via `reward_module` config |

> 🔗 Multiple rewards can be **composed with configurable weights** — GenRL supports both *reward-weighted* and *advantage-weighted* composition modes.

---

## 📊 Performance

<!-- TODO: Fill in actual numbers from your experiments -->

### 🎬 Text-to-Video (Wan2.1-T2V 1.3B)

| Method | HPSv3 ↑ | VideoAlign-MQ ↑ | VideoAlign-TA ↑ | Training Cost |
|--------|---------|-----------------|-----------------|---------------|
| Baseline (pretrained) | — | — | — | — |
| GenRL-GRPO (LoRA) | — | — | — | — |
| GenRL-GRPO (Full FT) | — | — | — | — |

### 📝 Video OCR

| Method | OCR Accuracy ↑ | Levenshtein Score ↑ | Training Cost |
|--------|---------------|---------------------|---------------|
| Baseline (pretrained) | — | — | — |
| GenRL-GRPO (LoRA) | — | — | — |

<!-- ### 🖼️ Text-to-Image -->
<!-- TODO: Add T2I benchmarks when available -->

> 📈 *Performance tables will be updated with results from ongoing experiments. Stay tuned!*

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.10+
- CUDA 12.x + PyTorch 2.6
- 8× A100/H100 GPUs (recommended for video training)

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3️⃣ Setup Reward Model Checkpoints

<details>
<summary>🎬 VideoAlign (for <code>videoalign_mq</code> / <code>videoalign_ta</code> rewards)</summary>

```bash
cd genrl/reward/VideoAlign/checkpoints
git lfs install
git clone https://huggingface.co/KwaiVGI/VideoReward
mv VideoReward/* .
mv VideoReward/.* . 2>/dev/null || true
rm -rf VideoReward
cd ../../../..
```
</details>

<details>
<summary>📝 PaddleOCR (for <code>video_ocr</code> reward)</summary>

```bash
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)"
```
</details>

<details>
<summary>🖼️ HPSv3 (for <code>hpsv3_general</code> / <code>hpsv3_percentile</code> rewards)</summary>

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```
</details>

### 4️⃣ Launch Training

```bash
# Single node, 8 GPUs (LoRA + FSDP)
accelerate launch train.py --config config/default.yaml

# Multi-node (8 nodes × 8 GPUs)
torchrun --nnodes=4 --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py --config config/longcat.yaml
```

---

## 🏗️ Architecture

```
GenRL/
├── 🚀 train.py                        # Entry point
├── 📁 config/                          # YAML configs
│   ├── default.yaml                    #   Default (OCR, FlowGRPO)
│   └── longcat.yaml                    #   Multi-reward, LongCat
├── 📁 genrl/
│   ├── config.py                       # Config schema & loader
│   ├── constants.py                    # Global constants
│   ├── data.py                         # Dataset & dataloaders
│   ├── rewards.py                      # Multi-reward composition
│   ├── advantages.py                   # Advantage computation (GRPO)
│   ├── stat_tracking.py                # Per-prompt stat tracking
│   ├── ema.py                          # EMA wrapper
│   ├── 📁 trainer/
│   │   ├── base_trainer.py             #   Abstract base trainer
│   │   ├── wan_trainer.py              #   Wan model trainer
│   │   ├── sampling.py                 #   Sampling epoch logic
│   │   ├── evaluation.py               #   Eval & video logging
│   │   ├── diffusion.py                #   Log-prob computation
│   │   └── embeddings.py               #   Text embedding utils
│   ├── 📁 reward/
│   │   ├── ocr.py                      #   OCR reward
│   │   ├── hpsv3.py                    #   HPSv3 reward
│   │   ├── videoalign.py               #   VideoAlign rewards
│   │   ├── 📁 HPSv3/                   #   HPSv3 submodule
│   │   └── 📁 VideoAlign/              #   VideoAlign submodule
│   └── 📁 diffusers_patch/
│       └── wan_pipeline_with_logprob.py  # SDE step with log-prob
├── 📁 datasets/                        # Prompt datasets
└── 📁 scripts/
    └── launch.sh                       # Launch script
```

---

## ⚙️ Configuration

All training behavior is controlled by a single YAML file. Key sections:

| Section | What it controls |
|---------|-----------------|
| `reward_fn` | Reward functions & weights (e.g., `video_ocr: 1.0`, `hpsv3_general: 1.0`) |
| `sample` | Sampling: batch size, num steps, guidance scale, SDE type, noise level |
| `train` | Training: learning rate, clip range, advantage clipping, LoRA rank, EMA |
| `accelerate` | Distributed: FSDP, mixed precision, num GPUs/nodes |
| `paths` | Model path, dataset path, save directory, resume checkpoint |

<details>
<summary>📄 Example config (<code>config/default.yaml</code>)</summary>

```yaml
run_name: my_experiment
seed: 42
num_epochs: 100000
height: 240
width: 416
frames: 33

reward_fn:
  video_ocr: 1.0

trainer: wan
use_lora: true

sample:
  batch_size: 8
  num_steps: 20
  guidance_scale: 4.5
  sde_type: flow_sde

train:
  learning_rate: 1.0e-4
  clip_range: 1.0e-3
  lora_r: 32
  ema: true

accelerate:
  distributed_type: FSDP
  mixed_precision: bf16
  num_processes: 8
```
</details>

---

## 📂 Output Structure

```
logs/
└── <experiment>/
    └── <run_name>_<timestamp>/
        ├── 📁 checkpoints/                    # Periodic checkpoints
        │   └── checkpoint-{step}/
        │       ├── ema/                        # EMA states
        │       ├── unwrapped_model/transformer/ # Model weights
        │       └── metadata.json               # Step & config metadata
        ├── 📁 final_model/                    # Final trained model
        │   └── transformer/
        │       ├── adapter_config.json         # LoRA config (if LoRA)
        │       └── adapter_model.safetensors   # LoRA weights (if LoRA)
        ├── 📁 eval_videos/                    # Evaluation videos
        └── 📁 sample_videos/                  # Training sample videos
```

---

## 🔑 Key Features at a Glance

| Feature | Details |
|---------|---------|
| 🎯 RL Algorithm | GRPO with per-prompt stat tracking & advantage clipping |
| 🧬 SDE Types | `flow_sde`, `flow_cps` — unified SDE formulation for rectified flow |
| 🪟 Windowed Training | `sde_window_size` / `sde_window_range` for timestep sub-sampling |
| 📊 Reward Composition | Multi-reward weighted sum, advantage-weighted mode |
| 🧮 KL Regularization | Optional KL reward to constrain policy drift |
| 🎚️ Guidance | Configurable classifier-free guidance for sampling & evaluation |
| 💾 Checkpointing | Periodic + final model saves with FSDP sharded state dict |
| 📈 Logging | WandB integration with training curves, sample videos, eval videos |
| 🔁 EMA | Exponential moving average with configurable decay & update interval |
| 🧩 LoRA | PEFT LoRA with configurable rank, alpha, and target modules |
| 🔒 Reproducibility | Deterministic seeding with `SEED_EPOCH_STRIDE` for all stochastic ops |

---

<!-- ## 📝 Citation -->

<!-- ```bibtex -->
<!-- @article{genrl2026, -->
<!--   title={GenRL: Reinforcement Learning Framework for Visual Generation}, -->
<!--   author={}, -->
<!--   year={2026} -->
<!-- } -->
<!-- ``` -->

## 📝 TODO

- **Model support**
  - Extend support for more text-to-image / image-to-image backbones beyond the current Wan / Hunyuan family
- **Algorithmic extensions**
  - Integrate more **GRPO-family** variants and related online RL algorithms
  - Add DPO / OnlineDPO, SFT / OnlineSFT style objectives alongside GRPO-style training
- **Rollout & parallelism**
  - Integrate **LightX2V** inference framework for accelerated rollout
  - Multi-level parallel rollout (e.g., **SP**, **HSDP**) for better hardware utilization
  - **Asynchronous rollout** workers with decoupled sampling/training pipelines
  - Improved multi-node orchestration utilities and monitoring for large-scale runs

---

## 🙏 Acknowledgements

GenRL is built upon the excellent work of the open-source community. We would like to thank:

- **[Flow-GRPO](https://github.com/yifan123/flow_grpo)** — We reference their implementation for the GRPO-based algorithm and training framework.

---

## 📄 License

GenRL is licensed under the **Apache License 2.0**.  
See `LICENSE.txt` for the full license text.

---

<div align="center">

**If you find GenRL useful, please give us a ⭐!**

</div>
