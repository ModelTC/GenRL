# VideoGRPO

## Environment
1. Python 3.10+ (use an isolated virtualenv/conda env).
2. Install base deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize and update submodules (for `videoalign_mq` and `videoalign_ta` rewards):
   ```bash
   git submodule update --init --recursive
   ```
4. Setup VideoAlign checkpoints (required for `videoalign_mq` and `videoalign_ta` rewards):
   ```bash
   cd video_grpo/reward/VideoAlign/checkpoints
   git lfs install
   git clone https://huggingface.co/KwaiVGI/VideoReward
   # Move all files from VideoReward to checkpoints directory
   mv VideoReward/* .
   mv VideoReward/.* . 2>/dev/null || true  # Move hidden files, ignore errors if none exist
   # Remove the empty VideoReward directory
   rmdir VideoReward
   cd ../../../..  # Return to project root
   ```
5. OCR extras (for `video_ocr` reward):
   ```bash
   pip install paddlepaddle-gpu==2.6.2
   pip install paddleocr==2.9.1
   pip install python-Levenshtein
   # pre-download OCR model
   python - <<'PY'
   from paddleocr import PaddleOCR
   ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
   PY
   ```

## Run
Single node example (default `config/default.yaml`, LoRA + FSDP):
```bash
accelerate launch train.py --config config/default.yaml
```

## Directory Structure

After training, the output directory structure is organized as follows:

```
logs/
└── video_ocr/
    └── wan_flow_grpo_2026.01.12_21.48.08/    # Run directory (save_dir/run_name)
        ├── checkpoints/                       # Training checkpoints
        │   └── checkpoint-{step}/
        │       ├── ema/                       # EMA states (if enabled)
        │       ├── unwrapped_model/
        │       │   └── transformer/           # Unwrapped model weights
        │       └── metadata.json              # Checkpoint metadata
        ├── final_model/                       # Final model after training completes
        │   └── transformer/                  # Final model weights
        │       ├── adapter_config.json        # LoRA config (if using LoRA)
        │       └── adapter_model.safetensors  # LoRA weights (if using LoRA)
        │       # OR full transformer weights (if full finetune)
        ├── eval_videos/                       # Evaluation videos
        └── sample_videos/                     # Training sample videos
```

**Notes:**
- The run directory name includes a timestamp: `{run_name}_{YYYY.MM.DD_HH.MM.SS}`
- `checkpoints/` contains periodic checkpoints saved during training
- `final_model/` contains the final trained model:
  - For LoRA training: Only LoRA adapter weights are saved
  - For full finetune: Complete transformer weights are saved
- Videos are saved in `eval_videos/` and `sample_videos/` directories

## Notes

- All config is YAML-driven (`config/default.yaml`): FSDP, LoRA/full finetune, rewards, data paths, etc.
- Train/eval rewards can differ via `reward_fn/reward_module` and `eval_reward_fn/eval_reward_module`.
- All outputs (checkpoints, videos, final model) are saved under `paths.save_dir/run_name/`:
  - Checkpoints: `paths.save_dir/run_name/checkpoints/checkpoint-{step}/`
  - Videos: `paths.save_dir/run_name/eval_videos/` and `sample_videos/`
  - Final model: `paths.save_dir/run_name/final_model/`

