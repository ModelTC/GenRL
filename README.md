# VideoGRPO

## Environment
1. Python 3.10+ (use an isolated virtualenv/conda env).
2. Install base deps:
   ```bash
   pip install -r requirements.txt
   ```
3. OCR extras (for `video_ocr` reward):
   ```bash
   pip install paddlepaddle-gpu==2.6.2
   pip install paddleocr==2.9.1
   pip install python-Levenshtein
   # optional: pre-download OCR model
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

Notes
- All config is YAML-driven (`config/default.yaml`): FSDP, LoRA/full finetune, rewards, data paths, etc.
- Train/eval rewards can differ via `reward_fn/reward_module` and `eval_reward_fn/eval_reward_module`.
- Checkpoints: `paths.save_dir/checkpoints/checkpoint-*`; logs & videos: `paths.logdir/run_name/`.

