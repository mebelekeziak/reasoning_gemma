# Gemma 3 12B math + logic finetuning with Prime RL

This folder wires a tiny-budget pipeline: LoRA SFT to warm up Gemma 3 (12B), then RL with Prime Intellect environments (`vf-math-python` for math, `kalomaze/alphabet-sort` as a lightweight logic puzzle). Both stages checkpoint aggressively so you can resume on a 2×H200 box.

## Prereqs
- Python 3.10+ and `uv` (or pip) installed.
- Two GPUs (H200 x2) on one node; config pins inference to GPU0 and trainer to GPU1.
- Hugging Face auth set (`huggingface-cli login`) for Gemma weights.

## Install
```powershell
uv venv
uv pip install -e .
# pull environments from the hub (existing, no custom code)
uvx vf-install --env vf-math-python
uvx vf-install wordle --from-repo
```
Environment install helper uses the hub-backed IDs described in Prime’s env docs. citeturn2search1turn2search9

## Stage 1 — LoRA SFT (cheap warm‑start)
```powershell
uv run python scripts/run_sft.py `
  --model-name google/gemma-3-12b-it `
  --dataset gsm8k `
  --max-steps 400 `
  --save-steps 50 `
  --output-dir checkpoints/sft-gemma3-12b
```
- Uses `trl.SFTTrainer` + PEFT LoRA (r=16, alpha=32) in bf16.
- Checkpoints every 50 steps so you can resume: `--resume-from checkpoints/sft-gemma3-12b/checkpoint-150`.

## Stage 2 — RL on math + logic environments
Config: `configs/rl/math_logic.toml` (shared by orchestration, trainer, and inference).
Key bits:
- `inference_gpu_ids=[0]`, `trainer_gpu_ids=[1]` to split work across the two H200s.
- `ckpt.interval=50` writes `checkpoints/rl/step-*.pt`, keeping the last four. Prime’s ckpt flags are documented in the training guide. citeturn1search0
- `lora_path="checkpoints/sft-gemma3-12b"` seeds RL with the SFT adapters. citeturn2search2
- Environments: `vf-math-python` (math) and `wordle` (logic puzzle), both installed from the hub/repo. citeturn1search2turn2search9

Run:
```powershell
uv run rl `
  --trainer @configs/rl/math_logic.toml `
  --orchestrator @configs/rl/math_logic.toml `
  --inference @configs/rl/math_logic.toml
```
Resume after pre-emption without losing steps:
```powershell
uv run rl `
  --trainer @configs/rl/math_logic.toml `
  --orchestrator @configs/rl/math_logic.toml `
  --inference @configs/rl/math_logic.toml `
  --ckpt.path checkpoints/rl `
  --ckpt.resume true
```

## Notes for low budget
- If VRAM is tight, drop `orchestrator.batch_size` to 128 and `rollouts_per_example` to 4.
- To shorten runs, cut `max_steps` in the config; checkpoints still rotate.
- You can offline log runs by setting `WANDB_MODE=offline` before launching.

## File map
- `scripts/run_sft.py` – LoRA SFT driver with frequent checkpointing.
- `configs/rl/math_logic.toml` – Prime RL config (math + logic envs, LoRA init, ckpt cadence).
