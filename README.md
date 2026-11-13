# Bareq (بارق)

A small image-restoration model for dark, noisy and old (faded, yellowed, grainy) photos. It runs in the browser with ONNX Runtime Web, so photos never leave the device. Demo: https://www.yazeed.quest/bareq

By Yazeed Alhifzi and Yahya Majrashi. Vibe coded with Claude; trained on free Kaggle GPUs.

Work in progress.

- `bareq/model.py`: a NAFNet-style network (after Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022), about 4.8M parameters.
- `bareq/prepare.py`, `notebooks/01_data.ipynb`: builds the train, val and test data. Splits come first, LOL v1/v2 duplicates and same-scene pairs are removed from training, SIDD is split by scene.
- `bareq/degrade.py`: the synthetic "old photo" and noise effects.
- `bareq/train.py`, `notebooks/02_train.ipynb`: one training script for the four runs (dark, noisy, old, all). Checkpoints every 15 minutes and resumes exactly.
- `bareq/metrics.py`: PSNR, SSIM and the tiled inference the web page also uses.
- `tools/export_onnx.py`, `web/test.html`: ONNX export and the browser check on WebGPU and WASM.

Data: LOL v1 and v2 (Wei et al. 2018; Yang et al. 2021), SIDD (Abdelhamed et al. 2018), DIV2K (Agustsson and Timofte 2017).
