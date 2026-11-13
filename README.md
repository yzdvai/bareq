# Bareq (بارق)

A small image-restoration model for dark, noisy and old (faded, yellowed, grainy) photos. It runs in the browser with ONNX Runtime Web, so photos never leave the device. Demo: https://www.yazeed.quest/bareq

By Yazeed Alhifzi and Yahya Majrashi. Vibe coded with Claude; trained on free Kaggle GPUs.

Work in progress.

- `bareq/model.py`: a NAFNet-style network (after Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022), about 4.8M parameters.
- `tools/export_onnx.py`: exports to ONNX and checks it against PyTorch.
- `web/test.html`: loads the ONNX file on WebGPU and WASM and times one tile.
