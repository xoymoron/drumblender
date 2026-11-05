# Environment

Use Python 3.10–3.12 in a virtual environment. Run commands from the repository root.

| Environment | Install |
| --- | --- |
| CPU development and evaluation | `python -m pip install -r requirements/local.txt` |
| Linux server with CUDA 12.6 support | `python -m pip install -r requirements/server.txt` |

The server profile includes the optional `train` dependencies for the
joint training and export entry points. The core package supports preprocessing,
modal extraction, and WAV evaluation without Lightning. Both profiles include
the tools for lint and the focused data and analysis tests.

The Linux CUDA PyTorch distribution supplies Triton for the fused FP32 modal
block carry. CPU rendering uses the PyTorch carry implementation.
