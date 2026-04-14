#!/usr/bin/env bash
set -euo pipefail

python -m pip install "git+https://github.com/JonaRuthardt/SteerViT.git"
python -m pip install -e .
echo "SteerViT and TopoSteer scaffold installed."
