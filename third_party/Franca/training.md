## Training Details

### Data

- **Environment Setup:** Follow the instructions in the [`README`](README.md#installation) to configure the training environment.
- **Datasets:** We use ImageNet-21K (winter release) and LAION-600M (LAION-COCO). For details on dataset preparation, refer to [`dataset_prep.md`](dataset_prep.md).

### Model Architectures

- **ViT-B (86M parameters):**
  Patch size: 14 &nbsp;|&nbsp; Embedding dimension: 768 &nbsp;|&nbsp; Attention heads: 12 &nbsp;|&nbsp; MLP: SwiGLU
- **ViT-L (0.3B parameters):**
  Patch size: 14 &nbsp;|&nbsp; Embedding dimension: 1024 &nbsp;|&nbsp; Attention heads: 16 &nbsp;|&nbsp; MLP: SwiGLU
- **ViT-g (1.1B parameters):**
  Patch size: 14 &nbsp;|&nbsp; Embedding dimension: 1536 &nbsp;|&nbsp; Attention heads: 24 &nbsp;|&nbsp; MLP: SwiGLU


### Configuration

- **Precision:**
  - Backbone: `fp16` with FSDP mixed precision
  - Projection heads: `fp32`

- **Optimizer:** AdamW
- **Batch Size:** 2048 (Base), 3072 (Large, Giant)
- **Learning Rate:** 1e-3 (Base), 3.5e-4 (Large, Giant)
- **Stochastic depth regularization:** 0.1 (Base), 0.4 (Large, Giant)
- **Training Iterations:** 625K iterations
- **Warmup:** 100K iterations


### Pretraining Strategy

- **Objective:** Models are pretrained on ImageNet-21K and LAION-600M.
- **Launch Instructions:**
  To start training a model variant, use one of the scripts under:

```bash
scripts/train/IN21K/franca-<variant>.sh
scripts/train/LAION/franca-<variant>.sh
```

To ensure full reproducibility, please ensure your training follows the one as shown in our log files

| Model     | Dataset | # GPUs | Logfile                                               |
|-----------|---------|--------|--------------------------------------------------------|
| DINOv2-B  | IN21K   | 32     | [dinov2_B.txt](logfiles/In21K/dinov2_B.txt)               |
| DINOv2-L  | IN21K   | 64     | [dinov2-L.txt](logfiles/In21K/dinov2_L.txt)               |
| Franca-B  | IN21K   | 32     | [Franca-B.txt](logfiles/In21K/Franca_B.txt)               |
| Franca-L  | IN21K   | 64     | [Franca-L.txt](logfiles/In21K/Franca_L.txt)               |
| Franca-G  | IN21K   | 128     | [Franca-L.txt](logfiles/In21K/Franca_G.txt)               |
| Franca-L  | LAION-600M   | 64     | [Franca-L.txt](logfiles/In21K/Franca_L_LAION.txt)               |
| Franca-G  | LAION-600   | 128     | [Franca-L.txt](logfiles/In21K/Franca_G_LAION.txt)               |
