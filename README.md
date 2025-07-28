# XAI $\times$ Knowledge Distillation

This repository implements various knowledge distillation (KD) frameworks combined with explainable AI (XAI) techniques. It provides implementations of state-of-the-art KD methods for model compression and knowledge transfer.

## 📚 Knowledge Distillation Frameworks

This repository implements **7 major knowledge distillation frameworks**:

1. **KD** (Hinton et al., 2015) - Classic knowledge distillation
2. **FitNet** (Romero et al., 2015) - Hint-based intermediate layer transfer
3. **Attention Transfer** (Zagoruyko & Komodakis, 2017) - Spatial attention maps
4. **VID** (Ahn et al., 2019) - Variational information distillation
5. **SPKD** (Tung & Mori, 2019) - Similarity-preserving knowledge distillation
6. **DKD** (Zhao et al., 2021) - Decoupled knowledge distillation
7. **VkD** (Miles et al., 2024) - Orthogonal projections

### 🚀 Quick Start Guide

- **[Quick Reference](docs/kd_frameworks_quick_reference.md)** - Framework selection and implementation tips
- **[Detailed Comparison](docs/knowledge_distillation_frameworks_comparison.md)** - Comprehensive pros/cons analysis

### 🎯 Framework Selection

| Use Case | Recommended Framework |
|----------|----------------------|
| **Beginner/Baseline** | KD (Hinton 2015) |
| **Best Performance** | VkD (Miles 2024) |
| **Classification** | DKD (Zhao 2021) |
| **Computer Vision** | Attention Transfer |
| **Architecture Transfer** | FitNet |

## 🛠️ Setup

```bash
nix-shell -p poetry python311
```

We use `poetry` for dependency management.

Installing all dependencies: `poetry install`.

Activate environment: `poetry shell` or run commands via `poetry run ....`

## Available Models
- `cifar10-resnet-p1`
- `cifar100-resnet-p1`

## 📖 Documentation

### 📚 [Complete Documentation](docs/README.md)

**Quick Access:**
- **[Quick Start Guide](docs/kd_frameworks_quick_reference.md)** - Framework selection and tips
- **[Detailed Comparison](docs/knowledge_distillation_frameworks_comparison.md)** - Comprehensive analysis
- **[Technical Analysis](docs/kd_frameworks_technical_analysis.md)** - Research and production insights

# Resources
- notebook to train teacher models for cifar100
  - https://colab.research.google.com/drive/13NNSnXyRuN4vti22kKE2-vpQpTFGO0ta#scrollTo=Xsk-KCxTf07F&uniqifier=1
    *Remark* currently, it contains key for wandb, and it should not be shared.


## Things to Do/Check when adding a new model
- make the architecture `Ditsllable`
  - check prediction
  - update `constants` for layer dimension
- implement generator
- implement attributors
   - check attributor



## Apptainer

Unittests
```
apptainer run --bind /home/space/datasets/cifar100:/datasets  --nv containers/main.sif poetry run pytest tests/*
```

todo:
- Figure out how to run the script with apptainers!
