# Distilling Lightweight Domain Experts from Large ML Models by Identifying Relevant Subspaces

Authors: Pattarawat Chormai, Ali Hashemi, Klaus-Robert Müller, Grégoire Montavon

[![arXiv](https://img.shields.io/badge/arXiv-2601.05913-b31b1b.svg)](https://arxiv.org/abs/2601.05913)

--- 
The repo contains code for the manuscript above. 

Distillation experiments in the paper were run with tag `v0.8.16`, while the XAI Analysis was performed the code from branch `add-captum`.




# Requirements 
1. `wandb` service setup
2. Python environment with dependencies in `pyproject.toml`.
  In our case, we construct the enviorment using Apptainer (via `./containers/py311.def`)



# Usage 

```
python ./scripts/distill-layerwise.py  \
  --output-dir /tmp \
  --seed 1 \
  --epochs 100   \
  --training-size 0.8 \
  --lambda-layer 100  \
  --batch-size 32 \
  --dataset imagenet-wading-bird \
  --teacher imagenet-resnet101-tv  \
  --student student-mobilenetv4-small \
  --layers layer1:blocks.1.1,layer2:blocks.2.3,layer3:blocks.3.1,layer4:blocks.3.5 \
  --distillation-policy <POLICY>
```
*Remark:* in our case, we run the script in the  Apptainer environment (via `./runpy`).

Possible options for `<POLICY>` 
```
- basis-center-rotationv2:prcaposdef # this is the name of our SubDistill policy in the code
- attention-transfer
- vid
- vkd
```



---
Please do not hesitate to contact me or create an issue if there is any questions.