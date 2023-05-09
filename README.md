# XAI $\times$ Knowledge Distillation

We use `peotry` for deps management.

Installing all deps `peotry install`.

Activative env `peotry shell` or run commands via `peotry run ....`

## Available Models
- `cifar10-resnet-p1`
- `cifar100-resnet-p1`


## Coding

### Sprint 5
- [x] bases implementation (via ABC?)
- [x] activation subtractions (via Zennits); 
  - we need to define layers to investigate
- [x] setup cluster ml-server
- [x] implement PRCA-variants bases
    - [x] PRCA abs, recon
    - [x] add tests for learner
- [x] use `<dataset>-<arch>-<variant>` for model
- [x] Implement experiments for Invidiual Layers
    - [x] projection in each basis class
    - [x] layer dimension
    - [x] need to check how long does it takes for each run?
    - [x] random basis
- [x] make `act_mean` save in the root!
- [x] run `extract` and `accuracy` scripts (CIFAR100, 4 layers)
- [x] verify accuracies via jupyter notebook
- [x] two-classes analysis (focus on CIFAR100)
  - [x] implement ModelType and DatasetType; check pass_ctx that works with this.
  - [x] run hard pairs
  - [x] refactor colab notebook and compare results!
  - [x] verify auroc calculation
  - [x] run with random bases
- [] two-class with imagenet (monarch vs ...)
    - [x] ringlet (322) vs {monarch (323), notebook (681)}: https://github.com/p16i/concept-xai/blob/dev/cxai/config/imagenet-label-mapping.csv#L324
    - [x] where imagenet are in the cluster!; it is at `/home/space/datasets`
        - actually, for torchvision, it is at `/home/space/datasets/imagenet_torchvision/data`
        - [] check with Lorenz how to use imagenet on the cluster
    - [x] cross-check dev run cifar100-35vs98
      ```
      diff ./tmp/dev/cifar100-35vs98/cifar100-resnet18-p1/layer2/pca--centered/stats.json ./artifacts/2023-05-S5/experiment-binary-task/cifar100-35vs98/cifar100-resnet18-p1/layer2/pca--centered/stats.json 
      ```
    - [] why auroc is very high? 
      - [] confusion mat for butterfly classes
    - [] cross-check everything again before running experiments!

   
- [] grafting vs layerwise
  - remark: don't fotget shuffle and frozen parameters.
  - [] baseline: train from stach; hinston disllation


**Remarks**
- `prca-abs` is very sensitive to `eps`.
- `prca-recon` is very slow. Why?

### Sprint 6
- [] implementing baseline for TPAMI, Interpolative, ...
- [] ...

