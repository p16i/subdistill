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
- [] two-classes analysis (focus on CIFAR100)
  - [x] implement ModelType and DatasetType; check pass_ctx that works with this.
  - [] verify auroc calculation
  - [] run hard pairs
  - [] refactor colab notebook and compare results!
   
- [] grafting vs layerwise


**Remarks**
- `prca-abs` is very sensitive to `eps`.
- `prca-recon` is very slow. Why?

### Sprint 6
- [] implementing baseline for TPAMI, Interpolative, ...
- [] ...

