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
- [x] two-class with imagenet (monarch vs ...)
    - [x] ringlet (322) vs {monarch (323), notebook (681)}: https://github.com/p16i/concept-xai/blob/dev/cxai/config/imagenet-label-mapping.csv#L324
    - [x] where imagenet are in the cluster!; it is at `/home/space/datasets`
        - actually, for torchvision, it is at `/home/space/datasets/imagenet_torchvision/data`
        - [] ~~check with Lorenz how to use imagenet on the cluster~~
    - [x] cross-check dev run cifar100-35vs98
      ```
      diff ./tmp/dev/cifar100-35vs98/cifar100-resnet18-p1/layer2/pca--centered/stats.json ./artifacts/2023-05-S5/experiment-binary-task/cifar100-35vs98/cifar100-resnet18-p1/layer2/pca--centered/stats.json 
      ```
    - [] why auroc is very high? 
      - [] ~~confusion mat for butterfly classes~~
    - [x] cross-check everything again before running experiments!
    - Experiment 1: Binary Task from `cifar100` and `imagenet`

      **Conclusion**
        - `prca-abs` shows promising trends, especially layer3.
          - for early layer, not much different; perhaps, generic features?
        - there is a klink in the accuracy plot. Why?
        - `prca` (perhaps `prca-abs` as well) is sensitive to the way we compute `logodd`; order matter; see`imagenet-324vs325` vs `imagenet-325vs324`
    - Experiment 2: influence of number training samples used.


      **Hypothesis:** incorporting relevance allows us to find a meaningful subspace with much less data.

      **Conclusion:**

      - The current results do NOT seem to indicate that. 
      - (Ali's comment) what is the intuition that incorporating attribution signal could help reduce sample complexity.

      *Coding* (`branch=s5-binary-task-few-short`)
      - [x] generalize subset dataset, output dir `imagenet-2vs5--n20`;
#### Sprint 5.2
- [] comparison between `prca-recon` and others (`branch: fix-prca-recon`)
  - [x] bug fix  
  - [x] sweeping also 0,1,2,3 + range(4, ...)
  - [x] add `prca-recon` in the list!
  
  Experiment: cifar100, imagenet

  Conclusion: it seems that `prca-recon` behaves quite similar to `prca-abs`.

- [x] grafting vs layerwise
  - remark: don't fotget shuffle and frozen parameters.
  - [x] baseline: train from stach; hinston disllation
  - **Remark:** This needs to be revisit.

- `prca-recon` is very slow. Why? 
  - well,  it is slow because we forgot to neglect the sign in the objective

**Remarks**
- `prca-abs` is very sensitive to `eps`.

### Sprint 6
- [x] bug fixed on AUROC quantity
- [x] rerun imagenet class
  - [x] compare results
- [] toy problem on merged dataset (5h?)
  - [x] prototype in jupyter
  - [x] Parameters
        - dataset generation:
          - `eps` controls hardness
          - `seed`
          - `sample_per_dataset`
        - model (2-Layer MLP): model size
  - [x] write code to rerun with multiple seeds and architecture?
    - [x] each run need take parameterized by `seed` and `eps`, 
        - we have an inter loop in the script to run all `models`, `layers`
        - artifact
          - dataset
          - curves at different `k`
          - (jupyter) decision boundary of teacher
          - (juypter) decision boundary at `k={1, 2, 3, 4, 5}`
    - (next step): rerun five seeds on cluster
  - [x] check results on cluster:
    - eps: `1.0`: `mlp64,128,256,512`
    - model: `mlp64`; eps: `0.5,0.1`
  - [x] check result `uncentered` (`mlp64`, `eps=0.1`)
  - [] add regularizer to prca-recon
    - [x] trial run: with `eps0.1`; `dir=...`
    - [ ] check w/ seeds (`job: 243545_1`)

### Sprint 7 (2023/06/01)
- refactor
  - [] prca-abs,recon,reg with the same fit but adding assersion
- experiment: beta
  - 
- new toy dataset: mixture of gaussian
  - 


### Backlog
- [] refactor
  - the way we construct loader
  - projector in basis
- [] rerun some experiment again!
- [] implementing baseline for TPAMI, Interpolative, ...