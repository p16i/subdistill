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
  - [x] prca-abs,recon,reg with the same fit but adding assersion
- experiment: beta
  - eps: `1.0`
  - basis: `pca,prca-abs,prca-reconreg0.0,prca-reconreg0.001,prca-reconreg0.01,prca-reconreg0.1,prca-reconreg1.0,prca-reconreg10.0,prca-reconreg100.0,prca-reconreg1000.0`
    - run `249288_ (eps1.0), 249303_(eps10.0), 249308_ (eps0.1)`
- why stderr is high? how to reduce variance? is the model not that stable? 
  - this large stderr would be larger in  gaussian blobs
- new toy dataset: mixture of gaussian
  - increase epochs from 20 to 200
- [x] Experiment: compression \propto eps
  - cases: covdiag=True, covdiag=False
  - eps={0.75, 1.0} (1 seems to high)
- [x] Add rel basis
- [x] not training new model if `lighting_log` exists
- [x] Experiment: comparison between pca,prca-abs,prca-recon,rel
  - data: covDiag=False, eps=0.5,0.75, 1.0
  - basis-choice:
    - [x] `rel,rel-abs`, 

    - `rel,rel-abs,prca-reconreg0.0,prca-reconreg0.001,prca-reconreg0.01,prca-reconreg0.1,prca-reconreg1.0,prca-reconreg10.0,prca-reconreg100.0,prca-reconreg1000.0`
      - [x] job: `249927_, 249922_, 249917_`
- [x] proof of concept for kernel and approximation
  - MLP
    - training
    - extract activation
    - training svm
    - svm direction from samples
  - [x] to what extent we can use kernel to approximate module in DNNs?

### Sprint 8 (2023/06/13)
- [] toy dataset 10 seeds:
  - ./logs/array/254632_1
    - [] error bar on beta sweeping
- [] Toy Dataset (100 classes)
  - ./logs/array/253593_1.out
    - [] the results quite different why?
- [] approximation
  - focus: layer3 and layer4
  - approximation module: only one residue block
  - unittests:
    - test that feature_extractor, classification head the same output
  - trial experiments
    - layer3
      - n={600}, compression=0.1 (264765_*), compression=0.01 (264802_*)
        - epochs=100 and compression=0.01 (265266)
      - n={60}, compression=0.1 (264807_*), compression=0.01 (264844_*)
        - epochs=100 and compression=0.01 (265156); 
      - n={6}, compression=0.01 (265151_*)
        - epochs=100 and compression=0.01 (done)
    - layer4
      - n={60}, compression=0.01 (264839)

    - imagenet (imagenet-385vs386):
      - layer3 : n={100,  python ./scripts/baseline.py --epochs 31000} (epochs=100); compression 0.1, 0.01
        - n=100; compression 0.1 (268036*) 
        - n=100; compression 0.01 (268033*)
        - n=10; compression 0.1 (268445*)
        - n=10; compression 0.01 (268449*)
    - baseline training
      - cifar100-35vs98  (268523*) 
      - imagenet-385vs386 (268528*) 

    - baseline training with pretrained
      - cifar100-35vs98  (268552*) 
      - imagenet-385vs386 (n10,100: 268564*) (n1000: 269070_)

  - implementation cifar100 with coarse labels
    - [x] implement cifar100 subdataset
      - unittest
    - [x] check accuracy of coarse labels
    - [x] logit modifier
    - [ ] question: does model make mistakes across different superclasses
      - if not, pca migth be as good as prca.
    - experiment: (2023-06-s8/distill-superclass, layer=layer3)
      - [x] n600: compr=0.1 (270755*) 
        - [x] expected: all bases reach acc=0.5x
          - yes, indeed that is the case
      - [x] n600: compr=0.01 (271516) 

      - [x] n60: compr=0.1 (270757*)
        - [x] expected: pca or prca have highest acc
      - [x] n60: compression0.25 (271018*)
      - [x] n60: compression0.05 (270758*)
      - [x] n60: compression0.01 (270759)
    - experiment: (2023-06-s8/distill-superclass, layer=layer4)
      - [x] n500: compr=0.1 (271567*) 
      - [x] n500: compr=0.01 (271572*)
      - [x] n50: compr=0.1 (271602*)
      - [x] n50: compression0.25 (271743*)
      - [x] n50: compression0.05 (271742*)
      - [x] n50: compression0.01 (271597*)
    - hypotheses:
      - few samples (overfitting) -> poor accuracy
      - with good basis -> good accuracy with few samples
  - baseline from sctach
    - [x] 271522*
  - accuracy basis:
    - [x] cifar100-people 271985
  - logit modifier: comparision between target label or selected classes
    - [x] q: which one is better? (next step)
        - oneclass is better!
    - [x] try to vary number of training samples
  - experiment: (2023-06-s8/distill-superclass-wd, layer=layer3)
    - goal: investigate the effect of overfitting.
    - [ ] n50, compr=0.01, weight-decay=0.0 (272836*)
    - [ ] n50, compr=0.01, weight-decay=0.1 (272838*)
  - [ ] experiment: accuracy_basis (cifar100-*)
    - goal: see whether PRCA-abs is better than PCA.
    ```
     SEEDFILE=./xaikd/resources/cifar100-datasets.txt sbatch -p cpu-5h --array=1-20  ./slurm/job_array_cpu.sh ./runpy ./scripts/accuracy_basis.py --model cifar100-resnet18-p1 --output-dir ./artifacts/2023-06-s8/accuracy --logit-modifier oneclass --num-training-samples 50 --dataset {}
    ```
    - job: `273800`

  - eigenvector of classes in people
  - visualization
      - teacher
      - teacher w/ bottle neck
       - distilation with different basis
  - implementation inaturlist


- [] activation for different compression is the same
- [] write torchvision.dataset for UFI

### Backlog
- [] refactor
  - the way we construct loader
  - projector in basis
- [] rerun some experiment again!
- [] implementing baseline for TPAMI, Interpolative, ...