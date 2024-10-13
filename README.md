# XAI $\times$ Knowledge Distillation


```
nix-shell -p poetry python311
```

We use `peotry` for deps management.

Installing all deps `peotry install`.

Activative env `peotry shell` or run commands via `peotry run ....`

## Available Models
- `cifar10-resnet-p1`
- `cifar100-resnet-p1`


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
