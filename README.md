# XAI $\times$ Knowledge Distillation

We use `peotry` for deps management.

Installing all deps `peotry install`.

Activative env `peotry shell` or run commands via `peotry run ....`

## Available Models
- `cifar10-resnet-p1`
- `cifar100-resnet-p1`


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
