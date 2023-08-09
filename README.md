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
