import torch

from . import register_model

# from robustbench.utils import load_model
from xaikd import utils


@register_model("cifar100-modas2021-robustbench")
def _Modas2021PRIMEResNet18():
    model = utils.robustbench.load_model(
        model_name="Modas2021PRIMEResNet18",
        dataset="cifar100",
        threat_model="corruptions",
    )

    # we handle this in dataloader
    model.register_buffer("mu", torch.tensor(0.0))
    model.register_buffer("sigma", torch.tensor(1.0))

    setattr(model, "num_classes", 100)
    setattr(model, "__last_layer", model.linear)

    return model
