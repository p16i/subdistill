from dataclasses import dataclass
from xaikd import approximators


@dataclass
class ExperimentConfiguration:
    basis_name: str
    compression_ratio: float
    approximator_mode: approximators.ApproximatorMode


@dataclass
class LayerDistillInfo:
    layer_name: str
    num_input_channels: int
    num_output_channels: int


def get_distill_infor(
    arch: str, layer: str, compression_ratio: float
) -> LayerDistillInfo:
    assert arch == "cifar100-resnet18-p1" or arch == "imagenet-resnet18-tv"
    assert compression_ratio >= 1.0

    info = dict(
        zip(
            ["layer3", "layer4"],
            [
                LayerDistillInfo(
                    layer_name="layer3",
                    num_input_channels=128,
                    num_output_channels=approximators.compute_compressed_dimension(
                        256, compression_ratio
                    ),
                ),
                LayerDistillInfo(
                    layer_name="layer4",
                    num_input_channels=256,
                    num_output_channels=approximators.compute_compressed_dimension(
                        512, compression_ratio
                    ),
                ),
            ],
        )
    )

    return info[layer]
