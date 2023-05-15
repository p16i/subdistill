import os
import typing
import numpy as np
import numpy.typing as npt

from pathlib import Path

from scipy.stats import ortho_group


import torch
from torch.nn import functional as F
from abc import ABC

from . import learners
from xaikd import utils


BASES = dict()


def register_basis(name):
    """Decorator to register a data modality provider."""

    def wrapped(cls):
        """Wrapped function to register a data modality provider with name `name`"""
        BASES[name] = cls

        return cls

    return wrapped


class Basis(ABC):
    artifact_keys: list
    mean: torch.Tensor

    def __init__(self, alias, centering: bool = True, **kwargs):
        self.centering = centering
        self.alias = alias
        self.artifact: dict

        self.kwargs = kwargs

    def fit(
        self,
        activation: npt.NDArray,
        context: typing.Union[npt.NDArray, None],
        mean: typing.Union[npt.NDArray, None],
        device: str,
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
        pass

    def __str__(self) -> str:
        prefix = self.alias

        suffix = "centered" if self.centering else "uncentered"

        return "--".join([prefix, suffix])

    def save(self, output_dir: Path):
        if hasattr(self, "artifact") is None:
            raise ValueError("Artifact is NONE! Please fit first.")

        output_dir = output_dir / f"{self}"

        os.makedirs(output_dir, exist_ok=True)

        for k, v in self.artifact.items():
            np.save(output_dir / f"{k}", v)

    def load(self, artifact_dir: Path, device="cpu"):
        """_summary_

        Remark: although, artifacts are saved as `numpy.NDArray`, here, for convenience,
        we directly load artifacts as `torch.Tensor`.



        Args:
            artifact_dir (str): _description_
            device (str, optional): _description_. Defaults to "cpu".
        """
        artifact = dict()

        slug = f"{self}"

        for item in self.artifact_keys:
            mat = torch.from_numpy(np.load(artifact_dir / slug / f"{item}.npy")).float()
            mat = mat.to(device)

            artifact[item] = mat

        setattr(self, "artifact", artifact)

        mean = np.load(artifact_dir / "act_mean.npy")

        if not self.centering:
            mean = np.zeros_like(mean)

        self.mean = torch.from_numpy(mean).float().to(device)

    def construct_fh_rank_k_projection(self, k: int) -> typing.Callable:
        """_summary_

        Assumption: this generates a hook for 4d tensors!

        Args:
            k (_type_): _description_

        Returns:
            _type_: _description_
        """
        U = self.artifact["eigvecs"][:, :k]
        mu = self.mean

        if not self.centering:
            assert torch.allclose(mu, torch.zeros_like(mu))

        assert U.shape == (mu.shape[0], k)

        UUT = U @ U.T

        UUT = UUT.unsqueeze(2).unsqueeze(3)

        mu = mu.reshape((1, -1, 1, 1))

        def fh(mod, input, output):
            assert isinstance(output, torch.Tensor)

            projected = F.conv2d(output - mu, UUT)

            return projected + mu

        return fh

    def contruct_rank_d_decoder(self, k: int) -> torch.nn.Module:
        U = self.artifact["eigvecs"][:, :k]

        decoder = torch.nn.Conv2d(k, U.shape[0], kernel_size=1)
        decoder.weight = torch.nn.Parameter(U.unsqueeze(2).unsqueeze(3))
        decoder.bias = torch.nn.Parameter(self.mean.reshape((1, -1, 1, 1)))

        return decoder

    def __str__(self) -> str:
        return getattr(self, "__name")


def get_basis(slug, **kwargs) -> Basis:
    name_slug, centering_slug = slug.split("--")
    centering = True if centering_slug == "centered" else False

    if "random" in name_slug:
        seed = int(name_slug.replace("random", ""))
        basis = BASES["random"](
            alias=name_slug, centering=centering, seed=seed, **kwargs
        )
    else:
        assert centering_slug in ["uncentered", "centered"], f"Value `{centering_slug}`"
        basis = BASES[name_slug](alias=name_slug, centering=centering, **kwargs)

    setattr(basis, "__name", slug)

    return basis


@register_basis("pca")
class PCA(Basis):
    artifact_keys = ["eigvecs", "eigvals"]

    def fit(
        self, activation: npt.NDArray, context: npt.NDArray, mean: npt.NDArray, **kwargs
    ):
        """_summary_

        Args:
            activation (npt.NDArray): _description_
            context (npt.NDArray): We do NOT use this here!
        """

        n, d = activation.shape

        if self.centering:
            activation = activation - mean

        eigvals, eigvecs = np.linalg.eigh(activation.T @ activation / n)

        indices = np.argsort(eigvals)[::-1]

        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, eigvals)))

        return eigvecs, eigvals


@register_basis("prca")
class PRCA(Basis):
    artifact_keys = ["eigvecs", "eigvals"]

    def fit(
        self, activation: npt.NDArray, context: npt.NDArray, mean: npt.NDArray, **kwargs
    ) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """_summary_ Summary

        Args:
            activation (npt.NDArray): _description_
            context (npt.NDArray): _description_

        Returns:
            typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: _description_
        """
        n, d = activation.shape

        if self.centering:
            activation = activation - mean

        eigvals, eigvecs = np.linalg.eigh(
            ((activation.T @ context + context.T @ activation)) / n
        )

        indices = np.argsort(eigvals)[::-1]

        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, eigvals)))

        return eigvecs, eigvals


@register_basis("random")
class Random(Basis):
    def load(self, artifact_dir: str, device="cpu"):
        """_summary_

        Remark: although, artifacts are saved as `numpy.NDArray`, here, for convenience,
        we directly load artifacts as `torch.Tensor`.



        Args:
            artifact_dir (str): _description_
            device (str, optional): _description_. Defaults to "cpu".
        """

        mean = torch.from_numpy(np.load(Path(artifact_dir) / "act_mean.npy")).float()

        if not self.centering:
            mean = torch.zeros_like(mean)

        mean = mean.to(device)

        d = mean.shape[0]

        seed = self.kwargs["seed"]

        np.random.seed(seed)

        mat = ortho_group.rvs(d)
        mat = torch.from_numpy(mat).float()
        mat = mat.to(device)

        setattr(self, "artifact", dict(eigvecs=mat))

        self.mean = mean


class PRCAVariant(Basis):
    artifact_keys = ["eigvecs"]
    mode: str

    def fit(
        self, activation: npt.NDArray, context: npt.NDArray, mean: npt.NDArray, **kwargs
    ) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """_summary_ Summary

        Args:
            activation (npt.NDArray): _description_
            context (npt.NDArray): _description_

        Returns:
            typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: _description_
        """
        _, d = activation.shape

        if not self.centering:
            mean = np.zeros(d)

        activation = activation - mean

        learner = learners.PRCAGreedyLeaner(mode=self.mode)

        U = learner.fit(activation, context, **kwargs)

        self.artifact = dict(zip(self.artifact_keys, (U, mean)))

        return U, None


@register_basis("prca-abs")
class PRCAAbs(PRCAVariant):
    mode = "abs"


@register_basis("prca-recon")
class PRCARelRecon(PRCAVariant):
    mode = "recon"
