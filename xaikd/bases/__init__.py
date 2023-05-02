import os
import typing
import numpy as np
import numpy.typing as npt

from scipy.stats import ortho_group


import torch
from torch.nn import functional as F
from torch.utils import hooks
from abc import ABC

from . import learners


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

    def __init__(self, alias, centering: bool = True, **kwargs):
        self.centering = centering
        self.alias = alias
        self.artifact: dict

        self.kwargs = kwargs

    def fit(
        self, activation: np.ndarray, context: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def __str__(self) -> str:
        prefix = self.alias

        suffix = "centered" if self.centering else "uncentered"

        return "--".join([prefix, suffix])

    def save(self, output_dir: str):
        if hasattr(self, "artifact") is None:
            raise ValueError("Artifact is NONE! Please fit first.")
        output_dir = f"{output_dir}/{self}"

        os.makedirs(output_dir, exist_ok=True)

        for k, v in self.artifact.items():
            np.save(f"{output_dir}/{k}", v)

    def load(self, artifact_dir: str, device="cpu"):
        """_summary_

        Remark: although, artifacts are saved as `numpy.NDArray`, here, for convenience,
        we directly load artifacts as `torch.Tensor`.



        Args:
            artifact_dir (str): _description_
            device (str, optional): _description_. Defaults to "cpu".
        """
        artifact = dict()
        for k in self.artifact_keys:
            mat = torch.from_numpy(np.load(f"{artifact_dir}/{self}/{k}.npy")).float()
            mat = mat.to(device)

            artifact[k] = mat

        setattr(self, "artifact", artifact)

    def construct_fh_rank_k_projection(self, k: int) -> typing.Callable:
        """_summary_

        Assumption: this generates a hook for 4d tensors!

        Args:
            k (_type_): _description_

        Returns:
            _type_: _description_
        """
        U = self.artifact["eigvecs"][:, :k]
        mu = self.artifact["mean"]

        assert U.shape == (mu.shape[0], k)

        UUT = U @ U.T

        UUT = UUT.unsqueeze(2).unsqueeze(3)

        mu = mu.reshape((1, -1, 1, 1))

        def fh(mod, input, output):
            assert isinstance(output, torch.Tensor)

            projected = F.conv2d(output - mu, UUT)

            return projected + mu

        return fh


def get_basis(name, **kwargs) -> Basis:
    name, centering_slug = name.split("--")
    centering = True if centering_slug == "centered" else False

    if "random" in name:
        seed = int(name.replace("random", ""))
        return BASES["random"](alias=name, centering=centering, seed=seed, **kwargs)
    else:
        assert centering_slug in ["uncentered", "centered"], f"Value `{centering_slug}`"
        return BASES[name](alias=name, centering=centering, **kwargs)


@register_basis("pca")
class PCA(Basis):
    artifact_keys = ["eigvecs", "mean", "eigvals"]

    def fit(self, activation: np.ndarray, context: np.ndarray, **kwargs):
        """_summary_

        Args:
            activation (np.ndarray): _description_
            context (np.ndarray): We do NOT use this here!
        """

        n, d = activation.shape

        if self.centering:
            mean = np.mean(activation, axis=0)
        else:
            mean = np.zeros(d)

        activation = activation - mean
        eigvals, eigvecs = np.linalg.eigh(activation.T @ activation / n)

        index = np.arange(d)[::-1]

        eigvals = eigvals[index]
        eigvecs = eigvecs[:, index]

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, mean, eigvals)))

        return eigvecs, mean, eigvals


@register_basis("prca")
class PRCA(Basis):
    artifact_keys = ["eigvecs", "mean", "eigvals"]

    def fit(
        self, activation: np.ndarray, context: np.ndarray, **kwargs
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_ Summary

        Args:
            activation (np.ndarray): _description_
            context (np.ndarray): _description_

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        n, d = activation.shape

        if self.centering:
            mean = np.mean(activation, axis=0)
        else:
            mean = np.zeros(d)

        activation = activation - mean
        eigvals, eigvecs = np.linalg.eigh(
            ((activation.T @ context + context.T @ activation)) / n
        )

        index = np.arange(d)[::-1]

        eigvals = eigvals[index]
        eigvecs = eigvecs[:, index]

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, mean, eigvals)))

        return eigvecs, mean, eigvals


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

        if self.centering:
            slug = "pca--centered"
        else:
            slug = "pca--uncentered"

        mean = torch.from_numpy(np.load(f"{artifact_dir}/{slug}/mean.npy")).float()

        if not self.centering:
            assert torch.allclose(mean, torch.tensor(0))

        mean = mean.to(device)

        d = mean.shape[0]

        seed = self.kwargs["seed"]

        np.random.seed(seed)

        mat = ortho_group.rvs(d)
        mat = torch.from_numpy(mat).float()
        mat = mat.to(device)

        setattr(self, "artifact", dict(mean=mean, eigvecs=mat))


class PRCAVariant(Basis):
    artifact_keys = ["eigvecs", "mean"]
    mode: str

    def fit(
        self, activation: np.ndarray, context: np.ndarray, **kwargs
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_ Summary

        Args:
            activation (np.ndarray): _description_
            context (np.ndarray): _description_

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        n, d = activation.shape

        if self.centering:
            mean = np.mean(activation, axis=0)
        else:
            mean = np.zeros(d)

        activation = activation - mean

        learner = learners.PRCAGreedyLeaner(mode=self.mode)

        U = learner.fit(activation, context, **kwargs)

        self.artifact = dict(zip(self.artifact_keys, (U, mean)))

        return U, mean, None


@register_basis("prca-abs")
class PRCAAbs(PRCAVariant):
    mode = "abs"


@register_basis("prca-recon")
class PRCARelRecon(PRCAVariant):
    mode = "recon"
