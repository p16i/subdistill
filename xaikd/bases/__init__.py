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


from enum import Enum

EPS = 1e-6
BASES = dict()

AdapterMode = Enum("AdapterMode", ["ENCODER", "DECODER"])


class Adapter(torch.nn.Module):
    def __init__(
        self,
        U: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        device: str,
        mode: AdapterMode,
    ) -> None:
        super().__init__()

        d, k = U.shape

        assert std.shape[0] == k
        assert mean.shape[0] == d

        self.mat_encoder = U.T.unsqueeze(2).unsqueeze(3).to(device)
        self.mat_decoder = U.unsqueeze(2).unsqueeze(3).to(device)

        self.mean = mean.reshape((1, -1, 1, 1)).to(device)
        self.std = std.reshape((1, -1, 1, 1)).to(device)

        self.mode = mode

    def forward(self, x) -> torch.Tensor:
        if self.mode == AdapterMode.ENCODER:
            return self.encoder(x)
        elif self.mode == AdapterMode.DECODER:
            return self.decoder(x)
        else:
            raise ValueError(f"[mode={self.mode}] doesn't exist!")

    def encoder(self, x):
        out = x - self.mean
        out = F.conv2d(out, self.mat_encoder)
        out = out / (self.std + EPS)
        return out

    def decoder(self, x):
        out = x * (self.std + EPS)
        out = F.conv2d(x, self.mat_decoder)
        out = out + self.mean
        return out


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

        if self.centering:
            mean = np.load(artifact_dir / "act_mean.npy")
        else:
            d = artifact[self.artifact_keys[0]].shape[0]
            mean = np.zeros(d)

        self.mean = torch.from_numpy(mean).float().to(device)

    def construct_adapter(self, k: int, mode: AdapterMode, device: str) -> Adapter:
        U: torch.Tensor = self.artifact["eigvecs"][:, :k]
        std = self.artifact["std"][:k]

        return Adapter(U=U, mean=self.mean, std=std, mode=mode, device=device)

    def construct_fh_rank_k_projection(self, k: int, device: str) -> typing.Callable:
        encoder = self.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device)
        decoder = self.construct_adapter(k=k, mode=AdapterMode.DECODER, device=device)

        def fh(mod, input, output):
            assert isinstance(output, torch.Tensor)
            return decoder(encoder(output))

        return fh

    def __str__(self) -> str:
        return getattr(self, "__name")


def get_basis(slug, **kwargs) -> Basis:
    name_slug, centering_slug = slug.split("--")
    centering = True if centering_slug == "centered" else False

    assert (
        centering
    ), "Since Sprint S9 (2023-07), we conclude that `centering=True` is the fixed parameter."

    if "random" in name_slug:
        seed = int(name_slug.replace("random", ""))
        basis = BASES["random"](
            alias=name_slug, centering=centering, seed=seed, **kwargs
        )
    else:
        assert centering_slug in ["uncentered", "centered"], f"Value `{centering_slug}`"

        if "prca-reconreg" in name_slug:
            beta = float(name_slug.split("reg")[-1])

            name_slug = "prca-reconreg"

            kwargs["beta"] = beta

        basis = BASES[name_slug](alias=name_slug, centering=centering, **kwargs)

    setattr(basis, "__name", slug)

    return basis


@register_basis("pca")
class PCA(Basis):
    artifact_keys = ["eigvecs", "std"]

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

        std = np.std(activation @ eigvecs, axis=0)

        np.testing.assert_allclose(std, eigvals**0.5, atol=1e-3)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, std)))

        return eigvecs, std


@register_basis("identity")
class Identity(Basis):
    artifact = dict()
    artifact_keys = []

    def construct_fh_rank_k_projection(self, k: int, device: str):
        def fh(module, input, output):
            pass

        return fh

    def load(self, artifact_dir: Path, device="cpu"):
        pass

    def save(self, output_dir: Path):
        pass


@register_basis("rel")
class Rel(Basis):
    artifact_keys = ["eigvecs", "std"]

    def _relevance_preprocessing(self, x: npt.NDArray) -> npt.NDArray:
        return x

    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        mean: typing.Union[npt.NDArray, None],
        device: str,
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
        n, d = activation.shape

        if self.centering:
            activation = activation - mean

        relevance: npt.NDArray = self._relevance_preprocessing(activation * context)
        eigvals = np.mean(relevance, axis=0)

        # large relevance first
        indices = np.argsort(-eigvals)

        eigvecs = np.eye(d)[:, indices]

        std = np.std(activation @ eigvecs, axis=0)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, std)))

        return eigvecs, std


@register_basis("rel-abs")
class RelAbs(Rel):
    def _relevance_preprocessing(self, x: npt.NDArray) -> npt.NDArray:
        # todo: add test?
        return np.abs(x)


@register_basis("prca")
class PRCA(Basis):
    artifact_keys = ["eigvecs", "std"]

    def fit(
        self, activation: npt.NDArray, context: npt.NDArray, mean: npt.NDArray, **kwargs
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
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

        std = np.std(activation @ eigvecs, axis=0)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, std)))

        return eigvecs, std


@register_basis("random")
class Random(Basis):
    artifact_keys = ["eigvecs", "std"]

    def fit(
        self, activation: npt.NDArray, context: npt.NDArray, mean: npt.NDArray, **kwargs
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
        if self.centering:
            activation = activation - mean

        _, d = activation.shape
        seed = self.kwargs["seed"]

        np.random.seed(seed)

        U = ortho_group.rvs(d)

        std = np.std(activation @ U, axis=0)

        setattr(self, "artifact", dict(eigvecs=U, std=std))

        return U, std


class PRCAVariant(Basis):
    artifact_keys = ["eigvecs", "std"]
    mode: str

    def fit(
        self, activation: npt.NDArray, context: npt.NDArray, mean: npt.NDArray, **kwargs
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
        """_summary_ Summary

        Args:
            activation (npt.NDArray): (uncenter)
            context (npt.NDArray): _description_

        Returns:
            typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]: _description_
        """
        _, d = activation.shape

        if not self.centering:
            mean = np.zeros(d)

        activation = activation - mean

        learner = learners.PRCAGreedyLeaner(mode=self.mode)

        U = learner.fit(activation, context, **kwargs, beta=self.beta)

        std = np.std(activation @ U, axis=0)

        self.artifact = dict(zip(self.artifact_keys, (U, std)))

        return U, std


@register_basis("prca-abs")
class PRCAAbs(PRCAVariant):
    mode = "abs"
    beta = 0.0


@register_basis("prca-recon")
class PRCARelRecon(PRCAVariant):
    mode = "recon"
    beta = 0.0


@register_basis("prca-reconreg")
class PRCARelReconReg(PRCAVariant):
    mode = "recon"

    def __init__(self, beta=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta


@register_basis("pcaprca-abs")
class PCAPRCAVariant(Basis):
    artifact_keys = ["eigvecs", "std"]
    mode = "abs"
    beta = 0.0

    def fit(
        self, activation: npt.NDArray, context: npt.NDArray, mean: npt.NDArray, **kwargs
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
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

        cov = np.cov(activation.T)
        _, E = np.linalg.eigh(cov)
        E = np.copy(E[:, ::-1])

        learner = learners.PRCAGreedyLeaner(mode=self.mode)

        activation = activation @ E
        context = context @ E

        U = learner.fit(activation, context, **kwargs, beta=self.beta)

        # combining the eigvectors of cov(x) and the vectors from PRCA
        # -> X @ (E@U)
        U = E @ U

        std = np.std(activation @ U, axis=0)

        self.artifact = dict(zip(self.artifact_keys, (U, std)))

        return U, std


@register_basis("pcaprca-recon")
class PCAPRCARecon(PCAPRCAVariant):
    mode = "recon"
    beta = 0.0
