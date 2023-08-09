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
        # remark: we don't use any `std` here.
        # todo: perhaps, at some point, we should remove it!
        self.std = std.reshape((1, -1, 1, 1)).to(device)

        self.mode = mode

    def forward(self, x) -> torch.Tensor:
        if self.mode == AdapterMode.ENCODER:
            return self.encode(x)
        elif self.mode == AdapterMode.DECODER:
            return self.decode(x)
        else:
            raise ValueError(f"[mode={self.mode}] doesn't exist!")

    def encode(self, x):
        x = x - self.mean
        x = F.conv2d(x, self.mat_encoder)
        return x

    def decode(self, x):
        x = F.conv2d(x, self.mat_decoder)
        x = x + self.mean
        return x


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

    if name_slug in ["random", "randomperm"]:
        assert "seed" in kwargs, "`seed` must be specify for `random` basis."

        basis = BASES[name_slug](alias=name_slug, centering=centering, **kwargs)
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
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        mean: npt.NDArray,
        device: str,
    ):
        """_summary_

        Args:
            activation (npt.NDArray): _description_
            context (npt.NDArray): We do NOT use this here!
        """

        n, d = activation.shape

        if self.centering:
            activation = activation - mean

        assert not np.isnan(activation).any()

        cov = activation.T @ activation / n

        eigvals, eigvecs = np.linalg.eigh(cov)

        indices = np.argsort(eigvals)[::-1]

        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        std = np.std(activation @ eigvecs, axis=0)

        cond = eigvals < 0

        if cond.sum() > 0:
            print("[warning]: some eigenvalues are of PCA smaller than zero!")
            print(
                "Because this seems to be numerical issue, we set eigvals[eigvals < 0] = 0"
            )
            eigvals[cond] = 0

        assert not np.isnan(std).any()
        assert not (eigvals < 0).any()
        if self.centering:
            np.testing.assert_allclose(std, eigvals**0.5, atol=1e-3)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, std)))

        return eigvecs, std


@register_basis("identity")
class Identity(Basis):
    artifact_keys = ["std"]

    def construct_adapter(self, k: int, mode: AdapterMode, device: str) -> Adapter:
        std = self.artifact["std"]
        d = std.shape[0]

        print(
            f"[basis=identity] setting k={k} has no effect. The following forces k=d={d}!"
        )

        return Adapter(
            U=torch.eye(d),
            std=std,
            mean=self.mean,
            device=device,
            mode=mode,
        )

    def construct_fh_rank_k_projection(self, k: int, device: str):
        def fh(module, input, output):
            pass

        return fh

    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        mean: npt.NDArray,
        device: str,
    ) -> typing.Tuple[npt.NDArray]:
        if self.centering:
            activation = activation - mean

        std = np.std(activation, axis=0)

        setattr(self, "artifact", dict(zip(self.artifact_keys, [std])))

        return std


class CanonicalBasis(Basis):
    artifact_keys = ["eigvecs", "std"]

    def _computuing_maximization_objective(
        self, activation: npt.NDArray, context: npt.NDArray
    ) -> npt.NDArray:
        raise NotImplementedError("")

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

        objective: npt.NDArray = self._computuing_maximization_objective(
            activation, context
        )
        eigvals = np.mean(objective, axis=0)

        # argmax_i E[objective]
        indices = np.argsort(-eigvals)

        eigvecs = np.eye(d)[:, indices]

        std = np.std(activation @ eigvecs, axis=0)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, std)))

        return eigvecs, std


@register_basis("rel")
class Rel(CanonicalBasis):
    artifact_keys = ["eigvecs", "std"]

    def _computuing_maximization_objective(self, activation, context):
        # problem: argmax_i r_i
        return activation * context


@register_basis("rel-abs")
class RelAbs(CanonicalBasis):
    def _computuing_maximization_objective(self, activation, context):
        # problem: argmax_i |r_i|
        return np.abs(activation * context)


@register_basis("rel-recon")
class RelRecon(CanonicalBasis):
    def _computuing_maximization_objective(self, activation, context):
        # problem: argmin_i   \|r - r_i\|_2^2
        #       => argmax_i - \|r - r_i\|_2^2
        rel_per_dim = activation * context
        rel = np.sum(rel_per_dim, keepdims=True)

        recon = (rel - rel_per_dim) ** 2

        return -recon


@register_basis("rel-reconfixed")
class RelReconFixed(CanonicalBasis):
    def _computuing_maximization_objective(self, activation, context):
        # problem: argmin_i   \|r - r_i\|_2^2
        #       => argmax_i - \|r - r_i\|_2^2
        n, d = activation.shape

        rel_per_dim = activation * context
        rel = np.sum(rel_per_dim, axis=1)
        assert rel.shape == (n,)

        recon = (rel - rel_per_dim) ** 2

        return -recon


@register_basis("rel-recongreedyimproved2")
class RelReconGreedy(CanonicalBasis):
    @torch.no_grad()
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

        indices = []

        n, d = activation.shape

        activation = activation / (
            (np.mean(activation**2) ** (1 / 2)) * (d ** (1 / 4))
        )
        context = context / ((np.mean(context**2) ** (1 / 2)) * (d ** (1 / 4)))

        activation = torch.from_numpy(activation).float().to(device)
        context = torch.from_numpy(context).float().to(device)

        I = torch.eye(d).float().to(device)
        U = torch.zeros(d, d)
        U = U.to(device)

        indices = []

        for step in range(d):
            dimensions = list(set(range(d)).difference(indices))

            stats = []

            UUt = U @ U.T

            a_comp = activation @ (I - UUt)
            c_comp = context @ (I - UUt)

            assert a_comp.shape == c_comp.shape == (n, d)

            rel_total_left = (a_comp * c_comp).sum(axis=1)

            np.testing.assert_allclose(
                rel_total_left.cpu().numpy(),
                (activation[:, dimensions] * context[:, dimensions])
                .sum(axis=1)
                .cpu()
                .numpy(),
            )

            for i in dimensions:
                u = torch.zeros(d)
                u[i] = 1
                u = u.to(device)
                np.testing.assert_allclose((U.T @ u).cpu().numpy(), 0)
                rel_proj = (a_comp @ u) * (c_comp @ u)
                norm = (rel_total_left - rel_proj) ** 2
                stat = float(torch.mean(norm).detach().cpu().numpy())
                stats.append(stat)

            _k = dimensions[np.argmin(stats)]

            indices.append(_k)
            U[_k, step] = 1.0

        U = U.cpu().numpy()
        indices = np.argmax(U, axis=0)

        assert len(set(indices)) == d

        activation = activation.detach().cpu().numpy()

        eigvecs = np.eye(d)[:, indices]

        std = np.std(activation @ eigvecs, axis=0)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, std)))

        return eigvecs, std


@register_basis("act")
class Act(CanonicalBasis):
    def _computuing_maximization_objective(self, activation, context):
        # problem: argmax_i a_i
        return activation


@register_basis("act-abs")
class ActAbs(CanonicalBasis):
    def _computuing_maximization_objective(self, activation, context):
        # problem: argmax_i |a_i|
        return np.abs(activation)


@register_basis("act-recon")
class ActRecon(CanonicalBasis):
    def _computuing_maximization_objective(self, activation, context):
        # problem: argmin_i   \| a - (a^\tope_i) e_i \|^2_2
        #          argmin_i   a^Ta - 2 a^e_i + a_i^2
        #          argmax_i - (a^T a - 2 a^e_i + a_i^2)
        norm = np.linalg.norm(activation, axis=1, keepdims=True)
        criteria = norm**2 - 2 * activation + activation**2

        # convert to maximization
        return -criteria


@register_basis("act-reconfixed")
class ActReconFixed(CanonicalBasis):
    def _computuing_maximization_objective(self, activation, context):
        # problem: argmin_i   \| a - (a^\tope_i) e_i \|^2_2
        #          argmin_i   a^Ta - 2 a^e_i + a_i^2
        #          argmax_i - (a^T a - 2 a^e_i + a_i^2)
        norm = np.linalg.norm(activation, axis=1, keepdims=True)
        criteria = -(activation**2)

        # convert to maximization
        return -criteria


@register_basis("act-recongreedy")
class ActReconGreedy(CanonicalBasis):
    # def _computuing_maximization_objective(self, activation, context):
    #     # problem: argmin_i   \| a - (a^\tope_i) e_i \|^2_2
    #     #          argmin_i   a^Ta - 2 a^e_i + a_i^2
    #     #          argmax_i - (a^T a - 2 a^e_i + a_i^2)
    #     norm = np.linalg.norm(activation, axis=1, keepdims=True)
    #     criteria = norm**2 - 2 * activation + activation**2

    #     # convert to maximization
    #     return -criteria
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

        # objective: npt.NDArray = self._computuing_maximization_objective(
        #     activation, context
        # )
        # eigvals = np.mean(objective, axis=0)

        # # argmax_i E[objective]
        # indices = np.argsort(-eigvals)

        indices = []

        d = activation.shape[1]

        activation = torch.from_numpy(activation).float().to(device)

        while len(indices) < d:
            dimensions = list(set(range(d)).difference(indices))

            stats = []

            U = np.eye(d)[:, indices]
            U = torch.from_numpy(U).float().to(device)

            a_c = activation @ (torch.eye(d).float().to(device) - U @ U.T)
            for i in dimensions:
                u = torch.zeros(d)
                u[i] = 1
                u = u.to(device)
                uut = u.outer(u)
                norm = torch.linalg.norm(a_c - activation @ uut)
                stat = float(torch.mean(norm).detach().cpu().numpy())
                stats.append(stat)

            selected_i = dimensions[np.argmin(stats)]
            indices.append(selected_i)

        activation = activation.detach().cpu().numpy()

        eigvecs = np.eye(d)[:, indices]

        std = np.std(activation @ eigvecs, axis=0)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, std)))

        return eigvecs, std


@register_basis("prca")
class PRCA(Basis):
    artifact_keys = ["eigvecs", "std"]

    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        mean: npt.NDArray,
        device: str,
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
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        mean: npt.NDArray,
        device: str,
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
        if self.centering:
            activation = activation - mean

        _, d = activation.shape
        seed = self.kwargs["seed"]

        U = ortho_group.rvs(d, random_state=np.random.default_rng(seed))

        std = np.std(activation @ U, axis=0)

        setattr(self, "artifact", dict(eigvecs=U, std=std))

        return U, std


@register_basis("randomperm")
class RandomPerm(Basis):
    artifact_keys = ["eigvecs", "std"]

    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        mean: npt.NDArray,
        device: str,
    ) -> typing.Tuple[npt.NDArray, npt.NDArray]:
        if self.centering:
            activation = activation - mean

        _, d = activation.shape
        seed = self.kwargs["seed"]

        indices = np.random.default_rng(seed).permutation(d)
        U = np.eye(d)[:, indices]

        std = np.std(activation @ U, axis=0)

        setattr(self, "artifact", dict(eigvecs=U, std=std))

        return U, std


class PRCAVariant(Basis):
    artifact_keys = ["eigvecs", "std"]
    mode: str

    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        mean: npt.NDArray,
        device: str,
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

        precondition_mat = self.precondition_matrix(activation)

        learner = learners.PRCAGreedyLeaner(mode=self.mode)

        U = learner.fit(
            activation @ precondition_mat,
            context @ precondition_mat,
            beta=self.beta,
            seed=self.kwargs["seed"],
            device=device,
        )

        # remark: we do right multiplication
        #      Z = X @ (Precondition Mat) @ U
        # ; therefore, the final basis is
        #      U = Precondition Mat @ U
        U = precondition_mat @ U

        std = np.std(activation @ U, axis=0)

        self.artifact = dict(zip(self.artifact_keys, (U, std)))

        return U, std

    def precondition_matrix(self, activation: npt.NDArray) -> npt.NDArray:
        d = activation.shape[1]
        return np.eye(d)


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
class PCAPRCAVariant(PRCAVariant):
    artifact_keys = ["eigvecs", "std"]
    mode = "abs"
    beta = 0.0

    def precondition_matrix(self, activation: npt.NDArray) -> npt.NDArray:
        # remarks:
        # - we assume `activation` is processed accordingly to the mode (centered or uncentered)
        # - if mode=centered, then, this outer product is covariance
        _, E = np.linalg.eigh(activation.T @ activation / activation.shape[0])

        # reorder according to the descendence of eigenvalues
        E = np.flip(E, axis=1)

        return E


@register_basis("pcaprca-recon")
class PCAPRCARecon(PCAPRCAVariant):
    mode = "recon"
    beta = 0.0
