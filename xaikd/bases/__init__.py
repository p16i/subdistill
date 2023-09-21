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
        scale: torch.Tensor,
        device: str,
        mode: AdapterMode,
    ) -> None:
        super().__init__()

        d, k = U.shape

        assert scale.shape[0] == k
        assert mean.shape[0] == d

        self.mat_encoder = U.T.unsqueeze(2).unsqueeze(3).to(device)
        self.mat_decoder = U.unsqueeze(2).unsqueeze(3).to(device)

        self.mean = mean.reshape((1, -1, 1, 1)).to(device)
        # remark: we don't use any `std` here.
        # todo: perhaps, at some point, we should remove it!
        # also, if it is used again, be aware that the value is refactored
        # to be `variance` (when basis_mode=centered). So, one might need `\sqrt{}`.
        self.scale = scale.reshape((1, -1, 1, 1)).to(device)

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

    def _compute_scale(self, activation: npt.NDArray, U: npt.NDArray) -> npt.NDArray:
        # remark: if centering (i.e., `mean(activation)=0`), then
        # this expresssion is `standard deviation`
        return np.mean((activation @ U) ** 2, axis=0)

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
        scale = self.artifact["scale"][:k]

        return Adapter(U=U, mean=self.mean, scale=scale, mode=mode, device=device)

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


@register_basis("identity")
class Identity(Basis):
    artifact_keys = ["scale"]

    def construct_adapter(self, k: int, mode: AdapterMode, device: str) -> Adapter:
        scale = self.artifact["scale"]
        d = scale.shape[0]

        print(
            f"[basis=identity] setting k={k} has no effect. The following forces k=d={d}!"
        )

        return Adapter(
            U=torch.eye(d),
            scale=scale,
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

        _, d = activation.shape

        scale = self._compute_scale(activation, np.eye(d))

        setattr(self, "artifact", dict(zip(self.artifact_keys, [scale])))

        return scale


@register_basis("random")
class Random(Basis):
    artifact_keys = ["eigvecs", "scale"]

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

        scale = self._compute_scale(activation, U)

        setattr(self, "artifact", dict(eigvecs=U, scale=scale))

        return U, scale


@register_basis("randomperm")
class RandomPerm(Basis):
    artifact_keys = ["eigvecs", "scale"]

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

        scale = self._compute_scale(activation, U)

        setattr(self, "artifact", dict(eigvecs=U, scale=scale))

        return U, scale


class CanonicalBasis(Basis):
    artifact_keys = ["eigvecs", "scale"]

    def _solve_objective(
        self, activation: npt.NDArray, context: npt.NDArray
    ) -> npt.NDArray[int]:
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

        indices = self._solve_objective(activation, context)

        eigvecs = np.eye(d)[:, indices]

        scale = self._compute_scale(activation, eigvecs)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, scale)))

        return eigvecs, scale


@register_basis("act-raw")
class ActRaw(CanonicalBasis):
    def _solve_objective(self, activation, context):
        # problem: argmax_i E[a_i]
        criteria = np.mean(activation, axis=0)

        indices = np.argsort(-criteria)

        return indices


@register_basis("act-recon")
class ActRecon(CanonicalBasis):
    def _solve_objective(self, activation, context):
        # problem: argmin_i  E[ \| a - (a^\tope_i) e_i \|^2_2 ]
        #          argmin_i  E[ a^Ta - 2 a_i^2 + a_i^2        ] where a^\top e_i = a_i
        #          argmin_i  E[      - 2 a_i^2 + a_i^2        ]
        #          argmin_i  E[      -   a_i^2                ]

        _, d = activation.shape
        criteria = -np.mean((activation**2), axis=0)
        assert criteria.shape == (d,)

        indices = np.argsort(criteria)

        return indices


@register_basis("rel-raw")
class RelRaw(CanonicalBasis):
    artifact_keys = ["eigvecs", "scale"]

    def _solve_objective(self, activation, context):
        # problem: argmax_i E[r_i]
        _, d = activation.shape
        rel = np.mean(activation * context, axis=0)

        assert rel.shape == (d,)

        indices = np.argsort(-rel)

        return indices


@register_basis("rel-abs")
class RelAbs(CanonicalBasis):
    def _solve_objective(self, activation, context):
        # problem: argmax_i |E[r_i]|
        rel = np.mean(activation * context, axis=0)
        abs_rel = np.abs(rel)

        indices = np.argsort(-abs_rel)

        return indices


@register_basis("rel-reconnaive")
class RelReconNaive(CanonicalBasis):
    def _solve_objective(self, activation, context):
        # problem: argmin_i   E[(r - r_i)^2]
        n, d = activation.shape

        rel_per_dim = activation * context
        rel = np.sum(rel_per_dim, axis=1, keepdims=True)
        assert rel.shape == (n, 1)

        recon = (rel - rel_per_dim) ** 2

        criteria = np.mean(recon, axis=0)
        assert criteria.shape == (d,)

        indices = np.argsort(criteria)

        return indices


@register_basis("rel-recon")
class RelRecon(CanonicalBasis):
    def _solve_objective(self, activation, context):
        indices = []

        n, d = activation.shape

        indices = []

        rel_per_dim = activation * context

        assert rel_per_dim.shape == (n, d)

        possible_dimensions = set(list(range(d)))

        for _ in range(d):
            dimensions = list(possible_dimensions.difference(indices))

            stats = np.zeros(len(dimensions))
            stats = []

            rel_total_left = rel_per_dim[:, dimensions].sum(axis=1)
            assert rel_total_left.shape == (n,)

            for k in dimensions:
                rel_proj = rel_per_dim[:, k]
                norm = (rel_total_left - rel_proj) ** 2
                stats.append(np.mean(norm))

            indices.append(dimensions[np.argmin(stats)])

        assert len(set(indices)) == d

        return indices


@register_basis("pca")
class PCA(Basis):
    artifact_keys = ["eigvecs", "scale"]

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

        scale = self._compute_scale(activation, eigvecs)

        cond = eigvals < 0

        if cond.sum() > 0:
            print("[warning]: some eigenvalues are of PCA smaller than zero!")
            print(
                "Because this seems to be numerical issue, we set eigvals[eigvals < 0] = 0"
            )
            eigvals[cond] = 0

        assert not np.isnan(scale).any()
        assert not (eigvals < 0).any()

        np.testing.assert_allclose(scale**0.5, eigvals**0.5, atol=1e-3)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, scale)))

        return eigvecs, scale


@register_basis("pcainv")
class PCAInverse(Basis):
    artifact_keys = ["eigvecs", "scale"]

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

        indices = np.argsort(eigvals)

        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        scale = self._compute_scale(activation, eigvecs)

        cond = eigvals < 0

        if cond.sum() > 0:
            print("[warning]: some eigenvalues are of PCA smaller than zero!")
            print(
                "Because this seems to be numerical issue, we set eigvals[eigvals < 0] = 0"
            )
            eigvals[cond] = 0

        assert not np.isnan(scale).any()
        assert not (eigvals < 0).any()

        np.testing.assert_allclose(scale**0.5, eigvals**0.5, atol=1e-3)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, scale)))

        return eigvecs, scale


@register_basis("prca")
class PRCA(Basis):
    artifact_keys = ["eigvecs", "scale"]

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

        # sorted by descending of `criteria(eigvals)`
        indices = np.argsort(-self._criteria(eigvals))

        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        scale = self._compute_scale(activation, eigvecs)

        self.artifact = dict(zip(self.artifact_keys, (eigvecs, scale)))

        return eigvecs, scale

    def _criteria(self, eigvals: npt.NDArray) -> npt.NDArray:
        return eigvals


@register_basis("prca-sortabs")
class PRCASortAbs(PRCA):
    def _criteria(self, eigvals: npt.NDArray) -> npt.NDArray:
        return np.abs(eigvals)


@register_basis("prca-sortabsinv")
class PRCASortAbsInv(PRCA):
    def _criteria(self, eigvals: npt.NDArray) -> npt.NDArray:
        return -np.abs(eigvals)


class PRCAVariant(Basis):
    artifact_keys = ["eigvecs", "scale"]
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

        if self.centering:
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

        scale = self._compute_scale(activation, U)

        self.artifact = dict(zip(self.artifact_keys, (U, scale)))

        return U, scale

    def precondition_matrix(self, activation: npt.NDArray) -> npt.NDArray:
        d = activation.shape[1]
        return np.eye(d)


@register_basis("prca-abs")
class PRCAAbs(PRCAVariant):
    mode = "abs"
    beta = 0.0


@register_basis("prca-recon")
class PRCARecon(PRCAVariant):
    mode = "recon"
    beta = 0.0


@register_basis("prca-reconnaive")
class PRCAReconNaive(PRCAVariant):
    mode = "reconnaive"
    beta = 0.0


@register_basis("prca-reconreg")
class PRCAReconReg(PRCAVariant):
    mode = "recon"

    def __init__(self, beta=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta


class PCAPRCAVariant(PRCAVariant):
    def precondition_matrix(self, activation: npt.NDArray) -> npt.NDArray:
        # remarks:
        # - we assume `activation` is processed accordingly to the mode (centered or uncentered)
        # - if mode=centered, then, this outer product is covariance
        _, E = np.linalg.eigh(activation.T @ activation / activation.shape[0])

        # reorder according to the descendence of eigenvalues
        E = np.flip(E, axis=1)

        return E


@register_basis("pcaprca-abs")
class PCAPRCAAbs(PCAPRCAVariant):
    mode = "abs"
    beta = 0.0


@register_basis("pcaprca-recon")
class PCAPRCARecon(PCAPRCAVariant):
    mode = "recon"
    beta = 0.0
