import numpy as np
from numpy import typing as npt

from scipy.stats import norm as norm_gaussian, entropy

from xaikd import utils

from .interface import OrthogonalBasis

from .orthogonal import PRCAPosDef
from .register import register_basis


def estimate_std_wrt_ratio_maxent(
    arr_logits: npt.NDArray, threshold: float, ratio: float
) -> float:

    (N,) = arr_logits.shape

    # uniform distribution over all datapoints
    target_entropy = ratio * np.log(N)

    arr_std = []
    arr_percentile_entropies = []
    arr_candidates = np.linspace(1, 100, num=100) / 100

    vmin = np.min(arr_logits)
    vmax = np.max(arr_logits)

    for pth in arr_candidates:
        std = pth * (vmax - vmin) / 2

        weights = norm_gaussian.pdf(arr_logits, loc=threshold, scale=std)
        weights = weights / np.sum(weights)
        arr_std.append(std)
        arr_percentile_entropies.append(entropy(weights))

    arr_percentile_entropies = np.array(arr_percentile_entropies)

    best_ix = np.argmin(np.abs(arr_percentile_entropies - target_entropy))
    best_std = arr_std[best_ix]
    print(
        f"[entropy_ratio={ratio}]: best_ix={best_ix} best_std={best_std:.4e} best_candidate={arr_candidates[best_ix]}"
    )

    return best_std


class PRCAPosDefWeightSTDFromEntropy(PRCAPosDef):
    entropy_ratio = None

    def _compute_sample_weight(
        self, arr_logodd: npt.NDArray, logodd_threshold: float
    ) -> npt.NDArray:

        assert self.entropy_ratio is not None

        std = estimate_std_wrt_ratio_maxent(
            arr_logits=arr_logodd,
            threshold=logodd_threshold,
            ratio=self.entropy_ratio,
        )

        weights = norm_gaussian.pdf(
            arr_logodd,
            loc=logodd_threshold,
            scale=std,
        )

        weights = weights / np.sum(weights)

        assert weights.shape == arr_logodd.shape

        return weights

    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):

        weights = self._compute_sample_weight(
            arr_logodd=arr_logodd, logodd_threshold=logodd_threshold
        )

        # Remark: when arr_ctx is large, this leads to OOM.
        # The reason is that it creates a new array for the result.
        # One way to mitigate the issue is to perform inline assignment.
        # Consequently, one should make sure that the side effect is not
        # propagated to other parts.
        arr_ctx = arr_ctx * weights.reshape((-1, 1, 1))

        return super()._solve(
            arr_act=arr_act,
            arr_ctx=arr_ctx,
            arr_logodd=arr_logodd,
            logodd_threshold=logodd_threshold,
            **kwargs,
        )

    @classmethod
    def slug(cls):
        assert cls.entropy_ratio is not None

        return f"prcaposdef-entropy{cls.entropy_ratio}"


class PRCAPosDefUniformWeight(PRCAPosDefWeightSTDFromEntropy):
    # this class is for testing purpose
    def _compute_sample_weight(
        self, arr_logodd: npt.NDArray, logodd_threshold: float
    ) -> npt.NDArray:
        (N,) = arr_logodd.shape
        return np.ones(N) / N


@register_basis()
class PRCAPosDefWeightSTDWithH0_5(PRCAPosDefWeightSTDFromEntropy):
    entropy_ratio = 0.5


@register_basis()
class PRCAPosDefWeightSTDWithH0_6(PRCAPosDefWeightSTDFromEntropy):
    entropy_ratio = 0.6


@register_basis()
class PRCAPosDefWeightSTDWithH0_7(PRCAPosDefWeightSTDFromEntropy):
    entropy_ratio = 0.7


@register_basis()
class PRCAPosDefWeightSTDWithH0_8(PRCAPosDefWeightSTDFromEntropy):
    entropy_ratio = 0.8


@register_basis()
class PRCAPosDefWeightSTDWithH0_9(PRCAPosDefWeightSTDFromEntropy):
    entropy_ratio = 0.9


@register_basis()
class PRCAPosDefWeightSTDWithH0_95(PRCAPosDefWeightSTDFromEntropy):
    entropy_ratio = 0.95


@register_basis()
class PRCAPosDefWeightSTDWithH0_975(PRCAPosDefWeightSTDFromEntropy):
    entropy_ratio = 0.975


@register_basis()
class PRCAPosDefWeightSTDWithH0_99(PRCAPosDefWeightSTDFromEntropy):
    entropy_ratio = 0.99


@register_basis()
class GradPCAWeightSTDWithEntropy(OrthogonalBasis):
    entropy_ratio = 0.95

    def _compute_sample_weight(
        self, arr_logodd: npt.NDArray, logodd_threshold: float
    ) -> npt.NDArray:
        assert self.entropy_ratio is not None

        std = estimate_std_wrt_ratio_maxent(
            arr_logits=arr_logodd,
            threshold=logodd_threshold,
            ratio=self.entropy_ratio,
        )

        weights = norm_gaussian.pdf(
            arr_logodd,
            loc=logodd_threshold,
            scale=std,
        )

        weights = weights / np.sum(weights)

        assert weights.shape == arr_logodd.shape

        return weights

    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):

        weights = self._compute_sample_weight(
            arr_logodd=arr_logodd, logodd_threshold=logodd_threshold
        )
        arr_ctx = arr_ctx * weights.reshape((-1, 1, 1))

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        cov_c = arr_ctx.T @ arr_ctx

        _, eigvecs = utils.solve_eigh(cov_c)

        return eigvecs

    @classmethod
    def slug(cls):

        assert cls.entropy_ratio is not None

        return f"gradpca-entropy{cls.entropy_ratio}"
