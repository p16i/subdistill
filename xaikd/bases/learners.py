import torch

import numpy as np
import numpy.typing as npt


from tqdm import tqdm


class PRCAGreedyLeaner:
    def __init__(self, mode: str) -> None:
        if mode == "abs":
            self.obj_func = PRCAGreedyLeaner._obj_abs
        elif mode == "recon":
            self.obj_func = PRCAGreedyLeaner._obj_recon
        elif mode == "reconnaive":
            self.obj_func = PRCAGreedyLeaner._obj_recon_naive
        else:
            raise ValueError(f"No mode=`{mode}` available!")

        self.mode = mode

    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        epochs=1000,
        seed=1,
        eps=1e-5,  # todo: this parameter seem to be very important!
        device="cpu",
        beta=0.0,
    ) -> npt.NDArray:
        assert activation.shape == context.shape

        _, d = activation.shape

        activation = activation / (
            (np.mean(activation**2) ** (1 / 2)) * (d ** (1 / 4))
        )
        context = context / ((np.mean(context**2) ** (1 / 2)) * (d ** (1 / 4)))

        activation = torch.from_numpy(activation).float().to(device)
        context = torch.from_numpy(context).float().to(device)

        rng = torch.Generator()
        rng.manual_seed(seed)

        U = torch.zeros(d, d)
        U = U.to(device)

        I = torch.eye(d).to(device)

        for k in tqdm(
            range(d), total=d, desc=f"[mode={self.mode},beta={beta},device={device}]"
        ):
            UUt = U @ U.T

            # take a random vector
            v = torch.randn(d, generator=rng).to(device)

            v = (I - UUt) @ v

            v = v / torch.linalg.norm(v)
            v = v.to(device)

            for _ in range(epochs):
                v.requires_grad_(True)
                v.grad = None

                # projected out previous component
                # A_compt = activation @ (I - UUt)
                # C_compt = context @ (I - UUt)

                obj = self.obj_func(activation, context, I - UUt, v, beta)

                obj.backward()

                with torch.no_grad():
                    ov = v

                    # update v with gradient `ascent`.
                    v = v + v.grad
                    v = (I - UUt) @ v
                    v = v / torch.linalg.norm(v)

                    if (v @ ov).abs() > (1 - eps):
                        break

            # testing orthogonality
            np.testing.assert_allclose(
                (U.T @ v).detach().cpu().numpy(), np.zeros(U.shape[1]), atol=1e-3
            )

            U[:, k] = v.detach()

        np.testing.assert_allclose(
            (U.T @ U).detach().cpu().numpy(), np.eye(d), atol=1e-3
        )

        return U.detach().cpu().numpy()

    @staticmethod
    def _obj_abs(
        activation: torch.Tensor,
        context: torch.Tensor,
        IUUt: torch.Tensor,
        u: torch.Tensor,
        beta=0,
    ) -> torch.Tensor:
        assert beta == 0, f"setting beta={beta} has not effect here."

        activation = activation @ IUUt
        context = context @ IUUt

        activation_projected = activation.matmul(u)
        context_projected = context.matmul(u)

        assert len(activation_projected.shape) == len(context_projected.shape) == 1

        obj = (activation_projected * context_projected).abs()
        assert len(obj.shape) == 1 and obj.shape[0] == activation.shape[0]
        return obj.mean()

    @staticmethod
    def _obj_recon(
        activation: torch.Tensor,
        context: torch.Tensor,
        IUUt: torch.Tensor,
        u: torch.Tensor,
        beta=0,
    ) -> torch.Tensor:
        activation = activation @ IUUt
        context = context @ IUUt

        activation_projected = activation.matmul(u)
        context_projected = context.matmul(u)

        assert len(activation_projected.shape) == len(context_projected.shape) == 1

        relevance_original = (activation * context).sum(dim=1)
        relevance_projected = activation_projected * context_projected
        assert relevance_original.shape == relevance_projected.shape

        obj = (relevance_original - relevance_projected) ** 2

        assert len(obj.shape) == 1 and obj.shape[0] == activation.shape[0]

        # convert the problem into maximization problem.
        loss = -obj.mean()

        reg = (beta * torch.abs(activation_projected)).mean()

        return loss + reg

    @staticmethod
    def _obj_recon_naive(
        activation: torch.Tensor,
        context: torch.Tensor,
        IUUt: torch.Tensor,
        u: torch.Tensor,
        beta=0,
    ) -> torch.Tensor:
        # activation = activation @ IUUt
        # context = context @ IUUt

        activation_projected = activation.matmul(u)
        context_projected = context.matmul(u)

        assert len(activation_projected.shape) == len(context_projected.shape) == 1

        relevance_original = (activation * context).sum(dim=1)
        relevance_projected = activation_projected * context_projected
        assert relevance_original.shape == relevance_projected.shape

        obj = (relevance_original - relevance_projected) ** 2

        assert len(obj.shape) == 1 and obj.shape[0] == activation.shape[0]

        # convert the problem into maximization problem.
        loss = -obj.mean()

        reg = (beta * torch.abs(activation_projected)).mean()

        return loss + reg
