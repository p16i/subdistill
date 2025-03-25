import torch

import numpy as np
import numpy.typing as npt


from tqdm.autonotebook import tqdm


def atol(mode):
    if mode == "reconnaive":
        return 1e-2
    else:
        return 1e-5


def exponential_map(p, g, t):
    norm_g = torch.linalg.norm(g)
    return torch.cos(norm_g * t) * p + torch.sin(norm_g * t) * (g / (norm_g + 1e-16))


class PRCAGreedyLearner:
    def __init__(self, mode: str) -> None:
        if mode == "abs":
            self.obj_func = PRCARecon._obj_abs
        elif mode == "recon":
            self.obj_func = PRCARecon._obj_func
        elif mode == "recon-idv":
            self.obj_func = PRCARecon._obj_recon_idv
        elif mode == "reconnaive":
            self.obj_func = PRCARecon._obj_recon_naive
        else:
            raise ValueError(f"No mode=`{mode}` available!")

        self.mode = mode

    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        epochs=1000,
        seed=1,
        eps=1e-5,  # remark: this parameter seem to be very important!
        device="cpu",
        beta=0.0,
    ) -> npt.NDArray:
        assert activation.shape == context.shape

        _, d = activation.shape

        activation = activation / ((np.mean(activation**2) ** (1 / 2)) * (d ** (1 / 4)))
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
            U_complement = I - U @ U.T

            # take a random vector
            v = torch.randn(d, generator=rng).to(device)

            v = U_complement @ v

            v = v / torch.linalg.norm(v)
            v = v.to(device)

            for _ in range(epochs):
                v.requires_grad_(True)
                v.grad = None

                obj = self.obj_func(activation, context, U_complement, v, beta)

                obj.backward()

                with torch.no_grad():
                    ov = v

                    # update v with gradient `ascent`.
                    v = v + v.grad
                    v = U_complement @ v
                    v = v / torch.linalg.norm(v)

                    if (v @ ov).abs() > (1 - eps):
                        # stop if the solution isn't update anymore.
                        break

            # testing orthogonality
            np.testing.assert_allclose(
                (U.T @ v).detach().cpu().numpy(),
                np.zeros(U.shape[1]),
                atol=atol(self.mode),
            )

            U[:, k] = v.detach()

        np.testing.assert_allclose(
            (U.T @ U).detach().cpu().numpy(), np.eye(d), atol=atol(self.mode)
        )

        return U.detach().cpu().numpy()

    @staticmethod
    def _obj_abs(
        activation: torch.Tensor,
        context: torch.Tensor,
        U_complement: torch.Tensor,
        u: torch.Tensor,
        beta=0,
    ) -> torch.Tensor:
        assert beta == 0, f"setting beta={beta} has not effect here."

        activation = activation @ U_complement
        context = context @ U_complement

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
    def _obj_recon_idv(
        activation: torch.Tensor,
        context: torch.Tensor,
        IUUt: torch.Tensor,
        u: torch.Tensor,
        beta=0,
    ) -> torch.Tensor:
        activation = activation @ IUUt
        context = context @ IUUt

        uuT = u.outer(u)

        activation_projected = activation @ uuT
        context_projected = context @ uuT

        assert len(activation_projected.shape) == len(context_projected.shape) == 2

        assert activation.shape == activation_projected.shape

        relevance_original = activation * context
        relevance_projected = activation_projected * context_projected
        assert relevance_original.shape == relevance_projected.shape

        obj = (relevance_original - relevance_projected) ** 2

        assert len(obj.shape) == 2 and obj.shape == activation.shape

        # convert the problem into maximization problem.
        loss = -obj.mean()

        return loss

    @staticmethod
    def _obj_recon_naive(
        activation: torch.Tensor,
        context: torch.Tensor,
        IUUt: torch.Tensor,
        u: torch.Tensor,
        beta=0,
    ) -> torch.Tensor:
        raise
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


class PRCAReconGreedy:
    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        epochs=100,
        eps=1e-6,
        seed=1,
        largest_k=None,
        device="cpu",
    ) -> npt.NDArray:
        assert activation.shape == context.shape

        _, d = activation.shape

        activation = activation / ((np.mean(activation**2) ** (1 / 2)) * (d ** (1 / 4)))
        context = context / ((np.mean(context**2) ** (1 / 2)) * (d ** (1 / 4)))

        activation = torch.from_numpy(activation).float().to(device)
        context = torch.from_numpy(context).float().to(device)

        rng = torch.Generator()
        rng.manual_seed(seed)

        U = torch.zeros(d, d)
        U = U.to(device)

        I = torch.eye(d).to(device)

        if largest_k is None:
            largest_k = d

        tbar = tqdm(
            range(largest_k),
            total=largest_k,
            desc=f"PRCA Recon [device={device}]]",
        )

        for k in tbar:
            if k >= 1:
                Uk = U[:, :k]
                U_pinv = torch.linalg.pinv(Uk)
                U_complement = I - U_pinv.T @ Uk.T
            else:
                U_pinv = 0
                U_complement = I

            v = torch.randn(d, generator=rng).to(device)
            v = v / torch.linalg.norm(v)

            v = v.to(device)

            # take only activation and context that are not captured by the learned directions
            activation = activation @ U_complement
            context = context @ U_complement

            for _ in range(epochs):
                v.requires_grad_(True)
                v.grad = None

                obj = self._obj_func(activation, context, v)

                tbar.set_description_str(
                    f"PRCAReconGreedy [device={device}] obj={obj.detach().cpu().numpy():.4e}"
                )

                obj.backward()

                with torch.no_grad():
                    ov = v

                    grad = v.grad
                    grad = (I - torch.outer(v, v)) @ grad

                    @torch.no_grad()
                    def linesearch(p, g):
                        # todo: separate this from function
                        max_alpha = 2 * np.pi / torch.linalg.norm(g)

                        arr_steps = torch.linspace(0, max_alpha, 100).to(device)
                        norm_g = torch.linalg.norm(g)

                        # construct candidate from exponential map at different time steps
                        term_cos = torch.outer(p, torch.cos(norm_g * arr_steps))
                        term_sin = torch.outer(
                            (g / norm_g), torch.sin(norm_g * arr_steps)
                        )

                        arr_directions = term_cos + term_sin

                        ref_rel = (activation * context).sum(axis=1, keepdims=True)

                        rel_on_dir = (activation @ arr_directions) * (
                            context @ arr_directions
                        )

                        arr_obj_directions = (
                            -torch.mean((ref_rel - rel_on_dir) ** 2, axis=0)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        assert arr_obj_directions.shape == (
                            arr_steps.shape[0],
                        ), arr_obj_directions.shape

                        best_ix = np.argmax(arr_obj_directions)
                        best_lr = arr_steps[best_ix].detach().cpu()

                        return best_lr

                    if torch.linalg.norm(grad).detach().cpu().numpy() == 0:
                        break

                    lr = linesearch(v, grad)

                    assert lr >= 0

                    if lr == 0:
                        break

                    # update v with gradient `ascent`.
                    v = exponential_map(v, grad, lr)

                    # theorethically, the exponential map should return a vector with unitnorm,
                    # but sometimes there is some numerical instability.
                    v = v / torch.linalg.norm(v)

                    np.testing.assert_allclose(
                        torch.linalg.norm(v).detach().cpu(), 1.0, atol=1e-3
                    )

                    if (v @ ov).abs() > (1 - eps):
                        # stop if the solution isn't update anymore.
                        break

            U[:, k] = v.detach()

        with torch.no_grad():

            Q, R = torch.linalg.qr(U)
            U = Q
        np.testing.assert_allclose(
            (U.T @ U).detach().cpu().numpy(), np.eye(d), atol=1e-6
        )

        return U.detach().cpu().numpy()

    @staticmethod
    def _obj_func(
        activation: torch.Tensor,
        context: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:

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

        return loss


class PRCASignAlignGreedy:
    def fit(
        self,
        activation: npt.NDArray,
        context: npt.NDArray,
        epochs=100,
        eps=1e-6,
        seed=1,
        largest_k=None,
        device="cpu",
    ) -> npt.NDArray:
        assert activation.shape == context.shape

        _, d = activation.shape

        activation = activation / ((np.mean(activation**2) ** (1 / 2)) * (d ** (1 / 4)))
        context = context / ((np.mean(context**2) ** (1 / 2)) * (d ** (1 / 4)))

        activation = torch.from_numpy(activation).float().to(device)
        context = torch.from_numpy(context).float().to(device)

        ori_activation = activation.clone()
        ori_context = context.clone()

        ori_rel = (ori_activation * ori_context).sum(dim=1)

        rng = torch.Generator()
        rng.manual_seed(seed)

        U = torch.zeros(d, d)
        U = U.to(device)

        I = torch.eye(d).to(device)

        if largest_k is None:
            largest_k = d

        tbar = tqdm(
            range(largest_k),
            total=largest_k,
            desc=f"PRCASignAlignGreedy [device={device}]]",
        )

        for k in tbar:
            if k >= 1:
                Uk = U[:, :k]
                U_pinv = torch.linalg.pinv(Uk)
                U_complement = I - U_pinv.T @ Uk.T
            else:
                U_pinv = 0
                U_complement = I

            v = torch.randn(d, generator=rng).to(device)
            v = v / torch.linalg.norm(v)

            v = v.to(device)

            # take only activation and context that are not captured by the learned directions
            activation = activation @ U_complement
            context = context @ U_complement

            for _ in range(epochs):
                v.requires_grad_(True)
                v.grad = None

                obj = self._obj_func(activation, context, v)

                obj.backward()

                with torch.no_grad():
                    ov = v

                    grad = v.grad
                    grad = (I - torch.outer(v, v)) @ grad

                    @torch.no_grad()
                    def linesearch(p, g):
                        max_alpha = 2 * np.pi / torch.linalg.norm(g)

                        arr_steps = torch.linspace(0, max_alpha, 100).to(device)
                        norm_g = torch.linalg.norm(g)

                        # construct candidate from exponential map at different time steps
                        term_cos = torch.outer(p, torch.cos(norm_g * arr_steps))
                        term_sin = torch.outer(
                            (g / norm_g), torch.sin(norm_g * arr_steps)
                        )

                        arr_directions = term_cos + term_sin

                        ref_rel = (activation * context).sum(axis=1, keepdims=True)

                        rel_on_dir = (activation @ arr_directions) * (
                            context @ arr_directions
                        )

                        arr_obj_directions = (
                            torch.mean(ref_rel * rel_on_dir, axis=0)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        assert arr_obj_directions.shape == (
                            arr_steps.shape[0],
                        ), arr_obj_directions.shape

                        best_ix = np.argmax(arr_obj_directions)
                        best_lr = arr_steps[best_ix].detach().cpu()

                        return best_lr

                    if torch.linalg.norm(grad).detach().cpu().numpy() == 0:
                        break

                    lr = linesearch(v, grad)

                    assert lr >= 0

                    if lr == 0:
                        break

                    # update v with gradient `ascent`.
                    v = exponential_map(v, grad, lr)

                    # theorethically, the exponential map should return a vector with unitnorm,
                    # but sometimes there is some numerical instability.
                    v = v / torch.linalg.norm(v)

                    np.testing.assert_allclose(
                        torch.linalg.norm(v).detach().cpu(), 1.0, atol=1e-3
                    )

                    if (v @ ov).abs() > (1 - eps):
                        # stop if the solution isn't update anymore.
                        break

            U[:, k] = v.detach()

            with torch.no_grad():

                Uk = U[:, : (k + 1)]

                projected_rel = ((ori_activation @ Uk) * (ori_context @ Uk)).sum(dim=1)
                assert projected_rel.shape == ori_rel.shape

                sign_align = (torch.sign(ori_rel) * torch.sign(projected_rel)).mean()

                perc_sign_align = sign_align.detach().cpu().numpy()
                tbar.set_description_str(
                    f"PRCASignAlignGreedy [device={device}] perc_sign_align={perc_sign_align:.4f}"
                )

        with torch.no_grad():
            Q, R = torch.linalg.qr(U)
            U = Q
        np.testing.assert_allclose(
            (U.T @ U).detach().cpu().numpy(), np.eye(d), atol=1e-6
        )

        return U.detach().cpu().numpy()

    @staticmethod
    def _obj_func(
        activation: torch.Tensor,
        context: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:

        activation_projected = activation.matmul(u)
        context_projected = context.matmul(u)

        assert len(activation_projected.shape) == len(context_projected.shape) == 1

        relevance_original = (activation * context).sum(dim=1)
        relevance_projected = activation_projected * context_projected
        assert relevance_original.shape == relevance_projected.shape

        obj = relevance_original * relevance_projected

        assert len(obj.shape) == 1 and obj.shape[0] == activation.shape[0]

        loss = obj.mean()

        return loss


class PRCASignAlignGreedyV2:
    @torch.no_grad()
    def fit(
        self, arr_act: npt.NDArray, arr_ctx: npt.NDArray, device="cpu"
    ) -> npt.NDArray:
        n, d = arr_act.shape

        I = torch.eye(d).to(device)

        arr_act: torch.Tensor = torch.from_numpy(arr_act).to(device)
        arr_ctx: torch.Tensor = torch.from_numpy(arr_ctx).to(device)

        U_prev = self._solve_objective(arr_act, arr_ctx)
        assert U_prev.shape == (d, 1)

        for k in tqdm(range(1, d), desc="PRCASignGreedyV2"):

            U_comp = I - U_prev @ U_prev.T

            arr_act_on_U_comp = arr_act @ U_comp
            arr_ctx_on_U_comp = arr_ctx @ U_comp

            u_star = self._solve_objective(arr_act_on_U_comp, arr_ctx_on_U_comp)

            # stacking prev results and the eigenvector
            # shape : (d, k+1)
            U_prev_and_k = torch.hstack([U_prev, u_star])

            # performing gram-schmidt
            U_prev, _ = torch.linalg.qr(U_prev_and_k)

            assert U_prev.shape == (d, k + 1)

            # sanity check
            Vk = U_prev.detach().cpu().numpy()
            np.testing.assert_allclose(Vk.T @ Vk, np.eye(k + 1), atol=1e-3)

        U = U_prev.detach().cpu().numpy()

        # sanity_check
        np.testing.assert_allclose(U.T @ U, np.eye(d), atol=1e-3)

        return U

    def _solve_objective(
        self, arr_act: torch.Tensor, arr_ctx: torch.Tensor
    ) -> torch.Tensor:

        arr_rel = (arr_act * arr_ctx).sum(dim=1, keepdim=True)

        arr_act_with_rel = arr_act * arr_rel

        cov = arr_act_with_rel.T @ arr_ctx + arr_ctx.T @ arr_act_with_rel

        _, eigvecs = torch.linalg.eigh(cov)

        return eigvecs[:, -1].reshape((-1, 1))
