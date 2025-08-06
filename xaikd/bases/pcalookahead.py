import numpy as np
import numpy.typing as npt

from tqdm.autonotebook import tqdm
import torch


from torch import nn
from torch.nn import functional as F

from torch.nn.utils.parametrizations import orthogonal
from torch.optim.adam import Adam


from torch.nn.utils.parametrizations import orthogonal

from xaikd import datasets, utils, bases


def construct_projection_fh(mean: npt.NDArray, ortho_layer: nn.Module, device: str):
    ts_mean = torch.from_numpy(mean).reshape(1, -1, 1, 1).to(device)

    def fh(mod, inp, outp):

        U = ortho_layer.weight
        proj_mat = (U.T @ U).unsqueeze(2).unsqueeze(3)

        outp = outp - ts_mean
        outp.requires_grad_(True)
        outp = F.conv2d(outp, proj_mat)
        outp = outp + ts_mean

        return outp

    return fh


def fit_pcalookahead(
    model: nn.Module,
    layer: str,
    mean: npt.NDArray,
    Uinit: npt.NDArray,
    dataloader: datasets.DataLoader,
    device: str,
    num_epochs=10,
    lr=5e-3,
    verbose=True,
):
    d, k = Uinit.shape

    linear_layer = torch.nn.Linear(k, d, bias=False)

    linear_layer.weight = torch.nn.Parameter(torch.from_numpy(Uinit.T).float())

    ortho_layer = orthogonal(linear_layer).to(device)

    optimizer = Adam(ortho_layer.parameters(), lr=lr)

    fh = construct_projection_fh(mean, ortho_layer, device)

    module = utils.interceptor.get_module(model, layer)

    pgb = tqdm(range(num_epochs), disable=not verbose)

    for epoch in tqdm(pgb):
        for x, y in dataloader:
            optimizer.zero_grad()

            x = x.to(device)

            with torch.no_grad():
                target = model(x)

            hook = None
            try:
                hook = module.register_forward_hook(fh)

                recon = model(x)

                np.testing.assert_equal(recon.shape, target.shape)
                assert len(recon.shape) == 3

                loss = (recon - target) ** 2
                loss = torch.flatten(loss, start_dim=1).sum(dim=1).mean()

                loss.backward()
                optimizer.step()

                loss = loss.detach().cpu().numpy()

            finally:
                hook.remove()

            pgb.set_description(f"PCA-LH Optimization: k={k}; loss={loss:.4f} ")

    return ortho_layer.weight.T.detach().cpu().numpy()


@bases.register.register_basis()
class PCALookAhead(bases.orthogonal.PCA):

    def get_Uk(self, k: int) -> npt.NDArray[np.float32]:

        assert self.is_prepared

        # this U is from PCA
        Uinit = self.U[:, :k]

        dl = datasets.build_dataloader(
            self.ds_train,
            shuffle=True,
            batch_size=256,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        return fit_pcalookahead(
            model=self.model,
            layer=self.layer,
            mean=self.mean,
            Uinit=Uinit,
            dataloader=dl,
            device=self.device,
        )

    def set_model_layer_ds(
        self,
        model: nn.Module,
        layer: str,
        ds_train: datasets.Dataset,
        device: str,
    ):
        self.model = model
        self.layer = layer
        self.ds_train = ds_train
        self.device = device

        self.is_prepared = True
