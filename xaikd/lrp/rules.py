import torch
from zennit.core import expand
from zennit.rules import NoMod, ClampMod, GammaMod, Hook, BasicHook, zero_bias


def lrp_rule_ratio(nom, denom, eps) -> torch.Tensor:
    # Remark: for some reason, torch automatically remove the batch axis of context
    # could this be PyTorch's bug?
    if nom.shape[0] == 1 and len(nom.shape) == 4 and len(nom.shape) == 3:
        output = output.unsqueeze(0)

    # this trick combats getting nan from backprop of x/0.
    # see https://github.com/pytorch/pytorch/issues/4132
    nonzero_ix = denom.abs() > eps

    new_output = torch.zeros_like(nom)

    new_output[nonzero_ix] = nom[nonzero_ix] / denom[nonzero_ix]

    err_msg = f"min(abs(denom))={torch.abs(denom[nonzero_ix]).min().detach().cpu().numpy():.4f}"

    assert not torch.isnan(new_output).any(), err_msg

    return new_output


class SafeGamma(BasicHook):
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):
        mod_kwargs = {"zero_params": zero_params}
        mod_kwargs_nobias = {"zero_params": zero_bias(zero_params)}
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input,
            ],
            param_modifiers=[
                GammaMod(gamma, min=0.0, **mod_kwargs),
                GammaMod(gamma, max=0.0, **mod_kwargs_nobias),
                GammaMod(gamma, max=0.0, **mod_kwargs),
                GammaMod(gamma, min=0.0, **mod_kwargs_nobias),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    output * lrp_rule_ratio(nom=out_grad, denom=denom, eps=stabilizer)
                    for output, denom in (
                        [(outputs[4] > 0.0, sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0.0, sum(outputs[2:4]))] * 2
                    )
                ]
                + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(
                    input * gradient
                    for input, gradient in zip(inputs[:4], gradients[:4])
                )
            ),
        )


class GammaForPooling(BasicHook):
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):

        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0) * (1 + gamma),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0) * (1 + gamma),
                lambda input: input,
            ],
            param_modifiers=[
                NoMod(),
                NoMod(),
                NoMod(),
                NoMod(),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    output * lrp_rule_ratio(nom=out_grad, denom=denom, eps=stabilizer)
                    for output, denom in (
                        [(outputs[4] > 0.0, sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0.0, sum(outputs[2:4]))] * 2
                    )
                ]
                + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(
                    input * gradient
                    for input, gradient in zip(inputs[:4], gradients[:4])
                )
            ),
        )


class SafeGammaForPooling(BasicHook):
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0) * (1 + gamma),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0) * (1 + gamma),
                lambda input: input,
            ],
            param_modifiers=[
                NoMod(),
                NoMod(),
                NoMod(),
                NoMod(),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    # output * out_grad / stabilizer_fn(denom)
                    output * lrp_rule_ratio(nom=out_grad, denom=denom, eps=stabilizer)
                    for output, denom in (
                        [(outputs[4] > 0.0, sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0.0, sum(outputs[2:4]))] * 2
                    )
                ]
                + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(
                    input * gradient
                    for input, gradient in zip(inputs[:4], gradients[:4])
                )
            ),
        )


class SafeZBox(BasicHook):
    def __init__(self, low, high, stabilizer=1e-6, zero_params=None):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        mod_kwargs = {"zero_params": zero_params}

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: expand(low, input.shape, cut_batch_dim=True).to(input),
                lambda input: expand(high, input.shape, cut_batch_dim=True).to(input),
            ],
            param_modifiers=[
                NoMod(**mod_kwargs),
                ClampMod(min=0.0, **mod_kwargs),
                ClampMod(max=0.0, **mod_kwargs),
            ],
            output_modifiers=[lambda output: output] * 3,
            gradient_mapper=(
                lambda out_grad, outputs: (
                    lrp_rule_ratio(out_grad, sub(*outputs), eps=stabilizer),
                )
                * 3
            ),
            reducer=(
                lambda inputs, gradients: sub(
                    *(input * gradient for input, gradient in zip(inputs, gradients))
                )
            ),
        )


class SafeEpsilon(BasicHook):
    def __init__(self, epsilon=1e-6, zero_params=None):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(
                lambda out_grad, outputs: lrp_rule_ratio(
                    nom=out_grad, denom=outputs[0], eps=epsilon
                )
            ),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )


class PassWithConstantSign(Hook):
    """Unmodified pass-through rule.
    If the rule of a layer shall not be any other, is elementwise and shall not be the gradient, the `Pass` rule simply
    passes upper layer relevance through to the lower layer.
    """

    def backward(self, module, grad_input, grad_output):
        """Pass through the upper gradient, skipping the one for this layer."""

        if isinstance(module.constant, torch.Tensor):
            sign = torch.sign(module.constant.detach())
        else:
            sign = np.sign(module.constant)

        sign = float(sign)

        return tuple([sign * grad for grad in grad_output])
