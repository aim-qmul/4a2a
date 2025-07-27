import torch
from torchcomp import compexp_gain, db2amp, amp2db
from torchlpc import sample_wise_lpc
from torch.autograd import Function
from functools import reduce


def simple_filter(x: torch.Tensor, a1: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    return sample_wise_lpc(
        x + b1 * torch.cat([x.new_zeros(x.shape[0], 1), x[:, :-1]], dim=1),
        a1.broadcast_to(x.shape + (1,)),
    )


def chain_functions(*functions):
    return lambda x: reduce(lambda x, f: f(x), functions, x)


def arcsigmoid(x: torch.Tensor) -> torch.Tensor:
    return (x / (1 - x)).log()


def comp_gain(x, *args, **kwargs) -> torch.Tensor:
    return compexp_gain(x, *args, exp_ratio=0.9999, exp_thresh=-120, **kwargs)


def avg(rms: torch.Tensor, avg_coef: torch.Tensor):
    """Compute the running average of a signal.

    Args:
        rms (torch.Tensor): Input signal.
        avg_coef (torch.Tensor): Coefficient for the average RMS.

    Shape:
        - rms: :math:`(B, T)` where :math:`B` is the batch size and :math:`T` is the number of samples.
        - avg_coef: :math:`(B,)` or a scalar.

    """

    avg_coef = avg_coef.broadcast_to(rms.shape[0])
    assert torch.all(avg_coef > 0) and torch.all(avg_coef <= 1)

    return sample_wise_lpc(
        rms * avg_coef.unsqueeze(1),
        avg_coef[:, None, None].broadcast_to(rms.shape + (1,)) - 1,
    )


def avg_rms(audio: torch.Tensor, avg_coef: torch.Tensor) -> torch.Tensor:
    return avg(audio.square().clamp_min(1e-8), avg_coef).sqrt()


def gain_reduction(x, th, ratio, at, rt, make_up, delay: int = 0):
    # rms = avg_rms(x, avg_coef=avg_coef)
    gain = comp_gain(
        x.abs().clamp_min(1e-8),
        comp_ratio=ratio,
        comp_thresh=th,
        at=at,
        rt=rt,
    )
    if delay > 0:
        gain = torch.cat([gain[:, delay:], gain.new_ones(gain.shape[0], delay)], dim=1)
    return gain * db2amp(make_up.broadcast_to(x.shape[0], 1))


def compressor(x, *args, **kwargs):
    return gain_reduction(x, *args, **kwargs) * x


def compressor_inverse_filter(y, *args, **kwargs):
    return y / gain_reduction(*args, **kwargs)


def logits2comp_params(
    logits: torch.Tensor,
    ratio_func=lambda x: 1 + torch.exp(x),
    at_func=torch.sigmoid,
    rt_func=torch.sigmoid,
):
    th, ratio_logits, at_logits, rt_logits, make_up = torch.unbind(logits, dim=0)
    # avg_coef = torch.sigmoid(avg_coef_logits)
    # ratio = 1 + torch.exp(ratio_logits)
    ratio = ratio_func(ratio_logits)
    at = at_func(at_logits)
    rt = rt_func(rt_logits)
    # at = torch.sigmoid(at_logits)
    # rt = torch.sigmoid(rt_logits)
    return {
        # "avg_coef": avg_coef,
        "th": th,
        "ratio": ratio,
        "at": at,
        "rt": rt,
        "make_up": make_up,
    }


def simple_compressor(x, avg_coef, th, ratio, at, *args, **kwargs):
    return compressor(x, avg_coef, th, ratio, at, at, *args, **kwargs)


def esr(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.flatten()
    target = target.flatten()
    diff = pred - target
    return (diff @ diff) / (target @ target)
