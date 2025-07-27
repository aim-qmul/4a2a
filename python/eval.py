import torch
import yaml
import argparse
from pathlib import Path
from functools import partial
from itertools import starmap
from torchcomp import ms2coef, coef2ms
from scipy.interpolate import CubicSpline
import torchaudio

from utils import logits2comp_params, esr, compressor, chain_functions, simple_filter


SAMPLERATE = 44100
ATTACK_MIN = 0.1
ATTACK_MAX = 100
RELEASE_MIN = 10
RELEASE_MAX = 1000
RATIO_MIN = 1
RATIO_MAX = 20

m2c = partial(ms2coef, sr=44100)
c2m = partial(coef2ms, sr=44100)


def eval_loop(
    input_files,
    target_files,
    peak_reductions,
    params,
    name,
    device,
):
    esr_list = []
    prefilter = partial(
        simple_filter,
        a1=torch.tensor(-0.995, device=device),
        b1=torch.tensor(-1, device=device),
    )
    for input_file, target_file, param, peak in zip(
        input_files, target_files, params, peak_reductions
    ):
        x, sr = torchaudio.load(input_file)
        assert sr == SAMPLERATE
        x = x.to(device)
        y = torchaudio.load(target_file)[0].to(device)

        loss = esr(
            prefilter(compressor(x, **param)),
            prefilter(y),
        )
        print(f"Peak reduction: {peak}, ESR: {loss.item()}")
        esr_list.append(loss.item())
    print(f"{name} Average ESR: {sum(esr_list) / len(esr_list)}")
    return esr_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("dir", help="Directory containing the logs")
    parser.add_argument("csv", help="Output CSV file")
    parser.add_argument("--device", default="cpu", help="Device to use")
    args = parser.parse_args()

    logs_dir = Path(args.dir)
    device = args.device

    at_min = m2c(torch.tensor(ATTACK_MIN, device=device))
    at_max = m2c(torch.tensor(ATTACK_MAX, device=device))
    rt_min = m2c(torch.tensor(RELEASE_MIN, device=device))
    rt_max = m2c(torch.tensor(RELEASE_MAX, device=device))

    my_logits2comp_params = partial(
        logits2comp_params,
        ratio_func=lambda x: 1 + 19 * torch.sigmoid(x),
        at_func=lambda x: at_min + (at_max - at_min) * torch.sigmoid(x),
        rt_func=lambda x: rt_min + (rt_max - rt_min) * torch.sigmoid(x),
    )

    peak_reductions = list(range(40, 101, 5))

    input_files = []
    target_files = []
    params = []
    for log in map(lambda x: logs_dir / f"la2a_{x}", peak_reductions):
        with open(log / "config.yaml") as f:
            config = yaml.safe_load(f)
        range_details = config["compressor"]["range"]
        assert range_details["attack_ms"]["min"] == ATTACK_MIN
        assert range_details["attack_ms"]["max"] == ATTACK_MAX
        assert range_details["release_ms"]["min"] == RELEASE_MIN
        assert range_details["release_ms"]["max"] == RELEASE_MAX
        assert range_details["ratio"]["min"] == RATIO_MIN
        assert range_details["ratio"]["max"] == RATIO_MAX

        data_details = config["data"]["train"]
        input_file = data_details["input"]
        target_file = data_details["target"]

        input_files.append(input_file)
        target_files.append(target_file)
        params.append(
            my_logits2comp_params(torch.load(log / "logits.pt", map_location=device))
        )

    error_bounds = eval_loop(
        input_files,
        target_files,
        peak_reductions,
        params,
        "Ground Truth",
        device,
    )

    convert2ms = lambda d: dict(
        starmap(lambda k, v: (k, c2m(v) if k in ["at", "rt"] else v), d.items())
    )
    convert2coef = lambda d: dict(
        starmap(lambda k, v: (k, m2c(v) if k in ["at", "rt"] else v), d.items())
    )

    # test interpolation
    train_peaks = peak_reductions[::2]
    test_peaks = peak_reductions[1::2]
    train_params = params[::2]
    concat_one_param = lambda plist, k: torch.cat([p[k].view(1) for p in plist])
    linear_interp = lambda ps: [
        dict(zip(params[0].keys(), v))
        for v in zip(
            *[
                (x[:-1] + x[1:]) / 2
                for x in map(
                    partial(concat_one_param, ps),
                    params[0].keys(),
                )
            ]
        )
    ]
    spline_interp = lambda ps, xp, x: [
        dict(zip(params[0].keys(), v))
        for v in zip(
            *[
                torch.from_numpy(
                    CubicSpline(
                        xp,
                        concat_one_param(ps, k).cpu().numpy(),
                    )(x)
                ).to(device)
                for k in params[0].keys()
            ]
        )
    ]

    linear_interp_params = chain_functions(
        partial(map, convert2ms), list, linear_interp, partial(map, convert2coef), list
    )(train_params)
    spline_interp_params = chain_functions(
        partial(map, convert2ms),
        list,
        partial(spline_interp, xp=train_peaks, x=test_peaks),
        partial(map, convert2coef),
        list,
    )(train_params)

    linear_esr_list = eval_loop(
        input_files[1::2],
        target_files[1::2],
        test_peaks,
        linear_interp_params,
        "Linear Interpolation",
        device,
    )
    spline_esr_list = eval_loop(
        input_files[1::2],
        target_files[1::2],
        test_peaks,
        spline_interp_params,
        "Spline Interpolation",
        device,
    )

    linear_esr_list = sum([[x, None] for x in linear_esr_list], [None])
    spline_esr_list = sum([[x, None] for x in spline_esr_list], [None])

    with open(args.csv, "w") as f:
        f.write(
            "Peak Reduction,ESR,,,Threshold (dB),Ratio,Attack (ms),Release (ms), Make-up Gain (dB)\n"
        )
        f.write(",GT,Linear Interpolation,Spline Interpolation,,,,,\n")
        for peak, gt, linear, spline, param in zip(
            peak_reductions,
            error_bounds,
            linear_esr_list,
            spline_esr_list,
            map(convert2ms, params),
        ):
            th = param["th"].item()
            ratio = param["ratio"].item()
            at = param["at"].item()
            rt = param["rt"].item()
            make_up = param["make_up"].item()
            if linear is None:
                f.write(f"{peak},{gt},,,{th},{ratio},{at},{rt},{make_up}\n")
            else:
                f.write(
                    f"{peak},{gt},{linear},{spline},{th},{ratio},{at},{rt},{make_up}\n"
                )


if __name__ == "__main__":
    main()
