import numpy as np
from pedalboard import load_plugin, Pedalboard, ExternalPlugin
from pedalboard.io import AudioFile
import argparse
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from scipy.signal import lfilter

uad_vst_path = r"C:\Program Files\Common Files\VST3\uaudio_teletronix_la-2a_tc.vst3\Contents\x86_64-win\uaudio_teletronix_la-2a_tc.vst3"
ik_vst_path = r"C:\Program Files\Common Files\VST3\TR5 White 2A.vst3"
cakewalk_vst_path = r"C:\Program Files\Common Files\VST3\CA2ALevelingAmplifier\CA-2ALevelingAmplifier_64.vst3"


def find_time_offset(x: np.ndarray, y: np.ndarray):
    N = x.size
    M = y.size

    X = np.fft.rfft(x, n=N + M - 1)
    Y = np.fft.rfft(y, n=N + M - 1)
    corr = np.fft.irfft(X.conj() * Y)
    shifts = np.argmax(corr, axis=-1)
    return np.where(shifts >= N, shifts - N - M + 1, shifts)


def esr(signal, target, adaptive_scale=False):
    signal = lfilter([1, -1], [1, -0.995], signal)
    target = lfilter([1, -1], [1, -0.995], target)
    target_norm = target @ target
    if adaptive_scale:
        scaler = (target_norm / (signal @ signal)) ** 0.5
    else:
        scaler = 1
    signal *= scaler
    # print(scaler)
    diff = signal - target
    return diff @ diff / target_norm, scaler


def compare_loudness(output_file, target_file):
    output, _ = sf.read(output_file)
    target, _ = sf.read(target_file)
    return 20 * np.log10(np.linalg.norm(output) / np.linalg.norm(target))


def ik_set_mode_and_peak_reduction(
    plugin: ExternalPlugin, mode: int, peak_reduction: int
):
    plugin.limit_compress = "Limit" if mode else "Compress"
    plugin.peak_reduction = peak_reduction


def cakewalk_set_mode_and_peak_reduction(
    plugin: ExternalPlugin, mode: int, peak_reduction: int
):
    plugin.peak_reduction = peak_reduction / 100
    plugin.limit_mode = "Limit" if mode else "Compress"


def uad_set_mode_and_peak_reduction(
    plugin: ExternalPlugin, mode: int, peak_reduction: int
):
    plugin.peak_reduct = str(peak_reduction)
    plugin.comp_limit = "Limit" if mode else "Comp"


def waves_set_mode_and_peak_reduction(
    plugin: ExternalPlugin, mode: int, peak_reduction: int
):
    plugin.comp_limiter = "Limit " if mode else "Comp "
    plugin.peak_reduction = peak_reduction


def main():
    parser = argparse.ArgumentParser(description="data augmentation")
    parser.add_argument("signaltrain", type=str, help="path to signal train")
    parser.add_argument("output", type=str, help="path to output")
    parser.add_argument("--vst", type=str, help="path to vst plugin")
    parser.add_argument(
        "--brand",
        type=str,
        choices=["uad", "ik", "cakewalk", "waves"],
        help="brand of vst plugin",
        default="uad",
    )
    parser.add_argument(
        "--gain", type=float, help="input gain before the plugin", default=0.0
    )
    parser.add_argument(
        "--out-gain", type=int, help="output gain on the plugin", default=30
    )
    parser.add_argument("--mode", type=int, help="mode", default=0)
    parser.add_argument(
        "--adaptive-scale", action="store_true", help="use adaptive scale"
    )

    args = parser.parse_args()
    signaltrain = Path(args.signaltrain).resolve()
    vst_path = Path(args.vst).resolve()
    output = Path(args.output).resolve()
    match args.brand:
        case "uad":
            plugin = load_plugin(
                str(vst_path),
                parameter_values={"gain": str(args.out_gain)},
                # parameter_values={"gain": "17"}
            )
            set_mode_and_peak_reduction = uad_set_mode_and_peak_reduction
        case "ik":
            plugin = load_plugin(
                str(vst_path),
                parameter_values={
                    # "gain": 50,
                    "gain": args.out_gain,
                },
            )
            set_mode_and_peak_reduction = ik_set_mode_and_peak_reduction
        case "cakewalk":
            plugin = load_plugin(
                str(vst_path),
                parameter_values={
                    # "gain": 0.35,
                    "gain": args.out_gain / 100,
                    "r37": 0,
                },
            )
            set_mode_and_peak_reduction = cakewalk_set_mode_and_peak_reduction
        case "waves":
            plugin = load_plugin(
                str(vst_path),
                plugin_name="CLA-2A Mono",
                parameter_values={
                    "gain": args.out_gain,
                    "hi_freq": 100,
                },
            )
            set_mode_and_peak_reduction = waves_set_mode_and_peak_reduction
        case _:
            raise ValueError("Invalid brand")

    print(plugin.parameters)

    if args.gain != 0.0:
        gain = 10 ** (args.gain / 20)
    else:
        gain = 1

    target_files = list(
        filter(
            lambda x: sf.info(x).frames > 44100 * 600,
            signaltrain.glob(f"**/*3c__{args.mode}__*.wav"),
        )
    )
    esr_list = [0] * len(target_files)
    with tqdm(target_files) as pbar:
        for target_file in pbar:
            file_num = target_file.stem.split("_")[1]
            input_file = target_file.parent / f"input_{file_num}_.wav"
            assert input_file.exists(), f"input file {input_file} does not exist"
            _, mode, peak_reduction = target_file.stem.split("__")
            # if int(peak_reduction) % 20 != 0:
            # if int(peak_reduction) not in (70, 75, 80):
            #     continue
            # print(f"Processing {input_file} with mode {mode} and peak reduction {peak_reduction}")
            plugin.reset()
            # print(plugin.parameters)
            set_mode_and_peak_reduction(plugin, int(mode), int(peak_reduction))

            output_file = output / target_file.relative_to(signaltrain)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            max_peak = 0

            pred = []
            with AudioFile(str(input_file)) as af:
                with AudioFile(
                    str(output_file), "w", af.samplerate, af.num_channels, bit_depth=32
                ) as af_out:
                    while af.tell() < af.frames:
                        data = af.read(af.samplerate) * gain
                        processed = plugin(data, af.samplerate, reset=False)
                        max_peak = max(max_peak, np.max(np.abs(processed)))
                        pred.append(processed)
                        af_out.write(processed)
            pred = np.concatenate(pred, axis=1).squeeze(0)
            # db_diff = compare_loudness(output_file, target_file)

            target, _ = sf.read(target_file)
            print(pred.shape, target.shape)
            esr_loss, scaler = esr(
                pred[: target.size], target[: pred.size], args.adaptive_scale
            )
            scaler_db = 20 * np.log10(scaler)
            # esr_list.append(esr_loss)
            esr_list[int(peak_reduction) // 5] = esr_loss

            pbar.set_postfix(
                {
                    "max_peak": max_peak,
                    "mode": mode,
                    "peak_reduction": peak_reduction,
                    "esr_loss": esr_loss,
                    "scaler_db": scaler_db,
                }
            )
        print(", ".join([f"{x}" for x in esr_list]))
        print(f"Mean ESR loss: {np.mean(esr_list)}")


if __name__ == "__main__":
    main()
