from pathlib import Path
import yaml
import torch
from functools import partial
from torchcomp import ms2coef, coef2ms
import torchaudio
from tqdm import tqdm

from utils import logits2comp_params, compressor

SAMPLERATE = 44100
ATTACK_MIN = 0.1
ATTACK_MAX = 100
RELEASE_MIN = 10
RELEASE_MAX = 1000
RATIO_MIN = 1
RATIO_MAX = 20

m2c = partial(ms2coef, sr=44100)
c2m = partial(coef2ms, sr=44100)

at_min = m2c(torch.tensor(ATTACK_MIN))
at_max = m2c(torch.tensor(ATTACK_MAX))
rt_min = m2c(torch.tensor(RELEASE_MIN))
rt_max = m2c(torch.tensor(RELEASE_MAX))

comp_ckpts = Path("fitted/")
limit_ckpts = Path("limit_mode/")
signaltrain_path = Path(r"D:\Datasets\SignalTrain_LA2A_Dataset_1.1")
output_path = Path(r"D:\Datasets\SignalTrain_LA2A_Aug\4A2A_no_makeup")

my_logits2comp_params = partial(
    logits2comp_params,
    ratio_func=lambda x: 1 + 19 * torch.sigmoid(x),
    at_func=lambda x: at_min + (at_max - at_min) * torch.sigmoid(x),
    rt_func=lambda x: rt_min + (rt_max - rt_min) * torch.sigmoid(x),
)

peak_reductions = list(range(40, 101, 5))


@torch.no_grad()
def runner(ckpt_path, mode: int):
    for log, peak_reduction in tqdm(zip(map(lambda x: ckpt_path / f"la2a_{x}", peak_reductions), peak_reductions),
                                    total=len(peak_reductions)):
        with open(log / "config.yaml") as f:
            config = yaml.safe_load(f)
        range_details = config["compressor"]["range"]
        assert range_details["attack_ms"]["min"] == ATTACK_MIN
        assert range_details["attack_ms"]["max"] == ATTACK_MAX
        assert range_details["release_ms"]["min"] == RELEASE_MIN
        assert range_details["release_ms"]["max"] == RELEASE_MAX
        assert range_details["ratio"]["min"] == RATIO_MIN
        assert range_details["ratio"]["max"] == RATIO_MAX

        params = my_logits2comp_params(torch.load(log / "logits.pt", map_location="cpu"))
        params["make_up"].zero_()

        for match_file in signaltrain_path.glob(f"**/*c__{mode}__{peak_reduction}.wav"):
            input_file = match_file.parent / f"input_{match_file.stem.split('_')[1]}_.wav"
            assert input_file.exists(), f"input file {input_file} does not exist"

            output_file = output_path / match_file.relative_to(signaltrain_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"Processing {input_file} with peak reduction {peak_reduction}")

            x, sr = torchaudio.load(input_file)
            assert sr == SAMPLERATE

            pred = compressor(x, **params)

            torchaudio.save(output_file, pred, sr)


runner(limit_ckpts, 1)
runner(comp_ckpts, 0)
