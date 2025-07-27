import torch
from pathlib import Path
from tqdm import tqdm
import torchaudio

from train_gru import (
    SignalTrain4A2ADataset,
    source_dataset_path,
    target_dataset_path,
    DEVICE,
    SAMPLERATE,
)

SEQUENCE_LENGTH = 32768
MODEL_CKPT = "gru_jit_no_overlap_13_0.0205.pt"

name_pattern = "3c__0"
output_path = Path(r"D:\Datasets\SignalTrain_LA2A_Aug\4A2A_GRU")


@torch.no_grad()
def main():
    dataset = SignalTrain4A2ADataset(
        source_dataset_path, target_dataset_path, name_pattern=name_pattern
    )
    all_inputs = dataset.all_inputs.transpose(1, 2)

    peak_values = list(
        map(lambda x: int(x.stem.split("__")[-1]), dataset.all_input_files)
    )
    all_output_files = [
        output_path / f.relative_to(source_dataset_path)
        for f in dataset.all_input_files
    ]
    print(peak_values)

    model = torch.jit.load(MODEL_CKPT).to(DEVICE)
    model.eval()
    state = None

    pred_buf = torch.empty_like(all_inputs)
    n = 0
    for x in tqdm(all_inputs.split(SEQUENCE_LENGTH, 1)):
        pred, state = model(x.to(DEVICE), state)
        pred_buf[:, n : n + pred.size(1)] = pred.cpu()
        n += pred.size(1)
    pred = pred_buf.mT

    for output_file, pred in tqdm(zip(all_output_files, pred)):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_file, pred, SAMPLERATE)


if __name__ == "__main__":
    main()
