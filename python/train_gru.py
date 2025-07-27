from os import PathLike
from pathlib import Path
import torch
import torchaudio
from torch import nn
from tqdm import tqdm
from functools import partial
from typing import Optional

from utils import chain_functions, simple_filter

SAMPLERATE = 44100
SEQUENCE_LENGTH = 4096 + 8192
OVERLAP_LENGTH = 4096
HOP_LENGTH = SEQUENCE_LENGTH - OVERLAP_LENGTH
BATCH_SIZE = 25
LR = 5e-3
EPOCHS = 50
EARLY_STOPPING = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

source_dataset_path = Path(r"D:\Datasets\SignalTrain_LA2A_Aug\4A2A_no_makeup")
target_dataset_path = Path(r"D:\Datasets\SignalTrain_LA2A_Dataset_1.1")

prefilter = partial(
    simple_filter,
    a1=torch.tensor(-0.995, device=DEVICE),
    b1=torch.tensor(-1, device=DEVICE),
)


def esr_loss(*args):
    pred, target = tuple(
        map(
            chain_functions(
                prefilter,
                torch.flatten
            ),
            args
        )
    )
    diff = pred - target
    return diff @ diff / (target @ target + 1e-5)


class SignalTrain4A2ADataset(torch.utils.data.Dataset):
    def __init__(self, path24a2a: Path, path2signaltrain: Path, name_pattern: str = ""):
        super().__init__()

        self.all_input_files = list(
            filter(lambda f: torchaudio.info(f).num_frames > 44100 * 600, path24a2a.glob(f"**/*{name_pattern}*.wav")))
        all_target_files = [path2signaltrain / f.relative_to(path24a2a) for f in self.all_input_files]

        print("Loading files...")
        all_inputs = [torchaudio.load(f)[0] for f in tqdm(self.all_input_files)]
        min_len = min(map(lambda x: x.size(-1), all_inputs))
        self.all_inputs = torch.stack([x[:, :min_len] for x in all_inputs], 0)
        self.all_targets = torch.empty_like(self.all_inputs)
        for i, f in enumerate(tqdm(all_target_files)):
            self.all_targets[i] = torchaudio.load(f)[0][:, :min_len]
        print("Files loaded.")

        num_segments = [(min_len - OVERLAP_LENGTH) // HOP_LENGTH] * self.all_inputs.size(0)
        self.boundaries = torch.cumsum(torch.tensor([0] + num_segments), 0)

    def __len__(self):
        return self.boundaries[-1]

    def __getitem__(self, idx):
        bin_pos = torch.bucketize(idx, self.boundaries, right=True).item() - 1
        offset = (idx - self.boundaries[bin_pos]).item() * HOP_LENGTH
        x = self.all_inputs[bin_pos, :, offset:offset + SEQUENCE_LENGTH]
        y = self.all_targets[bin_pos, :, offset:offset + SEQUENCE_LENGTH]
        return x, y


class GRUAmp(nn.Module):
    def __init__(self, hidden_size):
        super(GRUAmp, self).__init__()
        self.rec = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        h, state = self.rec(x, state)
        return x + self.lin(h), state


if __name__ == "__main__":
    ckpt_prefix = "gru_jit_no_overlap"
    model = torch.jit.script(GRUAmp(8).to(DEVICE))
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    train_dataset = SignalTrain4A2ADataset(source_dataset_path, target_dataset_path, "3c")
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
    #                                            pin_memory=True)

    all_inputs = train_dataset.all_inputs.transpose(1, 2)
    all_targets = train_dataset.all_targets.squeeze(1)
    random_shifts = (torch.linspace(0, 1, all_inputs.size(0) + 1, device=DEVICE)[:-1] * all_inputs.size(1)).long()

    prev_avg_loss = float("inf")
    early_stopping_counter = 0
    for epoch in range(EPOCHS):
        n = 0
        avg_loss = 0
        random_shifts = random_shifts[torch.randperm(random_shifts.size(0))]
        for i in range(0, all_inputs.size(0)):
            all_inputs[i] = torch.roll(all_inputs[i], random_shifts[i].item(), 0)
            all_targets[i] = torch.roll(all_targets[i], random_shifts[i].item(), 0)

        # with tqdm(train_loader) as pbar:
        with tqdm(list(zip(all_inputs.split(SEQUENCE_LENGTH, 1), all_targets.split(SEQUENCE_LENGTH, 1)))) as pbar:
            state = None
            for x, y in pbar:
                # x, y = x.to(DEVICE).transpose(1, 2), y.to(DEVICE).squeeze(1)[:, OVERLAP_LENGTH:]
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.autocast(device_type="cuda"):
                    pred, state = model(x, state)
                    # pred = pred.squeeze(-1)[:, OVERLAP_LENGTH:]
                    pred = pred.squeeze(-1)
                    loss = esr_loss(pred, y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                state = state.detach()
                n += 1
                avg_loss += (loss.item() - avg_loss) / n
                pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

            if avg_loss < prev_avg_loss:
                torch.jit.save(model, f"{ckpt_prefix}_{epoch}_{avg_loss:.4f}.pt")
                prev_avg_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= EARLY_STOPPING:
                break
