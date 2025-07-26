import torch
import torch as tr
from torch import Tensor
from typing import List, Dict
from neutone_sdk import WaveformToWaveformBase
from neutone_sdk.utils import save_neutone_model


class GRUWrapper(WaveformToWaveformBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("state", torch.zeros(1, 1, 8))

    def get_model_name(self) -> str:
        return "GRU for 4A-2A"

    def get_model_authors(self) -> List[str]:
        return ["Chin-Yun Yu"]

    def get_model_short_description(self) -> str:
        return "LA-2A non-linear model."

    def get_model_long_description(self) -> str:
        return "A GRU model that learns to emulate the LA-2A non-linear behaviour."

    def get_technical_description(self) -> str:
        return self.get_model_long_description()

    def get_technical_links(self) -> Dict[str, str]:
        return {"Code": "https://github.com/aim-qmul/4a2a"}

    def get_tags(self) -> List[str]:
        return ["LA-2A", "GRU", "Compressor"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return True

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return True

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [44100]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, _: Dict[str, Tensor]) -> Tensor:
        x, state = self.model.forward(x.t().unsqueeze(0), self.state)
        self.state.copy_(state)
        return x.squeeze(0).t()

    def reset_model(self) -> bool:
        self.state.zero_()
        return True


if __name__ == "__main__":
    from pathlib import Path

    gru = torch.jit.load("gru_jit_no_overlap_13_0.0205.pt", map_location="cpu")
    neutone_gru = GRUWrapper(gru)
    print(neutone_gru)

    save_neutone_model(neutone_gru, Path("neutone_gru"), True, True)
