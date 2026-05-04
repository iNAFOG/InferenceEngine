#This file is for starup/runtime config for the engine itself
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    device: str = "auto"  # auto | cpu | cuda
