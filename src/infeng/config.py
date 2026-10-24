#This file is for starup/runtime config for the engine itself
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EngineConfig:
    # env vars let you switch model/device for the API without touching code,
    # e.g.  $env:INFENG_MODEL="Qwen/Qwen2.5-0.5B"
    model_name: str = field(default_factory=lambda: os.getenv("INFENG_MODEL", "sshleifer/tiny-gpt2"))
    device: str = field(default_factory=lambda: os.getenv("INFENG_DEVICE", "auto"))  # auto | cpu | cuda
    dtype: str = field(default_factory=lambda: os.getenv("INFENG_DTYPE", "float32"))  # float32 | float16 | bfloat16
    use_kv_cache: bool = True
