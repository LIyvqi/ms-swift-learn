#!/usr/bin/env python3
"""快速检查持久化 ms-swift/ROCm 环境是否健康。"""

from pathlib import Path
import os
import sys

import torch
from flash_attn import flash_attn_func
import datasets
import fla
import modelscope
import peft
import swift
import transformers
import trl


def main() -> None:
    project_root = Path(__file__).resolve().parent
    expected_venv = project_root / ".venv"
    if not Path(sys.prefix).resolve().is_relative_to(expected_venv.resolve()):
        raise RuntimeError("请先执行 `source ./activate.sh`")

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}, hip={torch.version.hip}")
    print(f"transformers={transformers.__version__}")
    print(f"datasets={datasets.__version__}")
    print(f"peft={peft.__version__}, trl={trl.__version__}")
    print(f"ms-swift={swift.__version__}, modelscope={modelscope.__version__}")
    print(f"flash-linear-attention={fla.__version__}")

    assert torch.cuda.is_available(), "ROCm GPU is not available"
    assert torch.cuda.device_count() == 1
    assert torch.cuda.is_bf16_supported()
    props = torch.cuda.get_device_properties(0)
    print(
        f"gpu_arch={props.gcnArchName}, "
        f"vram_gib={props.total_memory / 1024**3:.2f}"
    )

    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    (x @ x).float().mean().backward()

    q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    flash_attn_func(q, k, v, causal=True).float().mean().backward()
    torch.cuda.synchronize()

    for name in (
        "MODELSCOPE_CACHE",
        "MODELSCOPE_HOME",
        "HF_HOME",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        path = Path(os.environ[name]).resolve()
        assert path.is_relative_to(project_root), f"{name} is not persistent: {path}"

    print("BF16_MATMUL=OK")
    print("FLASH_ATTENTION=OK")
    print("FLASH_LINEAR_ATTENTION=OK")
    print("PERSISTENT_CACHE_PATHS=OK")
    print("ENVIRONMENT_CHECK=PASS")


if __name__ == "__main__":
    main()
