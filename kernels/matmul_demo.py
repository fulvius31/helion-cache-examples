"""
Helion matmul kernel demo (copy of Helion's CI-tested ``examples/matmul.py``).

Kernel correctness is inherited from the upstream test suite; every run here
additionally checks against ``torch.matmul`` before timing anything.  The copy
exists so the demos can edit the kernel *body* in place (case 1 / case 5b)
without touching the Helion checkout, and so the suite is frozen against
upstream example changes.

The kernel body and its decorator are preserved *exactly* as in the upstream
example (including ``autotune_config_overrides``) because the demos depend on
the config-space fingerprint and the kernel-source hash staying stable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
from helion._compat import use_tileir_tunables
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


@helion.kernel(
    # static_shapes=True gives a performance boost for matmuls
    static_shapes=True,
    # Disable autotuning over unrolling/range_num_stages
    # tl.dot is pipelined with num_stages
    # Note: range_unroll_factors and range_num_stages are not supported for tileir backend
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    }
    if not use_tileir_tunables()
    else {},
)
def matmul(
    x: Tensor,
    y: Tensor,
    epilogue: Callable[[Tensor, tuple[Tensor, ...]], Tensor] = lambda acc, tile: acc,
) -> Tensor:
    """Performs matrix multiplication of x and y with an optional epilogue.

    Args:
        x: Left matrix of shape [m, k].
        y: Right matrix of shape [k, n].
        epilogue: Function applied to the accumulator and tile indices after
            the matmul. Defaults to identity (no change).
    Returns:
        Resulting matrix of shape [m, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))
    return out


def check(m: int, k: int, n: int) -> None:
    """Check the copied kernel against torch.matmul (HALF_DTYPE, default epilogue)."""
    x = torch.randn([m, k], device=DEVICE, dtype=HALF_DTYPE)
    y = torch.randn([k, n], device=DEVICE, dtype=HALF_DTYPE)
    out = matmul(x, y)
    torch.testing.assert_close(out, torch.matmul(x, y), rtol=1e-2, atol=1e-1)
    print(f"matmul_demo check OK: ({m},{k},{n})")


def main() -> None:
    check(1024, 1024, 1024)


if __name__ == "__main__":
    main()
