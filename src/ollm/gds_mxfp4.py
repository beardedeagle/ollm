"""MXFP4 tensor unpacking helpers used by GDS-backed loaders."""

import math

import torch

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def convert_moe_packed_tensors(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    scales = scales.to(torch.int32) - 127
    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape[:-1]=} does not match {scales.shape=}"
    )
    lookup_table = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    *prefix_shape, group_count, block_size = blocks.shape
    rows_total = math.prod(prefix_shape) * group_count
    blocks = blocks.reshape(rows_total, block_size)
    scales = scales.reshape(rows_total, 1)
    output = torch.empty(rows_total, block_size * 2, dtype=dtype, device=blocks.device)

    for row_start in range(0, rows_total, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows_total)
        block_slice = blocks[row_start:row_end]
        exponent_slice = scales[row_start:row_end]
        index_low = (block_slice & 0x0F).to(torch.long)
        index_high = (block_slice >> 4).to(torch.long)
        output_slice = output[row_start:row_end]
        output_slice[:, 0::2] = lookup_table[index_low]
        output_slice[:, 1::2] = lookup_table[index_high]
        torch.ldexp(output_slice, exponent_slice, out=output_slice)

    output = output.reshape(*prefix_shape, group_count, block_size * 2).view(
        *prefix_shape, group_count * block_size * 2
    )
    return output.transpose(1, 2).contiguous()
