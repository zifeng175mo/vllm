# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch


def sanity_check_mm_encoder_outputs(
    mm_embeddings: object,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    :meth:`vllm.model_executor.models.SupportsMultiModal.get_multimodal_embeddings`.
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")


def scatter_mm_placeholders(
    embeds: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Scatter the multimodal embeddings into a contiguous tensor that represents
    the placeholder tokens.

    :class:`vllm.multimodal.processing.PromptUpdateDetails.is_embed`.

    Args:
        embeds: The multimodal embeddings.
          Shape: `(num_embeds, embed_dim)`
        is_embed: A boolean mask indicating which positions in the placeholder
          tokens need to be filled with multimodal embeddings.
          Shape: `(num_placeholders, num_embeds)`
    """
    if is_embed is None:
        return embeds

    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed] = embeds
    return placeholders


def gather_mm_placeholders(
    placeholders: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Reconstructs the embeddings from the placeholder tokens.

    This is the operation of :func:`scatter_mm_placeholders`.
    """
    if is_embed is None:
        return placeholders

    return placeholders[is_embed]

def biload_blocks(
    src: torch.Tensor, 
    dst: torch.Tensor, 
    block_mapping: torch.Tensor
) -> None:
    if src.device.type != 'cpu':
        raise ValueError(f"src must be on CPU, but got {src.device.type}")
    if dst.device.type != 'cpu':
        raise ValueError(f"dst must be on CPU, but got {dst.device.type}")
    if block_mapping.device.type != 'cpu':
        raise ValueError("block_mapping must be on CPU")

    src_np = src.numpy().view(np.uint8).ravel()
    dst_np = dst.numpy().view(np.uint8).ravel()

    block_size = src.element_size() * src.stride(0)

    num_blocks = block_mapping.size(0)
    for i in range(num_blocks):
        # copy by block.
        src_idx = int(block_mapping[i, 0].item())
        dst_idx = int(block_mapping[i, 1].item())
        src_start = src_idx * block_size
        dst_start = dst_idx * block_size
        if src_start + block_size > src_np.size or dst_start + block_size > dst_np.size:
            raise IndexError(f"block index out of range: [{src_idx}->{dst_idx}]")

        dst_np[dst_start: dst_start + block_size] = src_np[src_start: src_start + block_size]
