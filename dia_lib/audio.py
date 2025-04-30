# ComfyUI-DiaTTS/dia_lib/audio.py

import typing as tp

import torch


def build_delay_indices(B: int, T: int, C: int, delay_pattern: tp.List[int], device: torch.device | None = None) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    Negative t_idx => BOS; t_idx >= T => PAD.
    Creates tensors directly on the specified device.
    """
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device)

    t_idx_BxT = torch.broadcast_to(
        torch.arange(T, dtype=torch.int32, device=device)[None, :],
        [B, T],
    )
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_arr.view(1, 1, C) # Result inherits device

    b_idx_BxTxC = torch.broadcast_to(
        torch.arange(B, dtype=torch.int32, device=device).view(B, 1, 1),
        [B, T, C],
    )
    c_idx_BxTxC = torch.broadcast_to(
        torch.arange(C, dtype=torch.int32, device=device).view(1, 1, C),
        [B, T, C],
    )

    # We must clamp time indices to [0..T-1] so gather_nd equivalent won't fail
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1) # Inherits device

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_clamped_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        dim=1,
    ).long() # Ensure indices are long type, inherits device

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    bos_value: int,
    precomp: tp.Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Applies the delay pattern to batched audio tokens using precomputed indices,
    inserting BOS where t_idx < 0 and PAD where t_idx >= T.
    Assumes precomp tensors are already on the correct device.
    """
    device = audio_BxTxC.device
    t_idx_BxTxC, indices_BTCx3 = precomp

    # Verify devices just in case, but ideally they match 'device'
    if t_idx_BxTxC.device != device: t_idx_BxTxC = t_idx_BxTxC.to(device)
    if indices_BTCx3.device != device: indices_BTCx3 = indices_BTCx3.to(device)

    # Equivalent of tf.gather_nd using advanced indexing
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.shape)

    # Create masks on the correct device
    mask_bos = t_idx_BxTxC < 0
    mask_pad = t_idx_BxTxC >= audio_BxTxC.shape[1]

    # Create scalar tensors on the correct device
    bos_tensor = torch.tensor(bos_value, dtype=audio_BxTxC.dtype, device=device)
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)

    # If mask_bos, BOS; else if mask_pad, PAD; else original gather
    result_BxTxC = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))

    return result_BxTxC


def build_revert_indices(B: int, T: int, C: int, delay_pattern: tp.List[int], device: torch.device | None = None) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute indices for the revert operation using PyTorch.
    Creates tensors directly on the specified device.
    """
    delay_arr = torch.tensor(delay_pattern, dtype=torch.int32, device=device)

    t_idx_BT1 = torch.broadcast_to(torch.arange(T, dtype=torch.int32, device=device).unsqueeze(0), [B, T])
    t_idx_BT1 = t_idx_BT1.unsqueeze(-1)

    # Use torch.tensor for T-1 to ensure it's on the correct device
    T_minus_1_tensor = torch.tensor(T - 1, dtype=torch.int32, device=device)
    t_idx_BxTxC = torch.minimum(
        t_idx_BT1 + delay_arr.view(1, 1, C),
        T_minus_1_tensor, # Use tensor here
    )
    b_idx_BxTxC = torch.broadcast_to(torch.arange(B, dtype=torch.int32, device=device).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C, dtype=torch.int32, device=device).view(1, 1, C), [B, T, C])

    indices_BTCx3 = torch.stack(
        [
            b_idx_BxTxC.reshape(-1),
            t_idx_BxTxC.reshape(-1),
            c_idx_BxTxC.reshape(-1),
        ],
        axis=1,
    ).long() # Ensure indices are long type

    return t_idx_BxTxC, indices_BTCx3


def revert_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    precomp: tp.Tuple[torch.Tensor, torch.Tensor], # Assumes already on correct device
    T: int,
) -> torch.Tensor:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices (PyTorch version).
    Assumes precomp tensors are already on the correct device.
    """
    t_idx_BxTxC, indices_BTCx3 = precomp
    device = audio_BxTxC.device

    # Verify devices just in case, but ideally they match 'device'
    if t_idx_BxTxC.device != device: t_idx_BxTxC = t_idx_BxTxC.to(device)
    if indices_BTCx3.device != device: indices_BTCx3 = indices_BTCx3.to(device)

    # Using PyTorch advanced indexing
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.size())

    # Create pad_tensor and T_tensor on the correct device
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
    # Use T_idx_BxTxC's dtype for comparison tensor
    T_tensor = torch.tensor(T, dtype=t_idx_BxTxC.dtype, device=device)

    result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)

    return result_BxTxC


@torch.no_grad()
@torch.inference_mode()
def decode(
    model, # DAC model
    audio_codes, # Input codes tensor
):
    """
    Decodes the given frames into an output audio waveform
    """
    if len(audio_codes) != 1:
        raise ValueError(f"Expected one frame, got {len(audio_codes)}")

    # Ensure model and codes are on the same device before calling internal methods
    model_device = next(model.parameters()).device
    if audio_codes.device != model_device:
        print(f"Decode function: Moving audio_codes from {audio_codes.device} to model device {model_device}")
        audio_codes = audio_codes.to(model_device)

    try:
        # Now call internal DAC methods, expecting inputs to be on model_device
        audio_values = model.quantizer.from_codes(audio_codes)
        audio_values = model.decode(audio_values[0]) # model.decode expects [1, T_audio]? Check DAC source if needed.
        # The original call was model.decode(audio_values[0]), assuming audio_values was [B, D, T_z]
        # And decode expects [D, T_z]. Let's stick to that for now.

        return audio_values
    except Exception as e:
        # Print the error with more context
        print(f"Error in decode method (dac): {str(e)}")
        # Check devices right before the failing call if possible (difficult without modifying DAC lib)
        print(f"  - DAC model device: {model_device}")
        print(f"  - audio_codes device: {audio_codes.device}")
        raise