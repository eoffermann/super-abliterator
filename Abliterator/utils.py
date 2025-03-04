import torch
import torch.nn.functional as F
import gc
import functools
import einops
from itertools import islice
from jaxtyping import Float, Int
from transformer_lens.hook_points import HookPoint
from typing import Generator, Iterable, Dict


def batch(iterable: Iterable, n: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from an iterable.

    Args:
        iterable (Iterable): The input iterable to split into batches.
        n (int): The batch size.

    Yields:
        Generator[list, None, None]: Generator yielding lists of length `n` (or shorter for the last batch).
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def clear_mem() -> None:
    """Clear memory caches to free up GPU and CPU memory.

    This function explicitly runs garbage collection and empties the PyTorch CUDA cache.
    Useful for preventing memory fragmentation and optimizing GPU usage.
    """
    gc.collect()
    torch.cuda.empty_cache()


def measure_fn(measure: str, input_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Applies a specified measure function to the input tensor.

    Args:
        measure (str): The measure function to apply. Available options:
            - 'mean': Computes the mean of the tensor.
            - 'median': Computes the median of the tensor.
            - 'max': Computes the maximum value of the tensor.
            - 'stack': Stacks multiple tensors.
        input_tensor (torch.Tensor): The tensor on which the measure function is applied.
        *args: Additional positional arguments passed to the selected function.
        **kwargs: Additional keyword arguments passed to the selected function.

    Returns:
        torch.Tensor: The result of applying the specified measure function.

    Raises:
        NotImplementedError: If an invalid measure function is specified.
    """
    avail_measures: Dict[str, callable] = {
        'mean': torch.mean,
        'median': torch.median,
        'max': torch.max,
        'stack': torch.stack
    }
    
    if measure not in avail_measures:
        raise NotImplementedError(
            f"Unknown measure function '{measure}'. Available measures: " +
            ', '.join(f"'{fn}'" for fn in avail_measures.keys())
        )

    return avail_measures[measure](input_tensor, *args, **kwargs)


def directional_hook(
    activation: Float[torch.Tensor, "... d_model"],
    hook: HookPoint,
    direction: Float[torch.Tensor, "d_model"]
) -> Float[torch.Tensor, "... d_model"]:
    """Applies a directional projection hook to modify activation values.

    This function projects activations along a given direction and removes the projected component.

    Args:
        activation (Float[torch.Tensor, "... d_model"]): The activation tensor to modify.
        hook (HookPoint): The hook point in the Transformer model where this function is applied.
        direction (Float[torch.Tensor, "d_model"]): The directional vector to use for projection.

    Returns:
        Float[torch.Tensor, "... d_model"]: The modified activation tensor with the directional component removed.
    """
    if activation.device != direction.device:
        direction = direction.to(activation.device)

    # Compute the projection of `activation` onto `direction`
    proj = einops.einsum(
        activation, direction.view(-1, 1),
        '... d_model, d_model single -> ... single'
    ) * direction

    # Subtract projection from original activation
    return activation - proj
