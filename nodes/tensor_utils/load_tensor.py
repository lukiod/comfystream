import torch
import queue
from comfystream import tensor_cache
from comfystream.exceptions import ComfyStreamInputTimeoutError


class LoadTensor:
    CATEGORY = "ComfyStream/Loaders"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    DESCRIPTION = "Load image tensor from ComfyStream input with timeout and batch support."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "timeout_seconds": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "Timeout in seconds"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, batch_size: int = 1, timeout_seconds: float = 1.0):
        """
        Load tensor(s) from the tensor cache.
        If batch_size > 1, loads multiple tensors and stacks them into a batch.
        """
        if batch_size == 1:
            # Single tensor loading with timeout
            try:
                frame = tensor_cache.image_inputs.get(block=True, timeout=timeout_seconds)
                frame.side_data.skipped = False
                return (frame.side_data.input,)
            except queue.Empty:
                raise ComfyStreamInputTimeoutError("video", timeout_seconds)
        else:
            # Batch tensor loading
            batch_images = []
            
            # Collect images up to batch_size
            for i in range(batch_size):
                if not tensor_cache.image_inputs.empty():
                    try:
                        frame = tensor_cache.image_inputs.get(block=False, timeout=timeout_seconds)
                        frame.side_data.skipped = False
                        batch_images.append(frame.side_data.input)
                    except queue.Empty:
                        # If we don't have enough images, pad with the last available image
                        if batch_images:
                            batch_images.append(batch_images[-1])
                        else:
                            # If no images available, create a dummy tensor
                            dummy_tensor = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                            batch_images.append(dummy_tensor)
                else:
                    # If we don't have enough images, pad with the last available image
                    if batch_images:
                        batch_images.append(batch_images[-1])
                    else:
                        # If no images available, create a dummy tensor
                        dummy_tensor = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
                        batch_images.append(dummy_tensor)
            
            # Stack images into a batch
            if len(batch_images) > 1:
                batch_tensor = torch.cat(batch_images, dim=0)
            else:
                batch_tensor = batch_images[0]
                
            return (batch_tensor,)
