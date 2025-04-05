import json

import folder_paths

import torch
import bitsandbytes as bnb

from bitsandbytes.nn.modules import Params4bit
from bitsandbytes.functional import QuantState

import comfy.ops


def copy_quant_state(state: QuantState, device: torch.device = None) -> QuantState:
    """Copy a QuantState object to potentially a new device."""
    if state is None:
        return None

    device = device or state.absmax.device

    state2 = (
        QuantState(
            absmax=state.state2.absmax.to(device),
            shape=state.state2.shape,
            code=state.state2.code.to(device),
            blocksize=state.state2.blocksize,
            quant_type=state.state2.quant_type,
            dtype=state.state2.dtype,
        )
        if state.nested
        else None
    )

    return QuantState(
        absmax=state.absmax.to(device),
        shape=state.shape,
        code=state.code.to(device),
        blocksize=state.blocksize,
        quant_type=state.quant_type,
        dtype=state.dtype,
        offset=state.offset.to(device) if state.nested else None,
        state2=state2,
    )


def ensure_float_tensor(tensor):
    """Ensure tensor is one of the supported float types for BnB quantization."""
    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        return tensor.to(torch.float32)
    return tensor


def functional_linear_4bits(x, weight, bias):
    """Functional implementation of 4-bit linear layer."""
    # Ensure weight is properly formatted for quantization
    if not weight.bnb_quantized and weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        weight.data = weight.data.to(torch.float32)
        
    out = bnb.matmul_4bit(x, weight.t(), bias=bias, quant_state=weight.quant_state)
    out = out.to(x)
    return out


class RuntimeQuantizedParams4bit(bnb.nn.modules.Params4bit):
    """Extended Params4bit that supports on-the-fly quantization."""
    _torch_fn_depth = 0

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls._torch_fn_depth > 0 or func != torch._C.TensorBase.detach:
            return super().__torch_function__(func, types, args, kwargs or {})
        cls._torch_fn_depth += 1
        try:
            slf = args[0]
            n = cls(
                torch.nn.Parameter.detach(slf),
                requires_grad=slf.requires_grad,
                quant_state=copy_quant_state(slf.quant_state, slf.device),
                blocksize=slf.blocksize,
                compress_statistics=slf.compress_statistics,
                quant_type=slf.quant_type,
                quant_storage=slf.quant_storage,
                bnb_quantized=slf.bnb_quantized,
                module=slf.module
            )
            return n
        finally:
            cls._torch_fn_depth -= 1

    @classmethod
    def quantize_tensor(cls, tensor, quant_type="nf4", device=None):
        """Quantize a tensor to 4-bit format without saving."""
        device = device or tensor.device
        
        # Convert tensor to float32 if it's not already a supported float type
        tensor = ensure_float_tensor(tensor)
        
        if device.type != 'cuda':
            # Move tensor to CUDA for quantization
            tensor_cuda = tensor.to('cuda')
        else:
            tensor_cuda = tensor
            
        # Perform 4-bit quantization
        try:
            quantized_weight, quant_state = bnb.functional.quantize_4bit(
                tensor_cuda,
                quant_type=quant_type,
                compress_statistics=True,
            )
            
            # Create parameter
            param = cls(
                quantized_weight,
                requires_grad=False,
                quant_state=quant_state,
                blocksize=64,  # Default blocksize for 4-bit quantization 
                compress_statistics=True,
                quant_type=quant_type,
                # Don't specify quant_storage - let bitsandbytes use its default
                module=None  # Will be set later
            )
            
            return param
        except Exception as e:
            print(f"Quantization error with tensor of type {tensor.dtype}: {e}")
            # Fall back to float32 if quantization fails
            tensor = tensor.to(torch.float32)
            quantized_weight, quant_state = bnb.functional.quantize_4bit(
                tensor_cuda,
                quant_type=quant_type,
                compress_statistics=True,
            )
            
            param = cls(
                quantized_weight,
                requires_grad=False,
                quant_state=quant_state,
                blocksize=64,
                compress_statistics=True,
                quant_type=quant_type,
                module=None
            )
            
            return param

    def to(self, *args, copy=False, **kwargs):
        if copy:
            return self.clone().to(*args, **kwargs)
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        
        # Ensure we're working with a float tensor before quantization
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            self.data = self.data.to(torch.float32)
            
        if device is not None and device.type == "cuda" and not self.bnb_quantized:
            try:
                return self._quantize(device)
            except Exception as e:
                print(f"Error during quantization: {e}")
                # Try converting to float32 explicitly and retry
                self.data = self.data.to(torch.float32)
                return self._quantize(device)
        else:
            n = self.__class__(
                torch.nn.Parameter.to(self, device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=copy_quant_state(self.quant_state, device),
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                bnb_quantized=self.bnb_quantized,
                module=self.module
            )
            self.module.quant_state = n.quant_state
            self.data = n.data
            self.quant_state = n.quant_state
            return n


class ForgeParams4bit(Params4bit):
    _torch_fn_depth=0

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls._torch_fn_depth > 0 or func != torch._C.TensorBase.detach:
            return super().__torch_function__(func, types, args, kwargs or {})
        cls._torch_fn_depth += 1
        try:
            slf = args[0]
            n = cls(
                    torch.nn.Parameter.detach(slf),
                    requires_grad=slf.requires_grad,
                    quant_state=copy_quant_state(slf.quant_state, slf.device),
                    blocksize=slf.blocksize,
                    compress_statistics=slf.compress_statistics,
                    quant_type=slf.quant_type,
                    quant_storage=slf.quant_storage,
                    bnb_quantized=slf.bnb_quantized,
                    module=slf.module
                )
            return n
        finally:
            cls._torch_fn_depth -= 1

    def to(self, *args, copy=False, **kwargs):
        if copy:
            return self.clone().to(*args, **kwargs)
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        
        # Convert to float32 before quantization if needed
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            self.data = self.data.to(torch.float32)
            
        if device is not None and device.type == "cuda" and not self.bnb_quantized:
            try:
                return self._quantize(device)
            except Exception as e:
                print(f"Error during quantization: {e}")
                # Try with explicit float32 conversion
                self.data = self.data.to(torch.float32)
                return self._quantize(device)
        else:
            n = self.__class__(
                torch.nn.Parameter.to(self, device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=copy_quant_state(self.quant_state, device),
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                bnb_quantized=self.bnb_quantized,
                module=self.module
            )
            self.module.quant_state = n.quant_state
            self.data = n.data
            self.quant_state = n.quant_state
            return n
    
    def _quantize(self, device):
        """Overriding the _quantize method to handle type conversion."""
        # Ensure we're working with a float tensor for quantization
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            self.data = self.data.to(torch.float32)
        
        # Call the parent's _quantize method which does the actual quantization
        # We wrap this in a try/except to handle any remaining conversion issues
        try:
            quantized = super()._quantize(device)
            return quantized
        except ValueError as e:
            if "Blockwise quantization only supports 16/32-bit floats" in str(e):
                # Convert explicitly to float32 and retry
                print("Converting tensor to float32 for quantization")
                self.data = self.data.to(torch.float32)
                return super()._quantize(device)
            else:
                raise e


class RuntimeQuantLoader4Bit(torch.nn.Module):
    """Module that can load OR quantize weights on-the-fly."""
    def __init__(self, *, device, dtype, quant_type, **kwargs):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.weight = None
        self.quant_state = None
        self.bias = None
        self.quant_type = quant_type
        self.quantize_on_load = kwargs.get('quantize_on_load', False)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is not None:
            for k, v in quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()
        return

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        quant_state_keys = {k[len(prefix + "weight."):] for k in state_dict.keys() if k.startswith(prefix + "weight.")}

        # Check if the weight is already quantized (has bitsandbytes keys)
        if any('bitsandbytes' in k for k in quant_state_keys):
            # Model is already quantized, load it normally
            quant_state_dict = {k: state_dict[prefix + "weight." + k] for k in quant_state_keys}

            # Ensure the weight is in a compatible data type
            if prefix + 'weight' in state_dict:
                weight_data = state_dict[prefix + 'weight']
                if weight_data.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                    state_dict[prefix + 'weight'] = weight_data.to(torch.float32)

            self.weight = RuntimeQuantizedParams4bit.from_prequantized(
                data=state_dict[prefix + 'weight'],
                quantized_stats=quant_state_dict,
                requires_grad=False,
                device=self.dummy.device,
                module=self
            )
            self.quant_state = self.weight.quant_state

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        elif hasattr(self, 'dummy'):
            if prefix + 'weight' in state_dict:
                weight_tensor = state_dict[prefix + 'weight']
                
                # Ensure weight tensor is in a compatible data type
                if weight_tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                    weight_tensor = weight_tensor.to(torch.float32)
                
                # Decide whether to quantize now or load as-is
                if self.quantize_on_load and (len(weight_tensor.shape) == 2):  # Only quantize linear layers
                    # Perform on-the-fly quantization
                    self.weight = RuntimeQuantizedParams4bit.quantize_tensor(
                        weight_tensor.to(self.dummy),
                        quant_type=self.quant_type,
                        device=self.dummy.device
                    )
                    self.weight.module = self
                    self.quant_state = self.weight.quant_state
                else:
                    # Load normally but prepare for quantization when needed
                    self.weight = RuntimeQuantizedParams4bit(
                        weight_tensor.to(self.dummy),
                        requires_grad=False,
                        compress_statistics=True,
                        quant_type=self.quant_type,
                        module=self,
                    )
                    self.quant_state = self.weight.quant_state

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class ForgeLoader4Bit(torch.nn.Module):
    def __init__(self, *, device, dtype, quant_type, **kwargs):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.weight = None
        self.quant_state = None
        self.bias = None
        self.quant_type = quant_type

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is not None:
            for k, v in quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()
        return

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        quant_state_keys = {k[len(prefix + "weight."):] for k in state_dict.keys() if k.startswith(prefix + "weight.")}

        if any('bitsandbytes' in k for k in quant_state_keys):
            quant_state_dict = {k: state_dict[prefix + "weight." + k] for k in quant_state_keys}

            # Ensure weight is in a compatible format
            if prefix + 'weight' in state_dict:
                weight_data = state_dict[prefix + 'weight']
                if weight_data.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                    state_dict[prefix + 'weight'] = weight_data.to(torch.float32)

            self.weight = ForgeParams4bit.from_prequantized(
                data=state_dict[prefix + 'weight'],
                quantized_stats=quant_state_dict,
                requires_grad=False,
                device=self.dummy.device,
                module=self
            )
            self.quant_state = self.weight.quant_state

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        elif hasattr(self, 'dummy'):
            if prefix + 'weight' in state_dict:
                weight_tensor = state_dict[prefix + 'weight']
                
                # Ensure tensor is in a compatible format
                if weight_tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                    weight_tensor = weight_tensor.to(torch.float32)
                
                self.weight = ForgeParams4bit(
                    weight_tensor.to(self.dummy),
                    requires_grad=False,
                    compress_statistics=True,
                    quant_type=self.quant_type,
                    quant_storage=torch.float32,
                    module=self,
                )
                self.quant_state = self.weight.quant_state

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def make_runtime_quantized_ops(loader_class=RuntimeQuantLoader4Bit, current_device=None, current_dtype=None, 
                             current_manual_cast_enabled=False, current_bnb_dtype=None, quantize_on_load=False):
    """Create custom operations for runtime quantization."""
    
    class RuntimeQuantOPS(comfy.ops.manual_cast):
        class Linear(loader_class):
            def __init__(self, *args, device=None, dtype=None, **kwargs):
                super().__init__(
                    device=device, 
                    dtype=dtype, 
                    quant_type=current_bnb_dtype, 
                    quantize_on_load=quantize_on_load
                )
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                self.weight.quant_state = self.quant_state

                if self.bias is not None and self.bias.dtype != x.dtype:
                    self.bias.data = self.bias.data.to(x.dtype)

                if not self.parameters_manual_cast:
                    return functional_linear_4bits(x, self.weight, self.bias)
                elif not self.weight.bnb_quantized:
                    assert x.device.type == 'cuda', 'BNB Must Use CUDA as Computation Device!'
                    layer_original_device = self.weight.device
                    
                    # Ensure weight is in the correct format before quantization
                    if self.weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        self.weight.data = self.weight.data.to(torch.float32)
                        
                    try:
                        self.weight = self.weight._quantize(x.device)
                    except Exception as e:
                        print(f"Quantization error: {e}")
                        # Try with explicit conversion
                        self.weight.data = self.weight.data.to(torch.float32)
                        self.weight = self.weight._quantize(x.device)
                        
                    bias = self.bias.to(x.device) if self.bias is not None else None
                    out = functional_linear_4bits(x, self.weight, bias)
                    self.weight = self.weight.to(layer_original_device)
                    return out
                else:
                    raise RuntimeError("Unexpected state in forward")

    return RuntimeQuantOPS


def make_ops(loader_class, current_device = None, current_dtype = None, current_manual_cast_enabled = False, current_bnb_dtype = None):
    class OPS(comfy.ops.manual_cast):
        class Linear(loader_class):
            def __init__(self, *args, device=None, dtype=None, **kwargs):
                super().__init__(device=device, dtype=dtype, quant_type=current_bnb_dtype)
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                self.weight.quant_state = self.quant_state

                if self.bias is not None and self.bias.dtype != x.dtype:
                    self.bias.data = self.bias.data.to(x.dtype)

                if not self.parameters_manual_cast:
                    return functional_linear_4bits(x, self.weight, self.bias)
                elif not self.weight.bnb_quantized:
                    assert x.device.type == 'cuda', 'BNB Must Use CUDA as Computation Device!'
                    layer_original_device = self.weight.device
                    
                    # Ensure weight is in correct format before quantization
                    if self.weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        self.weight.data = self.weight.data.to(torch.float32)
                        
                    try:
                        self.weight = self.weight._quantize(x.device)
                    except Exception as e:
                        print(f"Quantization error: {e}")
                        # Try with explicit conversion
                        self.weight.data = self.weight.data.to(torch.float32)
                        self.weight = self.weight._quantize(x.device)
                        
                    bias = self.bias.to(x.device) if self.bias is not None else None
                    out = functional_linear_4bits(x, self.weight, bias)
                    self.weight = self.weight.to(layer_original_device)
                    return out
                else:
                    raise RuntimeError("Unexpected state in forward")

    return OPS


# Rest of the code remains the same (CheckpointLoaderNF4, RuntimeQuantizedCheckpointLoader, etc.)
class CheckpointLoaderNF4:
    NodeId = 'CheckpointLoaderNF4'
    NodeName = 'Load FP4 or NF4 Quantized Checkpoint Model'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "bnb_dtype": (("default", "nf4", "fp4"), {"default": "default"}),
         }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, bnb_dtype="default"):
        if bnb_dtype == "default":
            bnb_dtype = None
        ops = make_ops(ForgeLoader4Bit, current_bnb_dtype = bnb_dtype)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options={"custom_operations": ops})
        return out[:3]


class RuntimeQuantizedCheckpointLoader:
    NodeId = 'RuntimeQuantizedCheckpointLoader'
    NodeName = 'Load and Quantize Checkpoint Model (Runtime)'
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "bnb_dtype": (("nf4", "fp4"), {"default": "nf4"}),
            "quantize_immediately": (("True", "False"), {"default": "True"}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, bnb_dtype="nf4", quantize_immediately="True"):
        # Create custom operations with runtime quantization
        quantize_on_load = (quantize_immediately == "True")
        
        ops = make_runtime_quantized_ops(
            loader_class=RuntimeQuantLoader4Bit, 
            current_bnb_dtype=bnb_dtype,
            quantize_on_load=quantize_on_load
        )
        
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        # Set model_options to use our custom operations
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings"), 
            model_options={"custom_operations": ops}
        )
        
        return out[:3]


class UNETLoaderNF4:
    NodeId = 'UNETLoaderNF4'
    NodeName = 'Load FP4 or NF4 Quantized Diffusion or UNET Model'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
            "bnb_dtype": (("default", "nf4", "fp4"), {"default": "default"}),
         }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, bnb_dtype="default"):
        if bnb_dtype == "default":
            bnb_dtype = None
        ops = make_ops(ForgeLoader4Bit, current_bnb_dtype = bnb_dtype)
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={"custom_operations": ops})
        return (model,)


class RuntimeQuantizedUNETLoader:
    NodeId = 'RuntimeQuantizedUNETLoader'
    NodeName = 'Load and Quantize Diffusion or UNET Model (Runtime)'
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
            "bnb_dtype": (("nf4", "fp4"), {"default": "nf4"}),
            "quantize_immediately": (("True", "False"), {"default": "True"}),
        }}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, bnb_dtype="nf4", quantize_immediately="True"):
        # Create custom operations with runtime quantization
        quantize_on_load = (quantize_immediately == "True")
        
        ops = make_runtime_quantized_ops(
            loader_class=RuntimeQuantLoader4Bit, 
            current_bnb_dtype=bnb_dtype,
            quantize_on_load=quantize_on_load
        )
        
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        # Set model_options to use our custom operations
        model = comfy.sd.load_diffusion_model(
            unet_path, 
            model_options={"custom_operations": ops}
        )
        
        return (model,)


class CLIPLoaderNF4:
    # WIP
    NodeId = 'CLIPLoaderNF4'
    NodeName = 'Load FP4 or NF4 Quantized Text Encoder'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
            "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos"], ),
            "bnb_dtype": (("default", "nf4", "fp4"), {"default": "default"}),
         }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, type):
        if type == "stable_cascade":
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        elif type == "mochi":
            clip_type = comfy.sd.CLIPType.MOCHI
        elif type == "ltxv":
            clip_type = comfy.sd.CLIPType.LTXV
        elif type == "pixart":
            clip_type = comfy.sd.CLIPType.PIXART
        else:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options={"custom_operations": OPS})
        return (clip,)


class DualCLIPLoaderNF4:
    # WIP
    NodeId = 'DualCLIPLoaderNF4'
    NodeName = 'Load FP4 or NF4 Quantized Text Encoders'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
            "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
            "type": (["sdxl", "sd3", "flux", "hunyuan_video", "custom_flux", "fluxmod"], ),
            "bnb_dtype": (("default", "nf4", "fp4"), {"default": "default"}),
         }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        elif type == "hunyuan_video":
            clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO
        elif type == "custom_flux":
            clip_type = comfy.sd.CLIPType.FLUXC
        elif type == "fluxmod":
            clip_type = comfy.sd.CLIPType.FLUXMOD

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options={"custom_operations": OPS})
        return (clip,)

node_list = [
    # Checkpoint model loaders
    CheckpointLoaderNF4,
    RuntimeQuantizedCheckpointLoader,
    # Diffusion model loaders
    UNETLoaderNF4,
    RuntimeQuantizedUNETLoader,
    # Text encoder model loaders
    # WIP CLIPLoaderNF4,
    # WIP DualCLIPLoaderNF4,
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node in node_list:
    NODE_CLASS_MAPPINGS[node.NodeId] = node
    NODE_DISPLAY_NAME_MAPPINGS[node.NodeId] = node.NodeName
