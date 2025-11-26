import torch
import torch.nn as nn
import comfy.ops
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.model_detection
import folder_paths
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit, QuantState
import json
import logging

# ==============================================================================
# Helpers for Metadata Parsing
# ==============================================================================

def tensor_to_dict(tensor_data):
    """Decodes the JSON metadata stored in a uint8 tensor."""
    try:
        byte_data = bytes(tensor_data.tolist())
        json_str = byte_data.decode('utf-8')
        return json.loads(json_str)
    except Exception as e:
        logging.warning(f"Failed to decode NF4 metadata: {e}")
        return {}

# ==============================================================================
# Bitsandbytes Custom Parameter & Loader Classes
# ==============================================================================

def functional_linear_4bits(x, weight, bias):
    out = bnb.matmul_4bit(x, weight.t(), bias=bias, quant_state=weight.quant_state)
    out = out.to(x)
    return out

def copy_quant_state(state: QuantState, device: torch.device = None) -> QuantState:
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

class ForgeParams4bit(Params4bit):
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

    def to(self, *args, copy=False, **kwargs):
        if copy:
            return self.clone().to(*args, **kwargs)
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == "cuda" and not self.bnb_quantized:
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
        # Identify quantization keys for this layer
        quant_state_keys = {k[len(prefix + "weight."):] for k in state_dict.keys() if k.startswith(prefix + "weight.")}

        if any('bitsandbytes' in k for k in quant_state_keys):
            # This is an NF4/FP4 loaded layer
            quant_state_dict = {k: state_dict[prefix + "weight." + k] for k in quant_state_keys}
            
            # The 'weight' key contains the packed uint8 data
            weight_data = state_dict[prefix + 'weight']
            
            self.weight = ForgeParams4bit.from_prequantized(
                data=weight_data,
                quantized_stats=quant_state_dict,
                requires_grad=False,
                device=self.dummy.device,
                module=self
            )
            self.quant_state = self.weight.quant_state

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))
            
            # Cleanup keys to prevent unexpected key errors in strict loading
            for k in quant_state_keys:
                key_name = prefix + "weight." + k
                if key_name in missing_keys: missing_keys.remove(key_name)
                # We don't remove from unexpected_keys because ComfyUI logic might handle that, 
                # but we consumed them.

            del self.dummy

        elif hasattr(self, 'dummy'):
            # Fallback for standard loading or on-the-fly quantization
            if prefix + 'weight' in state_dict:
                self.weight = ForgeParams4bit(
                    state_dict[prefix + 'weight'].to(self.dummy),
                    requires_grad=False,
                    compress_statistics=True,
                    quant_type=self.quant_type,
                    quant_storage=torch.uint8,
                    module=self,
                )
                self.quant_state = self.weight.quant_state

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def make_ops(loader_class, current_device=None, current_dtype=None, current_manual_cast_enabled=False, current_bnb_dtype=None):
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
                    self.weight = self.weight._quantize(x.device)
                    bias = self.bias.to(x.device) if self.bias is not None else None
                    out = functional_linear_4bits(x, self.weight, bias)
                    self.weight = self.weight.to(layer_original_device)
                    return out
                else:
                    return functional_linear_4bits(x, self.weight, self.bias)
                    
    return OPS

# ==============================================================================
# Model Loading Logic with Shape Fixing
# ==============================================================================

def create_shape_patched_state_dict(sd):
    """
    Scans the state dict for bitsandbytes metadata, retrieves original shapes,
    and returns a shallow copy of the state dict where quantized weights are 
    replaced with empty tensors of the CORRECT original shape.
    """
    patched_sd = {}
    
    # Map of weight_key -> original_shape
    restored_shapes = {}

    # First pass: find metadata and extract original shapes
    for k, v in sd.items():
        if k.endswith(".quant_state.bitsandbytes__nf4"):
            weight_key = k.replace(".quant_state.bitsandbytes__nf4", "")
            meta = tensor_to_dict(v)
            if "shape" in meta:
                restored_shapes[weight_key] = meta["shape"]

    # Second pass: build patched dict
    for k, v in sd.items():
        if k in restored_shapes:
            # Replace packed uint8 tensor with empty tensor of original shape/dtype
            # This tricks ComfyUI into inferring the correct model config
            original_shape = restored_shapes[k]
            # We use float16 as a safe default for config inference
            patched_sd[k] = torch.empty(original_shape, dtype=torch.float16, device="meta") 
        else:
            patched_sd[k] = v
            
    return patched_sd

def load_bnb_model(ckpt_path, bnb_dtype):
    sd = comfy.utils.load_torch_file(ckpt_path)
    
    # 1. Patch shapes for detection
    patched_sd = create_shape_patched_state_dict(sd)
    
    # 2. Detect config using patched SD
    try:
        model_config = comfy.model_detection.model_config_from_unet(patched_sd, ckpt_path)
        
        # 3. Initialize model with patched SD (so Flux checks pass)
        # We pass patched_sd to get_model so it sees correct axes_dim/shapes during init
        ops = make_ops(ForgeLoader4Bit, current_bnb_dtype=bnb_dtype)
        model = model_config.get_model(patched_sd, "", model_options={"custom_operations": ops})
        
        # 4. Load the ACTUAL quantized weights
        # strict=False allows ignoring the metadata keys we don't need in standard load
        model.load_state_dict(sd, strict=False)
        
        return model
    except Exception as e:
        logging.error(f"Error loading NF4 model: {e}")
        raise e

def load_bnb_checkpoint(ckpt_path, bnb_dtype):
    # Similar to comfy.sd.load_checkpoint_guess_config but with shape patching
    sd = comfy.utils.load_torch_file(ckpt_path)
    
    # Extract UNet/Diffusion parts for patching
    # (Simplified assumption: The quantized parts are mainly in the diffusion model)
    diffusion_sd = {}
    other_sd = {}
    
    # Common prefixes for ComfyUI checkpoints
    prefix_map = ["model.diffusion_model.", "model.", ""]
    
    # We need to find the specific keys that are quantized
    patched_sd = create_shape_patched_state_dict(sd)
    
    # Detect config
    try:
        # We pass the FULL patched_sd to detection logic
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options={"custom_operations": make_ops(ForgeLoader4Bit, current_bnb_dtype=bnb_dtype)}
        )
        # Note: calling load_checkpoint_guess_config normally will fail if we don't intercept 
        # the SD it uses. But we can't easily inject patched_sd into that function without 
        # rewriting it entirely.
        
        # ALTERNATIVE STRATEGY for Checkpoints:
        # Since load_checkpoint_guess_config loads the file internally, we have to 
        # replicate its high-level logic or use a "Loader" object if available.
        # Given the constraint, we will manually construct the return values.
        
        model_config = comfy.model_detection.model_config_from_unet(patched_sd, ckpt_path)
        
        # Init Model with patched SD
        ops = make_ops(ForgeLoader4Bit, current_bnb_dtype=bnb_dtype)
        model = model_config.get_model(patched_sd, "", model_options={"custom_operations": ops})
        
        # Load weights
        model.load_state_dict(sd, strict=False)
        
        # Load Clip/VAE (Best effort fallback to standard loader functions for these parts)
        # This is a simplification; for a full checkpoint loader, you might need to extract CLIP/VAE keys
        # We assume standard keys for CLIP/VAE.
        
        # Re-using Comfy's logic for CLIP/VAE is hard if we hold the SD. 
        # However, usually NF4 checkpoints are just the UNet. 
        # If it's a full checkpoint, we can use Comfy's CLIP loading utilities on the 'sd' dict.
        
        clip = None
        vae = None
        
        # Try to load CLIP
        try:
             # This is hacky: we rely on Comfy identifying CLIP from the dict
             # If keys exist in SD
             clip = comfy.sd.load_clip(ckpt_path, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        except:
             pass 

        # Try to load VAE
        try:
             vae_sd = comfy.utils.state_dict_prefix_replace(sd, {"first_stage_model.": ""}, replace_prefix=True)
             vae = comfy.sd.VAE(sd=vae_sd)
        except:
             pass

        return (model, clip, vae)
        
    except Exception as e:
        # Fallback: If the patching didn't help (maybe it's not quantized?), try standard load
        logging.warning(f"NF4 Custom Load failed, trying standard: {e}")
        ops = make_ops(ForgeLoader4Bit, current_bnb_dtype=bnb_dtype)
        return comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options={"custom_operations": ops}
        )[:3]

# ==============================================================================
# Node Definitions
# ==============================================================================

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
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        # Use our robust loader that patches shapes before config detection
        return load_bnb_checkpoint(ckpt_path, bnb_dtype)

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
        unet_path = folder_paths.get_full_path("unet", unet_name)
        
        # Use our robust loader
        model = load_bnb_model(unet_path, bnb_dtype)
        return (model,)

node_list = [
    CheckpointLoaderNF4,
    UNETLoaderNF4,
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node in node_list:
    NODE_CLASS_MAPPINGS[node.NodeId] = node
    NODE_DISPLAY_NAME_MAPPINGS[node.NodeId] = node.NodeName