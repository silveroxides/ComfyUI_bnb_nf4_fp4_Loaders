#shamelessly taken from forge

import folder_paths

import torch
import bitsandbytes as bnb

from bitsandbytes.nn.modules import Params4bit, QuantState


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
        quant_state_keys = {k[len(prefix + "weight."):] for k in state_dict.keys() if k.startswith(prefix + "weight.")}

        if any('bitsandbytes' in k for k in quant_state_keys):
            quant_state_dict = {k: state_dict[prefix + "weight." + k] for k in quant_state_keys}

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


import comfy.ops

def make_ops(loader_class, current_device = None, current_dtype = None, current_manual_cast_enabled = False, current_bnb_dtype = None):

    class OPS(comfy.ops.manual_cast):
        class Linear(loader_class):
            def __init__(self, *args, device=None, dtype=None, **kwargs):
                super().__init__(device=device, dtype=dtype, quant_type=current_bnb_dtype)
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                self.weight.quant_state = self.quant_state

                if self.bias is not None and self.bias.dtype != x.dtype:
                    # Maybe this can also be set to all non-bnb ops since the cost is very low.
                    # And it only invokes one time, and most linear does not have bias
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
                    raise RuntimeError("Unexpected state in forward")

    return OPS

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
        unet_path = folder_paths.get_full_path("unet", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={"custom_operations": ops})
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
    # Diffusion model loaders
    UNETLoaderNF4,
    # Text encoder model loaders
    # WIP CLIPLoaderNF4,
    # WIP DualCLIPLoaderNF4,
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node in node_list:
    NODE_CLASS_MAPPINGS[node.NodeId] = node
    NODE_DISPLAY_NAME_MAPPINGS[node.NodeId] = node.NodeName
