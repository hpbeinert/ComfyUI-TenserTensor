# (c) TenserTensor <tenser.tensor@proton.me> || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

import fnmatch
import math
import os
from datetime import datetime
from pathlib import Path
from typing import override, Any

import numpy
import torch
from PIL import Image, ImageOps, ImageSequence
from spandrel import ModelLoader, ImageModelDescriptor

import folder_paths
from comfy import model_management, utils, samplers
from comfy_api.latest import ComfyExtension, IO, ui
from comfy_api.latest._io import UploadType
from node_helpers import conditioning_set_values
from node_helpers import pillow
from .nodes_latent import MEGAPIXELS

CATEGORY = "TenserTensor/Image"
RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
FILENAME_FORMATS = ["name-###", "date-name-###", "name-datetime"]
FORMAT_EXT = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", }


class SingleCondCFGGuider(samplers.CFGGuider):

    @override
    def set_conds(self, conditioning):
        self.inner_set_conds({"positive": conditioning})

    def get_conds(self, key="positive"):
        return [[c.get("cross_attn", None), c] for c in self.original_conds[key]]


def align_to_step(size, scale_factor, dimension_step):
    return round(size * scale_factor / dimension_step) * dimension_step


# crop = "center" / "disabled"
def resize_image(timage, to_width, to_height, upscale_method, dimension_step=1, crop="disabled") -> torch.Tensor:
    is_mask = timage.ndim == 3
    if is_mask:
        timage = timage.unsqueeze(-1)

    samples = timage.movedim(-1, 1)
    orig_height, orig_width = samples.shape[2:]

    if crop == "disabled":
        scale_factor = max(to_width / orig_width, to_height / orig_height)
        final_width = align_to_step(orig_width, scale_factor, dimension_step)
        final_height = align_to_step(orig_height, scale_factor, dimension_step)
    else:
        final_width = round(to_width / dimension_step) * dimension_step
        final_height = round(to_height / dimension_step) * dimension_step

    final_width, final_height = int(max(final_width, dimension_step)), int(max(final_height, dimension_step))
    resized = utils.common_upscale(samples, final_width, final_height, upscale_method, crop).movedim(1, -1)

    return resized.squeeze(-1) if is_mask else resized


def resize_image_to_megapixels(timage, resize_method, megapixels, dimension_step=1, crop="disabled") -> torch.Tensor:
    is_mask = timage.ndim == 3
    if is_mask:
        timage = timage.unsqueeze(-1)

    mp_value = float(megapixels.split()[0])
    total_pixels = int(mp_value * 1_000_000)
    samples = timage.movedim(-1, 1)
    orig_height, orig_width = samples.shape[2:]
    scale_factor = math.sqrt(total_pixels / (orig_width * orig_height))

    if crop == "disabled":
        final_width = align_to_step(orig_width, scale_factor, dimension_step)
        final_height = align_to_step(orig_height, scale_factor, dimension_step)
    else:
        to_width, to_height = orig_width * scale_factor, orig_height * scale_factor
        final_width = round(to_width / dimension_step) * dimension_step
        final_height = round(to_height / dimension_step) * dimension_step

    final_width, final_height = int(max(final_width, dimension_step)), int(max(final_height, dimension_step))
    resized = utils.common_upscale(samples, final_width, final_height, resize_method, crop).movedim(1, -1)

    return resized.squeeze(-1) if is_mask else resized


def get_torch_device(device=None):
    return (
        model_management.get_torch_device()
        if device is None or device == "default"
        else torch.device(device)
    )


def load_image(filename, create_mask, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    tdevice = get_torch_device(device)
    path = folder_paths.get_annotated_filepath(filename)
    pil_image = pillow(Image.open, path)

    images, masks = [], []
    width, height = None, None

    for frame in ImageSequence.Iterator(pil_image):
        pil_frame = pillow(ImageOps.exif_transpose, frame)

        if pil_frame.mode == "I":
            pil_frame = pil_frame.point(lambda i: i * (1 / 255))

        if len(images) == 0:
            width, height = pil_frame.size

        if pil_frame.size != (width, height,):
            continue

        if create_mask:
            alpha = numpy.zeros((height, width,), dtype=numpy.float32)
            if 'A' in pil_frame.getbands():
                alpha = numpy.array(pil_frame.getchannel('A'), dtype=numpy.float32) / 255.0
            elif pil_frame.mode == 'P' and 'transparency' in pil_frame.info:
                alpha = numpy.array(pil_frame.convert('RGBA').getchannel('A'), dtype=numpy.float32) / 255.0
            tmask = torch.from_numpy(1.0 - alpha).to(tdevice)
            masks.append(tmask)

        rgb_frame = pil_frame.convert("RGB")
        np_image = numpy.array(rgb_frame, dtype=numpy.float32) / 255.0
        timage = torch.from_numpy(np_image)[None].to(tdevice)

        images.append(timage)

        if pil_image.format == "MPO":
            break

    if len(images) == 0:
        raise ValueError(f"ERROR: No frames found in {path}")

    o_image = torch.cat(images, dim=0) if len(images) > 1 else images[0]

    match len(masks):
        case 0:
            o_mask = torch.zeros((1, height, width), dtype=torch.float32)
        case 1:
            o_mask = masks[0].unsqueeze(0)
        case _:
            o_mask = torch.cat(masks, dim=0)

    return o_image, o_mask,


def get_image_files():
    return sorted([
        f.name for f in os.scandir(folder_paths.get_input_directory())
        if f.is_file() and Path(f.name).suffix.lower() in (".png", ".jpg", ".webp", ".bmp")
    ])


class TT_ImageLoaderResizerNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_ImageLoaderResizerNode",
            display_name="TT Image Loader / Resizer",
            category=CATEGORY,
            description="",
            inputs=[
                IO.Boolean.Input("resize_image", default=False, label_on="Scale", label_off="Keep"),
                IO.Combo.Input("resize_method", options=RESIZE_METHODS, advanced=True),
                IO.Int.Input("dimension_step", default=1, min=1, max=256, advanced=True),
                IO.Combo.Input("megapixels", options=MEGAPIXELS, advanced=True),
                IO.Boolean.Input("create_mask", default=False, label_on="Mask", label_off="Empty"),
                IO.Combo.Input("image", options=get_image_files(), upload=UploadType.image)
            ],
            outputs=[
                IO.Image.Output(display_name="IMAGE"),
                IO.Mask.Output(display_name="MASK"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> dict[str, Any]:
        path = folder_paths.get_annotated_filepath(kwargs.get("image"))
        resize = kwargs.get("resize_image")
        result = {
            "image": path + str(os.path.getmtime(path)),
            "resize_image": resize,
            "resize_method": kwargs.get("resize_method"),
            "dimension_step": kwargs.get("dimension_step"),
            "create_mask": kwargs.get("create_mask"),
        }

        if resize:
            result["megapixels"] = kwargs.get("megapixels")

        return result

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        path = folder_paths.get_annotated_filepath(kwargs.get("image"))
        if not os.path.isfile(path):
            return f"Image file not found: {path}"

        return True

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        timage, tmask = load_image(kwargs.get("image"), kwargs.get("create_mask"))
        if kwargs.get("resize_image"):
            resize_method, megapixels, dimension_step = (
                kwargs.get("resize_method"),
                kwargs.get("megapixels"),
                kwargs.get("dimension_step")
            )
            timage = resize_image_to_megapixels(
                timage,
                resize_method,
                megapixels,
                dimension_step
            )
            tmask = resize_image_to_megapixels(
                tmask,
                resize_method,
                megapixels,
                dimension_step
            )

        return IO.NodeOutput(timage, tmask, ui=ui.PreviewImage(timage, cls=cls))


class InvalidSavePathError(Exception):
    pass


def build_filename(save_path, filename_mask) -> str:
    count = sum(1 for f in os.scandir(save_path) if fnmatch.fnmatch(f.name, filename_mask)) + 1

    return filename_mask.replace("*", f"{count:05}")


def build_save_path(**kwargs) -> tuple[str, str]:
    output_path, subfolder, image_format, filename_prefix = (
        kwargs.get("output_path"),
        kwargs.get("subfolder"),
        kwargs.get("image_format"),
        kwargs.get("filename_prefix"),
    )
    save_path = os.path.join(output_path, subfolder)

    if os.path.commonpath((output_path, os.path.abspath(save_path))) != output_path:
        raise InvalidSavePathError("Saving image outside the output folder is not allowed.")

    os.makedirs(save_path, exist_ok=True)

    ext = FORMAT_EXT[image_format]
    filename = ""
    match kwargs.get("filename_format"):
        case "name-###":
            filename = build_filename(save_path, f"{filename_prefix}-*{ext}")
        case "date-name-###":
            date = datetime.now().strftime("%Y-%m-%d")
            filename = build_filename(save_path, f"{date}-{filename_prefix}-*{ext}")
        case "name-datetime":
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{filename_prefix}-{timestamp}{ext}"
        case _:
            filename = f"{filename_prefix}{ext}"

    return save_path, filename


def save_image(**kwargs) -> None:
    kwargs["output_path"] = folder_paths.get_output_directory()
    image = kwargs.get("image")
    for frame in image:
        np_frame = frame.cpu().numpy() * 255.0
        tframe = Image.fromarray(numpy.clip(np_frame, 0, 255).astype(numpy.uint8))
        save_path, filename = build_save_path(**kwargs)
        image_format = kwargs.get("image_format")
        params = (
            {"compress_level": kwargs.get("compression_level")}
            if kwargs.get("image_format") == "PNG"
            else {"quality": kwargs.get("image_quality")}
        )
        tframe.save(os.path.join(save_path, filename), format=image_format, **params)


class TT_ImagePreviewSaveNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_ImagePreviewSaveNode",
            display_name="TT Image Preview / Save",
            category=CATEGORY,
            description="",
            is_output_node=True,
            inputs=[
                IO.Image.Input("image"),
                IO.Boolean.Input("save_image", default=True, label_on="Save image", label_off="Only preview"),
                IO.String.Input("filename_prefix", default="TT"),
                IO.Combo.Input("filename_format", options=FILENAME_FORMATS, default="name-datetime", advanced=True),
                IO.String.Input("subfolder", default="", advanced=True),
                IO.Combo.Input("image_format", options=list(FORMAT_EXT.keys()), default="PNG", advanced=True),
                IO.Int.Input("image_quality", default=75, min=1, max=100, advanced=True),
                IO.Int.Input("compression_level", default=6, min=0, max=9, advanced=True),
            ],
            outputs=[]
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        timage = kwargs.get("image")

        if kwargs.get("save_image"):
            save_image(**kwargs)

        return IO.NodeOutput(ui=ui.PreviewImage(timage, cls=cls))


DEVICES = ["default", "cpu"]


def get_upscale_models() -> list[str]:
    return folder_paths.get_filename_list("upscale_models")


def load_upscale_model(upscale_model) -> ImageModelDescriptor:
    model_path = folder_paths.get_full_path_or_raise("upscale_models", upscale_model)

    try:
        sd = utils.load_torch_file(model_path, safe_load=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load model file {model_path}: {e}")

    sd = utils.state_dict_prefix_replace(sd, {"module.": ""})
    upscale_model = ModelLoader().load_from_state_dict(sd).eval()

    if not isinstance(upscale_model, ImageModelDescriptor):
        raise TypeError(f"Expected ImageModelDescriptor, got {type(upscale_model).__name__}")

    return upscale_model


BUFFER_FACTOR = 384.0


def upscale(timage, **kwargs):
    device_name, upscale_tile, upscale_overlap = (
        kwargs.get("upscaler_device", None),
        kwargs.get("upscale_tile"),
        kwargs.get("upscale_overlap")
    )
    device = get_torch_device(device_name)
    model = load_upscale_model(kwargs.get("upscale_model"))
    i_image = timage.movedim(-1, 1).to(device)
    batch, _, height, width = i_image.shape
    m_model = model_management.module_size(model.model)
    m_upscale_tile = (upscale_tile * upscale_tile * 3) * i_image.element_size() * max(model.scale, 1) * BUFFER_FACTOR
    m_total = m_model + m_upscale_tile + i_image.nelement() * i_image.element_size()
    model_management.free_memory(m_total, device)
    model.model.to(device)

    progress_bar = None
    try:
        while True:
            try:
                steps = batch * utils.get_tiled_scale_steps(
                    width, height,
                    tile_x=upscale_tile,
                    tile_y=upscale_tile,
                    overlap=upscale_overlap
                )
                progress_bar = utils.ProgressBar(steps)
                scaled = utils.tiled_scale(
                    i_image,
                    model,
                    tile_x=upscale_tile,
                    tile_y=upscale_tile,
                    overlap=upscale_overlap,
                    upscale_amount=model.scale,
                    pbar=progress_bar
                )
                break
            except model_management.OOM_EXCEPTION as e:
                progress_bar = None
                upscale_tile //= 2
                if upscale_tile < 128:
                    raise e
    finally:
        if device.type != "cpu":
            model.model.to("cpu")

    return torch.clamp(scaled.movedim(1, -1), min=0.0, max=1.0)


class TT_ImagePreviewUpscaleSaveNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_ImagePreviewUpscaleSaveNode",
            display_name="TT Image Preview / Upscale / Save",
            category=CATEGORY,
            description="",
            is_output_node=True,
            inputs=[
                IO.Image.Input("image"),
                IO.Boolean.Input("save_image", default=True, label_on="Save image", label_off="Only preview"),
                IO.Boolean.Input("upscale_image", default=True, label_on="Upscale image", label_off="Keep size"),
                IO.String.Input("filename_prefix", default="TT"),
                IO.Combo.Input("filename_format", options=FILENAME_FORMATS, default="name-datetime", advanced=True),
                IO.String.Input("subfolder", default="", advanced=True),
                IO.Combo.Input("image_format", options=list(FORMAT_EXT.keys()), default="PNG", advanced=True),
                IO.Int.Input("image_quality", default=75, min=1, max=100, advanced=True),
                IO.Int.Input("compression_level", default=6, min=0, max=9, advanced=True),
                IO.Combo.Input("upscaler_device", options=DEVICES, advanced=True),
                IO.Combo.Input("upscale_model", options=get_upscale_models(), advanced=True),
                IO.Int.Input("upscale_tile", default=512, min=128, max=4096, step=64, advanced=True),
                IO.Int.Input("upscale_overlap", default=64, min=8, max=256, step=8, advanced=True),
            ],
            outputs=[]
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        timage = kwargs.get("image")

        if kwargs.get("upscale_image"):
            timage = upscale(timage, **kwargs)

        if kwargs.get("save_image"):
            kwargs["image"] = timage
            save_image(**kwargs)

        return IO.NodeOutput(ui=ui.PreviewImage(timage, cls=cls))


TILE_SIZE, OVERLAP = 512, 64


class TT_GuiderImageReferenceNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TT_GuiderImageReferenceNode",
            display_name="TT Guider Image Reference",
            category=CATEGORY,
            description="",
            inputs=[
                IO.Vae.Input("vae"),
                IO.Guider.Input("guider"),
                IO.Combo.Input("megapixels", options=MEGAPIXELS),
                IO.Combo.Input("resize_method", options=RESIZE_METHODS, advanced=True),
                IO.Int.Input("dimension_step", default=1, min=1, max=256, advanced=True),
                IO.Combo.Input("image", options=get_image_files(), upload=UploadType.image)
            ],
            outputs=[
                IO.Vae.Output("VAE"),
                IO.Guider.Output("GUIDER"),
            ]
        )

    @classmethod
    def execute(cls, **kwargs) -> IO.NodeOutput:
        timage, _ = load_image(kwargs.get("image"), create_mask=False)
        vae, guider, resize_method, megapixels, dimension_step = (
            kwargs.get("vae"),
            kwargs.get("guider"),
            kwargs.get("resize_method"),
            kwargs.get("megapixels"),
            kwargs.get("dimension_step")
        )

        resized = resize_image_to_megapixels(timage, resize_method, megapixels, dimension_step)
        samples = vae.encode_tiled(
            resized,
            tile_x=TILE_SIZE,
            tile_y=TILE_SIZE,
            overlap=OVERLAP
        )

        conditioning = guider.get_conds()

        if conditioning is not None:
            conditioning = conditioning_set_values(conditioning, {"reference_latents": [samples]}, append=True)
            guider.set_conds(conditioning)
        else:
            raise ValueError("ERROR: Guider has no conditioning")

        return IO.NodeOutput(vae, guider, ui=ui.PreviewImage(timage, cls=cls))


# ==============================================================================
# V3 entrypoint — registers context nodes with ComfyUI
# ==============================================================================

class ImageNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            TT_ImageLoaderResizerNode,
            TT_ImagePreviewSaveNode,
            TT_ImagePreviewUpscaleSaveNode,
            TT_GuiderImageReferenceNode,
        ]


async def comfy_entrypoint() -> ImageNodesExtension:
    return ImageNodesExtension()


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "TT_ImageLoaderResizerNode",
    "TT_ImagePreviewSaveNode",
    "TT_ImagePreviewUpscaleSaveNode",
    "TT_GuiderImageReferenceNode",
]
