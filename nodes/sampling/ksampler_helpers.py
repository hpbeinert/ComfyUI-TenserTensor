# (c) TenserTensor || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

import comfy
import latent_preview as LP
import torch
from comfy import sample as S, utils as U, model_management as MM, nested_tensor as NT


def sample_latents(**kwargs):
    model = kwargs.get("model", None)
    latent = kwargs.get("latent", None)

    if model is None:
        raise ValueError("Model is required for sampling")
    if latent is None:
        raise ValueError("Latent image is required for sampling")

    steps = kwargs.get("steps", 0)
    seed = kwargs.get("seed", 0)

    latent_img = comfy.sample.fix_empty_latent_channels(model, latent["samples"])

    add_noise = kwargs.get("add_noise", True)
    if add_noise:
        bat_idx = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_img, seed, bat_idx)
    else:
        noise = torch.zeros(latent_img.size(), dtype=latent_img.dtype, layout=latent_img.layout, device="cpu")

    callback = LP.prepare_callback(model, steps)

    args = {
        "model": model,
        "noise": noise,
        "steps": steps,
        "cfg": kwargs.get("cfg", 0),
        "sampler_name": kwargs.get("sampler_name", None),
        "scheduler": kwargs.get("scheduler", None),
        "positive": kwargs.get("positive", None),
        "negative": kwargs.get("negative", None),
        "latent_image": latent_img,
        "denoise": kwargs.get("denoise", 1.0),
        "disable_noise": not add_noise,
        "start_step": kwargs.get("start_step", 0),
        "last_step": kwargs.get("last_step", 10000),
        "force_full_denoise": kwargs.get("full_denoise", True),
        "noise_mask": latent.get("noise_mask", None),
        "disable_pbar": not comfy.utils.PROGRESS_BAR_ENABLED,
        "callback": callback,
        "seed": seed,
    }

    samples = comfy.sample.sample(**args)

    retval = latent.copy()
    retval["samples"] = samples

    return retval


def guided_sample_latents(latent, guider, sigmas, sampler, random_noise):
    model_patcher = guider.model_patcher
    latent_dict = latent.copy()
    latent_tensor = S.fix_empty_latent_channels(model_patcher, latent_dict["samples"], latent.get("downscale_ratio_spacial", None))
    latent_dict["samples"] = latent_tensor

    noise_mask = latent.get("noise_mask", None)

    interim_buffer = {}
    callback = LP.prepare_callback(model_patcher, sigmas.shape[-1] - 1, interim_buffer)
    disable_pbar = not U.PROGRESS_BAR_ENABLED

    samples = guider.sample(
        random_noise.generate_noise(latent),
        latent_tensor,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=random_noise.seed
    ).to(MM.intermediate_device())

    retval = latent_dict.copy()
    retval.pop("downscale_ratio_spacial", None)
    retval["samples"] = samples

    return retval
