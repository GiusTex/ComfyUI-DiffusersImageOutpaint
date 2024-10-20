# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import cv2
import PIL.Image
import torch
import gc
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor

from .controlnet_union import ControlNetModel_Union
from comfy.utils import ProgressBar


def latents_to_rgb(latents):
    weights = ((60, -60, 25, -70), (60, -5, 15, -50), (60, 10, -5, -35))

    weights_tensor = torch.t(
        torch.tensor(weights, dtype=latents.dtype).to(latents.device)
    )
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(
        latents.device
    )
    rgb_tensor = torch.einsum(
        "...lxy,lr -> ...rxy", latents, weights_tensor
    ) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)  # Change the order of dimensions

    denoised_image = cv2.fastNlMeansDenoisingColored(image_array, None, 10, 10, 7, 21)
    blurred_image = cv2.GaussianBlur(denoised_image, (5, 5), 0)
    final_image = PIL.Image.fromarray(blurred_image)

    width, height = final_image.size
    final_image = final_image.resize(
        (width * 8, height * 8), PIL.Image.Resampling.LANCZOS
    )

    return final_image


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class StableDiffusionXLFillPipeline(DiffusionPipeline, StableDiffusionMixin):
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )
        self.register_to_config(
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt
        )
        self.controlnet_model = None

    def prepare_image(self, image, device, dtype, do_classifier_free_guidance=False):
        image = self.control_image_processor.preprocess(image).to(dtype=torch.float32)

        image_batch_size = image.shape[0]

        image = image.repeat_interleave(image_batch_size, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        latents = randn_tensor(shape, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @torch.no_grad()
    def __call__(
        self,
        controlnet_model,
        device,
        dtype,
        keep_model_device,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor,
        image: PipelineImageInput = None,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.5,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    ):
        self.controlnet = controlnet_model
        self._guidance_scale = guidance_scale

        # 2. Define call parameters
        batch_size = 1

        # 4. Prepare image
        if isinstance(self.controlnet, ControlNetModel_Union):
            image = self.prepare_image(
                image=image,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            height, width = image.shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
        )
        
        # 7 Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds

        add_time_ids = negative_add_time_ids = torch.tensor(
            image.shape[-2:] + torch.Size([0, 0]) + image.shape[-2:]
        ).unsqueeze(0)
        
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size, 1)

        controlnet_image_list = [0, 0, 0, 0, 0, 0, image, 0]
        union_control_type = (
            torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0])
            .to(device, dtype=prompt_embeds.dtype)
            .repeat(batch_size * 2, 1)
        )

        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
            "control_type": union_control_type,
        }
        
        controlnet_prompt_embeds = prompt_embeds.to(device)
        controlnet_added_cond_kwargs = added_cond_kwargs

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        ComfyUI_ProgressBar = ProgressBar(int(num_inference_steps))

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # controlnet(s) inference
                control_model_input = latent_model_input
                
                self.controlnet.to(device)
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond_list=controlnet_image_list,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=False,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )
                
                if keep_model_device:
                    self.controlnet.to('cpu')

                try:
                    # predict the noise residual
                    self.unet.to(device)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=None,
                        cross_attention_kwargs={},
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    if keep_model_device:
                        self.unet.to('cpu')
                except torch.cuda.OutOfMemoryError as e: # Free vram when OOM
                    self.unet.to('cpu')
                    print('\033[93m', 'Gpu is out of memory!', '\033[0m')
                    raise e
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if i == 2:
                    prompt_embeds = prompt_embeds[-1:]
                    add_text_embeds = add_text_embeds[-1:]
                    add_time_ids = add_time_ids[-1:]
                    union_control_type = union_control_type[-1:]

                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                        "control_type": union_control_type,
                    }

                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs

                    image = image[-1:]
                    controlnet_image_list = [0, 0, 0, 0, 0, 0, image, 0]

                    self._guidance_scale = 0.0

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    ComfyUI_ProgressBar.update(1)
                    #yield latents_to_rgb(latents)
        
        del self.unet
        del self.controlnet
        gc.collect()
        torch.cuda.empty_cache()
        
        latents = latents / 0.13025
        yield latents
