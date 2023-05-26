from typing import Union
import torch
import numpy as np
from PIL import Image

        
def preprocess_image(image, height=512, width=512, left=0, right=0, top=0, bottom=0):
    if isinstance(image, str):
        image = np.array(Image.open(image))
    else:
        image = np.array(image)
        
    if image.ndim == 3:
        image = image[:, :, :3]
        h, w, _ = image.shape
    else:
        h, w = image.shape
        
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    
    if image.ndim == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((height, width)))
    return image


class DDIMInversion():
    def __init__(self, unet, vae, tokenizer, text_encoder, scheduler,
                 num_ddim_steps=50, device="cuda"):
        self.unet = unet
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_ddim_steps)
        self.num_ddim_steps = num_ddim_steps
        self.device = device
        
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if isinstance(image, Image.Image):
                image = np.array(image)
            if isinstance(image, torch.Tensor) and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
    
    @torch.no_grad()
    def ddim_loop(self, latent):
        _, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    def invert(self, image: Union[str, Image.Image], prompt, height, width, offsets):
        self.init_prompt(prompt)
        if isinstance(image, str):
            image = preprocess_image(image, height, width, *offsets)
        latent = self.image2latent(image)
        ddim_latents = self.ddim_loop(latent)
        return ddim_latents