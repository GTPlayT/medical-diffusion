from torch import nn
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.schedulers import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers.models.clip.modeling_clip import CLIPTextModel
from typing import Literal, Union

from config import Datatypes, scheduler_types

class DiffusionModel (nn.Module):
    def __init__(
            self, 
            model_id: str = "runwayml/stable-diffusion-v1-5", 
            scheduler: scheduler_types = DDPMScheduler, 
            datatype: Datatypes = Datatypes.FLOAT32, 
            device: Union[Literal['cpu'], Literal['cuda'], Literal['xpu'], Literal['cuda']] = 'cpu'
            ):
        super().__init__()
        
        self.scheduler = scheduler.from_pretrained(model_id, subfolder='scheduler', torch_dtype=datatype.to_torch_dtype())
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=datatype.to_torch_dtype())
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=datatype.to_torch_dtype())
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=datatype.to_torch_dtype())
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=datatype.to_torch_dtype())
        self.feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor", torch_dtype=datatype.to_torch_dtype())

        self._num_inference_steps = 20
        retrieve_timesteps(self.scheduler, self._num_inference_steps, device)
    
        self.device = device
        self.to(device)

    @property
    def num_inference_steps(self):
        return self._num_inference_steps

    @num_inference_steps.setter
    def num_inference_steps(self, steps):
        retrieve_timesteps(self.scheduler, steps, self.device)
        self._num_inference_steps = steps


if __name__ == "__main__":
    hf_token = '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'
    model_id = 'runwayml/stable-diffusion-v1-5'
    diffusion_model = DiffusionModel(model_id)

