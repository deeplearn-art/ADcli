from transformers import CLIPProcessor, CLIPModel
import torch
from controlnet_aux import OpenposeDetector
from diffusers import ControlNetModel
from animatediff.utils import model
from animatediff.utils.util import prepare_ip_adapter
from pathlib import Path
model.get_base_model( "runwayml/stable-diffusion-v1-5",local_dir=Path("/src/data/models/huggingface"))
CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/src/.cache",local_files_only=False)
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/src/.cache", local_files_only=False)
ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_openpose',cache_dir='/src/.cache/huggingface/hub', torch_dtype=torch.float16)
OpenposeDetector.from_pretrained('lllyasviel/Annotators',cache_dir='/src/.cache/huggingface/hub')
#prepare_ip_adapter()