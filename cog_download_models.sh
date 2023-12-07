#!/bin/bash

mkdir -p data/share/Stable-diffusion/
mkdir -p data/share/lora/

#wget -c https://civitai.com/api/download/models/78775 -O data/share/Stable-diffusion/toonyou_beta3.safetensors || false
#wget -c https://civitai.com/api/download/models/72396 -O data/share/Stable-diffusion/lyriel_v16.safetensors || false
#wget -c https://civitai.com/api/download/models/71009 -O data/share/Stable-diffusion/rcnzCartoon3d_v10.safetensors || false
#wget -c https://civitai.com/api/download/models/79068 -O data/share/Stable-diffusion/majicmixRealistic_v5Preview.safetensors || false
#wget -c https://civitai.com/api/download/models/29460 -O data/share/Stable-diffusion/realisticVisionV40_v20Novae.safetensors || false
wget -c https://civitai.com/api/download/models/56071 -O data/share/Stable-diffusion/darkSushiMixMix_colorful.safetensors
#wget -c https://civitai.com/api/download/models/96101 -O data/share/Stable-diffusion/3RDEdNEWILLUSTRO_illustroV3.safetensors
wget -c https://civitai.com/api/download/models/144249/ -O data/share/Stable-diffusion/animerge_v25.safetensors
wget -c https://civitai.com/api/download/models/158852/ -O data/share/Stable-diffusion/fantasticHackerStyleBailing_v1029.safetensors
wget -c https://civitai.com/api/download/models/11193 -O data/share/Stable-diffusion/illuminatiDiffusionV1_v11.safetensors
wget -c https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -P data/share/Stable-diffusion/
wget -c https://huggingface.co/OedoSoldier/detail-tweaker-lora/resolve/main/add_detail.safetensors -P data/share/lora/

# Download Motion_Module models
#wget -O data/models/motion-module/mm_sd_v14.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt || false
#wget -O data/models/motion-module/mm_sd_v15.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt || false
wget -O data/models/motion-module/mm_sd_v15_v2.safetensors https://huggingface.co/guoyww/animatediff/resolve/refs%2Fpr%2F3/mm_sd_v15_v2.safetensors?download=true
