#!/bin/bash

mkdir -p data/share/Stable-diffusion/

wget -c https://civitai.com/api/download/models/78775 -O data/share/Stable-diffusion/toonyou_beta3.safetensors || false
wget -c https://civitai.com/api/download/models/72396 -O data/share/Stable-diffusion/lyriel_v16.safetensors || false
wget -c https://civitai.com/api/download/models/71009 -O data/share/Stable-diffusion/rcnzCartoon3d_v10.safetensors || false
wget -c https://civitai.com/api/download/models/79068 -O data/share/Stable-diffusion/majicmixRealistic_v5Preview.safetensors || false
wget -c https://civitai.com/api/download/models/29460 -O data/share/Stable-diffusion/realisticVisionV40_v20Novae.safetensors || false
wget -c https://civitai.com/api/download/models/56071 -O data/share/Stable-diffusion/DarkSushiMixMix_colorful.safetensors || true

# Download Motion_Module models
wget -O data/models/motion-module/mm_sd_v14.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt || false
wget -O data/models/motion-module/mm_sd_v15.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt || false
wget -O data/models/motion-module/mm_sd_v15_v2.ckpt https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt || true
