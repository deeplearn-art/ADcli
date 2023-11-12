# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import glob
import os
from pathlib import Path
import re
import shutil
import subprocess

from cog import BasePredictor, Input
from cog import Path as CogPath

import sys
import gdown
import torch


FAKE_PROMPT_TRAVEL_JSON = """
{{
  "name": "sample",
  "path": "{dreambooth_path}",
  "motion_module": "models/motion-module/mm_sd_v15_v2.ckpt",
  "compile": false,
  "seed": [
    {seed}
  ],
  "scheduler": "{scheduler}",
  "steps": {steps},
  "guidance_scale": {guidance_scale},
  "clip_skip": {clip_skip},
  "prompt_fixed_ratio": {prompt_fixed_ratio},
  "head_prompt": "{head_prompt}",
  "prompt_map": {{
    {prompt_map}
  }},
  "tail_prompt": "{tail_prompt}",
  "n_prompt": [
    "{negative_prompt}"
  ],
  "controlnet_map": {{
    "input_image_dir": "controlnet_image/xeno",
    "max_samples_on_vram": 200,
    "max_models_on_vram": 3,
    "save_detectmap": true,
    "preprocess_on_gpu": true,
    "is_loop": {loop},
    "controlnet_openpose": {
        "enable": true,
        "use_preprocessor": true,
        "guess_mode": false,
        "controlnet_conditioning_scale": {controlnet_conditioning_scale},
        "control_guidance_start": 0.0,
        "control_guidance_end": 1.0,
        "control_scale_list": []
    }}
  }},
  "output":{{
    "format" : "{output_format}",
    "fps" : {playback_frames_per_second},
    "encode_param":{{
      "crf": 10
    }}
  }}
}}
"""


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

    def download_custom_model(self, custom_base_model_url: str):
        # Validate the custom_base_model_url to ensure it's from "civitai.com"
        if not re.match(r"^https://civitai\.com/api/download/models/\d+$", custom_base_model_url):
            raise ValueError(
                "Invalid URL. Only downloads from 'https://civitai.com/api/download/models/' are allowed."
            )

        # cmd = ["pget", custom_base_model_url, "data/share/Stable-diffusion/custom.safetensors"]
        cmd = ["wget", "-O", "data/share/Stable-diffusion/custom.safetensors", custom_base_model_url]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_output, stderr_output = process.communicate()

        print("Output from pget command:")
        print(stdout_output)
        if stderr_output:
            print("Errors from wget command:")
            print(stderr_output)

        if process.returncode:
            raise ValueError(f"Failed to download the custom model. Wget returned code: {process.returncode}")
        return "custom"

    def transform_prompt_map(self, prompt_map_string: str):
        """
        Transform the given prompt_map string into a formatted string suitable for JSON injection.

        Parameters
        ----------
        prompt_map_string : str
            A string containing animation prompts in the format 'frame number : prompt at this frame',
            separated by '|'. Colons inside the prompt description are allowed.

        Returns
        -------
        str
            A formatted string where each prompt is represented as '"frame": "description"'.
        """

        segments = prompt_map_string.split("|")

        formatted_segments = []
        for segment in segments:
            frame, prompt = segment.split(":", 1)
            frame = frame.strip()
            prompt = prompt.strip()

            formatted_segment = f'"{frame}": "{prompt}"'
            formatted_segments.append(formatted_segment)

        return ", ".join(formatted_segments)

    def find_recent_directory(self, base_path):
        return max(
            (
                os.path.join(base_path, d)
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ),
            key=os.path.getmtime,
        )

    def find_media_file(self, directory, extensions):
        media_files = [f for f in os.listdir(directory) if f.endswith(extensions)]
        if not media_files:
            raise ValueError(f"No media files with extensions {extensions} found in directory: {directory}")
        return os.path.join(directory, media_files[0])

    def find_png_folder(self, directory, exclude_folder):
        contents = os.listdir(directory)
        png_folder = next(
            (
                item
                for item in contents
                if item != exclude_folder and os.path.isdir(os.path.join(directory, item))
            ),
            None,
        )
        if png_folder is None:
            raise ValueError(f"No PNG folder found in directory: {directory}")
        return os.path.join(directory, png_folder)

    def copy_dir_contents(self,src_dir,dest_dir):
        if os.path.exists(dest_dir):  # Empty out the output directory
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir,dirs_exist_ok=True)


    def predict(
        self,
        prompt: str = Input(
            default=None
        ),
        base_video: CogPath = Input(
            default=None,
        ),

        controlnet_conditioning_scale: float = Input(
            description="Strength of ControlNet. The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original UNet",
            default=0.18,
        ),

        negative_prompt: str = Input(
            default="(worst quality, low quality:1.4), black and white, b&w, sunny, clear skies, calm seas, beach, daytime, ((bright colors)), cartoonish, modern ships, sketchy, unfinished, modern buildings, trees, island",
        ),
        frames: int = Input(
            description="Length of the video in frames (playback is at 8 fps e.g. 16 frames @ 8 fps is 2 seconds)",
            default=128,
            ge=1,
            le=1024,
        ),
        width: int = Input(
            description="Width of generated video in pixels, must be divisable by 8",
            default=256,
            ge=64,
            le=2160,
        ),
        height: int = Input(
            description="Height of generated video in pixels, must be divisable by 8",
            default=384,
            ge=64,
            le=2160,
        ),
        base_model: str = Input(
            description="Choose the base model for animation generation. If 'CUSTOM' is selected, provide a custom model URL in the next parameter",
            default="DarkSushiMixMix_colorful",
            choices=[
                "realisticVisionV40_v20Novae",
                "lyriel_v16",
                "majicmixRealistic_v5Preview",
                "rcnzCartoon3d_v10",
                "toonyou_beta3",
                "DarkSushiMixMix_colorful",
                "CUSTOM",
            ],
        ),
        custom_base_model_url: str = Input(
            description="Only used when base model is set to 'CUSTOM'. URL of the custom model to download if 'CUSTOM' is selected in the base model. Only downloads from 'https://civitai.com/api/download/models/' are allowed",
            default="",
        ),
        prompt_fixed_ratio: float = Input(
            description="Defines the ratio of adherence to the fixed part of the prompt versus the dynamic part (from prompt map). Value should be between 0 (only dynamic) to 1 (only fixed).",
            default=0.5,
            ge=0,
            le=1,
        ),
        scheduler: str = Input(
            description="Diffusion scheduler",
            default="euler_a",
            choices=[
                "ddim",
                "pndm",
                "heun",
                "unipc",
                "euler",
                "euler_a",
                "lms",
                "k_lms",
                "dpm_2",
                "k_dpm_2",
                "dpm_2_a",
                "k_dpm_2_a",
                "dpmpp_2m",
                "k_dpmpp_2m",
                "dpmpp_sde",
                "k_dpmpp_sde",
                "dpmpp_2m_sde",
                "k_dpmpp_2m_sde",
            ],
        ),
        steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=100,
            default=25,
        ),
        guidance_scale: float = Input(
            description="Guidance Scale. How closely do we want to adhere to the prompt and its contents",
            ge=0.0,
            le=20,
            default=7.5,
        ),
        clip_skip: int = Input(
            description="Skip the last N-1 layers of the CLIP text encoder (lower values follow prompt more closely)",
            default=2,
            ge=1,
            le=6,
        ),
        context: int = Input(
            description="Number of frames to condition on (default: max of <length> or 32). max for motion module v1 is 24",
            default=16,
            ge=1,
            le=32,
        ),
        output_format: str = Input(
            description="Output format of the video. Can be 'mp4' or 'gif'",
            default="mp4",
            choices=["mp4", "gif"],
        ),
        playback_frames_per_second: int = Input(default=8, ge=1, le=60),
        film_interpolation: bool = Input(
            description="Whether to use FILM for between-frame interpolation (film-net.github.io)",
            default=False,
        ),
        num_interpolation_steps: int = Input(
            description="Number of steps to interpolate between animation frames",
            default=3,
            ge=1,
            le=50,
        ),
        seed: int = Input(
            description="Seed for different images and reproducibility. Leave blank to randomise seed",
            default=None,
        ),
    ) -> CogPath:
        """
        Animate Diff Prompt Walking CLI w/ QR Monster ControlNet
        NOTE: lora_map, motion_lora_map are NOT supported
        """

        if seed is None or seed < 0:
            seed = -1

        if base_video:
            input_img_dir = "data/controlnet_image/xeno"
            print("Preparing base video")
            path_to_controlnet_video = str(base_video)
            output_dir = f"{input_img_dir}/img"
            if os.path.exists(output_dir):  # Empty out the output directory
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            output_pattern = os.path.join(output_dir, "%04d.png")

            # Run the ffmpeg command to extract frames
            subprocess.run(
                ["ffmpeg", "-i", path_to_controlnet_video, "-vframes", str(frames), output_pattern], check=True
            )

            #copy to controlnet subfolders
            self.copy_dir_contents(output_dir,f"{input_img_dir}/controlnet_openpose")


        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if base_model.upper() == "CUSTOM":
            base_model = self.download_custom_model(custom_base_model_url)

        prompt_travel_json = FAKE_PROMPT_TRAVEL_JSON.format(
            dreambooth_path=f"share/Stable-diffusion/{base_model}.safetensors",
            output_format=output_format,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            prompt_fixed_ratio=prompt_fixed_ratio,
            head_prompt="",
            tail_prompt="",
            negative_prompt=negative_prompt,
            playback_frames_per_second=playback_frames_per_second,
            prompt_map=f"0:(prompt_map)",
            scheduler=scheduler,
            clip_skip=clip_skip,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            loop="false",
        )

        print(f"{'-'*80}")
        print(prompt_travel_json)
        print(f"{'-'*80}")

        file_path = "config/prompts/custom_prompt_travel.json"
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, "w") as file:
            file.write(prompt_travel_json)

        torch.cuda.empty_cache()
        cmd = [
            "animatediff",
            "generate",
            "-c",
            str(file_path),
            "-W",
            str(width),
            "-H",
            str(height),
            "-L",
            str(frames),
            "-C",
            str(context),
        ]
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        (
            stdout_output,
            stderr_output,
        ) = process.communicate()

        print(stdout_output)
        if stderr_output:
            print(f"Error: {stderr_output}")

        if process.returncode:
            raise ValueError(f"Command exited with code: {process.returncode}")

        print("Identifying the GIF path from the generated outputs...")
        output_base_path = "output"
        recent_dir = self.find_recent_directory(output_base_path)
        media_path = self.find_media_file(recent_dir, (".gif", ".mp4"))
        png_folder_path = self.find_png_folder(recent_dir, "00_detectmap")

        # Print the identified paths
        print(f"Identified directory: {recent_dir}")
        print(f"Identified Media Path: {media_path}")
        print(f"Identified PNG Folder Path: {png_folder_path}")

        # Use the new directory with interpolated frames for the ffmpeg input
        input_pattern = str(png_folder_path / "%08d.png")
        output_video = str(Path(recent_dir) / "output_video.mp4")
        ffmpeg_command = [
            "ffmpeg",
            "-r",
            str(
                playback_frames_per_second
            ),
            "-i",
            input_pattern,
            "-vcodec",
            "libx264",
            "-crf",
            "1",  # High quality CRF value
            "-pix_fmt",
            "yuv420p",
            output_video,
        ]

        # Execute the ffmpeg command
        subprocess.run(ffmpeg_command, check=True)

        # Return the path to the new video
        return CogPath(output_video)
