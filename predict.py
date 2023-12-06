# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import glob
import os
from pathlib import Path
#from random import randrange
import re
import shutil
import subprocess

from cog import BasePredictor, Input
from cog import Path as CogPath

import sys
#import gdown
import torch


CONFIG_JSON = """
{{
  "name": "sample",
  "path": "{dreambooth_path}",
  "apply_lcm_lora": {lcm_lora},
  "lcm_lora_scale": 1.0,
  "motion_module": "models/motion-module/mm_sd_v15_v2.safetensors",
  "vae_path": "{vae_path}",
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
  "lora_map": {{
    "share/lora/add_detail.safetensors": {detail}
  }},
  "controlnet_map": {{
    "input_image_dir": "controlnet_image/xeno",
    "max_samples_on_vram": 200,
    "max_models_on_vram": 3,
    "save_detectmap": true,
    "preprocess_on_gpu": true,
    "is_loop": {loop},
    "controlnet_openpose": {{
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
    "fps" : {fps},
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

    def leave_every_nth_png(self,n, dir):
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            if filename.endswith(".png") and filename.startswith("0"):
                file_base, file_ext = os.path.splitext(filename)
                file_number = int(file_base)
                try:
                    if file_number % n != 0:
                        os.remove(file_path)
                        print(f"Dropped: {file_path}")
                except ValueError:
                    print(f"Invalid file format: {file_path}")

    def predict(
        self,
        prompt: str = Input(
            default=None
        ),
        base_video: CogPath = Input(
            default=None
        ),
        lcm_lora: str = Input(
            default="false",
             choices=[
                "true",
                "false"
             ]
        ),
        controlnet_conditioning_scale: float = Input(
            default=1.0,
        ),

        negative_prompt: str = Input(
            default="(worst quality, low quality:1.4)",
        ),

        duration : int = Input( default=2,ge=2,le=60),
        width: int = Input(
            default=216,
            ge=64,
            le=2160,
        ),
        height: int = Input(
            default=384,
            ge=64,
            le=2160,
        ),
        base_model: str = Input(
            default="darkSushiMixMix_colorful",
            choices=[
                "realisticVisionV40_v20Novae",
                "lyriel_v16",
                "majicmixRealistic_v5Preview",
                "rcnzCartoon3d_v10",
                "toonyou_beta3",
                "darkSushiMixMix_colorful",
                "CUSTOM",
            ],
        ),
        custom_base_model_url: str = Input(
            default="",
        ),
        prompt_fixed_ratio: float = Input(
            default=0.5,
            ge=0,
            le=1,
        ),
        scheduler: str = Input(
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
            default=2,
            ge=1,
            le=6,
        ),
        context: int = Input(
            default=16,
            ge=1,
            le=32,
        ),
        output_format: str = Input(
            default="mp4",
            choices=["mp4", "gif"],
        ),
        fps: int = Input(default=12, ge=1, le=60),

        seed: int = Input(
            default=None,
        ),
        detail: float = Input(
            default=1.0,
            ge=0,
            le=1,
        ),
        loose: int = Input(default=1)
    ) -> CogPath:
        """
        Animate Diff Prompt Walking CLI w/ ControlNet (openpose)
        # builds upon the work of neggle, s9roll7, zsxkib
        """

        output_dir = "output"
        if os.path.exists(output_dir):  # Empty outputs
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        audio_file = None
        if seed is None or seed < 0:
            seed = -1

        if height % 8 != 0:
            height =  height + (8-height) % 8
            print(f"height rounded to {height}")
        if width % 8 != 0:
            width =  width + (8-width) % 8
            print(f"width rounded to {width}")

        if base_video:
            input_img_dir = "data/controlnet_image/xeno"
            os.makedirs(input_img_dir,exist_ok=True)
            print("Preparing base video")
            controlnet_img_dir = f"{input_img_dir}/img"
            if os.path.exists(controlnet_img_dir): # empty imgs dir
                shutil.rmtree(controlnet_img_dir)
            os.makedirs(controlnet_img_dir)

            print("Preprocessing ")
            result = subprocess.run( # Check if the input video has an audio stream
                ["ffprobe", "-i", base_video, "-show_streams", "-select_streams", "a", "-loglevel", "error"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            has_audio_stream = bool(result.stdout)

            if has_audio_stream:
                print("Extracting audio")
                command = ['ffmpeg', '-i', str(base_video), '-vn', '-c:a', 'copy', 'audio.mp4',]
                audio_file = f"{output_dir}/audio.mp4"
                subprocess.run(command)

            command1 = [
                'ffmpeg',
                '-y',
                '-i', str(base_video),
                '-vf', f'fps={fps},scale={width}:{height}',
                '-f', 'ismv',
                '-'
            ]

            # break into init frames
            command2 = [
                'ffmpeg',
                '-i', '-',  # Use input from the pipe
                '-start_number', '0',
                '-vsync', '0',
                f'{controlnet_img_dir}/%08d.png'
            ]

            # Run the first command and capture its output
            process1 = subprocess.Popen(command1, stdout=subprocess.PIPE)

            # Run the second command, using the output of the first as input
            subprocess.run(command2, stdin=process1.stdout)

            # Wait for the first process to finish
            process1.wait()
            print("Preprocessing finished")

            if loose>1:
                print("Dropping frames")
                self.leave_every_nth_png(loose, controlnet_img_dir)
            #copy to controlnet subfolders
            print("Copying frames")
            self.copy_dir_contents(controlnet_img_dir,f"{input_img_dir}/controlnet_openpose")

        if base_model.upper() == "CUSTOM":
            base_model = self.download_custom_model(custom_base_model_url)

        prompt_travel_json = CONFIG_JSON.format(
            lcm_lora=lcm_lora,
            dreambooth_path=f"share/Stable-diffusion/{base_model}.safetensors",
            vae_path="share/Stable-diffusion/vae-ft-mse-840000-ema-pruned.safetensors",
            output_format=output_format,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            prompt_fixed_ratio=prompt_fixed_ratio,
            head_prompt="",
            tail_prompt="",
            negative_prompt=f'{negative_prompt} easynegative',
            fps=fps,
            prompt_map=f'"0":"{prompt}"',
            scheduler=scheduler,
            clip_skip=clip_skip,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            loop="false",
            detail=detail
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
        frames_n = duration * fps

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
            str(frames_n),
            "-C",
            str(context),
        ]
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)

        # Read stdout line by line
        with process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')

        # Wait for the process to finish and get stderr
        stdout_output, stderr_output = process.communicate()

        # Print remaining output
        print(stdout_output, end='')

        # Print stderr
        if stderr_output:
            print(f"Error: {stderr_output}")

        if process.returncode:
            raise ValueError(f"Command exited with code: {process.returncode}")
        print("Identifying the video path from the generated outputs...")
        output_base_path = "output"
        recent_dir = self.find_recent_directory(output_base_path)
        media_path = self.find_media_file(recent_dir, (".gif", ".mp4"))
        png_folder_path = self.find_png_folder(recent_dir, "00_detectmap")

        # Print the identified paths
        print(f"Identified directory: {recent_dir}")
        print(f"Identified Media Path: {media_path}")
        print(f"Identified PNG Folder Path: {png_folder_path}")

        if audio_file:
            media_path_with_audio = f'{media_path.split(".")[0]}_a.mp4'
            # restore audio
            print(f"Postprocessing, duration : {duration}")
            command3 = [
                "ffmpeg",
                '-y',
                "-i", audio_file,
                "-i", media_path,
                "-c:v", "copy",
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "20",
                "-vf", "format=yuv420p",
                "-c:a", "copy",
                "-t", str(duration),
                "-map", "1:v",
                "-map", "0:a",
                f'{media_path_with_audio}'
            ]
            subprocess.run(command3)
            return CogPath(media_path_with_audio)

        return CogPath(media_path)

