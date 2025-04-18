import os
import cv2
import datetime
import time
import pickle
import numpy as np
import subprocess
import torch
from PIL import Image
from omegaconf import OmegaConf
import base64
import shutil
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory

# Assume these paths are correctly set based on your environment
FFMPEG = "ffmpeg"  # Or the full path to ffmpeg if needed
DEFAULT_CONFIG_ONNX = "configs/onnx_infer.yaml"
DEFAULT_CONFIG_ONNX_MP = "configs/onnx_mp_infer.yaml"
DEFAULT_CONFIG_TRT = "configs/trt_infer.yaml"
DEFAULT_CONFIG_TRT_MP = "configs/trt_mp_infer.yaml"
VIDEO_OUTPUT_DIR = "/FasterLivePortrait/results/"  # Ensure this directory exists

app = Flask(__name__)
live_portrait_pipeline = None

class LivePortraitAnimator:
    def __init__(self, config_path=DEFAULT_CONFIG_ONNX, use_mediapipe=False):
        cfg_path = config_path
        if config_path == "onnx":
            cfg_path = DEFAULT_CONFIG_ONNX_MP if use_mediapipe else DEFAULT_CONFIG_ONNX
        elif config_path == "trt":
            cfg_path = DEFAULT_CONFIG_TRT_MP if use_mediapipe else DEFAULT_CONFIG_TRT
        else:
            cfg_path = config_path  # Use the provided path directly

        infer_cfg = OmegaConf.load(cfg_path)
        from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline
        self.pipeline = GradioLivePortraitPipeline(infer_cfg)
        self.is_animal = False # Initialize animal flag

    def _ensure_directory(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def image_to_video(self, source_image_path, driving_video_path, output_path="output_video.mp4",
                       relative_motion=False, do_crop=True, paste_back=True, driving_multiplier=1.0,
                       stitching=True, crop_driving_video=False, video_editing_head_rotation=False,
                       animation_region="all", source_crop_scale=2.3, source_crop_x=0.0,
                       source_crop_y=-0.125, driving_crop_scale=2.2, driving_crop_x=0.0,
                       driving_crop_y=-0.1, motion_smooth_observation_variance=1e-7, cfg_scale=4.0, uuid=None):
        """
        Takes a source image and a driving video to generate an animated video.
        """
        output_path = self._ensure_directory(output_path)
        args_user = {
            'source': source_image_path,
            'driving': driving_video_path,
            'flag_relative_motion': relative_motion,
            'flag_do_crop': do_crop,
            'flag_pasteback': paste_back,
            'driving_multiplier': driving_multiplier,
            'flag_stitching': stitching,
            'flag_crop_driving_video': crop_driving_video,
            'flag_video_editing_head_rotation': video_editing_head_rotation,
            'src_scale': source_crop_scale,
            'src_vx_ratio': source_crop_x,
            'src_vy_ratio': source_crop_y,
            'dri_scale': driving_crop_scale,
            'dri_vx_ratio': driving_crop_x,
            'dri_vy_ratio': driving_crop_y,
            'driving_smooth_observation_variance': motion_smooth_observation_variance,
            'animation_region': animation_region,
            'cfg_scale': cfg_scale
        }
        update_ret = self.pipeline.update_cfg(args_user)
        video_path, _, _ = self.pipeline.run_video_driving(driving_video_path, source_image_path, update_ret=update_ret)

        if video_path and os.path.exists(video_path):
            os.rename(video_path, output_path)
            return output_path
        else:
            return None

    def image_to_image(self, source_image_path, driving_image_path, output_path="output_image.png",
                       do_crop=True, source_crop_scale=2.3, source_crop_x=0.0, source_crop_y=-0.125,
                       driving_crop_scale=2.2, driving_crop_x=0.0, driving_crop_y=-0.1,
                       relative_motion=False, paste_back=True, driving_multiplier=1.0,
                       stitching=True, crop_driving_video=False, video_editing_head_rotation=False,
                       animation_region="all", motion_smooth_observation_variance=1e-7, cfg_scale=4.0, uuid=None):
        """
        Takes a source image and a driving image to generate a new animated image.
        """
        output_path = self._ensure_directory(output_path)
        print(f">>> Processing image to image for UUID: {uuid}, Source Image: {source_image_path}, Driving Image: {driving_image_path}, Output: {output_path}")
        args_user = {
            'source':  source_image_path ,
            'driving':driving_image_path ,
            'flag_relative_motion': relative_motion,
            'flag_do_crop': do_crop,
            'flag_pasteback': paste_back,
            'driving_multiplier': driving_multiplier,
            'flag_stitching': stitching,
            'flag_crop_driving_video': crop_driving_video,
            'flag_video_editing_head_rotation': video_editing_head_rotation,
            'src_scale': source_crop_scale,
            'src_vx_ratio': source_crop_x,
            'src_vy_ratio': source_crop_y,
            'dri_scale': driving_crop_scale,
            'dri_vx_ratio': driving_crop_x,
            'dri_vy_ratio': driving_crop_y,
            'driving_smooth_observation_variance': motion_smooth_observation_variance,
            'animation_region': animation_region,
            'cfg_scale': cfg_scale
        }
        update_ret = self.pipeline.update_cfg(args_user)
        image_path, _ , _= self.pipeline.run_image_driving(driving_image_path, source_image_path, update_ret=update_ret)

        if image_path and os.path.exists(image_path):
            os.rename(image_path, output_path)
            return output_path
        else:
            return None

    def speech_to_video(self, source_image_path, audio_path, output_path="output_video_from_speech.mp4",
                        do_crop=True, source_crop_scale=2.3, source_crop_x=0.0, source_crop_y=-0.125,
                        relative_motion=False, paste_back=True, driving_multiplier=1.0,
                        stitching=True, animation_region="all", cfg_scale=4.0, voice_name='af', uuid=None):
        """
        Takes a source image and an audio file to generate an animated video.
        """
        output_path = self._ensure_directory(output_path)
        print(f"Processing speech to video for UUID: {uuid}, Source Image: {source_image_path}, Audio: {audio_path}, Output: {output_path}")
        args_user = {
            'source': source_image_path,
            'driving': audio_path,
            'flag_relative_motion': relative_motion,
            'flag_do_crop': do_crop,
            'flag_pasteback': paste_back,
            'driving_multiplier': driving_multiplier,
            'flag_stitching': stitching,
            'src_scale': source_crop_scale,
            'src_vx_ratio': source_crop_x,
            'src_vy_ratio': source_crop_y,
            'animation_region': animation_region,
            'cfg_scale': cfg_scale
        }
        update_ret = self.pipeline.update_cfg(args_user)
        video_path, _, _ = self.pipeline.run_audio_driving(audio_path, source_image_path, update_ret=update_ret)

        if video_path and os.path.exists(video_path):
            os.rename(video_path, output_path)
            return output_path
        else:
            return None

def save_input_data(image_data, input_file_path, uuid_str, input_type="video"):
    print(f"Saving input data for UUID: {uuid_str}, Type: {input_type}")
    if input_type == "image":
        image_filename = os.path.join(VIDEO_OUTPUT_DIR, f"{uuid_str}_input.png")
        if "base64" in image_data:
            image_data = image_data.split(',')[1]
            with open(image_filename, "wb") as fh:
                fh.write(base64.b64decode(image_data))
        else:
            source_file_name = image_data.split('/')[-1]
            # //FasterLivePortrait/assets/examples/assets/examples
            image_data = os.path.join("/FasterLivePortrait/static/source",source_file_name)
            print("Saving image data from:", image_data, "to:", image_filename)
            shutil.copy(image_data, image_filename)
        return image_filename, None
    elif input_type == "video":
        video_path = os.path.join("/FasterLivePortrait/assets/examples/driving", input_file_path)
        video_path_file = os.path.join(VIDEO_OUTPUT_DIR, f"{uuid_str}_video_path.txt")
        with open(video_path_file, "w") as f:
            f.write(video_path)
        return None, video_path
    elif input_type == "audio":
        audio_path = os.path.join("/FasterLivePortrait/assets/examples/audio", input_file_path)
        audio_path_file = os.path.join(VIDEO_OUTPUT_DIR, f"{uuid_str}_audio_path.txt")
        with open(audio_path_file, "w") as f:
            f.write(audio_path)
        return None, audio_path
    return None, None

def process_live_portrait(uuid_str, animator, workload_type, image_path, input_path, extra_params=None):
    output_file = os.path.join(VIDEO_OUTPUT_DIR, f"{uuid_str}_output.{'mp4' if workload_type != 'image_to_image' else 'png'}")
    status_file = os.path.join(VIDEO_OUTPUT_DIR, f"{uuid_str}_status.txt")
    try:
        print(f"Processing {workload_type} for UUID: {uuid_str}, Image Path: {image_path}, Input Path: {input_path}, output: {output_file}")
        if workload_type == "image_to_video":
            animator.image_to_video(image_path, input_path, output_file, **(extra_params or {}))
        elif workload_type == "image_to_image":
            animator.image_to_image(image_path, input_path, output_file, **(extra_params or {}))
        elif workload_type == "speech_to_video":
            animator.speech_to_video(image_path, input_path, output_file, **(extra_params or {}))
        with open(status_file, "w") as f:
            f.write("ready")
    except Exception as e:
        print(f"Error during {workload_type} processing for UUID {uuid_str}: {e}")
        with open(status_file, "w") as f:
            f.write("error")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if live_portrait_pipeline is None:
      return jsonify({'status': 'error','message':'Live Portrait Pipeline initialization failed'})

    data = request.get_json()
    image_data = data.get('image')
    input_file = data.get('input_file') # Could be video or audio filename or driving image data/path
    workload_type = data.get('workload_type')
    uuid_str = str(data.get('uuid'))
    
    print(f"reqiuest received with following params : input file {input_file}, workload type {workload_type}, uuid : {uuid_str}" )
    
    extra_params = {k: v for k, v in data.items() if k not in ['image', 'input_file', 'workload_type']}

    if not all([image_data, input_file, workload_type]):
        return jsonify({'status': 'error', 'message': 'Missing input data', 'uuid': uuid_str})

    image_path, input_path = "", ""
    print(f"Received request for UUID: {uuid_str}, Workload Type: {workload_type}, Image Data: {image_data}, Input File: {input_file}")
    if workload_type == "image_to_video":
        image_path, _ = save_input_data(image_data, None, uuid_str, "image")
        _, input_path = save_input_data(None, input_file, uuid_str, "video")
    elif workload_type == "image_to_image":
        image_path, _ = save_input_data(image_data, None, uuid_str, "image")
        input_path, _ = save_input_data(input_file, None, uuid_str, "image") # Treat input_file as driving image data
    elif workload_type == "speech_to_video":
        image_path, _ = save_input_data(image_data, None, uuid_str, "image")
        _, input_path = save_input_data(None, input_file, uuid_str, "audio")
    else:
        return jsonify({'status': 'error', 'message': 'Invalid workload type', 'uuid': uuid_str})

    if not image_path or not input_path:
        return jsonify({'status': 'error', 'message': 'Failed to save input files', 'uuid': uuid_str})

    animator = LivePortraitAnimator(config_path="trt") # Initialize animator for each request
    process_live_portrait(uuid_str, animator, workload_type, image_path, input_path, extra_params)
    return jsonify({'status': 'processing', 'uuid': uuid_str})

@app.route('/status/<uuid_str>', methods=['GET'])
def status(uuid_str):
    status_file = os.path.join(VIDEO_OUTPUT_DIR, f"{uuid_str}_status.txt")
    output_filename = f"{uuid_str}_output.{'mp4' if 'video' in request.args.get('type', '') else 'png'}"
    print(f"Checking status for UUID: {uuid_str}, Status file: {status_file}, output filename: {output_filename}, status file exists: {os.path.exists(status_file)}")
    output_path = f"/output/{output_filename}"

    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status = f.read().strip()
        if status == "ready":
            return jsonify({'status': 'ready', 'output_path': output_path})
        elif status == "error":
            return jsonify({'status': 'error'})
        else:
            return jsonify({'status': 'processing'})
    else:
        return jsonify({'status': 'processing'})

@app.route('/output/<filename>', methods=['GET'])
def serve_output(filename):
    return send_from_directory(VIDEO_OUTPUT_DIR, filename)

if __name__ == '__main__':
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    animator_init = LivePortraitAnimator(config_path="trt")
    live_portrait_pipeline = animator_init.pipeline # Initialize the pipeline once
    app.run(debug=True, host="0.0.0.0", port=9898, ssl_context='adhoc')
    

# if __name__ == '__main__':
#     # Example Usage:

#     # Initialize the animator
#     animator = LivePortraitAnimator(config_path="trt") # or "trt"

#     # Create dummy input files if they don't exist
#     source_image_path_1 = "/FasterLivePortrait/assets/examples/source/s0.jpg"
#     source_image_path_2 = "/FasterLivePortrait/assets/examples/source/s5.jpg"
#     driving_video_path = "/FasterLivePortrait/assets/examples/driving/d6.mp4"
#     driving_audio_path = "/FasterLivePortrait/assets/examples/sample_audio.wav"
#     dummy_image = source_image_path_1
#     dummy_video = driving_video_path
#     dummy_audio = driving_audio_path

#     # if not os.path.exists(dummy_image):
#     #     Image.new('RGB', (256, 256), color='red').save(dummy_image)
#     # if not os.path.exists(dummy_video):
#     #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     #     out = cv2.VideoWriter(dummy_video, fourcc, 20.0, (256, 256))
#     #     for i in range(100):
#     #         out.write(np.zeros((256, 256, 3), dtype=np.uint8))
#     #     out.release()
#     # if not os.path.exists(dummy_audio):
#     #     import soundfile as sf
#     #     sf.write(dummy_audio, np.random.rand(24000 * 5) * 0.1, 24000) # 5 seconds of random audio

#     # # Image to Video
#     # output_video_path = animator.image_to_video(
#     #     source_image_path=dummy_image,
#     #     driving_video_path=dummy_video,
#     #     output_path="/FasterLivePortrait/results/animated_video.mp4"
#     # )
#     # if output_video_path:
#     #     print(f"Generated video: {output_video_path}")
#     # else:
#     #     print("Failed to generate video.")

#     # # Image to Image
#     # output_image_path = animator.image_to_image(
#     #     source_image_path=dummy_image,
#     #     driving_image_path=dummy_image, # Using the same dummy image for simplicity
#     #     output_path="/FasterLivePortrait/results/animated_image.png"
#     # )
#     # if output_image_path:
#     #     print(f"Generated image: {output_image_path}")
#     # else:
#     #     print("Failed to generate image.")

#     # Speech to Video
#     output_speech_video_path = animator.speech_to_video(
#         source_image_path=dummy_image,
#         audio_path=dummy_audio,
#         output_path="/FasterLivePortrait/results/speech_driven_video.mp4"
#     )
#     if output_speech_video_path:
#         print(f"Generated speech-driven video: {output_speech_video_path}")
#     else:
#         print("Failed to generate speech-driven video.")
