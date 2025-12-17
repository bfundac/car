
import os

import shutil

import json

import random

import time

from typing import List

from cog import BasePredictor, Input, Path

from comfyui import ComfyUI

from cog_model_helpers import optimise_images

import requests



# 屏蔽 verify_ssl 警告

requests.packages.urllib3.disable_warnings()



OUTPUT_DIR = "/tmp/outputs"

INPUT_DIR = "/tmp/inputs"

COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]



# Flux Dev 官方权重 (fp8 能够显著减少下载时间和显存，但这里我们用 fp16 保证质量，使用 Replicate 缓存)

FLUX_WEIGHTS_URL = "https://weights.replicate.delivery/default/flux-dev/flux1-dev.safetensors"



class Predictor(BasePredictor):

    def setup(self, weights: str = None):

        self.comfyUI = ComfyUI("127.0.0.1:8188")

        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        

        # 1. 设置 Flux 模型 (运行时下载)

        ckpt_dir = "ComfyUI/models/checkpoints"

        os.makedirs(ckpt_dir, exist_ok=True)

        self.flux_path = os.path.join(ckpt_dir, "flux1-dev.safetensors")



        if not os.path.exists(self.flux_path):

            print(f"⏳ Flux model not found. Downloading...")

            start = time.time()

            # 优先尝试 pget (Replicate 高速下载器)

            if os.system(f"pget {FLUX_WEIGHTS_URL} {self.flux_path}") != 0:

                print("⚠️ pget failed, using wget...")

                os.system(f"wget -O {self.flux_path} {FLUX_WEIGHTS_URL}")

            print(f"✅ Downloaded Flux in {time.time() - start:.2f}s")



    def cleanup(self):

        self.comfyUI.clear_queue()

        for directory in ALL_DIRECTORIES:

            if os.path.exists(directory):

                shutil.rmtree(directory)

            os.makedirs(directory)



    def update_workflow(self, workflow_json, input_filename):

        """

        核心修复逻辑：遍历 workflow，找到所有 LoadImage 节点，

        强制将文件名修改为我们上传的文件名。

        """

        wf = json.loads(workflow_json)

        for node_id, node in wf.items():

            if node["class_type"] == "LoadImage":

                print(f"🔧 Fixing LoadImage node [{node_id}]: {node['inputs']['image']} -> {input_filename}")

                node["inputs"]["image"] = input_filename

            

            # 针对 SAM2 节点的潜在修复，如果它有 hidden input 引用了图片

            # 通常 SAM2 是通过连线获取 IMAGE 的，所以不需要改 SAM2 节点本身的参数

            

            # 针对随机种子 (KSampler, Florence2)

            if "inputs" in node and "seed" in node["inputs"]:

                node["inputs"]["seed"] = random.randint(1, 10000000000)

                

        return wf



    def predict(

        self,

        workflow_json: str = Input(description="ComfyUI API Format JSON", default=""),

        input_file: Path = Input(description="Input image", default=None),

        return_temp_files: bool = Input(description="Return temp files", default=False),

        output_format: str = optimise_images.predict_output_format(),

        output_quality: int = optimise_images.predict_output_quality(),

        randomise_seeds: bool = Input(description="Randomise seeds", default=True),

        force_reset_cache: bool = Input(description="Force reset cache", default=False),

    ) -> List[Path]:

        

        self.cleanup()



        # 1. 处理输入文件

        input_filename = "input.jpg"

        if input_file:

            target_path = os.path.join(INPUT_DIR, input_filename)

            shutil.copy(input_file, target_path)

        

        # 2. 动态修正 Workflow

        try:

            wf = self.update_workflow(workflow_json, input_filename)

        except json.JSONDecodeError:

            raise ValueError("❌ Invalid JSON provided. Please export 'API Format' JSON from ComfyUI.")



        # 3. 执行

        print("🚀 Sending workflow to ComfyUI...")

        self.comfyUI.connect()

        self.comfyUI.run_workflow(wf)

        

        # 4. 获取结果

        return optimise_images.optimise_image_files(

            output_format, 

            output_quality, 

            self.comfyUI.get_files([OUTPUT_DIR])

        )

