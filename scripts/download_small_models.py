
import os

from huggingface_hub import snapshot_download



# 定义绝对路径，防止相对路径出错

COMFY_PATH = "/src/ComfyUI" 

# 注意：ComfyUI 里的路径逻辑有时候很奇怪，这里我们用绝对路径最稳

if not os.path.exists(COMFY_PATH):

    COMFY_PATH = "ComfyUI" # 兼容本地测试



SAM2_PATH = os.path.join(COMFY_PATH, "models/sam2")

FLORENCE_PATH = os.path.join(COMFY_PATH, "models/LLM")



# 1. 下载 SAM 2.1 (严格匹配截图文件名)

print(f"Downloading SAM2.1 models to {SAM2_PATH}...")

os.makedirs(SAM2_PATH, exist_ok=True)

os.system(f"wget -O {SAM2_PATH}/sam2.1_hiera_large.safetensors https://huggingface.co/kijai/sam2-safetensors/resolve/main/sam2.1_hiera_large.safetensors")



# 2. 下载 Florence-2 (严格目录结构)

# 插件期望路径：ComfyUI/models/LLM/microsoft/Florence-2-large-ft/config.json

print(f"Downloading Florence-2 models to {FLORENCE_PATH}...")

target_folder = os.path.join(FLORENCE_PATH, "microsoft/Florence-2-large-ft")

os.makedirs(target_folder, exist_ok=True)



snapshot_download(repo_id="microsoft/Florence-2-large-ft", 

                  local_dir=target_folder,

                  local_dir_use_symlinks=False)



print("✅ All small models baked successfully!")

