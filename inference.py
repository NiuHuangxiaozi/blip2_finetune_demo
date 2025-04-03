'''
Author: riverman nanjing.com
Date: 2025-04-03 23:07:07
LastEditors: riverman nanjing.com
LastEditTime: 2025-04-03 23:31:55
FilePath: /wsj/bliptime/inference.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def infer_image(base_model_path, lora_model_path, image_path):
    """
    使用 LoRA 微调的 BLIP2 模型对 PNG 图片进行推理并打印结果。

    参数：
    - base_model_path: 原始 BLIP2 预训练模型的路径
    - lora_model_path: LoRA 微调后模型的路径
    - image_path: PNG 图片路径
    """
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载原始 BLIP2 模型（8-bit 量化加速）
    model = AutoModelForVision2Seq.from_pretrained(base_model_path, load_in_8bit=True, device_map="auto")

    # 加载 PEFT (LoRA) 适配器
    model = PeftModel.from_pretrained(model, lora_model_path)
    model.to(device)  # 移动到 GPU

    # 加载处理器
    processor = AutoProcessor.from_pretrained(base_model_path)

    # 加载并预处理图片
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 进行推理
    with torch.no_grad():
        output_ids = model.generate(pixel_values=inputs["pixel_values"])

    # 解码生成的文本
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    print("\n=== 推理结果 ===")
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP2 LoRA Image Captioning Inference")
    parser.add_argument("--base_model_path", type=str, required=True, help="原始 BLIP2 预训练模型路径")
    parser.add_argument("--lora_model_path", type=str, required=True, help="LoRA 微调后的模型路径")
    parser.add_argument("--image_path", type=str, required=True, help="要推理的 PNG 图片路径")
    args = parser.parse_args()

    infer_image(args.base_model_path, args.lora_model_path, args.image_path)
