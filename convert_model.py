import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

def convert_to_onnx():
    # 1. Load the model and feature extractor from Hugging Face
    model_name = "Qwen2-VL-2B-Instruct"
    print(f"Loading model: {model_name}...")
    local_dir = "/home/xsf/fl/edge/models/Qwen2-VL-2B-Instruct"
    model = AutoModelForImageClassification.from_pretrained(local_dir)
    processor = AutoImageProcessor.from_pretrained(local_dir)

    # 2. Define a dummy input with the correct dimensions
    # 兼容 image_size、height/width 或默认值
    # 这部分逻辑是为了从模型的预处理器配置中，智能地找出模型期望的输入图像高度和宽度。
    # 不同的模型可能会用不同的字段名来存储这个信息，所以这里做了兼容性处理。

    if hasattr(processor, "image_size"):
        # 检查是否存在 'image_size' 属性
        size = processor.image_size
        if isinstance(size, dict):
            # 如果 'image_size' 是一个字典，例如 {'height': 224, 'width': 224}
            height = size.get("height", 224)
            width = size.get("width", 224)
        else:
            # 如果 'image_size' 是一个整数，例如 224
            height = width = size
    elif hasattr(processor, "size") and "height" in processor.size and "width" in processor.size:
        # 兼容另一种常见的配置格式，其中尺寸信息存储在 'size' 字典里
        height = processor.size["height"]
        width = processor.size["width"]
    else:
        # 如果以上都找不到，则使用一个最常见的默认值 224x224
        height = width = 224  # 默认值

    # 这是最终创建虚拟输入的地方
    # torch.randn 创建一个符合标准正态分布的随机张量
    # (1, 3, height, width) 这个形状代表：
    # - 1: Batch size 为 1
    # - 3: 颜色通道为 3 (R, G, B)
    # - height, width: 图像的高度和宽度
    dummy_input = torch.randn(1, 3, height, width)

    # 3. Set the model to evaluation mode
    model.eval()

    # 4. Define input and output names for the ONNX graph
    input_names = ["input"]
    output_names = ["output"]
    onnx_path = "mobilenetv2.onnx"

    # 5. Export the model to ONNX
    print(f"Exporting model to {onnx_path}...")
    # NOTE: The 'dynamic_axes' argument is only effective when dynamo=False.
    # PyTorch 2.9+ uses dynamo export logic by default, so 'dynamic_axes' will be ignored.
    # In future versions, consider migrating to the 'dynamic_shapes' argument.
    torch.onnx.export(
        model,
        dummy_input,
        opset_version=11,  # 常用的 opset 版本
        input_names=input_names,
        output_names=output_names,
        opset_version=11,  # A commonly supported opset version
        dynamic_axes={
            'input': {0: 'batch_size'},  # Allow for dynamic batch size
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    print("Model conversion successful!")

if __name__ == "__main__":
    convert_to_onnx()
