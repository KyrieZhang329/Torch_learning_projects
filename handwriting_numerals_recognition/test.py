import torch
from Net import HandwritingNet
from utils import load_model_weights, predict_single_image, predict_batch_images
import config
import os


def main():
    print("开始测试手写数字识别模型")
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = HandwritingNet().to(device)
    model = load_model_weights(model, config.MODEL_SAVE_PATH)
    print("模型加载成功")
    
    test_folders = ['./numerals/num_1', './numerals/num_2', './numerals/yh']
    
    for folder in test_folders:
        if os.path.exists(folder):
            print(f"\n正在测试文件夹: {folder}")
            results = predict_batch_images(folder, model, device=device)
            
            if results:
                for filename, predicted_class in results:
                    print(f"  {filename} → {predicted_class}")
            else:
                print(f"  无有效图片")
        else:
            print(f"文件夹不存在: {folder}")


if __name__ == '__main__':
    main()