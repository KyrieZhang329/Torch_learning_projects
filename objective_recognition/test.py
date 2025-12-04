import torch
from Net import ObjectRecognitionNet
from utils import load_model_weights, predict_single_image, predict_batch_images
import config
import os


def main():
    print("开始测试物体识别模型")
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = ObjectRecognitionNet().to(device)
    model = load_model_weights(model, config.MODEL_SAVE_PATH)
    print("模型加载成功")
    
    test_folder = './test_data'
    
    if os.path.exists(test_folder):
        print(f"\n正在测试文件夹: {test_folder}")
        results = predict_batch_images(test_folder, model, device=device)
        
        if results:
            for filename, class_idx, class_name, confidence in results:
                print(f"  {filename}: {class_name} ({confidence:.4f})")
        else:
            print(f"  无有效图片")
    else:
        print(f"文件夹不存在: {test_folder}")


if __name__ == '__main__':
    main()