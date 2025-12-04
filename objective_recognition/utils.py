import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import config


CIFAR10_CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)


def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    trainset = datasets.CIFAR10(
        root=config.DATA_PATH,
        train=True,
        download=True,
        transform=transform
    )
    
    testset = datasets.CIFAR10(
        root=config.DATA_PATH,
        train=False,
        download=True,
        transform=transform
    )
    
    train_size = len(trainset)
    test_size = len(testset)
    
    train_loader = DataLoader(
        trainset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, test_loader, (train_size, test_size)


def load_model_weights(model, model_path=config.MODEL_SAVE_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def predict_single_image(image_path, model, device='cpu'):
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    class_name = CIFAR10_CLASSES[predicted_class]
    confidence_score = confidence.item()
    
    return predicted_class, class_name, confidence_score


def predict_batch_images(image_dir, model, file_extensions=('.jpg', '.png', '.jpeg', '.bmp'), 
                        device='cpu'):
    results = []
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"文件夹不存在: {image_dir}")
    
    for filename in os.listdir(image_dir):
        if filename.endswith(file_extensions):
            image_path = os.path.join(image_dir, filename)
            class_idx, class_name, confidence = predict_single_image(image_path, model, device)
            results.append((filename, class_idx, class_name, confidence))
    
    return results
