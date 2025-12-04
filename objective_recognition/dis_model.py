import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import config
from Net import ObjectRecognitionNet
from utils import get_data_loaders


def train():
    print("开始训练物体识别模型")
    train_loader, test_loader, (train_size, test_size) = get_data_loaders()
    print(f"训练集大小: {train_size} 张")
    print(f"测试集大小: {test_size} 张")
    
    model = ObjectRecognitionNet()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    writer = SummaryWriter(config.LOGS_DIR)
    total_train_steps = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_steps += 1
            
            if total_train_steps % 100 == 0:
                print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Step {total_train_steps} | Loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), total_train_steps)
        
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_test_loss += loss.item()
                
                accuracy = (outputs.argmax(1) == labels).sum().item()
                total_accuracy += accuracy
        
        avg_test_loss = total_test_loss / len(test_loader)
        accuracy_rate = 100 * (total_accuracy / test_size)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy_rate:.2f}%")
        
        writer.add_scalar("test_loss", avg_test_loss, epoch + 1)
        writer.add_scalar("accuracy", total_accuracy / test_size, epoch + 1)
    
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"模型已保存")
    writer.close()


if __name__ == '__main__':
    train()