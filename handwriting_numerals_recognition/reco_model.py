import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import config
from Net import HandwritingNet
from utils import get_data_loaders


def train():
    print("开始训练手写数字识别模型")
    train_loader, test_loader = get_data_loaders()
    print(f"训练集大小: {len(train_loader.dataset)} 张")
    print(f"测试集大小: {len(test_loader.dataset)} 张")
    
    model = HandwritingNet()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(config.LOGS_DIR)
    total_train_steps = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_train_steps += 1
            
            if total_train_steps % 100 == 0:
                print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Step {total_train_steps} | Loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), total_train_steps)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {running_loss / len(train_loader):.4f}")
        
        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_test_loss = total_test_loss / len(test_loader)
        
        print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
        writer.add_scalar("test_loss", avg_test_loss, epoch + 1)
        writer.add_scalar("accuracy", correct / total, epoch + 1)
    
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"模型已保存")
    writer.close()


if __name__ == '__main__':
    train()


