import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import config
from utils import (
    load_text_data, build_vocab, save_vocab,
    encode_text, create_sequences
)
from Net import TextGeneratorRNN


class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


def train():
    print("开始训练文本生成模型")
    try:
        text = load_text_data(config.DATA_PATH, config.ENCODING)
        print(f"数据加载完成: {len(text)}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    char2id, id2char, vocab_size = build_vocab(text)
    config.VOCAB_SIZE = vocab_size
    print(f"词汇表大小: {vocab_size}")
    save_vocab(char2id, id2char, 'vocab.json')
    
    encoded_text = encode_text(text, char2id)
    print(f"编码完成: {len(encoded_text)} 个ID")

    sequences, targets = create_sequences(encoded_text, config.SEQUENCE_LENGTH)
    print(f"创建样本: {len(sequences)} 个")
    
    dataset = TextDataset(sequences, targets)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0 
    )
    
    model = TextGeneratorRNN(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (batch_sequences, batch_targets) in enumerate(dataloader):
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            outputs, _ = model(batch_sequences)

            loss = loss_fn(
                outputs.reshape(-1, vocab_size),
                batch_targets
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'embedding_dim': config.EMBEDDING_DIM,
                    'hidden_dim': config.HIDDEN_DIM,
                    'num_layers': config.NUM_LAYERS,
                    'dropout': config.DROPOUT
                }
            }, config.MODEL_SAVE_PATH)
            print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | Loss: {avg_loss:.4f} | ✓ 模型已保存")
        else:
            print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | Loss: {avg_loss:.4f}")
    
    print(f"训练完成，最佳损失: {best_loss:.4f}")
    print(f"模型已保存")



if __name__ == '__main__':
    train()
