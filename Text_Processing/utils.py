import json
import os


def load_text_data(file_path, encoding='utf-8'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        text = f.read()
    
    return text


def build_vocab(text):
    unique_chars = sorted(set(text))
    char2id = {char: idx for idx, char in enumerate(unique_chars)}
    id2char = {idx: char for char, idx in char2id.items()}
    vocab_size = len(unique_chars)
    return char2id, id2char, vocab_size


def save_vocab(char2id, id2char, save_path='vocab.json'):
    vocab = {
        'char2id': char2id,
        'id2char': {int(k): v for k, v in id2char.items()} 
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(load_path='vocab.json'):
    with open(load_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    char2id = vocab['char2id']
    id2char = {int(k): v for k, v in vocab['id2char'].items()}
    
    return char2id, id2char


def encode_text(text, char2id):
    encoded = [char2id[char] for char in text if char in char2id]
    return encoded


def decode_text(ids, id2char):
    if hasattr(ids, 'cpu'):
        ids = ids.cpu().numpy().tolist()
    elif hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if isinstance(ids[0], list):
        ids = [item for sublist in ids for item in sublist]
    
    text = ''.join([id2char[int(idx)] for idx in ids if int(idx) in id2char])
    return text


def create_sequences(encoded_text, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(encoded_text) - seq_length):
        seq = encoded_text[i:i + seq_length]
        target = encoded_text[i + seq_length]
        
        sequences.append(seq)
        targets.append(target)
    
    return sequences, targets

