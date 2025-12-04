import torch
import config
from utils import load_vocab, encode_text, decode_text
from Net import TextGeneratorRNN


def load_model(model_path=config.MODEL_SAVE_PATH):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config_dict = checkpoint['config']
        model_state = checkpoint['model_state_dict']
        model = TextGeneratorRNN(
            vocab_size=config_dict['vocab_size'],
            embedding_dim=config_dict['embedding_dim'],
            hidden_dim=config_dict['hidden_dim'],
            num_layers=config_dict['num_layers'],
            dropout=config_dict['dropout']
        )
        model.load_state_dict(model_state)
        model.eval()
    except FileNotFoundError:
        print(f"请先训练模型")
        return None, None, None

    char2id, id2char = load_vocab('vocab.json')
    return model, char2id, id2char


def generate_text(model, seed_text, generate_length, char2id, id2char,
                 device='cpu', temperature=0.8):
    model = model.to(device)
    model.eval()
    
    generated = seed_text
    
    with torch.no_grad():
        seed_ids = encode_text(seed_text, char2id)
        input_ids = torch.tensor([seed_ids], dtype=torch.long).to(device)
        hidden = None
        
        for _ in range(generate_length):
            next_char_id, hidden = model.predict_next_char(
                input_ids,
                hidden,
                temperature=temperature
            )
            next_char = id2char[next_char_id]
            generated += next_char
            seed_ids = seed_ids[1:] + [next_char_id]
            input_ids = torch.tensor([seed_ids], dtype=torch.long).to(device)
    
    return generated


def interactive_generate(model, char2id, id2char, device='cpu'):
    seed = config.SEED_TEXT if hasattr(config, 'SEED_TEXT') else "你好"
    generated_text = generate_text(
        model,
        seed,
        config.GENERATE_LENGTH,
        char2id,
        id2char,
        device=device,
        temperature=config.TEMPERATURE
    )
    print("生成结果:")
    print(generated_text)


def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model, char2id, id2char = load_model()
    
    if model is None:
        return

    interactive_generate(model, char2id, id2char, device=device)


if __name__ == '__main__':
    main()
