
import torch
import json
from transformers import AutoTokenizer, AutoConfig, MT5EncoderModel
from typing import Union, List
from tqdm import tqdm
import os


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class MTRanker(torch.nn.Module):
    def __init__(self, encoder_config):
        super().__init__()
        self.encoder = MT5EncoderModel(encoder_config)
        self.num_classes = 2
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, self.num_classes)
    
    def forward(self, input_ids, attention_mask):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        seq_lengths = torch.sum(attention_mask, keepdim=True, dim=1)
        pooled_hidden_state = torch.sum(encoder_output * attention_mask.unsqueeze(-1).expand(-1, -1, self.encoder.config.hidden_size), dim=1)
        pooled_hidden_state /= seq_lengths
        prediction_logit = self.classifier(pooled_hidden_state)
        return prediction_logit


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    backbone = config.get('backbone', 'google/mt5-base')

    
    encoder_config = AutoConfig.from_pretrained(backbone)
    
    model = MTRanker(encoder_config)
    
    index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
    single_checkpoint_path = os.path.join(model_path, 'pytorch_model.bin')
    
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        state_dict = {}
        shard_files = sorted(list(set(index['weight_map'].values())))
        
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            shard_state_dict = torch.load(shard_path, map_location='cpu')
            state_dict.update(shard_state_dict)
            print(f"Loaded shard: {shard_file}")
        
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded all weights from {model_path}")
    elif os.path.exists(single_checkpoint_path):
        state_dict = torch.load(single_checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded weights from {single_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Could not find pytorch_model.bin or pytorch_model.bin.index.json in {model_path}")
    
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    return tokenizer, model


def func_call(
    model_path: str,
    src_list: list[str],
    mt_list: list[list[str]],
    src_langs: Union[str, List[str]] = None,
    trg_langs: Union[str, List[str]] = None,
    batch_size: int = 32,
    tokenizer=None,
    model=None,
    **kwargs,
):
    if len(src_list) != len(mt_list):
        raise ValueError("src_list and mt_list must have the same length.")

    if model is None or tokenizer is None:
        tokenizer, model = load_model_and_tokenizer(model_path)

    test_data = []
    for src_text, mt_texts in zip(src_list, mt_list):
        if len(mt_texts) != 2:
            raise ValueError(
                "mt-ranker requires exactly 2 translation candidates per source."
            )
        test_data.append(
            {
                "src_text": src_text,
                "mt_text_0": mt_texts[0],
                "mt_text_1": mt_texts[1],
            }
        )

    out_list = []

    for batch_samples in tqdm(batch(test_data, batch_size), desc="Processing batches"):
        input_texts = []
        for sample in batch_samples:
            input_text = f"Source: {sample['src_text']} Translation 0: {sample['mt_text_0']} Translation 1: {sample['mt_text_1']}"
            input_texts.append(input_text)

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs)

        probs = torch.softmax(logits, dim=-1).tolist()
        out_list.extend(probs)

    return {"scores": out_list, "responses": None}


if __name__ == "__main__":
    model_path = "/mnt/nfs06/yangs/LLM/ibraheemmoosa/mt-ranker-large"

    src_list = [
        "Le chat est sur le tapis.",
        "Hello, how are you?",
        "我喜欢喝苹果汁！",
        "今日所有餐品七五折"
    ]

    mt_list = [
        ["The cat is on the bed.", "The cat is on the carpet."],
        ["Bonjour, comment ça va?", "Bonjour, comment allez-vous?"],
        ["I like to drink apple juice!", "I don't like to drink orange juice!"],
        ["All meals today are 25% off!", "All meals today are 75% off!"]
    ]

    print("Testing mt-ranker inference script...")
    print(f"Source texts: {src_list}")
    print(f"Translation pairs: {mt_list}")

    result = func_call(model_path, src_list, mt_list, batch_size=2)

    print(f"\nResults:")
    for i, (src, mts, scores) in enumerate(zip(src_list, mt_list, result["scores"])):
        print(f"\nSample {i+1}:")
        print(f"Source: {src}")
        print(f"Translation 0: {mts[0]} (score: {scores[0]:.4f})")
        print(f"Translation 1: {mts[1]} (score: {scores[1]:.4f})")
        if scores[0] > scores[1]:
            print("Translation 0 is preferred.")
        else:
            print("Translation 1 is preferred.")
