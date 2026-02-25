from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os

import torch
from tqdm import tqdm

from utils.config import LANG_MAP
from typing import Union, Dict


def load_direct_prompt(
    tokenizer, src_lang, trg_lang, src_text, mt_text, chat_template=True
):
    if len(src_lang) == 2:
        src_lang = LANG_MAP[src_lang]
    if len(trg_lang) == 2:
        trg_lang = LANG_MAP[trg_lang]
    prompt = f"Translate the following text from {src_lang} into {trg_lang}:\n{src_lang}: {src_text}\n{trg_lang}:"

    if chat_template:
        messages = [
            {"role": "user", "content": prompt},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt += " "

    full_prompt = f"{prompt}{mt_text}{tokenizer.eos_token}"
    return full_prompt


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]




def load_model_tokenizer(model_path, trust_remote_code=False):
    from safetensors import safe_open
    from trl import AutoModelForCausalLMWithValueHead

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        config=model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    module = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    try:
        vhead_file = os.path.join(model_path, "value_head.safetensors")
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            v_params = {key: f.get_tensor(key) for key in f.keys()}
        module.load_state_dict(v_params, strict=False)
    except:
        raise
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    module = module.eval()
    return module, tokenizer


def func_call(
    src_texts: list,
    mt_texts: list,
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    batch_size: int = 16,
    chat_template: bool = True,
    tokenizer=None,
    model=None,
    model_path: str = None,
):

    if model_path is None:
        assert model is not None and tokenizer is not None
    else:
        model, tokenizer = load_model_tokenizer(model_path)

    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_texts)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_texts)

    assert len(src_texts) == len(mt_texts)

    test_data = []

    for src_text, mt_text, src_lang, trg_lang in zip(
        src_texts, mt_texts, src_langs, trg_langs
    ):
        test_data.append(
            {
                "src_text": src_text,
                "mt_text": mt_text,
                "src_lang": src_lang,
                "trg_lang": trg_lang,
            }
        )

    example_prompt = load_direct_prompt(
        tokenizer,
        test_data[0]["src_lang"],
        test_data[0]["trg_lang"],
        test_data[0]["src_text"],
        test_data[0]["mt_text"],
    )

    print("Use prompt: {}".format(example_prompt))

    out_list = []

    for batch_samples in tqdm(batch(test_data, batch_size), desc="Processing batches"):
        input_texts = []
        for sample in batch_samples:
            assert (
                "src_text" in sample
                and "mt_text" in sample
                and "src_lang" in sample
                and "trg_lang" in sample
            )
            src_text = sample["src_text"]
            mt_text = sample["mt_text"]
            src_lang = sample["src_lang"]
            trg_lang = sample["trg_lang"]

            input_text = load_direct_prompt(
                tokenizer,
                src_lang,
                trg_lang,
                src_text,
                mt_text,
                chat_template=chat_template,
            )

            input_texts.append(input_text)

        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            values = model(**inputs, return_dict=True, use_cache=False)[-1]

        scores = values.gather(
            dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1)
        )

        scores = scores.squeeze(-1).tolist()
        out_list.extend(scores)

    return {"scores": out_list}


# def main(
#     model_path: str,
#     src_path: str,
#     mt_path: str,
#     src_lang: str,
#     trg_lang: str,
#     batch_size: int = 16,
#     chat_template: bool = True,
# ):
#     """
#     return_pivot: return pivot translation instead of target translation
#     """

#     src_texts = []
#     mt_texts = []

#     with open(src_path, "r") as f_src, open(mt_path, "r") as f_mt:
#         for src_line, mt_line in zip(f_src, f_mt):
#             src_texts.append(src_line.strip())
#             mt_texts.append(mt_line.strip())

#     return func_call(
#         src_texts,
#         mt_texts,
#         src_lang,
#         trg_lang,
#         batch_size,
#         chat_template,
#         model_path=model_path,
#     )

