from typing import Dict, Union
import warnings
from tqdm import tqdm

from utils.client import get_client
from inference.run_oss_GQM import get_prompt, extract_response

client = None
model_name = None



def run_GQM_eval(src_text:str, mt_texts:list[str], src_lang:str, trg_lang:str, temperature:float=0.0, top_p:float=0.0):
    prompt = get_prompt(src_lang, trg_lang, src_text, mt_texts)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=8192,
            # response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"Exception in run_mt_eval: {e}")
        return None

    assessment = chat_completion.choices[0].message.content
    assessment = extract_response(assessment, len(mt_texts))

    return assessment

def func_call(
    src_list: list,
    mt_list: list[list],
    src_langs: Union[str, list],
    trg_langs: Union[str, list],
    temperature: float = 0.4,
    top_p: float = 1.0,
    retry:int = 3,
    model: str = "gemini-2.5-pro"
):
    assert len(src_list) == len(mt_list)
    if isinstance(src_langs, str):
        src_langs = [src_langs] * len(src_list)
    if isinstance(trg_langs, str):
        trg_langs = [trg_langs] * len(src_list)

    
    out_data = {}
    out_data["scores"] = []
    out_data["response"] = []

    # init client
    global client, model_name
    assert model in ["gemini-2.5-pro", "DeepSeek-R1-0528"]
    model_name = model
    print(f"using model: {model}")
    client = get_client("azure")


    for src_text, mt_texts, src_lang, trg_lang in tqdm(zip(src_list, mt_list, src_langs, trg_langs), total=len(src_list)):
        retry_temp = temperature
        for _ in range(retry):
            assessment = run_GQM_eval(src_text, mt_texts, src_lang, trg_lang, temperature=retry_temp, top_p=top_p)
            if assessment is not None:
                break
            warnings.warn(f"Evaluation failed, retry...")
            retry_temp = 1.0
        
        if assessment is None:
            warnings.warn(f"Evaluation failed, src_text: {src_text}")
            assessment = {"scores": None, "analysis": None}

        out_data["scores"].append(assessment["scores"])
        out_data["response"].append(assessment)
    
    return out_data
    





    

if __name__ == "__main__":
    # import fire

    # fire.Fire(main)
    test_src_list = [
        "今天所有餐品七五折",
        "今天所有餐品七五折",
    ]
    test_mt_list = [
        "All menu items are 25% off today.",
        "All menu items are 75% off today.",
    ]
    
    # func_call(
    #     test_src_list[:1],
    #     [test_mt_list],
    #     "zh",
    #     "en",
    #     temperature=0.4,
    #     top_p=1.0,
    #     retry=3
    # )

    import pandas as pd
    df = pd.read_parquet("parquet_data/zh2en.parquet")
    test_src_list = [
        df.iloc[-1].src_text
    ]
    test_mt_list = [
        df.iloc[-1].src2trg_sampling_text[:4]
    ]
    func_call(
        test_src_list,
        test_mt_list,
        "zh",
        "en",
        temperature=0.4,
        top_p=1.0,
        retry=3
    )