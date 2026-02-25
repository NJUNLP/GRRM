import pandas as pd
from utils.config import TOWER_LANGS
import random
random.seed(114514)
from inference.run_mt import get_prompt
import fire



def run_prepare(df, testset=False):

    data_source = "towerblocks"
    src_lang = []
    trg_lang = []

    output = []

    for _, row in df.iterrows():
        src_lang = row["src_lang"]
        trg_lang = row["trg_lang"]
        src_text = row["src_text"]
        prompt = get_prompt('codeblock-think', src_lang, trg_lang, src_text)
        data_item = {
            "prompt": [
                {'role': 'user', 'content': prompt}
            ],
            "ability": "translation",
            "data_source": data_source,
            "extra_info": {
                "src_lang": src_lang,
                "trg_lang": trg_lang,
                "src_text": src_text,
                "trg_text": row["trg_text"],
            }
        }
        if testset:
            data_item["reward_model"] = {"style": "rule", "ground_truth": row["trg_text"]}
        output.append(data_item)

    out_df = pd.DataFrame(output)

    if not testset:
        out_df = out_df.sample(frac=1, random_state=114514)

    return out_df

def run_prepare_towerx(df, testset=False, trg_lang_num=1):
    """
    prepare x2x data
    """

    data_source = "towerblocks"
    src_lang = []
    trg_lang = []

    output = []

    for _, row in df.iterrows():
        src_lang = row["src_lang"]
        trg_lang = row["trg_lang"]
        langs_candidate = TOWER_LANGS.copy()
        langs_candidate.remove(src_lang)
        langs_candidate.remove(trg_lang)
        if trg_lang_num > len(langs_candidate):
            trg_lang_num = len(langs_candidate)
        trg_langs = random.sample(langs_candidate, trg_lang_num)

        for trg_lang in trg_langs:
            src_text = row["src_text"]
            prompt = get_prompt('codeblock-think', src_lang, trg_lang, src_text)
            data_item = {
                "prompt": [
                    {'role': 'user', 'content': prompt}
                ],
                "ability": "translation",
                "data_source": data_source,
                "extra_info": {
                    "src_lang": src_lang,
                    "trg_lang": trg_lang,
                    "src_text": src_text,
                    "trg_text": row["trg_text"],
                }
            }
            if testset:
                data_item["reward_model"] = {"style": "rule", "ground_truth": row["trg_text"]}
            output.append(data_item)

    out_df = pd.DataFrame(output)

    if not testset:
        out_df = out_df.sample(frac=1, random_state=114514)

    return out_df


def construct_tower(data_path, output_path, testset=False):
    df = pd.read_parquet(data_path)
    df = run_prepare(df, testset)
    df.to_parquet(output_path, index=False)

    

def construct_towerx(data_path, output_path, testset=False):
    df = pd.read_parquet(data_path)
    en_df = run_prepare(df, testset)
    x2x_df = run_prepare_towerx(df, testset, trg_lang_num=1)
    # x2x_df_test = x2x_df.sample(n=128, random_state=114514)
    # x2x_df_test.to_parquet("parquet_data/training_data/towerx2_all_mt-code-block-think.verl/test.parquet", index=False)
    df = pd.concat([en_df, x2x_df], axis=0)
    df = df.sample(frac=1, random_state=114514)
    df.to_parquet(output_path, index=False)



if __name__ == "__main__":
    fire.Fire({
        "construct_tower": construct_tower,
        "construct_towerx": construct_towerx,
    })
    
    # data_path = "parquet_data/raw/tower_all.parquet"
    # output_path = "parquet_data/training_data/towerx_v2_all_mt-code-block-think.verl/train.parquet"
