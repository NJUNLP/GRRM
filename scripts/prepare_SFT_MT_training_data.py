import fire
import pandas as pd
from typing import List, Tuple, Union
from inference.run_mt import get_prompt


def main(data_path:str, output_path:str, response_key: str):
    df = pd.read_parquet(data_path)
    data_items = []

    for _, row in df.iterrows():
        src_text = row["src_text"]
        src_lang = row["src_lang"]
        trg_lang = row["trg_lang"]
        response = row[response_key].strip()
        prompt = get_prompt('codeblock-think', src_lang, trg_lang, src_text)
        data_items.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        })
    out_df = pd.DataFrame(data_items)
    out_df = out_df.sample(frac=1.0, random_state=114514)
    out_df.to_parquet(output_path, index=False)

        

if __name__ == "__main__":
    fire.Fire(main)