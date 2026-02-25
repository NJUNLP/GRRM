import fire
import pandas as pd
from utils.config import LANG_MAP
from typing import List, Tuple, Union
from inference.run_rm_GQM import get_prompt
from utils.helpers import _score_to_rank
from utils.config import candidate_identifiers


def _remove_last_line_if_contains_keyword(text: str, keywords: list[str]) -> str:
    """
    Remove the last line of the input text if it contains any of the given keywords.

    :param text: The input string (possibly multi-line)
    :param keywords: A list of keywords to check in the last line
    :return: The modified string
    """
    # Split the text into lines
    lines = text.splitlines()

    if not lines:
        return text  # empty input, nothing to do

    last_line = lines[-1]

    # Check if any keyword appears in the last line
    if any(keyword in last_line for keyword in keywords):
        # Remove the last line
        lines = lines[:-1]

    # Rejoin the remaining lines
    return "\n".join(lines)




def _get_cand_orders(rank_str: str) -> List[str]:
    """
    Get the candidate orders from the ranking string.
    """
    rank_str = rank_str.strip()
    cand_list = []
    cand_idx = 0
    while cand_idx < len(rank_str):
        cand_char = rank_str[cand_idx]
        assert cand_char in candidate_identifiers, f"Invalid rank string {rank_str} at index {cand_idx}"
        cand_list.append(cand_char)
        cand_idx += 4 # interval is 4, e.g. B = C > A
    assert len(cand_list) > 0
    return cand_list

def get_response(analysis, scores:list[int], prompt_type: str = "ranking_score"):
    if prompt_type == "score":
        scores_str = [f"{candidate_identifiers[i]}: {scores[i]}" for i in range(len(scores))]
        scores_str = ", ".join(scores_str)
        return f"{analysis}\n\n{scores_str}\n"
    elif prompt_type == "ranking":
        analysis = analysis.strip()
        analysis = _remove_last_line_if_contains_keyword(analysis, ["Score", "score", "Scoring", "scoring", "Ranking", "ranking"])
        score_dict = {candidate_identifiers[i]: scores[i] for i in range(len(scores))}
        rank_str = _score_to_rank(score_dict)
        return f"{analysis}\n\n### Final Ranking:\n\n{rank_str}\n"
    elif prompt_type == "ranking_score":
        analysis = analysis.strip()
        analysis = _remove_last_line_if_contains_keyword(analysis, ["Score", "score", "Scoring", "scoring", "Ranking", "ranking"])
        score_dict = {candidate_identifiers[i]: scores[i] for i in range(len(scores))}
        rank_str = _score_to_rank(score_dict)
        cand_orders = _get_cand_orders(rank_str)
        scores_str = [f"{cand_orders[i]}: {score_dict[cand_orders[i]]}" for i in range(len(cand_orders))]
        scores_str = ", ".join(scores_str)
        return f"{analysis}\n\n### Final Ranking:\n\n{rank_str}\n\n### Scores:\n\n{scores_str}\n"

def main(data_path:str, output_path:str, mt_key: str, score_key: str, analysis_key: str, prompt_type: str = "ranking_score", add_example: bool = False):
    assert prompt_type in ["score", "ranking", "ranking_score"]
    df = pd.read_parquet(data_path)

    data_items = []

    for _, row in df.iterrows():
        src_text = row["src_text"]
        mt_texts = row[mt_key]
        src_lang = row["src_lang"]
        trg_lang = row["trg_lang"]
        analysis = row[analysis_key]
        scores = [int(score) for score in row[score_key]]
        if len(src_lang) == 2:
            src_lang = LANG_MAP[src_lang]
        if len(trg_lang) == 2:
            trg_lang = LANG_MAP[trg_lang]
        if len(mt_texts) != len(scores):
            raise ValueError(f"mt_texts ({len(mt_texts)}) and scores ({len(scores)}) should have the same length.")
        prompt = get_prompt(src_lang, trg_lang, src_text, mt_texts, prompt_type, add_example)
        response = get_response(analysis, scores, prompt_type)
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