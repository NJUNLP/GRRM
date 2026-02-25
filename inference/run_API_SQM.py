from typing import Dict, Union
import warnings
from utils.config import LANG_MAP

from tqdm import tqdm

from utils.client import get_client
from utils.helpers import find_int_in_string

client = None
model_name = None


promt = """Given a source text in {} and a translation text in {}. Perform a step by step analysis of translation quality and assign a integer score on a scale from 0 to 10. Output your analysis first and output the score in the end line.

Source text:
```
{}
```

Translation text:
```
{}
```
"""


strict_promt = """Given a source text in {} and a translation text in {}. Perform a step by step analysis of translation quality and assign a integer score on a scale from 0 to 10. Output your analysis first and output the score in the end line (do not output any other text after the score so I can easily extract the score).

Source text:
```
{}
```

Translation text:
```
{}
```
"""

def get_prompt(source_lang, target_lang, source_text, mt_text):


    return promt.format(source_lang, target_lang, source_text, mt_text)



def extract_response(response:str):
    response = response.strip()
    last_line_index = response.rfind("\n")
    if last_line_index == -1:
        print("no score line in response:")
        return None
    score_line = response[last_line_index+1:].strip()
    scores = find_int_in_string(score_line)
    if len(scores) == 0:
        print("no score in string: ", score_line)
        return None
    if len(scores) > 1:
        print(f"find multiple scores in string: {score_line}. Use the first one.")

    return {"analysis": response[:last_line_index].strip(), "score": scores[0]}

def run_SQM_eval(src_text:str, mt_text:str, src_lang:str, trg_lang:str, temperature:float=0.0, top_p:float=0.0):
    if len(src_lang) == 2:
        src_lang = LANG_MAP[src_lang]
    if len(trg_lang) == 2:
        trg_lang = LANG_MAP[trg_lang]
    prompt = get_prompt(src_lang, trg_lang, src_text, mt_text)

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
        print("Exception in run_mt_eval: ", e)
        return None

    assessment = chat_completion.choices[0].message.content
    assessment = extract_response(assessment)

    return assessment

def func_call(
    src_list: list,
    mt_list: list,
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
    if model == "gemini-2.5-pro":
        client = Gemini_client()
    elif model == "DeepSeek-R1-0528":
        client = Deepseek_client()
        # switch to strict promt
        global promt
        promt = strict_promt

    for src_text, mt_text, src_lang, trg_lang in tqdm(zip(src_list, mt_list, src_langs, trg_langs), total=len(src_list)):
        retry_temp = temperature
        for _ in range(retry):
            assessment = run_SQM_eval(src_text, mt_text, src_lang, trg_lang, temperature=retry_temp, top_p=top_p)
            if assessment is not None:
                break
            warnings.warn(f"Evaluation failed, retry...")
            retry_temp = 1.0
        
        if assessment is None:
            warnings.warn(f"Evaluation failed, src_text: {src_text}, mt_text: {mt_text}")
            assessment = {"score": None, "analysis": None}

        out_data["scores"].append(assessment["score"])
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
    # out = func_call(test_src_list, test_mt_list, "zh", "en", top_p=0.9)
    # print(out)

    cot="""
    Here is a step-by-step analysis of the translation quality.\n\n### Step-by-Step Analysis\n\n1.  **Component Breakdown of the Source Text:**\n    *   `今天` (jīntiān): This means "today".\n    *   `所有` (suǒyǒu): This means "all".\n    *   `餐品` (cānpǐn): This refers to all food and drink items on a menu, literally "meal products" or "menu items".\n    *   `七五折` (qīwǔ zhé): This is a specific Chinese term for a discount. It means you pay 75% of the original price. The `折` (zhé) system indicates the percentage of the original price to be paid.\n\n2.  **Component-to-Component Mapping and Evaluation:**\n    *   `今天` has been correctly translated to `today`. The placement at the end of the English sentence is natural and idiomatic.\n    *   `所有` has been correctly translated to `All`.\n    *   `餐品` has been translated to `menu items`. This is an excellent choice. While "dishes" or "food items" would be acceptable, "menu items" is more comprehensive and professional, perfectly capturing the meaning of `餐品` in a restaurant context.\n    *   `七五折` has been translated to `25% off`. This is the most crucial and impressive part of the translation. A literal translation would be confusing (e.g., "7.5 fold discount" or "pay 75%"). The translator correctly understood that `七五折` means paying 75% of the price, which is equivalent to a 25% discount in the standard English way of expressing sales. This demonstrates not just linguistic translation but also cultural and commercial localization.\n\n3.  **Evaluation of Fluency and Naturalness:**\n    *   The translated sentence, "All menu items are 25% off today," is perfectly fluent and natural in English. It is exactly how a restaurant or café would advertise this promotion to an English-speaking audience. The grammar, syntax, and word choice are flawless.\n\n4.  **Accuracy and Meaning Preservation:**\n    *   The core message of the source text is fully and accurately preserved. The key information—what is on sale (`All menu items`), the discount amount (`25% off`), and the timeframe (`today`)—is conveyed without any loss or distortion of meaning.\n\n### Conclusion\n\nThis is an exemplary translation. It is not only accurate in its transfer of information but also demonstrates a high level of skill in cultural adaptation by converting the Chinese discount convention (`七五折`) into its standard English equivalent (`25% off`). The language is fluent, natural, and perfectly suited for the context. There are no errors.
    """

    print(cot)
