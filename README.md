# GRRM: Group Relative Reward Modeling for Machine Translation

<a href="https://arxiv.org/abs/2602.14028">
  <img src="https://img.shields.io/badge/Paper-arXiv-blue"></a>
<a href="https://huggingface.co/collections/double7/grrm">
  <img src="https://img.shields.io/badge/Models-Hugging%20Face-brightgreen"></a>
<a href="https://huggingface.co/collections/double7/grrm">
  <img src="https://img.shields.io/badge/Data-Download-orange"></a>
<a href="LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-yellow"></a>

Official implementation of the paper: **GRRM: Group Relative Reward Modeling for Machine Translation**.

This repository contains the code for **GRRM (Group Relative Reward Model)**, a Generative Reward Model** instantiated under the proposed **Group Quality Metric (GQM) paradigm for **reinforcement learning-based machine translation (MT) optimization** with **GRPO**.

## Highlights

- **Improved reward accuracy:** large gains especially on challenging samples (idioms, slang, terminology) and restores reward variance needed by GRPO advantages. Also Robust to reward hacking.
- **GRRM**: efficient GQM-based reward model trained with SFT + RLVR.
- **Cross-lingual generalization**: GRRM trained on Zh–En can support multilingual MT optimization.
- **MT optimization**: GRPO training with GRRM yields strong improvements on WMT-style benchmarks and challenge sets.


Ranking accuracy performance and downstream translation performance on Seed-X-Challenge:
<p align="center">
  <img src="static/teaser.png" alt="Teaser" width="600">
</p>

## Method Overview


**Core idea:** Standard generative reward models often evaluate candidates independently (**Scalar Quality Metric, SQM**) and suffer from score saturation, which causes vanishing advantages in GRPO and stalls optimization.  
We propose **GQM**, which evaluates a *group* of candidates jointly to produce reliable fine-grained intra-group ranking, and implement it as **GRRM**, a high-throughput reward model with explicit comparative reasoning.

**GQM** evaluates the *entire group* together and outputs:
- comparative analysis
- predicted ranking
- scores consistent with that ranking

<p align="center">
  <img src="static/GRRM_framework.png" alt="GRRM Framework" width="600">
</p>

## Performance Overview


---

## Getting Started

### Installation

```bash
git clone https://github.com/NJUNLP/GRRM.git
cd GRRM
pip install -e . --no-build-isolation
```

Extra dependencies available:
- `infer`: install vllm for inference.
- `eval`: sacrebleu and bleurt for translation evaluation.

---

## Quick Use

### 1) Use GRRM to rank a group of translations (GQM inference)

[This script](inference/run_rm_GQM.py) performs Group Quality Metric (GQM) inference using vLLM to evaluate and rank multiple translation candidates. It includes prompt templates, result parsing and return the scores and raw model outputs.

Example usage:
```python
import inference.run_rm_GQM as rm_GQM

output = rm_GQM.func_call(
    model_path="double7/GRRM-Qwen2.5-7B",
    src_list=["I have a frog in my throat."],
    mt_list=[["我嗓子有点哑。", "我嗓子眼里好像有只青蛙。"]],
    src_langs=["en"],
    trg_langs=["zh"],
    temperature=1.0,
    top_p=1.0,
    max_new_tokens=8192,
    max_retries=6,
    prompt_type="ranking_score",
)

# output["scores"] -> [[8, 3]]  # scores for each candidate
# output["responses"] -> ["...model response text..."]
```

### 2) Use MT model for translation inference

[This script](inference/run_mt.py) performs machine translation inference using vLLM. It supports multiple prompt formats and answer extraction methods for different model types. It returns a dictionary with translation responses and raw model outputs.

Example usage:
```python
import inference.run_mt as mt

output = mt.func_call(
    model_path="double7/GRRM-MT-Qwen2.5-7B",
    src_list=["Hello, how are you?", "What is your name?"],
    src_langs=["en", "en"],
    trg_langs=["zh", "zh"],
    sampling_n=1,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=512,
    retry=4,
    prompt_type="codeblock-think",
    use_chat_template=True,
)

# output["responses"] -> ["你好，你好吗？", "你叫什么名字？"]
# output["raw_outputs"] -> ["...raw model output...", "...raw model output..."]
```


---

## Training Pipeline

Work in Progress...


## Citation

```bibtex
@misc{yang2026grrmgrouprelativereward,
      title={GRRM: Group Relative Reward Modeling for Machine Translation}, 
      author={Sen Yang and Shanbo Cheng and Lu Xu and Jianbing Zhang and Shujian Huang},
      year={2026},
      eprint={2602.14028},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.14028},
}
```