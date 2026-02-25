import pandas as pd
from pathlib import Path
import json

from typing import Optional, Dict, Any, List, Iterable
import wandb

from utils.config import MT_TEST_DATA_META_INFO
import numpy as np
from inference.run_mt import load_model_tokenizer
import inference.run_oss_eval as run_oss_eval

def log_results_to_wandb(
    datasets_metric_results: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    datasets_metric_none_counts: Optional[Dict[str, Dict[str, int]]] = None,
):
    """将多个数据集的结果以表格形式记录到 wandb。

    Args:
        datasets_metric_results: {dataset_name: {metric: value}}
        config: wandb config
        datasets_metric_none_counts: {dataset_name: {metric: none_count}}
    """

    project_name = "mt-eval"
    wandb.init(
        project=project_name,
        name=config["model_name"],
        config=config,
    )

    # 所有出现过的指标集合
    all_metrics: List[str] = []
    seen = set()
    for _ds, mr in datasets_metric_results.items():
        for m in mr.keys():
            if m not in seen:
                seen.add(m)
                all_metrics.append(m)

    # 以 config["metrics"] 为主，若为空则使用所有指标
    cfg_metrics = config.get("metrics") or []
    if cfg_metrics:
        metrics = [m for m in cfg_metrics if m in seen] or all_metrics
    else:
        metrics = all_metrics

    # 构建按数据集聚合后的表格，首列为 data_id（即 dataset_name）
    columns = ["data_id"] + metrics
    rows: List[List[Any]] = []
    for dataset_name, metric_results in datasets_metric_results.items():
        row = [dataset_name]
        for m in metrics:
            row.append(metric_results.get(m, np.nan))
        rows.append(row)

    table = wandb.Table(columns=columns, data=rows)

    # 记录到 wandb
    wandb.log({"metrics_by_dataset": table})

    # 同步到 summary 方便快速查看（data_id/metric）
    for dataset_name, metric_results in datasets_metric_results.items():
        for m, v in metric_results.items():
            wandb.run.summary[f"{dataset_name}/{m}"] = v

    # 记录 None 计数到 summary
    if datasets_metric_none_counts:
        for dataset_name, metric_none_counts in datasets_metric_none_counts.items():
            for m, cnt in metric_none_counts.items():
                key = f"none_count/{dataset_name}/{m}"
                try:
                    wandb.run.summary[key] = int(cnt)
                except Exception:
                    wandb.run.summary[key] = cnt


def _sanitize_filename_component(s: str) -> str:
    try:
        import re as _re
    except Exception:
        _re = None
    s = str(s).strip()
    if _re is not None:
        s = _re.sub(r"[\\/]+", "_", s)
    return s.replace(" ", "_")


def save_results_to_json(
    df: pd.DataFrame,
    mt_list_for_runs_nested: list[list[str]],
    per_item_metric_avgs: Dict[str, list[Optional[float]]],
    valid_metrics: list[str],
    dataset_name: str,
    model_name: str,
    model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    runs: int,
    prompt_type: str,
) -> Path:
    safe_model_name = _sanitize_filename_component(model_name)
    safe_dataset_name = _sanitize_filename_component(dataset_name)
    out_file = Path.cwd() / f"{safe_model_name}__{safe_dataset_name}.json"

    n = len(df)
    items = []
    for i in range(n):
        preds = [mt_list_for_runs_nested[r][i] for r in range(runs)]
        metrics_avg_item = {m: per_item_metric_avgs.get(m, [None] * n)[i] for m in valid_metrics}
        row = df.iloc[i]
        items.append({
            "index": int(i),
            "src_lang": str(row["src_lang"]),
            "trg_lang": str(row["trg_lang"]),
            "lang_pair": f"{row['src_lang']}-{row['trg_lang']}",
            "src_text": row["src_text"],
            "ref_text": row["trg_text"],
            "predictions": preds,
            "metrics_avg": metrics_avg_item,
        })

    json_payload = {
        "data_name": dataset_name,
        "model_name": model_name,
        "model_path": model_path,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "runs": runs,
        "prompt_type": prompt_type,
        "metrics": valid_metrics,
        "items": items,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    return out_file



def run_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    prompt_type: str,
    runs: int,
):
    import inference.run_mt as run_mt

    src_list = df["src_text"].tolist()
    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()

    mt_list_for_runs = []
    for _ in range(runs):        
        func_call_kwargs = {
            "model_path": model_path,
            "src_list": src_list,
            "src_langs": src_langs,
            "trg_langs": trg_langs,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "prompt_type": prompt_type,
            "model": model,
            "tokenizer": tokenizer,
        }
        if "Seed-X" in model_path:
            func_call_kwargs["use_chat_template"] = False
        output_dict = run_mt.func_call(**func_call_kwargs)
        mt_list = output_dict["responses"]
        if len(mt_list) != len(df):
            raise ValueError(
                f"mt_list must have the same length as src_list, but got {len(mt_list)} and {len(df)}"
            )

        mt_list_for_runs.append(mt_list)

    return mt_list_for_runs


def run_eval(
    df: pd.DataFrame,
    mt_list_for_runs: list[list[str]],
    runs: int,
    metrics: list[str],
    oss_models: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, float], list[str], Dict[str, int], Dict[str, list[Optional[float]]]]:
    # 展平 mt 的输出（runs x N -> N*runs）
    mt_list_for_runs = [item for sublist in mt_list_for_runs for item in sublist]

    n = len(df)
    if n == 0:
        raise ValueError("输入数据为空：df 行数为 0")

    # 构造参考译文与语言对的扁平列表，与 mt_list_for_runs 对齐
    ref_mt_list = df["trg_text"].tolist()
    refs_flat = ref_mt_list * runs

    src_langs = df["src_lang"].tolist()
    trg_langs = df["trg_lang"].tolist()
    src_texts = df["src_text"].tolist()

    src_flat = src_texts * runs
    src_langs_flat = src_langs * runs
    trg_langs_flat = trg_langs * runs

    # add evaluation hint to ref
    if "comment" in df.columns:
        comment_list = df["comment"].tolist()
        ref_hint_list = [f"{ref}\n评估重点：\n{comment}" for ref, comment in zip(refs_flat, comment_list)]
        ref_hint_flat = ref_hint_list * runs
    else:
        ref_hint_flat = None

    # 聚合结果：{ metric: dataset-level avg }
    metric_results: Dict[str, float] = {}
    valid_metrics: list[str] = []
    metric_none_counts: Dict[str, int] = {}
    # 每条样本的指标平均分（跨 runs 平均）
    per_item_metric_avgs: Dict[str, list[Optional[float]]] = {}

    def _normalize_metric_output(output: Any, n_items: int, n_runs: int) -> list[float]:
        # 支持多种返回形态：list、list[list]、dict 包含 scores、numpy 数组等
        try:
            import numpy as _np
        except Exception:
            _np = None

        if isinstance(output, dict):
            if "scores" in output:
                output = output["scores"]
            elif "bleurt_scores" in output:
                output = output["bleurt_scores"]

        if _np is not None and isinstance(output, _np.ndarray):
            output = output.tolist()

        if isinstance(output, list):
            if len(output) == n_runs and len(output) > 0 and isinstance(output[0], list):
                # 形如 [[...], [...]]
                return [v for sub in output for v in sub]
            # 扁平向量
            if len(output) == n_items * n_runs:
                return output

        raise ValueError(f"Unexpected metric output shape/type: type={type(output)}, len={getattr(output, '__len__', 'NA')}")

    def _average_overall(scores: list[float]) -> tuple[float, int]:
        vals: list[float] = []
        none_count: int = 0
        for s in scores:
            if s is None:
                none_count += 1
                continue
            try:
                vals.append(float(s))
            except Exception:
                # 跳过无法转换的值
                none_count += 1
                continue
        if not vals:
            return float("nan"), none_count
        return sum(vals) / len(vals), none_count

    def _average_per_item(scores: list[float], n_items: int, n_runs: int) -> list[Optional[float]]:
        avgs: list[Optional[float]] = []
        for i in range(n_items):
            vals: list[float] = []
            for r in range(n_runs):
                idx = r * n_items + i
                s = scores[idx]
                if s is None:
                    continue
                try:
                    vals.append(float(s))
                except Exception:
                    # 跳过无法转换的值
                    continue
            if vals:
                avgs.append(sum(vals) / len(vals))
            else:
                avgs.append(None)
        return avgs

    # 计算各指标并按语言对聚合
    for m in metrics:
        if m == "bleurt":
            try:
                # import eval.bleurt_eval_cli as bleurt_eval_cli
                import eval.bleurt_service as bleurt_eval_cli
            except Exception as e:
                raise ImportError(f"请求 BLEURT 指标，但未找到 bleurt_eval_cli：{e}")
            bleurt_path = "BLEURT-20"
            bleurt_output = bleurt_eval_cli.func_call(bleurt_path, mt_list_for_runs, refs_flat)
            bleurt_scores_flat = _normalize_metric_output(bleurt_output, n, runs)
            avg, none_count = _average_overall(bleurt_scores_flat)
            metric_results[m] = avg
            metric_none_counts[m] = none_count
            # 计算每条样本的平均分
            per_item_metric_avgs[m] = _average_per_item(bleurt_scores_flat, n, runs)
            valid_metrics.append(m)
        elif m == "oss" or m == "oss-120b":
            # 优先使用预加载的 OSS 模型；若未提供则按需加载（向后兼容）
            oss_key = "oss" if m == "oss" else "oss-120b"
            oss_model = None
            if oss_models is not None:
                oss_model = oss_models.get(oss_key)

            oss_model_path = "gpt-oss-20b" if m == "oss" else "gpt-oss-120b"
            if oss_model is None:
                oss_model = run_oss_eval.init_oss_model(oss_model_path)
            oss_output = run_oss_eval.func_call(
                src_list=src_flat,
                mt_list=mt_list_for_runs,
                src_langs=src_langs_flat,
                trg_langs=trg_langs_flat,
                ref_list=ref_hint_flat if ref_hint_flat is not None else refs_flat,
                model=oss_model,
                model_path=oss_model_path,
            )
            oss_scores_flat = _normalize_metric_output(oss_output, n, runs)
            avg, none_count = _average_overall(oss_scores_flat)
            metric_results[m] = avg
            metric_none_counts[m] = none_count
            per_item_metric_avgs[m] = _average_per_item(oss_scores_flat, n, runs)
            valid_metrics.append(m)
        else:
            # 未实现的指标可以在此扩展
            continue

    return metric_results, valid_metrics, metric_none_counts, per_item_metric_avgs


def _clear_mem():
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def main(
    data_id: tuple[str],
    model_path: str,
    model_name: str,
    temperature: float = 0.4,
    top_p: float = 0.7,
    max_new_tokens: int = 4096,
    metrics: list[str] = ["bleurt", "oss"],
    prompt_type: str = "Tower",
    runs: int = 1,
    save_results: bool = False,
):
    if isinstance(data_id, Iterable):
        data_id_list = tuple(data_id)
    elif isinstance(data_id, str):
        data_id_list = tuple(data_id.strip().split(","))
    if not data_id_list:
        raise ValueError("data_id 不能为空")

    # 先加载一次 MT 模型，后续复用，避免在 run_inference 内部重复加载
    model, tokenizer = load_model_tokenizer(model_path)

    # 第一阶段：对所有数据集先跑翻译推理，避免与评估（尤其是 oss）交替进行
    dfs: Dict[str, pd.DataFrame] = {}
    mt_lists_for_runs: Dict[str, list[list[str]]] = {}
    lang_pairs: Dict[str, str] = {}

    for did in data_id_list:
        if did not in MT_TEST_DATA_META_INFO:
            raise ValueError(
                f"data_id {did} not in MT_TEST_DATA_META_INFO: {MT_TEST_DATA_META_INFO.keys()}"
            )
        data_meta_info = MT_TEST_DATA_META_INFO[did]
        data_path = Path(data_meta_info["path"])
        lang_pair = f"{data_meta_info['src_lang']}2{data_meta_info['trg_lang']}"

        if not data_path.exists():
            raise ValueError(f"data_path {data_path} does not exist")

        df = pd.read_parquet(data_path)
        dfs[did] = df
        lang_pairs[did] = lang_pair

        mt_list_for_runs = run_inference(
            df,
            model,
            tokenizer,
            model_path,
            temperature,
            top_p,
            max_new_tokens,
            prompt_type=prompt_type,
            runs=runs,
        )
        mt_lists_for_runs[did] = mt_list_for_runs

    # MT 推理阶段结束，释放 MT 模型以腾出显存给 OSS 评估
    try:
        del model
        del tokenizer
        _clear_mem()
    except Exception:
        pass

    # 预加载 OSS 模型（按需加载使用到的指标）
    oss_models: Dict[str, Any] = {}
    if any(m in metrics for m in ("oss", "oss-120b")):
        if "oss" in metrics:
            oss_models["oss"] = run_oss_eval.init_oss_model("gpt-oss-20b")
        if "oss-120b" in metrics:
            oss_models["oss-120b"] = run_oss_eval.init_oss_model("gpt-oss-120b")

    # 第二阶段：对每个数据集分别做评估（包括 bleurt / oss 等）
    datasets_metric_results: Dict[str, Dict[str, float]] = {}
    datasets_metric_none_counts: Dict[str, Dict[str, int]] = {}
    datasets_valid_metrics: Dict[str, List[str]] = {}

    all_valid_metrics: List[str] = []
    seen_metrics = set()

    for did in data_id_list:
        df = dfs[did]
        mt_list_for_runs = mt_lists_for_runs[did]

        metric_results, valid_metrics, metric_none_counts, per_item_metric_avgs = run_eval(
            df,
            mt_list_for_runs,
            runs,
            metrics,
            oss_models=oss_models,
        )

        datasets_metric_results[did] = metric_results
        datasets_metric_none_counts[did] = metric_none_counts
        datasets_valid_metrics[did] = valid_metrics

        for m in valid_metrics:
            if m not in seen_metrics:
                seen_metrics.add(m)
                all_valid_metrics.append(m)

        # 按需保存所有推理结果及每条样本的指标平均分到当前目录
        if save_results:
            save_results_to_json(
                df=df,
                mt_list_for_runs_nested=mt_list_for_runs,
                per_item_metric_avgs=per_item_metric_avgs,
                valid_metrics=valid_metrics,
                dataset_name=did,
                model_name=model_name,
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                runs=runs,
                prompt_type=prompt_type,
            )

    # 统一构建 wandb config，支持单个或多个数据集
    if len(data_id_list) == 1:
        only_id = data_id_list[0]
        wandb_config = {
            "dataset_name": only_id,
            "model_path": model_path,
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "runs": runs,
            "metrics": datasets_valid_metrics.get(only_id, all_valid_metrics),
            "lang_pair": lang_pairs[only_id],
            "prompt_type": prompt_type,
        }
    else:
        wandb_config = {
            "dataset_names": data_id_list,
            "model_path": model_path,
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "runs": runs,
            "metrics": all_valid_metrics,
            "lang_pairs": lang_pairs,
            "prompt_type": prompt_type,
        }

    log_results_to_wandb(
        datasets_metric_results=datasets_metric_results,
        config=wandb_config,
        datasets_metric_none_counts=datasets_metric_none_counts,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
