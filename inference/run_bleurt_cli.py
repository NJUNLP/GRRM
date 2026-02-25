from typing import Dict
import subprocess
import warnings
import json
import os
import tempfile
from pathlib import Path

def preprocess_text(text:str):
    if "\n" in text:
        warnings.warn("output multiple lines, concatenate to one line.")
        text = text.replace("\n", " ")
    return text


def func_call(bleurt_path , mt_list, ref_list):
    assert len(mt_list) == len(ref_list)
    fd, test_file = tempfile.mkstemp(prefix="bleurt_", suffix=".temp")
    os.close(fd)
    try:
        with open(test_file, "w") as f_test:
            for mt_text, ref_text in zip(mt_list, ref_list):
                f_test.write(json.dumps({"candidate": preprocess_text(mt_text), "reference": preprocess_text(ref_text)}) + "\n")
        output = main(test_file, bleurt_path)
        if len(output["scores"]) != len(mt_list):
            raise ValueError(f"BLEURT output scores count ({len(output['scores'])}) does not match input count ({len(mt_list)})")
        return output
    finally:
        try:
            if os.path.exists(test_file):
                os.remove(test_file)
        except Exception:
            warnings.warn(f"Failed to remove {test_file}")


def main(test_file: str, bleurt_path: str = "BLEURT-20") -> Dict:
    def get_bleurt_python() -> str:
        env_python = os.environ.get("BLEURT_PYTHON")
        if env_python and os.path.exists(env_python):
            return env_python

        env_dir = os.environ.get("BLEURT_VENV_DIR")
        if env_dir:
            candidate = os.path.join(env_dir, "bin", "python")
            if os.path.exists(candidate):
                return candidate

        repo_root = Path(__file__).resolve().parents[1]
        default_python = repo_root / ".bleurt_venv" / "bin" / "python"
        if default_python.exists():
            return str(default_python)

        warnings.warn("BLEURT venv not found. Falling back to system python3. Set BLEURT_PYTHON or BLEURT_VENV_DIR to override.")
        return "python3"

    bleurt_py = get_bleurt_python()
    fd_scores, scores_file = tempfile.mkstemp(prefix="bleurt_", suffix=".scores")
    os.close(fd_scores)
    try:
        generation_command = (
            f'{bleurt_py} -m bleurt.score_files '
            f'-sentence_pairs_file="{test_file}" '
            f'-bleurt_checkpoint="{bleurt_path}" '
            f'-bleurt_batch_size 128 '
            f'-scores_file="{scores_file}"'
        )
        subprocess.run(generation_command, shell=True, check=True)
        scores = []
        with open(scores_file, "r") as f_scores:
            for line in f_scores:
                line = line.strip()
                if not line:
                    continue
                try:
                    scores.append(float(line))
                except ValueError:
                    warnings.warn(f"Invalid score line (expected float): {line}")
        return {"scores": scores}
    finally:
        try:
            if os.path.exists(scores_file):
                os.remove(scores_file)
        except Exception:
            warnings.warn(f"Failed to remove {scores_file}")



if __name__ == "__main__":
    import fire

    fire.Fire(main)
