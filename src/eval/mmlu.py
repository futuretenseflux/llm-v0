from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import get_dataset_config_names, load_dataset

from src.infer.inference import score_candidates_loglikelihood


_CHOICES = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class MMLUExample:
    question: str
    choices: Tuple[str, str, str, str]
    answer_index: int


def _normalize_answer_index(answer: Any) -> int:
    if isinstance(answer, int):
        if 0 <= answer <= 3:
            return answer
        raise ValueError(f"Answer index out of range: {answer}")
    if isinstance(answer, str):
        a = answer.strip()
        if a in _CHOICES:
            return _CHOICES.index(a)
        if a.isdigit():
            return _normalize_answer_index(int(a))
    raise ValueError(f"Unsupported answer format: {answer!r}")


def _row_to_example(row: Dict[str, Any]) -> MMLUExample:
    question = row.get("question")
    choices = row.get("choices")
    answer = row.get("answer")

    if not isinstance(question, str):
        raise ValueError("Expected 'question' to be a string")
    if not (isinstance(choices, (list, tuple)) and len(choices) == 4 and all(isinstance(x, str) for x in choices)):
        raise ValueError("Expected 'choices' to be a list of 4 strings")

    return MMLUExample(
        question=question,
        choices=(choices[0], choices[1], choices[2], choices[3]),
        answer_index=_normalize_answer_index(answer),
    )


def format_question_block(ex: MMLUExample) -> str:
    lines = [
        f"Question: {ex.question}",
        f"A. {ex.choices[0]}",
        f"B. {ex.choices[1]}",
        f"C. {ex.choices[2]}",
        f"D. {ex.choices[3]}",
        "Answer:",
    ]
    return "\n".join(lines)


def format_fewshot_example(ex: MMLUExample) -> str:
    return format_question_block(ex) + f" {_CHOICES[ex.answer_index]}\n\n"


def build_prompt(subject: str, fewshot: Iterable[MMLUExample], test_ex: MMLUExample) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    shots = "".join(format_fewshot_example(s) for s in fewshot)
    test = format_question_block(test_ex) + " "
    return header + shots + test


def list_mmlu_subjects(dataset_name: str) -> List[str]:
    return list(get_dataset_config_names(dataset_name))


def load_mmlu_split(dataset_name: str, subject: str, split: str):
    return load_dataset(dataset_name, subject, split=split)


def evaluate_subject(
    *,
    model_path: str,
    dataset_name: str,
    subject: str,
    k_shot: int = 5,
    max_examples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    dev_ds = load_mmlu_split(dataset_name, subject, "dev")
    test_ds = load_mmlu_split(dataset_name, subject, "test")

    fewshot_pool = [_row_to_example(dev_ds[i]) for i in range(min(len(dev_ds), k_shot))]

    n = len(test_ds) if max_examples is None else min(len(test_ds), int(max_examples))
    correct = 0

    for i in range(n):
        ex = _row_to_example(test_ds[i])
        prompt = build_prompt(subject, fewshot_pool, ex)
        scores = score_candidates_loglikelihood(
            model_path,
            prompt,
            _CHOICES,
            device=device,
        )
        pred = int(max(range(len(scores)), key=lambda j: scores[j]))
        if pred == ex.answer_index:
            correct += 1

    acc = correct / max(1, n)
    return {
        "subject": subject,
        "k_shot": k_shot,
        "n": n,
        "correct": correct,
        "accuracy": acc,
    }


def evaluate_mmlu(
    *,
    model_path: str,
    dataset_name: str,
    subjects: List[str],
    k_shot: int = 5,
    max_examples_per_subject: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    per_subject = []
    total_correct = 0
    total_n = 0

    for subject in subjects:
        res = evaluate_subject(
            model_path=model_path,
            dataset_name=dataset_name,
            subject=subject,
            k_shot=k_shot,
            max_examples=max_examples_per_subject,
            device=device,
        )
        per_subject.append(res)
        total_correct += int(res["correct"])
        total_n += int(res["n"])

    overall_acc = total_correct / max(1, total_n)
    return {
        "dataset": dataset_name,
        "k_shot": k_shot,
        "overall": {"n": total_n, "correct": total_correct, "accuracy": overall_acc},
        "per_subject": per_subject,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset", default="cais/mmlu")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--max-examples-per-subject", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    if args.subjects is None or len(args.subjects) == 0:
        subjects = list_mmlu_subjects(args.dataset)
    else:
        subjects = list(args.subjects)

    res = evaluate_mmlu(
        model_path=args.model_path,
        dataset_name=args.dataset,
        subjects=subjects,
        k_shot=args.k_shot,
        max_examples_per_subject=args.max_examples_per_subject,
        device=args.device,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    else:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
