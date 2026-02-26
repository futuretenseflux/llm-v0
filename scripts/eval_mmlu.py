import argparse
import json

from src.eval.mmlu import evaluate_mmlu, list_mmlu_subjects


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset", default="cais/mmlu")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--max-examples-per-subject", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    subjects = list_mmlu_subjects(args.dataset) if not args.subjects else list(args.subjects)

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
