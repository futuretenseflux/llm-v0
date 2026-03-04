
import argparse
import json
import os
import sys

# Add the project root to python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lm_eval
from src.infer import lm_eval_wrapper  # This registers 'my_custom_model'

def main():
    parser = argparse.ArgumentParser(description="Run lm-eval harness with local model")
    parser.add_argument("--model", default="my_custom_model", help="Model type to use (my_custom_model, sft_model)")
    parser.add_argument("--model-path", required=True, help="Path to the model checkpoint")
    parser.add_argument("--tasks", default="mmlu", help="Comma-separated list of tasks")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--device", default="cuda", help="Device to use (e.g. cuda, cpu)")
    parser.add_argument(
        "--long-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable long-context RoPE base (default: true)",
    )
    parser.add_argument("--reasoning", action="store_true", help="Enable reasoning mode for sft_model")
    parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for sft_model (ChatML)")
    parser.add_argument("--output-path", default=None, help="Path to save results JSON")
    parser.add_argument("--limit", type=float, default=None, help="Limit number of examples per task (for debugging)")
    parser.add_argument("--num-fewshot", type=int, default=None, help="Number of few-shot examples")
    
    args = parser.parse_args()
    
    tasks = args.tasks.split(",")
    
    print(f"Loading model ({args.model}) from {args.model_path}...")
    print(f"Running tasks: {tasks}")
    
    model_args = {
        "model_path": args.model_path,
        "device": args.device,
        "long_context": bool(args.long_context),
    }
    
    if args.reasoning:
        model_args["reasoning"] = True

    if args.system_prompt is not None:
        model_args["system_prompt"] = str(args.system_prompt)
    
    results = lm_eval.simple_evaluate(
        model=args.model,
        model_args=model_args,
        tasks=tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")
    else:
        print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()
