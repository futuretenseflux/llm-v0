import argparse

from src.infer.inference import run_chat_inference, run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--system", type=str, default="")
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.user is not None:
        messages = [
            {"role": "system", "content": str(args.system)},
            {"role": "user", "content": str(args.user)},
        ]
        out = run_chat_inference(
            args.model_path,
            messages,
            reasoning_on=bool(args.reasoning),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            long_context=True,
        )
    else:
        if args.prompt is None:
            raise SystemExit("Provide either --user (chat mode) or --prompt (raw mode).")
        out = run_inference(
            args.model_path,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
            long_context=True,
        )
    print(out)


if __name__ == "__main__":
    main()
