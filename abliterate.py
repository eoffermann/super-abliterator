#!/usr/bin/env python3
import argparse
import json
import sys

from Abliterator import (
    ModelAbliterator,
    get_harmful_instructions,
    get_harmless_instructions,
)

def load_instructions(json_path):
    """
    Loads additional instructions from a JSON file.
    Expects the JSON file to contain a list of strings or a dict with a key "instructions".
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("instructions", [])
        else:
            print(f"Unexpected JSON structure in {json_path}")
            return []
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description="Abliteration command line utility for model activation ablation."
    )
    # Global arguments
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model path or Hugging Face model identifier (e.g., meta-llama/Meta-Llama-3-70B-Instruct)"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--cache-fname", type=str, default=None, help="Filename for saving/loading activations")
    parser.add_argument(
        "--activation-layers", nargs="+",
        default=['resid_pre', 'resid_post', 'mlp_out', 'attn_out'],
        help="List of activation layers to monitor (default: resid_pre resid_post mlp_out attn_out)"
    )
    parser.add_argument(
        "--positive-toks", nargs="+", type=int, default=[23371, 40914],
        help="List of positive token IDs (default: 23371 40914)"
    )
    parser.add_argument(
        "--negative-toks", nargs="+", type=int, default=[4250],
        help="List of negative token IDs (default: 4250)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Cache subcommand: caches activations from dataset samples
    parser_cache = subparsers.add_parser("cache", help="Cache activations for dataset samples")
    parser_cache.add_argument("--N", type=int, default=512, help="Number of samples to cache (default: 512)")
    parser_cache.add_argument("--batch-size", type=int, default=8, help="Batch size for caching (default: 8)")
    parser_cache.add_argument(
        "--preserve-harmless", action="store_true",
        help="Preserve existing harmless cache when caching activations"
    )
    parser_cache.add_argument(
        "--harmful-hf-path", type=str, default="Undi95/orthogonal-activation-steering-TOXIC",
        help="Hugging Face dataset path for harmful instructions"
    )
    parser_cache.add_argument(
        "--harmless-hf-path", type=str, default="tatsu-lab/alpaca",
        help="Hugging Face dataset path for harmless instructions"
    )
    parser_cache.add_argument(
        "--harmful-json", type=str, default=None,
        help="Path to a JSON file with additional harmful instructions"
    )
    parser_cache.add_argument(
        "--harmless-json", type=str, default=None,
        help="Path to a JSON file with additional harmless instructions"
    )

    # Generate subcommand: generate text from a prompt
    parser_generate = subparsers.add_parser("generate", help="Generate text from a prompt")
    parser_generate.add_argument("--prompt", type=str, required=True, help="Prompt for text generation")
    parser_generate.add_argument("--max-tokens", type=int, default=64, help="Maximum tokens to generate (default: 64)")

    # Test subcommand: test model on a subset of the harmful dataset
    parser_test = subparsers.add_parser("test", help="Test model outputs on harmful test instances")
    parser_test.add_argument("--N", type=int, default=16, help="Number of test samples (default: 16)")
    parser_test.add_argument("--batch-size", type=int, default=4, help="Batch size for testing (default: 4)")

    # Find best refusal direction subcommand
    parser_find = subparsers.add_parser("find_best_dir", help="Find best refusal direction")
    parser_find.add_argument("--N", type=int, default=4, help="Number of samples per direction (default: 4)")
    parser_find.add_argument("--use-hooks", action="store_true", help="Use forward hooks for testing")
    parser_find.add_argument("--invert", action="store_true", help="Invert the computed refusal direction")
    parser_find.add_argument("--positive", action="store_true", help="Score using positive tokens instead")

    # Apply subcommand: apply a refusal direction (optionally finding the best one first)
    parser_apply = subparsers.add_parser("apply", help="Apply refusal direction modifications to the model")
    parser_apply.add_argument(
        "--layers", nargs="+", type=int, default=None,
        help="List of layer indices to apply modifications (default: all layers except layer 0)"
    )
    parser_apply.add_argument(
        "--find-best", action="store_true",
        help="Automatically find and apply the best refusal direction"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load additional instructions from JSON if provided.
    harmful_additional = load_instructions(args.harmful_json) if args.harmful_json else None
    harmless_additional = load_instructions(args.harmless_json) if args.harmless_json else None

    # Load datasets using custom Hugging Face paths and additional instructions.
    harmful_dataset = get_harmful_instructions(
        hf_path=args.harmful_hf_path, additional_instructions=harmful_additional
    )
    harmless_dataset = get_harmless_instructions(
        hf_path=args.harmless_hf_path, additional_instructions=harmless_additional
    )

    # Instantiate the ModelAbliterator with provided settings.
    model_instance = ModelAbliterator(
        model=args.model,
        dataset=[harmful_dataset, harmless_dataset],
        device=args.device,
        cache_fname=args.cache_fname,
        activation_layers=args.activation_layers,
        positive_toks=args.positive_toks,
        negative_toks=args.negative_toks
    )

    # Execute the requested subcommand.
    if args.command == "cache":
        print("Caching activations...")
        model_instance.cache_activations(
            N=args.N,
            batch_size=args.batch_size,
            preserve_harmless=args.preserve_harmless
        )
        if args.cache_fname:
            model_instance.save_activations(args.cache_fname)
            print(f"Activations saved to {args.cache_fname}")
        print("Caching complete.")
    elif args.command == "generate":
        output = model_instance.generate(
            args.prompt, max_tokens_generated=args.max_tokens, stop_at_eos=True
        )
        print("Generated output:")
        for line in output:
            print(line)
    elif args.command == "test":
        print("Running test generation on harmful test instances...")
        model_instance.test(N=args.N, batch_size=args.batch_size)
    elif args.command == "find_best_dir":
        print("Finding best refusal direction...")
        best_dirs = model_instance.find_best_refusal_dir(
            N=args.N, use_hooks=args.use_hooks, invert=args.invert, positive=args.positive
        )
        print("Best refusal directions (score, (activation name, direction)):")
        for score, (act_name, direction) in best_dirs:
            print(f"{act_name}: {score}")
    elif args.command == "apply":
        if args.find_best:
            print("Finding and applying best refusal direction...")
            best_dirs = model_instance.find_best_refusal_dir(N=4, use_hooks=True)
            if not best_dirs:
                print("No best direction found.")
                sys.exit(1)
            # Select the best direction from the sorted list.
            best_direction = best_dirs[0][1][1]
            model_instance.apply_refusal_dirs([best_direction], layers=args.layers)
            print("Best refusal direction applied.")
        else:
            print("No direction specified. Use --find-best to automatically find and apply a refusal direction.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
