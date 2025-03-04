# SuperAbliterator

SuperAbliterator is a Python toolkit for performing activation ablation experiments on large language models. It provides utilities to cache activations, calculate and apply “refusal directions” (i.e. modifications intended to alter undesirable model behaviors), and easily test the effects of these modifications. The project is based on the GitHub project [`FailSpy/abliterator`](https://github.com/FailSpy/abliterator) and aims to make the workflow more maintainable, easier to expand, and more user-friendly.

> **Note:** A Gradio GUI is under development. Documentation for the graphical interface will be coming soon.

## Table of Contents
- [Project Overview](#project-overview)
- [Background and Motivation](#background-and-motivation)
- [Key Features](#key-features)
- [Installation](#installation)
- [Command-Line Utility: abliterate.py](#command-line-utility-abliteratepy)
  - [Global Options](#global-options)
  - [Subcommands and Usage Examples](#subcommands-and-usage-examples)
    - [Cache Activations](#cache-activations)
    - [Generate Text](#generate-text)
    - [Test Model Behavior](#test-model-behavior)
    - [Find and Apply Refusal Directions](#find-and-apply-refusal-directions)
- [Example Python Usage](#example-python-usage)
- [Future Directions](#future-directions)

---

## Project Overview

SuperAbliterator is designed for researchers and practitioners who want to explore and manipulate the internals of transformer-based language models. By caching activations and computing directional modifications, users can test how changes in specific activation components affect overall model behavior.

## Background and Motivation

This project originally started as a set of IPython notebook experiments for activation ablation. Over time, it evolved into a more structured and extensible toolkit. The goal is to encapsulate common functionality—such as dataset handling, activation caching, and directional modifications—into a reusable library. While the original code (from [`FailSpy/`](https://github.com/FailSpy/)) laid the groundwork, this version strives for improved maintainability and ease-of-use.

## Key Features

- **Activation Caching:** Quickly cache model activations using samples from both harmful and harmless datasets.
- **Refusal Direction Computation:** Calculate and apply modifications (refusal directions) to alter model behavior.
- **Dataset Integration:** Seamlessly load harmful and harmless instruction datasets from Hugging Face, with support for custom JSON files containing additional instructions.
- **Flexible Command-Line Interface:** Execute core tasks (caching, testing, text generation, and modification) directly from the command line.
- **Context Management:** Safely experiment with modifications using Python context management to temporarily apply changes.

## Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/eoffermann/super-.git
cd super-
pip install -r requirements.txt
```

## Command-Line Utility: abliterate.py

The `abliterate.py` script provides a command-line interface (CLI) for all key operations of the toolkit. It not only wraps the core library functionality but also allows you to point the functions `get_harmful_instructions` and `get_harmless_instructions` to custom Hugging Face datasets, and to pass additional instructions via JSON files.

### Global Options

When running the script, you must provide several global arguments:
- `--model`: The model path or Hugging Face model identifier (e.g., `meta-llama/Meta-Llama-3-70B-Instruct`).
- `--device`: Device to use (default is `cuda`).
- `--cache-fname`: Filename for saving or loading cached activations.
- `--activation-layers`: List of activation layers to monitor (default: `resid_pre resid_post mlp_out attn_out`).
- `--positive-toks`: List of token IDs that are considered positive (default: `23371 40914`).
- `--negative-toks`: List of token IDs that are considered negative (default: `4250`).

### Subcommands and Usage Examples

The CLI supports several subcommands. Below are the most common ones:

#### Cache Activations

Caches model activations on both harmful and harmless instruction samples.

```bash
python abliterate.py --model meta-llama/Meta-Llama-3-70B-Instruct \
  cache --N 512 --batch-size 8 --preserve-harmless \
  --harmful-hf-path Undi95/orthogonal-activation-steering-TOXIC \
  --harmless-hf-path tatsu-lab/alpaca \
  --harmful-json path/to/harmful_instructions.json \
  --harmless-json path/to/harmless_instructions.json
```

- **Explanation:**
  - `--N`: Number of samples to cache.
  - `--preserve-harmless`: Keep the existing harmless cache if already set.
  - `--harmful-json`/`--harmless-json`: Custom JSON files with additional instructions to be merged with the respective Hugging Face datasets.
  - If a `--cache-fname` is provided, the cached activations are saved to the specified file.

#### Generate Text

Generate text responses from the model using a given prompt.

```bash
python abliterate.py --model meta-llama/Meta-Llama-3-70B-Instruct \
  generate --prompt "What is the capital of France?" --max-tokens 64
```

#### Test Model Behavior

Test the model on a subset of harmful test instances.

```bash
python abliterate.py --model meta-llama/Meta-Llama-3-70B-Instruct \
  test --N 16 --batch-size 4
```

#### Find and Apply Refusal Directions

Find the best refusal direction (i.e., modification that changes model behavior) and optionally apply it.

**Find Best Direction:**
```bash
python abliterate.py --model meta-llama/Meta-Llama-3-70B-Instruct \
  find_best_dir --N 4 --use-hooks
```

**Automatically Find and Apply:**
```bash
python abliterate.py --model meta-llama/Meta-Llama-3-70B-Instruct \
  apply --find-best --layers 1 2 3
```

- **Explanation:**
  - `find_best_dir`: Computes scores for different refusal directions.
  - `apply --find-best`: Automatically finds the best direction and applies it. The `--layers` option allows you to specify which layers should be modified.

## Example Python Usage

Below is an example snippet demonstrating how to use the library within Python code:

```python
import Abliterator

model = "meta-llama/Meta-Llama-3-70B-Instruct"
# Load harmful and harmless instruction datasets.
dataset = [
    Abliterator.get_harmful_instructions(),
    Abliterator.get_harmless_instructions()
]
device = 'cuda'
cache_fname = 'my_cached_point.pth'

# Instantiate the ModelAbliterator.
my_model = Abliterator.ModelAbliterator(
    model,
    dataset,
    device=device,
    cache_fname=cache_fname,
    activation_layers=['resid_pre', 'resid_post', 'attn_out', 'mlp_out'],
    chat_template="<system>\n{instruction}<end><assistant>",
    positive_toks=[23371, 40914],
    negative_toks=[4250]
)

# Cache activations on 512 samples.
my_model.cache_activations(N=512, batch_size=8, preserve_harmless=True)

# Generate text using the model.
output = my_model.generate("How much wood could a woodchuck chuck if a woodchuck could chuck wood?")
print("\n".join(output))
```

## Future Directions

- **Gradio GUI:** A graphical interface is in progress to allow users to interact with the toolkit via a web application.
- **Extended Datasets:** Future versions may include support for additional datasets and more nuanced instruction handling.
- **Enhanced Modifications:** The project will continue to improve the methods for calculating and applying refusal directions, making the process more robust.

---

Feel free to open an issue or submit a pull request if you have suggestions or need further assistance!
