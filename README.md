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
- [Gradio Interface (Beta)](#gradio-interface-beta)
- [Example Python Usage](#example-python-usage)
- [Future Directions](#future-directions)
  - [Fine-Grained Behavior Control and Extended Alignment](#1-fine-grained-behavior-control-and-extended-alignment)
  - [Enhanced Model Explainability and Debugging](#2-enhanced-model-explainability-and-debugging)
  - [Increasing Alignment Capabilities by Adding More Topics for Refusal](#3-increasing-alignment-capabilities-by-adding-more-topics-for-refusal)
  - [New Research Applications](#4-new-research-applications)

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
git clone https://github.com/eoffermann/super-abliterator.git
cd super-abliterator
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

## Gradio Interface (Beta)

**Status:** The Gradio interface is currently under development and is not fully functional yet. However, the planned feature set and UI organization are outlined below.

### Planned Feature Set

- **Custom Instruction Lists Editor:**
  - Edit and create harmful/harmless instruction lists.
  - Upload and download JSON files for easy sharing and persistence.
- **Activation Layers Selection:**
  - Checkboxes for selecting base activation layer types (e.g., `resid_pre`, `resid_post`, `mlp_out`, `attn_out`).
  - *TODO:* More granular per-layer selection options will be added in future updates.
- **Generate Text with Abliteration Comparison:**
  - Input prompt for text generation.
  - Side-by-side display of original model output and output after applying abliteration modifications.
- **Batch Analysis Panel:**
  - Run batch experiments on a harmful dataset.
  - Display both generated outputs and analysis metrics (such as positive/negative scores).
- **Abliteration Process & Export Panel:**
  - Model selection via default configurations or by searching Hugging Face directly.
  - Device selection (CPU or CUDA) and cache file loading (with a dropdown populated from the `Caches` directory).
  - Run the ablation process, logging progress, and applying the best refusal direction.
  - Export the modified (SUPERA) model state and simulate quant generation for various configurations (e.g., `Q8_0`, `Q6_K`, etc.).

### UI Organization

The planned Gradio interface will be organized into tabbed panels:

1. **Custom Instruction Lists:**  
   - Two editors (one for harmful and one for harmless instructions) with upload/download functionality.
2. **Activation Layers:**  
   - A checkbox group for selecting base activation layer types.
3. **Generate Text:**  
   - Prompt input with a "Generate" button that displays outputs side-by-side for comparison.
4. **Batch Analysis:**  
   - Batch run on harmful prompts with outputs and additional analysis metrics.
5. **Abliteration Process & Export:**  
   - Model selection, device settings, cache file dropdown, and ablation process with progress logging.
   - An export panel for saving the modified model and simulating quant generation.

---

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

This section outlines several planned enhancements and experimental avenues, along with detailed roadmaps for implementation. These improvements are designed to extend the toolkit’s capabilities in alignment, interpretability, and model control.

- **Gradio GUI:** The graphical interface is currently in development. Future updates will add a fully functional Gradio-based UI as outlined above.
- **Extended Datasets:** Future versions may include support for additional datasets and more nuanced instruction handling.
- **Enhanced Modifications:** The project will continue to improve the methods for calculating and applying refusal directions, making the process more robust.

### 1. **Fine-Grained Behavior Control and Extended Alignment**
- **Extended Alignment:**  
  The toolkit already computes “refusal directions” by comparing harmful and harmless activation patterns. With relatively minor modifications—such as incorporating additional datasets or token lists—you could expand this framework to target other misbehaviors (e.g., misinformation, hate speech, or biased language). For instance, you might curate new datasets for different unwanted topics and tweak the positive/negative token lists accordingly.
- **Style or Sentiment Adjustment:**  
  You could adapt the ablation techniques to modify stylistic or sentiment features of the model’s output. By isolating activations related to style, minor adjustments (like additional hooks or custom scoring functions) could let researchers steer responses toward more neutral or user-defined styles.

**Roadmap:**
- **Dataset Expansion:**  
  - **Short-Term:** Integrate additional datasets (or allow user-specified datasets) that target new harmful or undesirable behaviors (e.g., misinformation, hate speech).  
  - **Mid-Term:** Implement dataset filtering and curation tools that enable researchers to quickly combine or split datasets based on desired characteristics.
- **Token List and Scoring Adjustments:**  
  - **Short-Term:** Extend the current positive/negative token lists to include tokens representing new categories.  
  - **Mid-Term:** Develop a configuration interface (both CLI and within a Gradio UI) for dynamically updating token lists without modifying source code.
- **Hook Customization:**  
  - **Short-Term:** Allow for the easy addition of new forward hook functions that target specific activation components (e.g., sentiment or style markers).  
  - **Long-Term:** Integrate a plugin architecture so researchers can write and share custom hook modules that automatically integrate with the ablation pipeline.

### 2. **Enhanced Model Explainability and Debugging**
- **Activation Attribution:**  
  Researchers can repurpose the activation caching and directional projection functionalities to diagnose which parts of the model contribute most to particular outputs. With extra logging or visualization tools (for example, integrating matplotlib-based plots), one could build a dashboard that shows how different layers contribute to various decisions.
- **Bias Identification:**  
  By comparing activation differences for various subgroups or topics, the project could be extended to highlight potential biases in the model. Adding additional evaluation metrics wouldn’t require a major overhaul—just new data inputs and scoring functions.

**Roadmap:**
- **Activation Attribution and Visualization:**  
  - **Short-Term:** Develop utility functions that aggregate and log activation changes layer-by-layer.  
  - **Mid-Term:** Build a visualization dashboard (using matplotlib and/or a lightweight web framework) that plots activation distributions, directional projections, and the impact of ablations.
- **Bias Identification Tools:**  
  - **Short-Term:** Incorporate additional metrics in the `measure_scores` and `measure_scores_from_logits` functions to capture bias or imbalanced activations.  
  - **Long-Term:** Design experiments that compare activation differences across demographic or topical subgroups, and integrate these results into the visualization dashboard.
- **Automated Diagnostics:**  
  - **Mid-Term:** Implement context-aware logging and error handling within the activation caching routines to assist with debugging and reproducibility.

### 3. **Increasing Alignment Capabilities by Adding More Topics for Refusal**
- **Adding More Topics for Refusal:**  
  Increasing alignment by expanding the “refusal” topics is conceptually straightforward. The current implementation relies on datasets (loaded via functions like `get_harmful_instructions` and `get_harmless_instructions`) to define harmful versus harmless instructions. To add more topics:
  - **Dataset Augmentation:** Curate or synthesize additional instruction datasets for each new category.
  - **Token/Tuning Adjustments:** Extend the positive/negative token sets to capture these new topics.
  - **Hook Customization:** Potentially write custom hooks or scoring functions to evaluate the success of these additional refusal directions.
  
  Since the codebase is modular (with clear abstractions in the `ModelAbliterator` class and separate utility functions), these changes should be manageable for someone familiar with Python and transformer internals.

**Roadmap:**
- **Multi-Topic Refusal Extension:**  
  - **Short-Term:** Create configuration options (via command-line flags or a configuration file) to define multiple new refusal topics.  
  - **Mid-Term:** Extend the current implementation of `refusal_dirs` to accept multiple target directions, each associated with its own dataset and token set.
- **Dataset Augmentation and Customization:**  
  - **Short-Term:** Allow users to merge custom JSON files with additional instructions for each new topic.  
  - **Long-Term:** Develop a semi-automated pipeline to curate and validate additional harmful and harmless instruction sets for each new refusal topic.
- **Fine-Tuning Scoring Functions:**  
  - **Short-Term:** Update scoring functions to allow separate evaluation for each refusal category.  
  - **Long-Term:** Experiment with adaptive weighting schemes that balance modifications across multiple topics to avoid overfitting to any single behavior.

### 4. New Research Applications
- **Adversarial Testing:**  
  Researchers can use the toolkit to simulate adversarial attacks on models by identifying activation vulnerabilities and then applying countermeasures.
- **Activation-Based Fine-Tuning:**  
  Instead of using full-scale fine-tuning, one could explore targeted adjustments at specific layers (via the provided MLP and attention modifications) to see if lightweight “in-situ” tweaks can steer model behavior.
- **Comparative Studies:**  
  The project could serve as a framework to compare how different transformer architectures or training regimes respond to similar ablation strategies, offering insights into the robustness or safety of various models.

**Roadmap:**
- **Adversarial Testing:**  
  - **Short-Term:** Develop test suites to simulate adversarial prompts that target specific activation vulnerabilities.  
  - **Mid-Term:** Integrate countermeasure modules that apply targeted ablation or fine-tuning in response to detected adversarial behaviors.
- **Activation-Based Fine-Tuning:**  
  - **Short-Term:** Prototype lightweight fine-tuning routines that modify only specific layers or components identified via ablation.  
  - **Long-Term:** Run comparative studies across transformer architectures to determine which layers are most amenable to such targeted adjustments.
- **Comparative Studies of Model Behavior:**  
  - **Short-Term:** Enhance logging to capture detailed performance metrics and activation statistics under varied ablation settings.  
  - **Mid-Term:** Provide built-in support for batch experiments and comparative reports, allowing researchers to systematically explore the robustness and alignment of different models.
