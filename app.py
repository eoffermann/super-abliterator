import os
import json
import time
import torch
import gradio as gr
from datetime import datetime
from huggingface_hub import HfApi
from Abliterator import ModelAbliterator, get_harmful_instructions, get_harmless_instructions

# Directories for custom configs and caches
CUSTOM_LISTS_DIR = os.path.join("Configs", "CustomLists")
MODELS_CONFIG_PATH = os.path.join("Configs", "models.json")
CACHES_DIR = "Caches"

# Ensure necessary directories exist
os.makedirs(CUSTOM_LISTS_DIR, exist_ok=True)
os.makedirs(CACHES_DIR, exist_ok=True)

# Global state to hold the current abliterator instance
global_state = {"abliterator": None, "model_name": None}

# ----- Panel A: Custom Instruction Lists -----
def load_custom_list(file_obj):
    """
    Loads a custom instruction list from an uploaded JSON file.
    """
    if file_obj is None:
        return "No file uploaded."
    try:
        content = file_obj.read().decode("utf-8")
        data = json.loads(content)
        # Pretty-print JSON for the text editor
        return json.dumps(data, indent=4)
    except Exception as e:
        return f"Error reading file: {e}"

def save_custom_list(text, filename, list_type):
    """
    Saves the custom instruction list (harmful or harmless) to the CUSTOM_LISTS_DIR.
    """
    try:
        # Ensure valid JSON
        data = json.loads(text)
    except Exception as e:
        return f"Invalid JSON: {e}"
    save_path = os.path.join(CUSTOM_LISTS_DIR, f"{filename}.json")
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    return f"Saved {list_type} list to {save_path}"

def download_custom_list(text):
    """
    Returns the text for downloading.
    """
    return text

# ----- Panel B: Activation Layers -----
def get_activation_layers_selection(selected_layers):
    """
    Returns the currently selected activation layers.
    Currently we only support base types (resid_pre, resid_post, mlp_out, attn_out).
    TODO: Extend to allow per-layer granular selection.
    """
    return selected_layers

# ----- Panel C: Generate Text -----
def generate_text(prompt, activation_layers, cache_file):
    """
    Generates text side-by-side:
      - Left: Original generation.
      - Right: Generation after applying abliteration modifications.
    
    It uses the current global abliterator instance.
    """
    if global_state["abliterator"] is None:
        return "No model loaded. Please run the abliteration process first.", ""
    
    model_instance = global_state["abliterator"]
    
    # Set activation layers as selected by the user
    model_instance.activation_layers = activation_layers
    
    # Generate original text
    original_output = model_instance.generate(prompt, max_tokens_generated=64)
    
    # Use context management to temporarily apply abliteration
    try:
        best_dirs = model_instance.find_best_refusal_dir(N=4, use_hooks=True)
        if best_dirs:
            best_direction = best_dirs[0][1][1]
            with model_instance:
                model_instance.apply_refusal_dirs([best_direction])
                modified_output = model_instance.generate(prompt, max_tokens_generated=64)
        else:
            modified_output = ["No refusal direction found."]
    except Exception as e:
        modified_output = [f"Error during abliteration: {e}"]
    
    return "\n".join(original_output), "\n".join(modified_output)

# ----- Panel D: Batch Analysis with Harmful Prompts -----
def run_batch_analysis(batch_size, num_samples):
    """
    Runs a batch analysis on harmful prompts and returns generated outputs along with analysis metrics.
    """
    if global_state["abliterator"] is None:
        return "No model loaded. Please run the abliteration process first."
    
    model_instance = global_state["abliterator"]
    
    # Load harmful instructions using the default HF dataset
    harmful_train, harmful_test = get_harmful_instructions()
    test_prompts = harmful_test[:num_samples]
    
    outputs = []
    metrics_logs = []
    for prompt in test_prompts:
        output = model_instance.generate(prompt, max_tokens_generated=64)
        outputs.append(f"Prompt: {prompt}\nOutput: {output[0]}")
        try:
            scores = model_instance.measure_scores(N=4)
            metrics_logs.append(f"Prompt: {prompt}\nNegative Score: {scores['negative'].item():.4f}, Positive Score: {scores['positive'].item():.4f}")
        except Exception as e:
            metrics_logs.append(f"Prompt: {prompt}\nError measuring scores: {e}")
    
    return "\n\n".join(outputs), "\n\n".join(metrics_logs)

# ----- Panel E: Abliteration Process & Export Panel -----
def load_models_config():
    """
    Loads the default models from the Configs/models.json file.
    """
    if os.path.exists(MODELS_CONFIG_PATH):
        with open(MODELS_CONFIG_PATH, "r") as f:
            return json.load(f)
    else:
        default_models = [
            {"id": "meta-llama/Meta-Llama-3-70B-Instruct", "size": "70B"},
            {"id": "gpt2", "size": "117M"}
        ]
        with open(MODELS_CONFIG_PATH, "w") as f:
            json.dump(default_models, f, indent=4)
        return default_models

def search_huggingface_models(query):
    """
    Searches HuggingFace for models matching the query using HfApi.
    Returns a list of candidate model strings with size info.
    """
    api = HfApi()
    try:
        results = api.list_models(search=query)
        models = []
        for model in results[:10]:
            # For demonstration, we'll use the model id as a placeholder for size info.
            models.append(f"{model.modelId} (Size info not available)")
        if not models:
            return ["No models found."]
        return models
    except Exception as e:
        return [f"Error during search: {e}"]

def run_abliteration_process(model_choice, device_choice, cache_choice, harmful_list_text, harmless_list_text, activation_layers):
    """
    Runs the abliteration process:
      - Loads the chosen model.
      - Loads additional instructions from the custom lists.
      - Instantiates a ModelAbliterator.
      - Loads a cache if selected.
      - Finds and applies the best refusal direction.
    
    Returns a log string indicating progress.
    """
    log = []
    try:
        log.append(f"Loading model: {model_choice} on device {device_choice}...")
        global_state["model_name"] = model_choice

        harmful_additional = None
        harmless_additional = None
        if harmful_list_text.strip():
            try:
                harmful_additional = json.loads(harmful_list_text)
                log.append("Custom harmful instructions loaded.")
            except Exception as e:
                log.append(f"Error parsing harmful instructions: {e}")
        if harmless_list_text.strip():
            try:
                harmless_additional = json.loads(harmless_list_text)
                log.append("Custom harmless instructions loaded.")
            except Exception as e:
                log.append(f"Error parsing harmless instructions: {e}")
        
        harmful_dataset = get_harmful_instructions(additional_instructions=harmful_additional)
        harmless_dataset = get_harmless_instructions(additional_instructions=harmless_additional)
        
        if cache_choice == "None":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_fname = os.path.join(CACHES_DIR, f"{model_choice}_{timestamp}.pth")
            log.append(f"No cache loaded. New cache file will be: {cache_fname}")
        else:
            cache_fname = os.path.join(CACHES_DIR, cache_choice)
            log.append(f"Loading cache from file: {cache_fname}")
        
        model_instance = ModelAbliterator(
            model=model_choice,
            dataset=[harmful_dataset, harmless_dataset],
            device=device_choice,
            cache_fname=cache_fname,
            activation_layers=activation_layers
        )
        global_state["abliterator"] = model_instance
        log.append("ModelAbliterator instance created.")
        
        log.append("Finding best refusal direction...")
        best_dirs = model_instance.find_best_refusal_dir(N=4, use_hooks=True)
        if best_dirs:
            best_direction = best_dirs[0][1][1]
            model_instance.apply_refusal_dirs([best_direction])
            log.append("Best refusal direction applied.")
        else:
            log.append("No best refusal direction found.")
        
        log.append("Abliteration process completed successfully.")
    except Exception as e:
        log.append(f"Error during abliteration process: {e}")
    return "\n".join(log)

def list_cache_files():
    """
    Lists available cache files in the CACHES_DIR.
    """
    files = os.listdir(CACHES_DIR)
    pth_files = [f for f in files if f.endswith(".pth")]
    if not pth_files:
        return ["None"]
    return ["None"] + pth_files

def export_model(export_filename, quant_choice):
    """
    Exports the current modified model state to a file and simulates quant generation.
    """
    log = []
    if global_state["abliterator"] is None:
        return "No model loaded. Please run the abliteration process first."
    model_instance = global_state["abliterator"]
    try:
        export_path = os.path.join("Exports", f"{export_filename}.pth")
        os.makedirs("Exports", exist_ok=True)
        torch.save(model_instance.model.state_dict(), export_path)
        log.append(f"Model exported to {export_path}.")
        
        log.append(f"Simulating quant generation for quant type: {quant_choice}...")
        # TODO: Implement actual quant generation routine in future.
        time.sleep(1)
        log.append("Quant generation simulated successfully.")
    except Exception as e:
        log.append(f"Error exporting model: {e}")
    return "\n".join(log)

# ----- Gradio Interface Setup -----
with gr.Blocks(title="SuperAbliterator Gradio Interface") as demo:
    gr.Markdown("# SuperAbliterator Gradio Interface")
    
    with gr.Tabs():
        # Panel A: Custom Instruction Lists
        with gr.Tab("Custom Instruction Lists"):
            gr.Markdown("### Edit Custom Instruction Lists")
            with gr.Row():
                harmful_editor = gr.Textbox(label="Harmful Instructions (JSON)", lines=10)
                harmless_editor = gr.Textbox(label="Harmless Instructions (JSON)", lines=10)
            with gr.Row():
                harmful_upload = gr.File(label="Upload Harmful JSON", file_types=[".json"])
                harmless_upload = gr.File(label="Upload Harmless JSON", file_types=[".json"])
            with gr.Row():
                harmful_filename = gr.Textbox(label="Filename for Harmful List (without extension)")
                harmless_filename = gr.Textbox(label="Filename for Harmless List (without extension)")
            with gr.Row():
                harmful_save = gr.Button("Save Harmful List")
                harmless_save = gr.Button("Save Harmless List")
            with gr.Row():
                harmful_download = gr.Button("Download Harmful List")
                harmless_download = gr.Button("Download Harmless List")
            custom_list_log = gr.Textbox(label="Custom Lists Log", interactive=False)
            
            # Hidden constant inputs for list type
            harmful_const = gr.Textbox(value="harmful", visible=False)
            harmless_const = gr.Textbox(value="harmless", visible=False)
            
            harmful_upload.change(fn=load_custom_list, inputs=[harmful_upload], outputs=[harmful_editor])
            harmless_upload.change(fn=load_custom_list, inputs=[harmless_upload], outputs=[harmless_editor])
            
            harmful_save.click(fn=save_custom_list, inputs=[harmful_editor, harmful_filename, harmful_const], outputs=[custom_list_log])
            harmless_save.click(fn=save_custom_list, inputs=[harmless_editor, harmless_filename, harmless_const], outputs=[custom_list_log])
            
            harmful_download.click(fn=download_custom_list, inputs=[harmful_editor], outputs=[harmful_editor])
            harmless_download.click(fn=download_custom_list, inputs=[harmless_editor], outputs=[harmless_editor])
        
        # Panel B: Activation Layers Selection
        with gr.Tab("Activation Layers"):
            gr.Markdown("### Select Activation Layers")
            activation_layers_checkbox = gr.CheckboxGroup(
                choices=["resid_pre", "resid_post", "mlp_out", "attn_out"],
                value=["resid_pre", "resid_post", "mlp_out", "attn_out"],
                label="Activation Layers (Base Types)"
            )
            gr.Markdown("*(TODO: Implement more granular per-layer selection)*")
        
        # Panel C: Generate Text (Side-by-Side Comparison)
        with gr.Tab("Generate Text"):
            gr.Markdown("### Generate Text with and without Abliteration")
            prompt_input = gr.Textbox(label="Enter Prompt", placeholder="Type your prompt here...", lines=2)
            generate_button = gr.Button("Generate")
            gen_original = gr.Textbox(label="Original Output", interactive=False)
            gen_modified = gr.Textbox(label="Abliterated Output", interactive=False)
            generate_button.click(fn=generate_text, 
                                  inputs=[prompt_input, activation_layers_checkbox, gr.State(value="")],
                                  outputs=[gen_original, gen_modified])
        
        # Panel D: Batch Analysis with Harmful Prompts
        with gr.Tab("Batch Analysis"):
            gr.Markdown("### Batch Analysis on Harmful Prompts")
            batch_size_input = gr.Slider(label="Batch Size", minimum=1, maximum=32, step=1, value=4)
            num_samples_input = gr.Slider(label="Number of Samples", minimum=1, maximum=50, step=1, value=5)
            batch_button = gr.Button("Run Batch Analysis")
            batch_outputs = gr.Textbox(label="Batch Generation Outputs", interactive=False)
            batch_metrics = gr.Textbox(label="Analysis Metrics", interactive=False)
            batch_button.click(fn=run_batch_analysis, 
                               inputs=[batch_size_input, num_samples_input],
                               outputs=[batch_outputs, batch_metrics])
        
        # Panel E: Abliteration Process & Export Panel
        with gr.Tab("Abliteration Process & Export"):
            gr.Markdown("### Abliteration Process")
            with gr.Row():
                model_dropdown = gr.Dropdown(label="Select Model from Defaults", choices=[m["id"] for m in load_models_config()], value="meta-llama/Meta-Llama-3-70B-Instruct")
                hf_search_input = gr.Textbox(label="Or Search HuggingFace", placeholder="Enter search query for models")
                hf_search_button = gr.Button("Search")
            search_results = gr.Dropdown(label="Search Results", choices=[])
            
            def do_search(query):
                results = search_huggingface_models(query)
                return results
            hf_search_button.click(fn=do_search, inputs=[hf_search_input], outputs=[search_results])
            
            with gr.Row():
                device_radio = gr.Radio(label="Select Device", choices=["cuda", "cpu"], value="cuda")
                cache_dropdown = gr.Dropdown(label="Load Cache File", choices=list_cache_files(), value="None")
            abliterate_button = gr.Button("Run Abliteration Process")
            process_log = gr.Textbox(label="Process Log", interactive=False, lines=10)
            
            process_harmful_text = gr.Textbox(label="Custom Harmful Instructions (JSON)", lines=5, placeholder="Optional custom harmful instructions")
            process_harmless_text = gr.Textbox(label="Custom Harmless Instructions (JSON)", lines=5, placeholder="Optional custom harmless instructions")
            
            abliterate_button.click(fn=run_abliteration_process, 
                                    inputs=[model_dropdown, device_radio, cache_dropdown, process_harmful_text, process_harmless_text, activation_layers_checkbox],
                                    outputs=[process_log])
            
            gr.Markdown("### Export Panel")
            export_filename_input = gr.Textbox(label="Export Filename", placeholder="Default: <modelname>_SUPERA")
            quant_dropdown = gr.Dropdown(label="Select Quant Type", choices=["Q8_0", "Q6_K", "Q5_K_L", "Q5_K_M"], value="Q8_0")
            export_button = gr.Button("Export Model")
            export_log = gr.Textbox(label="Export Log", interactive=False, lines=6)
            export_button.click(fn=export_model,
                                inputs=[export_filename_input, quant_dropdown],
                                outputs=[export_log])
    
    gr.Markdown("### Note: Please ensure you have an active internet connection for HuggingFace model search.")
    
if __name__ == "__main__":
    demo.launch(inbrowser=True)
