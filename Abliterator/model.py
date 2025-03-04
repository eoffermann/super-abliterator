"""
Module for modifying and analyzing transformer model activations.

This module defines the ModelAbliterator class, which provides a suite of tools for:
    - Selective ablation of transformer model components (e.g., MLP and attention output weights)
    - Monitoring and caching activation changes across specified layers
    - Adjusting model responses by applying modifications based on positive and negative token lists

The class leverages a pretrained HookedTransformer from transformer_lens, integrates activation caching,
and provides utilities for generating text, testing model behavior, and evaluating activation modifications.

TODO:
    - Improve exception handling in methods such as get_layer_dirs and induce_refusal_dir.
"""

import torch
import torch.nn.functional as F
import functools
import einops
import re
from tqdm import tqdm
from typing import Callable, Dict, List, Set, Tuple, Union

from transformer_lens import HookedTransformer, utils as tl_utils, ActivationCache

from jaxtyping import Float, Int

from . import utils
from . import data
from . import chat


class ModelAbliterator:
    """
    Class for analyzing and modifying transformer model activations.

    This class provides methods to:
        - Initialize a transformer model for ablation studies.
        - Save and cache activations from harmful and harmless data samples.
        - Modify specific layers of the transformer (e.g., attention output, MLP output).
        - Generate and evaluate model responses with modified activations.
        - Compute directional metrics to quantify activation modifications.
    """

    def __init__(
        self,
        model: str,
        dataset: Union[Tuple[List[str], List[str]], List[Tuple[List[str], List[str]]]],
        device: str = 'cuda',
        n_devices: int = None,
        cache_fname: str = None,
        activation_layers: List[str] = ['resid_pre', 'resid_post', 'mlp_out', 'attn_out'],
        chat_template: str = None,
        positive_toks: Union[List[int], Tuple[int], Set[int], Int[torch.Tensor, '...']] = None,
        negative_toks: Union[List[int], Tuple[int], Set[int], Int[torch.Tensor, '...']] = None
    ) -> None:
        """
        Initializes the ModelAbliterator instance.

        Args:
            model (str): Path to the pretrained model.
            dataset (Union[Tuple[List[str], List[str]], List[Tuple[List[str], List[str]]]]):
                A tuple or list containing training and testing datasets.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.
            n_devices (int, optional): Number of devices to use. If None, determined automatically. Defaults to None.
            cache_fname (str, optional): Filename for cached activation data. Defaults to None.
            activation_layers (List[str], optional): List of activation layers to monitor. Defaults to common layers.
            chat_template (str, optional): Template for formatting text input. Defaults to a preset template.
            positive_toks (Union[List[int], Tuple[int], Set[int], Int[torch.Tensor, '...']], optional):
                Tokens representing 'positive' responses. Defaults to None.
            negative_toks (Union[List[int], Tuple[int], Set[int], Int[torch.Tensor, '...']], optional):
                Tokens representing 'negative' responses. Defaults to None.
        """
        self.MODEL_PATH = model

        # Determine the number of devices based on availability.
        if n_devices is None and torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
        elif n_devices is None:
            n_devices = 1

        torch.set_grad_enabled(False)
        # Load the model without additional preprocessing.
        self.model = HookedTransformer.from_pretrained_no_processing(
            model,
            n_devices=n_devices,
            device=device,
            dtype=torch.bfloat16,
            default_padding_side='left'
        )
        self.model.requires_grad_(False)
        self.model.tokenizer.padding_side = 'left'
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        # Set chat template for formatting prompts.
        self.chat_template = chat_template or chat.ChatTemplate(self, chat.LLAMA3_CHAT_TEMPLATE)

        self.hidden_size = self.model.cfg.d_model
        # Cache the original state of the model.
        self.original_state = {k: v.to('cpu') for k, v in self.model.state_dict().items()}
        self.harmful = {}
        self.harmless = {}
        self.modified_layers = {'mlp': {}, 'W_O': {}}
        self.checkpoints = []

        # Load cached activations and modifications if a cache filename is provided.
        if cache_fname is not None:
            outs = torch.load(cache_fname, map_location='cpu')
            self.harmful, self.harmless, modified_layers, checkpoints = outs[:4]
            self.checkpoints = checkpoints or []
            self.modified_layers = modified_layers

        # Prepare datasets for harmful and harmless instances.
        self.harmful_inst_train, self.harmful_inst_test = data.prepare_dataset(dataset[0])
        self.harmless_inst_train, self.harmless_inst_test = data.prepare_dataset(dataset[1])

        self.fwd_hooks = []  # Forward hooks for modifying activations during generation.
        self.modified = False
        # Ensure activation_layers is stored as a list.
        self.activation_layers = activation_layers if isinstance(activation_layers, list) else [activation_layers]

        # Set negative tokens, with a default warning if not provided.
        if negative_toks is None:
            print("WARNING: You've not set 'negative_toks', defaulting to tokens for Llama-3 vocab")
            self.negative_toks = {4250, 14931, 89735, 20451, 11660, 11458, 956}
        else:
            self.negative_toks = negative_toks

        # Set positive tokens, with a default warning if not provided.
        if positive_toks is None:
            print("WARNING: You've not set 'positive_toks', defaulting to tokens for Llama-3 vocab")
            self.positive_toks = {32, 1271, 8586, 96556, 78145}
        else:
            self.positive_toks = positive_toks

        self._blacklisted = set()  # Layers that are prevented from being modified.

    def __enter__(self) -> "ModelAbliterator":
        """
        Enables context management for safe modifications.

        Returns:
            ModelAbliterator: The current instance with context management enabled.

        Raises:
            Exception: If a context is already active.
        """
        if hasattr(self, "current_state"):
            # TODO: Refactor this exception into a custom exception type.
            raise Exception("Cannot do multi-contexting")
        # Save the current state and modifications.
        self.current_state = self.model.state_dict()
        self.current_layers = self.modified_layers.copy()
        self.was_modified = self.modified
        return self

    def __exit__(self, exc, exc_value, exc_tb) -> None:
        """
        Restores the model state upon exiting the context.
        """
        self.model.load_state_dict(self.current_state)
        del self.current_state
        self.modified_layers = self.current_layers
        del self.current_layers
        self.modified = self.was_modified
        del self.was_modified

    def reset_state(self) -> None:
        """
        Resets the model state to its original parameters.
        """
        self.modified = False
        self.modified_layers = {'mlp': {}, 'W_O': {}}
        self.model.load_state_dict(self.original_state)

    def checkpoint(self) -> None:
        """
        Saves the current model modification state as a checkpoint.
        """
        self.checkpoints.append(self.modified_layers.copy())

    # Utility functions

    def blacklist_layer(self, layer: Union[int, List[int]]) -> None:
        """
        Prevents a layer or a list of layers from being modified.

        Args:
            layer (Union[int, List[int]]): The layer index or list of indices to blacklist.
        """
        if isinstance(layer, list):
            for l in layer:
                self._blacklisted.add(l)
        else:
            self._blacklisted.add(layer)

    def whitelist_layer(self, layer: Union[int, List[int]]) -> None:
        """
        Removes a layer or a list of layers from the blacklist.

        Args:
            layer (Union[int, List[int]]): The layer index or list of indices to whitelist.
        """
        if isinstance(layer, list):
            for l in layer:
                self._blacklisted.discard(l)
        else:
            self._blacklisted.discard(layer)

    def save_activations(self, fname: str) -> None:
        """
        Saves the current activation modifications and checkpoints to a file.

        Args:
            fname (str): Filename to save the activations.
        """
        torch.save([
            self.harmful,
            self.harmless,
            self.modified_layers if self.modified_layers['mlp'] or self.modified_layers['W_O'] else None,
            self.checkpoints if len(self.checkpoints) > 0 else None
        ], fname)

    def get_whitelisted_layers(self) -> List[int]:
        """
        Retrieves a list of layer indices that are not blacklisted.

        Returns:
            List[int]: Whitelisted layer indices.
        """
        return [l for l in range(self.model.cfg.n_layers) if l not in self._blacklisted]

    def get_all_act_names(self, activation_layers: List[str] = None) -> List[Tuple[int, str]]:
        """
        Retrieves activation names from model layers based on activation layers provided.

        Args:
            activation_layers (List[str], optional): Specific activation layers to consider.
                Defaults to the instance's activation_layers.

        Returns:
            List[Tuple[int, str]]: List of tuples containing layer indices and activation names.
        """
        act_layers = activation_layers or self.activation_layers
        return [
            (i, tl_utils.get_act_name(act_name, i))
            for i in self.get_whitelisted_layers()
            for act_name in act_layers
        ]

    def calculate_mean_dirs(
        self,
        key: str,
        include_overall_mean: bool = False
    ) -> Dict[str, Float[torch.Tensor, 'd_model']]:
        """
        Computes mean activation directions for harmful and harmless instances.

        Args:
            key (str): The activation cache key for which to compute mean directions.
            include_overall_mean (bool, optional): Whether to compute an overall mean direction.
                Defaults to False.

        Returns:
            Dict[str, Float[torch.Tensor, 'd_model']]: Dictionary containing 'harmful_mean',
                'harmless_mean', and optionally 'mean_dir'.
        """
        dirs = {
            'harmful_mean': torch.mean(self.harmful[key], dim=0),
            'harmless_mean': torch.mean(self.harmless[key], dim=0)
        }
        if include_overall_mean:
            # TODO: Ensure that tensor shapes and device compatibility are robust.
            if self.harmful[key].shape != self.harmless[key].shape or self.harmful[key].device.type == 'cuda':
                dirs['mean_dir'] = torch.mean(torch.cat((self.harmful[key], self.harmless[key]), dim=0), dim=0)
            else:
                dirs['mean_dir'] = torch.mean(self.harmful[key] + self.harmless[key], dim=0) / 2.0
        return dirs

    def get_avg_projections(
        self,
        key: str,
        direction: Float[torch.Tensor, 'd_model']
    ) -> Tuple[Float[torch.Tensor, 'd_model'], Float[torch.Tensor, 'd_model']]:
        """
        Computes the dot product projections of harmful and harmless mean directions onto a given direction.

        Args:
            key (str): Activation cache key.
            direction (Float[torch.Tensor, 'd_model']): The direction vector.

        Returns:
            Tuple[Float[torch.Tensor, 'd_model'], Float[torch.Tensor, 'd_model']]:
                Tuple of dot products for harmful and harmless means.
        """
        dirs = self.calculate_mean_dirs(key)
        return (
            torch.dot(dirs['harmful_mean'], direction),
            torch.dot(dirs['harmless_mean'], direction)
        )

    def get_layer_dirs(
        self,
        layer: int,
        key: str = None,
        include_overall_mean: bool = False
    ) -> Dict[str, Float[torch.Tensor, 'd_model']]:
        """
        Retrieves mean activation directions for a specific layer.

        Args:
            layer (int): The layer index.
            key (str, optional): Activation key to use; if None, defaults to the first activation layer.
            include_overall_mean (bool, optional): Whether to include overall mean direction.
                Defaults to False.

        Returns:
            Dict[str, Float[torch.Tensor, 'd_model']]: Dictionary of mean directions.

        Raises:
            IndexError: If the specified layer is invalid.
        """
        act_key = key or self.activation_layers[0]
        # TODO: Verify if self.harmfuls should be self.harmful; potential bug.
        if len(self.harmful[act_key]) < layer:
            raise IndexError("Invalid layer")
        return self.calculate_mean_dirs(tl_utils.get_act_name(act_key, layer), include_overall_mean=include_overall_mean)

    def refusal_dirs(self, invert: bool = False) -> Dict[str, Float[torch.Tensor, 'd_model']]:
        """
        Computes the refusal direction for each activation key.

        Args:
            invert (bool, optional): If True, compute harmless_mean - harmful_mean.
                Otherwise, compute harmful_mean - harmless_mean. Defaults to False.

        Returns:
            Dict[str, Float[torch.Tensor, 'd_model']]: Dictionary of normalized refusal directions.

        Raises:
            IndexError: If no activation cache is available.
        """
        if not self.harmful:
            raise IndexError("No cache")
        refusal_dirs = {
            key: self.calculate_mean_dirs(key)
            for key in self.harmful if '.0.' not in key
        }
        if invert:
            refusal_dirs = {key: v['harmless_mean'] - v['harmful_mean'] for key, v in refusal_dirs.items()}
        else:
            refusal_dirs = {key: v['harmful_mean'] - v['harmless_mean'] for key, v in refusal_dirs.items()}
        # Normalize each refusal direction and move to CPU.
        return {key: (v / v.norm()).to('cpu') for key, v in refusal_dirs.items()}

    def scored_dirs(self, invert: bool = False) -> List[Tuple[str, Float[torch.Tensor, 'd_model']]]:
        """
        Returns a sorted list of activation directions with scores based on the mean of absolute values.

        Args:
            invert (bool, optional): Whether to invert the calculation in refusal_dirs.
                Defaults to False.

        Returns:
            List[Tuple[str, Float[torch.Tensor, 'd_model']]]: Sorted list of tuples (activation name, direction).
        """
        refusals = self.refusal_dirs(invert=invert)
        return sorted(
            [(ln, refusals[act_name]) for ln, act_name in self.get_all_act_names()],
            reverse=True, key=lambda x: abs(x[1].mean())
        )

    def get_layer_of_act_name(self, ref: str) -> Union[str, int]:
        """
        Extracts the layer index from an activation name string.

        Args:
            ref (str): The activation name string.

        Returns:
            Union[str, int]: The layer index if found; otherwise, the original string.
        """
        s = re.search(r"\.(\d+)\.", ref)
        return int(s[1]) if s else ref

    def layer_attn(
        self,
        layer: int,
        replacement: Float[torch.Tensor, "d_model"] = None
    ) -> Float[torch.Tensor, "d_model"]:
        """
        Gets or sets the attention output weight matrix for a specific layer.

        Args:
            layer (int): The layer index.
            replacement (Float[torch.Tensor, "d_model"], optional): Replacement matrix. If provided and the layer
                is not blacklisted, the attention output weight is replaced. Defaults to None.

        Returns:
            Float[torch.Tensor, "d_model"]: The current attention output weight matrix.
        """
        if replacement is not None and layer not in self._blacklisted:
            self.modified = True
            # Move replacement to the appropriate device.
            self.model.blocks[layer].attn.W_O.data = replacement.to(self.model.blocks[layer].attn.W_O.device)
            self.modified_layers['W_O'][layer] = self.modified_layers.get(layer, []) + [
                (self.model.blocks[layer].attn.W_O.data.to('cpu'), replacement.to('cpu'))
            ]
        return self.model.blocks[layer].attn.W_O.data

    def layer_mlp(
        self,
        layer: int,
        replacement: Float[torch.Tensor, "d_model"] = None
    ) -> Float[torch.Tensor, "d_model"]:
        """
        Gets or sets the MLP output weight matrix for a specific layer.

        Args:
            layer (int): The layer index.
            replacement (Float[torch.Tensor, "d_model"], optional): Replacement matrix. If provided and the layer
                is not blacklisted, the MLP output weight is replaced. Defaults to None.

        Returns:
            Float[torch.Tensor, "d_model"]: The current MLP output weight matrix.
        """
        if replacement is not None and layer not in self._blacklisted:
            self.modified = True
            self.model.blocks[layer].mlp.W_out.data = replacement.to(self.model.blocks[layer].mlp.W_out.device)
            self.modified_layers['mlp'][layer] = self.modified_layers.get(layer, []) + [
                (self.model.blocks[layer].mlp.W_out.data.to('cpu'), replacement.to('cpu'))
            ]
        return self.model.blocks[layer].mlp.W_out.data

    def tokenize_instructions_fn(self, instructions: List[str]) -> Int[torch.Tensor, 'batch_size seq_len']:
        """
        Tokenizes a list of instruction strings using the chat template.

        Args:
            instructions (List[str]): List of instruction strings.

        Returns:
            Int[torch.Tensor, 'batch_size seq_len']: Tokenized input tensor.
        """
        # Format each instruction using the chat template.
        prompts = [self.chat_template.format(instruction=instruction) for instruction in instructions]
        return self.model.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").input_ids

    def generate_logits(
        self,
        toks: Int[torch.Tensor, 'batch_size seq_len'],
        *args,
        drop_refusals: bool = True,
        stop_at_eos: bool = False,
        max_tokens_generated: int = 1,
        **kwargs
    ) -> Tuple[Float[torch.Tensor, 'batch_size seq_len d_vocab'], Int[torch.Tensor, 'batch_size seq_len']]:
        """
        Generates logits for given tokenized input and returns both logits and generated tokens.

        Args:
            toks (Int[torch.Tensor, 'batch_size seq_len']): Input token tensor.
            *args: Additional positional arguments to pass to the model.
            drop_refusals (bool, optional): If True, generation stops if a negative token is encountered.
                Defaults to True.
            stop_at_eos (bool, optional): If True, stops generation at the EOS token. Defaults to False.
            max_tokens_generated (int, optional): Maximum number of tokens to generate. Defaults to 1.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            Tuple[Float[torch.Tensor, 'batch_size seq_len d_vocab'], Int[torch.Tensor, 'batch_size seq_len']]:
                A tuple of logits and the corresponding token tensor.
        """
        # Create a tensor to store original tokens and generated tokens.
        all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated),
                                dtype=torch.long, device=toks.device)
        all_toks[:, :toks.shape[1]] = toks
        generating = list(range(toks.shape[0]))
        for i in range(max_tokens_generated):
            # Generate logits for the current tokens.
            logits = self.model(all_toks[generating, :-max_tokens_generated + i], *args, **kwargs)
            # Get the next token by taking the argmax over the vocabulary.
            next_tokens = logits[:, -1, :].argmax(dim=-1).to('cpu')
            all_toks[generating, -max_tokens_generated + i] = next_tokens
            # Check for negative tokens to possibly stop generation.
            if drop_refusals and any(negative_tok in next_tokens for negative_tok in self.negative_toks):
                break
            # Optionally stop at the EOS token.
            if stop_at_eos:
                generating = [i for i in range(toks.shape[0])
                              if all_toks[i][-1] != self.model.tokenizer.eos_token_id]
                if not generating:
                    break
        return logits, all_toks

    def generate(
        self,
        prompt: Union[List[str], str],
        *model_args,
        max_tokens_generated: int = 64,
        stop_at_eos: bool = True,
        **model_kwargs
    ) -> List[str]:
        """
        Generates text responses from the model given a prompt.

        Args:
            prompt (Union[List[str], str]): A prompt or list of prompts.
            *model_args: Additional positional arguments for model generation.
            max_tokens_generated (int, optional): Maximum tokens to generate. Defaults to 64.
            stop_at_eos (bool, optional): Whether to stop generation at the EOS token. Defaults to True.
            **model_kwargs: Additional keyword arguments for model generation.

        Returns:
            List[str]: List of generated text responses.
        """
        # Tokenize the prompt(s) using the provided chat template.
        if isinstance(prompt, str):
            gen = self.tokenize_instructions_fn([prompt])
        else:
            gen = self.tokenize_instructions_fn(prompt)
        logits, all_toks = self.generate_logits(
            gen,
            *model_args,
            stop_at_eos=stop_at_eos,
            max_tokens_generated=max_tokens_generated,
            **model_kwargs
        )
        return self.model.tokenizer.batch_decode(all_toks, skip_special_tokens=True)

    def test(self, *args, test_set: List[str] = None, N: int = 16, batch_size: int = 4, **kwargs) -> None:
        """
        Tests the model by generating outputs for a sample of test prompts.

        Args:
            *args: Additional positional arguments for generation.
            test_set (List[str], optional): A list of test prompts. Defaults to harmful test instances.
            N (int, optional): Maximum number of prompts to test. Defaults to 16.
            batch_size (int, optional): Batch size for processing prompts. Defaults to 4.
            **kwargs: Additional keyword arguments for generation.
        """
        test_set = test_set or self.harmful_inst_test
        # Process prompts in batches.
        for prompts in utils.batch(test_set[:min(len(test_set), N)], batch_size):
            for res in self.generate(prompts, *args, **kwargs):
                print(res)

    def run_with_cache(
        self,
        *model_args,
        names_filter: Callable[[str], bool] = None,
        incl_bwd: bool = False,
        device: str = None,
        remove_batch_dim: bool = False,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        fwd_hooks: List[str] = [],
        max_new_tokens: int = 1,
        **model_kwargs
    ) -> Tuple[Float[torch.Tensor, 'batch_size seq_len d_vocab'], Dict[str, Float[torch.Tensor, 'batch_size seq_len d_model']]]:
        """
        Runs the model with caching hooks to capture activations.

        Args:
            *model_args: Additional positional arguments for model generation.
            names_filter (Callable[[str], bool], optional): Function to filter activation layer names.
                Defaults to checking against instance's activation_layers.
            incl_bwd (bool, optional): Whether to include backward hooks. Defaults to False.
            device (str, optional): Device for caching. Defaults to None.
            remove_batch_dim (bool, optional): Whether to remove batch dimension in cache. Defaults to False.
            reset_hooks_end (bool, optional): Whether to reset hooks at the end. Defaults to True.
            clear_contexts (bool, optional): Whether to clear contexts after generation. Defaults to False.
            fwd_hooks (List[str], optional): Additional forward hooks to include. Defaults to empty list.
            max_new_tokens (int, optional): Maximum new tokens to generate. Defaults to 1.
            **model_kwargs: Additional keyword arguments for model generation.

        Returns:
            Tuple[Float[torch.Tensor, 'batch_size seq_len d_vocab'],
                  Dict[str, Float[torch.Tensor, 'batch_size seq_len d_model']]]:
                A tuple of model output logits and a dictionary of cached activations.
        """
        if names_filter is None and self.activation_layers:
            def activation_layering(namefunc: str) -> bool:
                return any(s in namefunc for s in self.activation_layers)
            names_filter = activation_layering

        cache_dict, fwd, bwd = self.model.get_caching_hooks(
            names_filter,
            incl_bwd,
            device,
            remove_batch_dim=remove_batch_dim,
            pos_slice=tl_utils.Slice(None)
        )
        # Combine provided forward hooks with caching hooks.
        fwd_hooks = fwd_hooks + fwd + self.fwd_hooks
        max_new_tokens = max_new_tokens or 1

        with self.model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd,
                              reset_hooks_end=reset_hooks_end, clear_contexts=clear_contexts):
            model_out, toks = self.generate_logits(
                *model_args,
                max_tokens_generated=max_new_tokens,
                **model_kwargs
            )
            if incl_bwd:
                model_out.backward()

        return model_out, cache_dict

    def apply_refusal_dirs(
        self,
        refusal_dirs: List[Float[torch.Tensor, 'd_model']],
        W_O: bool = True,
        mlp: bool = True,
        layers: List[str] = None
    ) -> None:
        """
        Applies the given refusal directions to specified layers, modifying the attention and/or MLP weights.

        Args:
            refusal_dirs (List[Float[torch.Tensor, 'd_model']]): List of refusal direction vectors.
            W_O (bool, optional): Whether to apply modifications to the attention output weights. Defaults to True.
            mlp (bool, optional): Whether to apply modifications to the MLP weights. Defaults to True.
            layers (List[str], optional): List of layer indices to modify. Defaults to all layers except layer 0.
        """
        if layers is None:
            layers = list(l for l in range(1, self.model.cfg.n_layers))
        for refusal_dir in refusal_dirs:
            for layer in layers:
                for modifying in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                    if modifying[0]:
                        matrix = modifying[1](layer)
                        # Ensure device compatibility.
                        if refusal_dir.device != matrix.device:
                            refusal_dir = refusal_dir.to(matrix.device)
                        proj = einops.einsum(
                            matrix, refusal_dir.view(-1, 1),
                            '... d_model, d_model single -> ... single'
                        ) * refusal_dir
                        modifying[1](layer, matrix - proj)

    def induce_refusal_dir(
        self,
        refusal_dir: Float[torch.Tensor, 'd_model'],
        W_O: bool = True,
        mlp: bool = True,
        layers: List[str] = None
    ) -> None:
        """
        Modifies the model weights by inducing a given refusal direction.

        Args:
            refusal_dir (Float[torch.Tensor, 'd_model']): The refusal direction vector to induce.
            W_O (bool, optional): Whether to modify the attention output weights. Defaults to True.
            mlp (bool, optional): Whether to modify the MLP weights. Defaults to True.
            layers (List[str], optional): List of layer indices to modify. Defaults to all layers except layer 0.

        TODO:
            - This method is incomplete and requires further work to properly balance the induced direction.
        """
        if layers is None:
            layers = list(l for l in range(1, self.model.cfg.n_layers))
        for layer in layers:
            for modifying in [(W_O, self.layer_attn), (mlp, self.layer_mlp)]:
                if modifying[0]:
                    matrix = modifying[1](layer)
                    if refusal_dir.device != matrix.device:
                        refusal_dir = refusal_dir.to(matrix.device)
                    proj = einops.einsum(
                        matrix, refusal_dir.view(-1, 1),
                        '... d_model, d_model single -> ... single'
                    ) * refusal_dir
                    avg_proj = refusal_dir * self.get_avg_projections(
                        tl_utils.get_act_name(self.activation_layers[0], layer),
                        refusal_dir
                    )
                    modifying[1](layer, (matrix - proj) + avg_proj)

    def test_dir(
        self,
        refusal_dir: Float[torch.Tensor, 'd_model'],
        activation_layers: List[str] = None,
        use_hooks: bool = True,
        layers: List[str] = None,
        **kwargs
    ) -> Dict[str, Float[torch.Tensor, 'd_model']]:
        """
        Tests the effect of a refusal direction on the model by measuring scores.

        Args:
            refusal_dir (Float[torch.Tensor, 'd_model']): The refusal direction vector to test.
            activation_layers (List[str], optional): Activation layers to consider. Defaults to instance's activation_layers.
            use_hooks (bool, optional): If True, uses forward hooks for evaluation (more memory efficient for larger models).
                If False, applies modifications directly. Defaults to True.
            layers (List[str], optional): List of layers to test. Defaults to whitelisted layers.
            **kwargs: Additional arguments to pass to measure_scores.

        Returns:
            Dict[str, Float[torch.Tensor, 'd_model']]: Dictionary with 'negative' and 'positive' score measurements.
        """
        before_hooks = self.fwd_hooks
        try:
            if layers is None:
                layers = self.get_whitelisted_layers()

            if activation_layers is None:
                activation_layers = self.activation_layers

            if use_hooks:
                hooks = self.fwd_hooks
                hook_fn = functools.partial(utils.directional_hook, direction=refusal_dir)
                # Add hooks for each activation layer.
                self.fwd_hooks = before_hooks + [(act_name, hook_fn) for ln, act_name in self.get_all_act_names()]
                return self.measure_scores(**kwargs)
            else:
                with self:
                    self.apply_refusal_dirs([refusal_dir], layers=layers)
                    return self.measure_scores(**kwargs)
        finally:
            # Restore original forward hooks.
            self.fwd_hooks = before_hooks

    def find_best_refusal_dir(
        self,
        N: int = 4,
        positive: bool = False,
        use_hooks: bool = True,
        invert: bool = False
    ) -> List[Tuple[float, str]]:
        """
        Finds the best refusal direction based on measured scores.

        Args:
            N (int, optional): Number of tokens to sample for evaluation. Defaults to 4.
            positive (bool, optional): Whether to use positive scoring. Defaults to False.
            use_hooks (bool, optional): Whether to use forward hooks during testing. Defaults to True.
            invert (bool, optional): Whether to invert the computed refusal direction. Defaults to False.

        Returns:
            List[Tuple[float, str]]: Sorted list of tuples containing score and corresponding activation direction.
        """
        dirs = self.refusal_dirs(invert=invert)
        if self.modified:
            print("WARNING: Modified; will restore model to current modified state each run")
        scores = []
        for direction in tqdm(dirs.items()):
            score = self.test_dir(direction[1], N=N, use_hooks=use_hooks)[int(positive)]
            scores.append((score, direction))
        return sorted(scores, key=lambda x: x[0])

    def measure_scores(
        self,
        N: int = 4,
        sampled_token_ct: int = 8,
        measure: str = 'max',
        batch_measure: str = 'max',
        positive: bool = False
    ) -> Dict[str, Float[torch.Tensor, 'd_model']]:
        """
        Measures model output scores based on cached activations for harmful test instances.

        Args:
            N (int, optional): Number of harmful test instances to use. Defaults to 4.
            sampled_token_ct (int, optional): Number of tokens to sample per instance. Defaults to 8.
            measure (str, optional): Measurement function to apply (e.g., 'max', 'mean'). Defaults to 'max'.
            batch_measure (str, optional): Measurement function for batch-level scoring. Defaults to 'max'.
            positive (bool, optional): Whether to return positive scores. Defaults to False.

        Returns:
            Dict[str, Float[torch.Tensor, 'd_model']]: Dictionary with keys 'negative' and 'positive' scores.
        """
        toks = self.tokenize_instructions_fn(instructions=self.harmful_inst_test[:N])
        logits, cache = self.run_with_cache(toks, max_new_tokens=sampled_token_ct, drop_refusals=False)

        negative_score, positive_score = self.measure_scores_from_logits(logits, sampled_token_ct, measure=batch_measure)

        negative_score = utils.measure_fn(measure, negative_score)
        positive_score = utils.measure_fn(measure, positive_score)
        return {'negative': negative_score.to('cpu'), 'positive': positive_score.to('cpu')}

    def measure_scores_from_logits(
        self,
        logits: Float[torch.Tensor, 'batch_size seq_len d_vocab'],
        sequence: int,
        measure: str = 'max'
    ) -> Tuple[Float[torch.Tensor, 'batch_size'], Float[torch.Tensor, 'batch_size']]:
        """
        Computes scores from logits for both positive and negative tokens.

        Args:
            logits (Float[torch.Tensor, 'batch_size seq_len d_vocab']): Logits from the model.
            sequence (int): The number of tokens at the end of the sequence to consider.
            measure (str, optional): Measurement function to apply. Defaults to 'max'.

        Returns:
            Tuple[Float[torch.Tensor, 'batch_size'], Float[torch.Tensor, 'batch_size']]:
                Tuple containing negative and positive scores for each batch.
        """
        # Normalize scores and restrict to selected tokens.
        normalized_scores = torch.softmax(logits[:, -sequence:, :].to('cpu'), dim=-1)[:, :, list(self.positive_toks) + list(self.negative_toks)]
        # Split scores for positive and negative tokens.
        normalized_positive, normalized_negative = torch.split(normalized_scores, [len(self.positive_toks), len(self.negative_toks)], dim=2)
        # Get maximum score per token for both negative and positive tokens.
        max_negative_score_per_sequence = torch.max(normalized_negative, dim=-1)[0]
        max_positive_score_per_sequence = torch.max(normalized_positive, dim=-1)[0]
        negative_score_per_batch = utils.measure_fn(measure, max_negative_score_per_sequence, dim=-1)[0]
        positive_score_per_batch = utils.measure_fn(measure, max_positive_score_per_sequence, dim=-1)[0]
        return negative_score_per_batch, positive_score_per_batch

    def do_resid(
        self,
        fn_name: str
    ) -> Tuple[Float[torch.Tensor, 'layer batch d_model'], Float[torch.Tensor, 'layer batch d_model'], List[str]]:
        """
        Decomposes the residual stream using a specified function on the cached activations.

        Args:
            fn_name (str): The function name to call on the activation caches (e.g., "decompose_resid" or "accumulated_resid").

        Returns:
            Tuple[Float[torch.Tensor, 'layer batch d_model'],
                  Float[torch.Tensor, 'layer batch d_model'],
                  List[str]]: Decomposed harmful residuals, harmless residuals, and associated labels.

        Raises:
            AssertionError: If residual streams are not available in the cache.
        """
        if not any("resid" in k for k in self.harmless.keys()):
            raise AssertionError("You need residual streams to decompose layers! Run cache_activations with None in `activation_layers`")
        resid_harmful, labels = getattr(self.harmful, fn_name)(apply_ln=True, return_labels=True)
        resid_harmless = getattr(self.harmless, fn_name)(apply_ln=True)
        return resid_harmful, resid_harmless, labels

    def decomposed_resid(self) -> Tuple[Float[torch.Tensor, 'layer batch d_model'],
                                          Float[torch.Tensor, 'layer batch d_model'],
                                          List[str]]:
        """
        Decomposes the residual stream using the decompose_resid function.

        Returns:
            Tuple[Float[torch.Tensor, 'layer batch d_model'],
                  Float[torch.Tensor, 'layer batch d_model'],
                  List[str]]: Decomposed harmful and harmless residuals and labels.
        """
        return self.do_resid("decompose_resid")

    def accumulated_resid(self) -> Tuple[Float[torch.Tensor, 'layer batch d_model'],
                                           Float[torch.Tensor, 'layer batch d_model'],
                                           List[str]]:
        """
        Accumulates the residuals using the accumulated_resid function.

        Returns:
            Tuple[Float[torch.Tensor, 'layer batch d_model'],
                  Float[torch.Tensor, 'layer batch d_model'],
                  List[str]]: Accumulated harmful and harmless residuals and labels.
        """
        return self.do_resid("accumulated_resid")

    def unembed_resid(
        self,
        resid: Float[torch.Tensor, "layer batch d_model"],
        pos: int = -1
    ) -> Float[torch.Tensor, "layer batch d_vocab"]:
        """
        Projects residuals back to vocabulary space using the unembedding matrix.

        Args:
            resid (Float[torch.Tensor, "layer batch d_model"]): Residual tensor.
            pos (int, optional): Position in the sequence to unembed; if None, unembeds all positions. Defaults to -1.

        Returns:
            Float[torch.Tensor, "layer batch d_vocab"]: The unembedded tensor.
        """
        W_U = self.model.W_U
        if pos is None:
            return einops.einsum(resid.to(W_U.device), W_U, "layer batch d_model, d_model d_vocab -> layer batch d_vocab").to('cpu')
        else:
            return einops.einsum(resid[:, pos, :].to(W_U.device), W_U, "layer d_model, d_model d_vocab -> layer d_vocab").to('cpu')

    def create_layer_rankings(
        self,
        token_set: Union[List[int], Set[int], Int[torch.Tensor, '...']],
        decompose: bool = True,
        token_set_b: Union[List[int], Set[int], Int[torch.Tensor, '...']] = None
    ) -> List[Tuple[int, int]]:
        """
        Creates rankings of layers based on unembedded residuals for specified tokens.

        Args:
            token_set (Union[List[int], Set[int], Int[torch.Tensor, '...']]): Set of tokens for ranking.
            decompose (bool, optional): Whether to use decomposed residuals or accumulated residuals.
                Defaults to True.
            token_set_b (Union[List[int], Set[int], Int[torch.Tensor, '...']], optional):
                Alternate set of tokens for harmless ranking. Defaults to None.

        Returns:
            List[Tuple[int, int]]: Rankings as tuples of indices.
        """
        decomposer = self.decomposed_resid if decompose else self.accumulated_resid
        decomposed_resid_harmful, decomposed_resid_harmless, labels = decomposer()
        W_U = self.model.W_U.to('cpu')
        unembedded_harmful = self.unembed_resid(decomposed_resid_harmful)
        unembedded_harmless = self.unembed_resid(decomposed_resid_harmless)
        sorted_harmful_indices = torch.argsort(unembedded_harmful, dim=1, descending=True)
        sorted_harmless_indices = torch.argsort(unembedded_harmless, dim=1, descending=True)
        harmful_set = torch.isin(sorted_harmful_indices, torch.tensor(list(token_set)))
        harmless_set = torch.isin(sorted_harmless_indices, torch.tensor(list(token_set if token_set_b is None else token_set_b)))
        indices_in_set = zip(harmful_set.nonzero(as_tuple=True)[1], harmless_set.nonzero(as_tuple=True)[1])
        return list(indices_in_set)

    def mse_positive(
        self,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1
    ) -> Dict[str, Float[torch.Tensor, 'd_model']]:
        """
        Computes the mean squared error (MSE) of cached activations against harmless instances.

        Args:
            N (int, optional): Maximum number of instances to use. Defaults to 128.
            batch_size (int, optional): Batch size for processing. Defaults to 8.
            last_indices (int, optional): Number of last indices in the sequence to average. Defaults to 1.

        Returns:
            Dict[str, Float[torch.Tensor, 'd_model']]: Dictionary mapping activation keys to their MSE loss.
        """
        # Tokenize combined harmful and harmless training instructions.
        toks = self.tokenize_instructions_fn(instructions=self.harmful_inst_train[:N] + self.harmless_inst_train[:N])
        splitpos = min(N, len(self.harmful_inst_train))
        # Select tokens corresponding to harmless instructions.
        toks = toks[splitpos:]
        self.loss_harmless = {}

        for i in tqdm(range(0, min(N, len(toks)), batch_size)):
            logits, cache = self.run_with_cache(toks[i:min(i + batch_size, len(toks))])
            for key in cache:
                if any(k in key for k in self.activation_layers):
                    tensor = torch.mean(cache[key][:, -last_indices:, :], dim=1).to('cpu')
                    if key not in self.loss_harmless:
                        self.loss_harmless[key] = tensor
                    else:
                        self.loss_harmless[key] = torch.cat((self.loss_harmless[key], tensor), dim=0)
            del logits, cache
            utils.clear_mem()

        return {k: F.mse_loss(self.loss_harmless[k].float()[:N], self.harmless[k].float()[:N])
                for k in self.loss_harmless}

    def create_activation_cache(
        self,
        toks,
        N: int = 128,
        batch_size: int = 8,
        last_indices: int = 1,
        measure_refusal: int = 0,
        stop_at_layer: int = None
    ) -> Tuple[ActivationCache, List[str]]:
        """
        Creates an activation cache for given tokens.

        Args:
            toks: Token tensor for which to cache activations.
            N (int, optional): Maximum number of tokens to process. Defaults to 128.
            batch_size (int, optional): Batch size for processing tokens. Defaults to 8.
            last_indices (int, optional): Number of last indices in each sequence to average. Defaults to 1.
            measure_refusal (int, optional): Number of tokens for measuring refusal. Defaults to 0.
            stop_at_layer (int, optional): Layer index at which to stop caching. Defaults to None.

        Returns:
            Tuple[ActivationCache, List[str]]: The activation cache and an optional list of labels for z scores.
        """
        base = dict()
        z_label = [] if measure_refusal > 1 else None
        for i in tqdm(range(0, min(N, len(toks)), batch_size)):
            logits, cache = self.run_with_cache(
                toks[i:min(i + batch_size, len(toks))],
                max_new_tokens=measure_refusal,
                stop_at_layer=stop_at_layer
            )
            if measure_refusal > 1:
                z_label.extend(self.measure_scores_from_logits(logits, measure_refusal)[0])
            for key in cache:
                if self.activation_layers is None or any(k in key for k in self.activation_layers):
                    tensor = torch.mean(cache[key][:, -last_indices:, :].to('cpu'), dim=1)
                    if key not in base:
                        base[key] = tensor
                    else:
                        base[key] = torch.cat((base[key], tensor), dim=0)
            del logits, cache
            utils.clear_mem()
        return ActivationCache(base, self.model), z_label

    def cache_activations(
        self,
        N: int = 128,
        batch_size: int = 8,
        measure_refusal: int = 0,
        last_indices: int = 1,
        reset: bool = True,
        activation_layers: int = -1,
        preserve_harmless: bool = True,
        stop_at_layer: int = None
    ) -> None:
        """
        Caches activations for harmful and harmless training instances.

        Args:
            N (int, optional): Maximum number of instances to process. Defaults to 128.
            batch_size (int, optional): Batch size for processing instances. Defaults to 8.
            measure_refusal (int, optional): Number of tokens for measuring refusal. Defaults to 0.
            last_indices (int, optional): Number of last indices in each sequence to average. Defaults to 1.
            reset (bool, optional): Whether to reset existing caches. Defaults to True.
            activation_layers (int, optional): Activation layers to cache; if -1, defaults to instance's activation_layers.
                Defaults to -1.
            preserve_harmless (bool, optional): Whether to preserve existing harmless cache if available. Defaults to True.
            stop_at_layer (int, optional): Layer index at which to stop caching. Defaults to None.
        """
        if hasattr(self, "current_state"):
            print("WARNING: Caching activations using a context")
        if self.modified:
            print("WARNING: Running modified model")

        if activation_layers == -1:
            activation_layers = self.activation_layers

        harmless_is_set = len(getattr(self, "harmless", {})) > 0
        preserve_harmless = harmless_is_set and preserve_harmless

        if reset or getattr(self, "harmless", None) is None:
            self.harmful = {}
            if not preserve_harmless:
                self.harmless = {}
            self.harmful_z_label = []
            self.harmless_z_label = []

        # Load combined training data for alignment.
        toks = self.tokenize_instructions_fn(instructions=self.harmful_inst_train[:N] + self.harmless_inst_train[:N])
        splitpos = min(N, len(self.harmful_inst_train))
        harmful_toks = toks[:splitpos]
        harmless_toks = toks[splitpos:]
        last_indices = last_indices or 1

        self.harmful, self.harmful_z_label = self.create_activation_cache(
            harmful_toks,
            N=N,
            batch_size=batch_size,
            last_indices=last_indices,
            measure_refusal=measure_refusal,
            stop_at_layer=None
        )
        if not preserve_harmless:
            self.harmless, self.harmless_z_label = self.create_activation_cache(
                harmless_toks,
                N=N,
                batch_size=batch_size,
                last_indices=last_indices,
                measure_refusal=measure_refusal,
                stop_at_layer=None
            )
