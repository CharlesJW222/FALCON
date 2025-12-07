"""
Model utilities for loading and activation extraction
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


def extract_hidden_states(model, inputs, module, no_grad=True):
    """Extract hidden states from specific module during forward pass"""
    cache = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None
    
    hook_handle = module.register_forward_hook(hook_fn)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    
    hook_handle.remove()
    return cache[0]


def select_parameters(model, layer_ids: List[int], param_ids: List[int]):
    """Select specific parameters from model layers"""
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_pretrained_model(model_path: str):
    """Load pre-trained model and tokenizer"""
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def get_target_modules(model_ref, model_train, module_str: str, layer_id: int):
    """Get target modules from both reference and training models"""
    ref_module = eval(module_str.format(model_name="model_ref", layer_id=layer_id))
    train_module = eval(module_str.format(model_name="model_train", layer_id=layer_id))
    return ref_module, train_module
