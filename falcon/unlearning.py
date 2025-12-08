"""
Main pipeline for unlearning
"""

import os
import datetime
import argparse
import numpy as np
import torch
from tqdm import tqdm
from zeta import SophiaG
import random

from .config import UnlearningConfig, prepare_batched_data
from .model_tools import load_pretrained_model, select_parameters, extract_hidden_states, get_target_modules
from .algorithms import (
    generate_steering_vector,
    compute_contrastive_loss,
    compute_retention_loss,
    resolve_gradient_conflict
)


def execute_unlearning(
    train_model,
    ref_model,
    tokenizer,
    forget_data,
    retain_data,
    config: UnlearningConfig
):
    """
    Execute the unlearning training process
    
    Args:
        train_model: Model to be updated
        ref_model: Frozen reference model
        tokenizer: Tokenizer for text processing
        forget_data: Data to forget (list of batches per topic)
        retain_data: Data to retain (list of batches per topic)
        config: Configuration object
    """
    config.print_config()
    
    train_model = train_model.train()
    params = select_parameters(train_model, config.layer_ids, config.param_ids)
    optimizer = SophiaG(params, lr=config.lr, rho=0.9, weight_decay=1e-3)
    
    # Get target modules
    ref_module, train_module = get_target_modules(
        ref_model, train_model, config.module_str, config.layer_id
    )
    
    # Determine number of batches
    num_batches = min(
        config.max_num_batches,
        min([len(f) for f in forget_data]),
        min([len(r) for r in retain_data]),
    )
    
    # Generate steering vectors for each batch
    print(f"Generating {num_batches} steering vectors...")
    steering_vectors = []
    for i in range(num_batches):
        # Sample from alternating topics
        sample_batch = forget_data[i%2][random.randint(0, len(forget_data[i%2]) - 1)]        
        inputs = tokenizer(
            sample_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(train_model.device)
        
        hidden_states = extract_hidden_states(
            train_model,
            inputs,
            module=train_module,
            no_grad=True
        )
        
        steering_vec = generate_steering_vector(train_model, hidden_states)
        steering_vectors.append(steering_vec)
    
    # Training loop
    print(f"Starting training for {num_batches} batches...")
    
    loss_values = []
    unlearn_loss_values = []
    retain_loss_values = []
    grad_cosine_similarities = []
    
    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"
    
    num_topics = len(forget_data)
    
    with tqdm(total=num_batches) as pbar:
        for idx in range(num_batches):
            topic_idx = idx % len(forget_data)
            batch_idx = idx // len(forget_data)
            steer_vec = steering_vectors[idx]
            # Get data batches
            forget_batch = forget_data[topic_idx][batch_idx]
            retain_batch = retain_data[topic_idx][batch_idx]
            
            # Tokenize
            max_length = 512 if topic_idx == 0 else 768
            forget_inputs = tokenizer(
                forget_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(train_model.device)
            
            retain_inputs = tokenizer(
                retain_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(train_model.device)
            
            # Extract activations for forget data
            train_forget_acts = extract_hidden_states(
                train_model, forget_inputs, train_module, no_grad=False
            )
            ref_forget_acts = extract_hidden_states(
                ref_model, forget_inputs, ref_module, no_grad=True
            )
            
            # Extract activations for retain data
            train_retain_acts = extract_hidden_states(
                train_model, retain_inputs, train_module, no_grad=False
            )
            ref_retain_acts = extract_hidden_states(
                ref_model, retain_inputs, ref_module, no_grad=True
            )
            
            steer_coeff = config.steering_coeffs[topic_idx]
            
            steering_expanded = steer_vec.expand(
                train_forget_acts.size(0),
                train_forget_acts.size(1),
                -1
            )
            steering_scaled = steering_expanded * steer_coeff 
            
            # Compute unlearning loss
            unlearn_loss = compute_contrastive_loss(
                train_forget_acts,
                steering_scaled,
                ref_forget_acts
            )
            
            # Compute retention loss
            retain_loss = compute_retention_loss(
                train_retain_acts,
                ref_retain_acts,
                weight=config.alpha[topic_idx]
            )
            
            # Compute gradients separately
            optimizer.zero_grad()
            unlearn_loss.backward(retain_graph=True)
            unlearn_grads = [param.grad.clone() for param in params]
            
            optimizer.zero_grad()
            retain_loss.backward(retain_graph=True)
            retain_grads = [param.grad.clone() for param in params]
            
            # Resolve gradient conflicts
            final_grads, cos_sims = resolve_gradient_conflict(
                unlearn_grads,
                retain_grads,
                conflict_w=config.conflict_weights,  
                align_w=config.align_weights
            )
            grad_cosine_similarities.extend(cos_sims)
            
            # Apply combined gradients
            for param, grad in zip(params, final_grads):
                param.grad = grad
            
            optimizer.step()
            
            # Record losses
            total_loss = unlearn_loss.item() + retain_loss.item()
            loss_values.append(total_loss)
            unlearn_loss_values.append(unlearn_loss.item())
            retain_loss_values.append(retain_loss.item())
            
            # Print progress
            print(f"loss: {total_loss:.4g} | "
                    f"unlearn_loss: {unlearn_loss.item():.4g} | "
                    f"retain_loss: {retain_loss.item():.4g} | "
                    f"param_change: {params[0].grad.abs().mean().item():.4g}")
            
            # Verbose logging
            if config.verbose:
                unlearn_cosine = torch.nn.functional.cosine_similarity(
                    train_forget_acts,
                    ref_forget_acts.to(train_forget_acts.device),
                    dim=-1
                ).mean()
                retain_cosine = torch.nn.functional.cosine_similarity(
                    train_retain_acts,
                    ref_retain_acts.to(train_retain_acts.device),
                    dim=-1
                ).mean()
                
                print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                print(f"retain_cosine_sim={retain_cosine.item()}")
                print(f"Topic {topic_idx} train_forget_acts.norm=",
                        torch.mean(train_forget_acts.norm(dim=-1).mean(dim=1), dim=0).item())
                print(f"Topic {topic_idx} ref_forget_acts.norm=",
                        torch.mean(ref_forget_acts.norm(dim=-1).mean(dim=1), dim=0).item())
                print(f"Topic {topic_idx} train_retain_acts.norm=",
                        torch.mean(train_retain_acts.norm(dim=-1).mean(dim=1), dim=0).item())
                print(f"Topic {topic_idx} ref_retain_acts.norm=",
                        torch.mean(ref_retain_acts.norm(dim=-1).mean(dim=1), dim=0).item())
            
            pbar.update(1)
    
    # Restore truncation setting
    tokenizer.truncation_side = truncation_side
    
    # Save model
    if config.output_dir:
        save_path = config.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_path = f"models/{config.model_path.split('/')[-1]}_alpha-{config.alpha}_batches-{num_batches}_layer-{config.layer_id}_{timestamp}"
    
    train_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved model to {save_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--module_str", type=str, default="{model_name}.model.layers[{layer_id}]")
    parser.add_argument("--output_dir", type=str, default=None)
    
    # Data arguments
    parser.add_argument("--retain_corpora", type=str, default="wikitext,wikitext",
                       help="comma-separated list of corpora to retain")
    parser.add_argument("--forget_corpora", type=str, default="bio-remove-dataset,cyber-forget-corpus",
                       help="comma-separated list of corpora to forget")
    
    # Gradient handling arguments
    parser.add_argument("--conflict_weights", type=str, default="0.8,1.2",
                       help="Gradient weights when conflict: unlearn_weight,retain_weight")
    parser.add_argument("--align_weights", type=str, default="0.1,1.9",
                       help="Gradient weights when aligned: unlearn_weight,retain_weight")
    
    # Hyperparameters
    parser.add_argument("--alpha", type=str, default="100,100", help="retain weight")
    parser.add_argument("--steering_coeffs", type=str, default="20,20",
                       help="Steer vector weight in order of topic")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true",
                       help="Logging the activations norms and cosine at each step")
    
    args = parser.parse_args()
    
    # Parse comma-separated arguments
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    conflict_w = args.conflict_weights.split(",")
    args.conflict_weights = (float(conflict_w[0]), float(conflict_w[1]))
    
    align_w = args.align_weights.split(",")
    args.align_weights = (float(align_w[0]), float(align_w[1]))
    
    return args


def main():
    args = parse_arguments()
    
    # Set random seeds
    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load models
    print("Loading models...")
    ref_model, tokenizer = load_pretrained_model(args.model_name_or_path)
    train_model, _ = load_pretrained_model(args.model_name_or_path)
    
    # Prepare data
    print("Loading data...")
    forget_data, retain_data = prepare_batched_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    
    # Create config
    config = UnlearningConfig.from_args(args)
    
    # Execute unlearning
    execute_unlearning(
        train_model,
        ref_model,
        tokenizer,
        forget_data,
        retain_data,
        config,
    )


if __name__ == "__main__":
    main()
