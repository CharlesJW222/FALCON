"""
Core algorithms: POV generation and loss computation
"""

import torch
import torch.nn.functional as F


def generate_steering_vector(model, hidden_states):
    """
    Generate irreversible steering vector using SVD projection
    
    Args:
        model: Model instance for device and dtype
        hidden_states: Input hidden states (batch_size, seq_len, hidden_dim)
    
    Returns:
        Steering vector with irreversible transformations
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    device, dtype = model.device, model.dtype
    
    with torch.no_grad():
        # Initialize base vector
        base_vector = torch.ones(1, 1, hidden_dim, dtype=dtype, device=device)
        base_vector = base_vector / torch.norm(base_vector)
        
        # Compute principal directions via SVD
        states_matrix = hidden_states.view(-1, hidden_dim).to(torch.float32)
        U, S, Vh = torch.linalg.svd(states_matrix, full_matrices=False)
        k = min(1000, Vh.shape[0])
        key_directions = Vh[:k].to(dtype)
        
        # Build projection matrix with strong weights
        projection = torch.eye(hidden_dim, dtype=dtype, device=device)
        for i in range(k):
            direction = key_directions[i].unsqueeze(0)
            weight = torch.sigmoid(S[i] / S[0]) * (-1000.0)
            projection -= weight * torch.matmul(direction.T, direction)
        
        # Add non-linear perturbation
        noise = torch.randn_like(projection) * 0.01
        projection = torch.tanh(projection + noise)
        
        # Apply iterative non-linear transformations
        final_vector = base_vector
        for _ in range(6):
            final_vector = torch.matmul(final_vector, projection)
            final_vector = torch.tanh(final_vector)
            final_vector = final_vector / (torch.norm(final_vector) + 1e-6)
    
    return final_vector.detach()


def compute_contrastive_loss(anchor, positive, negatives, temperature=0.7):
    """
    Compute InfoNCE contrastive loss
    
    Args:
        anchor: Anchor representations
        positive: Positive samples
        negatives: Negative samples
        temperature: Temperature parameter for scaling
    
    Returns:
        Contrastive loss value
    """
    device = anchor.device
    positive = positive.to(device)
    negatives = negatives.to(device)
    
    # Normalize on last dimension
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    
    negatives = negatives.unsqueeze(2)
    
    # Compute positive similarity
    pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True)
    
    # Compute negative similarities
    neg_sim = torch.einsum("bld,blnd->bln", anchor, negatives)
    
    # Concatenate and scale
    logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
    
    # Labels: positive sample is always at index 0
    labels = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=device)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss


def compute_retention_loss(updated_acts, frozen_acts, weight=1.0):
    """
    Compute retention loss based on cosine similarity
    
    Args:
        updated_acts: Activations from updated model
        frozen_acts: Activations from frozen model
        weight: Loss weight coefficient
    
    Returns:
        Weighted retention loss
    """
    retention_loss = 1 - F.cosine_similarity(
        updated_acts,
        frozen_acts.to(updated_acts.device),
        dim=-1
    ).mean()
    return retention_loss * weight


def resolve_gradient_conflict(unlearn_grads, retain_grads, conflict_w=(0.8, 1.2), align_w=(0.1, 1.9)):
    """
    Resolve gradient conflicts between unlearning and retention objectives
    
    Args:
        unlearn_grads: Gradients from unlearning loss
        retain_grads: Gradients from retention loss
        conflict_w: Weights when gradients conflict
        align_w: Weights when gradients align
    
    Returns:
        Combined gradients and cosine similarities
    """
    combined_grads = []
    cosine_sims = []
    
    for u_grad, r_grad in zip(unlearn_grads, retain_grads):
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(
            u_grad.view(-1),
            r_grad.view(-1),
            dim=0
        )
        cosine_sims.append(cos_sim.item())
        
        if cos_sim < 0:
            # Gradients conflict: project to orthogonal plane
            proj_grad = u_grad - (torch.dot(u_grad.view(-1), r_grad.view(-1)) / 
                                 torch.dot(r_grad.view(-1), r_grad.view(-1))) * r_grad
            combined_grad = conflict_w[0] * proj_grad + conflict_w[1] * r_grad
        else:
            # Gradients align: increase retention weight
            combined_grad = align_w[0] * u_grad + align_w[1] * r_grad
        
        combined_grads.append(combined_grad)
    
    return combined_grads, cosine_sims
