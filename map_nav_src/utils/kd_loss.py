import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(s_inputs, t_inputs, t_sample_weights=None, loss_type='sum', **kwargs):
    # Calculate per-sample loss
    per_sample_loss = (s_inputs - t_inputs) ** 2
    
    # Apply sample weights if provided
    if t_sample_weights is not None:
        if per_sample_loss.shape[0] == t_sample_weights.shape[0]:
            # Ensure t_sample_weights is correctly broadcasted
            t_sample_weights = t_sample_weights.view(-1, *([1] * (per_sample_loss.dim() - 1)))
            per_sample_loss = per_sample_loss * t_sample_weights
        else:
            raise ValueError("Shape mismatch between sample weights and inputs")
    
    # Aggregate the loss based on the loss_type
    if loss_type == 'sum':
        return per_sample_loss.sum()
    elif loss_type == 'mean':
        return per_sample_loss.mean()
    else:
        raise ValueError("Unsupported loss_type. Choose 'sum' or 'mean'.")

def kd_loss(student_logits, teacher_logits, temperature=1, epsilon=1e-6, t_sample_weights=None, loss_type='sum', **kwargs):
    # Mask or threshold -inf values
    student_logits = torch.where(student_logits == float('-inf'), torch.full_like(student_logits, -1e6), student_logits)
    teacher_logits = torch.where(teacher_logits == float('-inf'), torch.full_like(teacher_logits, -1e6), teacher_logits)
    
    soft_teacher_output = torch.softmax(teacher_logits / temperature, dim=1)
    soft_student_output = torch.log_softmax(student_logits / temperature, dim=1)
    t_scale = temperature ** 2
    if t_sample_weights is None:
        if loss_type == 'sum':
            kd_loss = nn.KLDivLoss(reduction='sum')(soft_student_output, soft_teacher_output)
        elif loss_type == 'mean':
            kd_loss = nn.KLDivLoss(reduction='mean')(soft_student_output, soft_teacher_output)
    else:
        kd_loss = torch.kl_div(soft_student_output, soft_teacher_output).sum(1)
        if kd_loss.dim() > 1:
            # Add singleton dimensions to t_sample_weights for broadcasting
            weight_shape = (-1,) + (1,) * (kd_loss.dim() - 1)
            t_sample_weights = t_sample_weights.view(weight_shape)
        else:
            # If kd_loss is 1D, ensure t_sample_weights is also 1D
            t_sample_weights = t_sample_weights.view(-1)
        if loss_type == 'sum':
            weighted_kd_loss = (kd_loss * t_sample_weights).sum()
        elif loss_type == 'mean':
            weighted_kd_loss = (kd_loss * t_sample_weights).mean()
        return weighted_kd_loss * t_scale
    return kd_loss * t_scale 

def exponential_decay(t_sample_losses, decay_rate=0.1):
    return torch.exp(-decay_rate * t_sample_losses)

def invert_normalized_losses(t_sample_losses, **kwargs):
    # Normalize losses to be between 0 and 1
    min_loss = torch.min(t_sample_losses)
    max_loss = torch.max(t_sample_losses)
    normalized_losses = (t_sample_losses - min_loss) / (max_loss - min_loss)

    # Invert the losses so that higher original losses give lower weights
    inverted_losses = 1 - normalized_losses
    return inverted_losses