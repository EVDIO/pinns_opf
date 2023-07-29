import torch
import torch.nn.functional as F

def pinns_loss(v_pred, v_lower, v_upper):
    """
    Computes the loss for inequality constraints v_lower <= v_pred <= v_upper.
    
    Arguments:
    v_pred -- predicted vector of size 5x1
    v_lower -- lower bound vector of size 5x1
    v_upper -- upper bound vector of size 5x1
    
    Returns:
    loss -- the computed loss
    """
    # Compute violation of lower bound
    violation_lower = torch.relu(v_lower - v_pred)
    
    # Compute violation of upper bound
    violation_upper = torch.relu(v_pred - v_upper)
    
    # Compute the total violation
    total_violation = torch.sum(violation_lower) + torch.sum(violation_upper)
    
    # Compute the hinge loss
    hinge_loss = F.relu(total_violation)
    
    return hinge_loss

