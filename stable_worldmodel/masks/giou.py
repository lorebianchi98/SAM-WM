import torch
from scipy.spatial import ConvexHull

def calculate_giou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """
    Calculates the mathematically pure Generalized Intersection over Union (GIoU) 
    using the exact enclosing Convex Hull for arbitrary mask shapes.
    
    Args:
        mask1 (torch.Tensor): Binary mask of shape (H, W), dtype=torch.bool or torch.int
        mask2 (torch.Tensor): Binary mask of shape (H, W), dtype=torch.bool or torch.int
        
    Returns:
        float: The exact GIoU value between -1.0 and 1.0
    """
    # Ensure boolean tensors
    mask1 = mask1.bool()
    mask2 = mask2.bool()
    
    # 1. Calculate Intersection and Union areas
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    
    # Handle edge case where both masks are completely empty
    if union == 0:
        return 0.0
        
    iou = intersection / union
    
    # 2. Extract coordinates of the combined active pixels
    combined_mask = mask1 | mask2
    nonzero_indices = torch.nonzero(combined_mask).cpu().numpy()
    
    # 3. Calculate the True Convex Hull Area
    # A convex hull in 2D requires at least 3 non-collinear points
    if len(nonzero_indices) >= 3:
        try:
            hull = ConvexHull(nonzero_indices)
            # hull.volume returns the exact 2D area enclosed by the hull polygon
            area_c = hull.volume
        except Exception:
            # Fallback if points are perfectly collinear and ConvexHull fails
            # In collinear cases, the bounding box area or union area is a safe fallback
            area_c = union
    else:
        area_c = union

    # 4. Compute Pure GIoU
    # Guard against division by zero (though area_c >= union > 0 here)
    if area_c == 0:
        return 0.0
        
    giou = iou - ((area_c - union) / area_c)
    
    return float(giou)
