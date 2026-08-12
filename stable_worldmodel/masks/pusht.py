import torch

def generate_pusht_t_mask(goal_state: torch.Tensor, img_shape=(224, 224), local_x_offset=-0.5, local_y_offset=13) -> torch.Tensor:
    """
    Generates a boolean mask of shape (H, W) representing the T-shaped block.
    Offsets are applied in the local frame of the block so they stay consistent across rotations.
    """
    h, w = img_shape
    
    # 1. Read base values (Scaling global coordinates from 512x512 space)
    original_space = 512.0 
    block_x = (goal_state[2] / original_space) * w
    block_y = (goal_state[3] / original_space) * h
    angle = goal_state[4]  # Radians
    
    # 2. Dimensions in target pixel space
    scale_x = w / original_space
    scale_y = h / original_space
    
    w1, h1 = 120.0 * scale_x, 30.0 * scale_y   # Top bar
    w2, h2 = 30.0 * scale_x,  90.0 * scale_y   # Vertical stem
    
    # 3. Create grid matching pixel coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), 
        torch.arange(w, dtype=torch.float32), 
        indexing='ij'
    )
    
    # 4. Translate grid to block global position
    dx = x_coords - block_x
    dy = y_coords - block_y
    
    # 5. Rotate grid to line up with the block's local alignment
    # Gym-pusht rotates clockwise in world space; we invert to un-rotate into local space
    cos_a = torch.cos(-angle)
    sin_a = torch.sin(-angle)
    
    local_x = dx * cos_a - dy * sin_a
    local_y = dx * sin_a + dy * cos_a
    
    # 6. Apply local offsets HERE (They will now rotate with the block automatically)
    local_x = local_x - local_x_offset
    local_y = local_y - local_y_offset
    
    # 7. Gym-PushT Precise Center of Mass Bounds
    # In the original gym-pusht environment, the Pymunk body origin (0,0) is located 
    # exactly at the center of mass of the T-shape, which is slightly lower than the top bar.
    # If the default bounds below are slightly off, adjust `local_x_offset` and `local_y_offset`.
    
    # Top Bar: Centered horizontally, sitting above the stem
    top_bar_mask = (local_x >= -w1 / 2) & (local_x <= w1 / 2) & \
                   (local_y >= -h1)     & (local_y <= 0)
                   
    # Vertical Stem: Centered horizontally, extending downwards
    stem_mask = (local_x >= -w2 / 2) & (local_x <= w2 / 2) & \
                (local_y >= 0)       & (local_y <= h2)
                
    # 8. Merge masks
    t_mask = top_bar_mask | stem_mask
    
    return t_mask


def generate_pusht_ball_mask(goal_state: torch.Tensor, img_shape=(224, 224), local_x_offset=0.0, local_y_offset=0.0) -> torch.Tensor:
    """
    Generates a boolean mask of shape (H, W) representing the circular agent end-effector.
    """
    h, w = img_shape
    
    # 1. Read agent base coordinates from goal_state (indices 0 and 1)
    original_space = 512.0
    agent_x = (goal_state[0] / original_space) * w + local_x_offset
    agent_y = (goal_state[1] / original_space) * h + local_y_offset
    
    # 2. Scale the radius to your current pixel space
    # Default circle radius in gym-pusht is 15.0 pixels in a 512x512 space
    base_radius = 15.0
    scale_r = (w / original_space)
    radius = base_radius * scale_r
    
    # 3. Create grid coordinates matching pixel coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), 
        torch.arange(w, dtype=torch.float32), 
        indexing='ij'
    )
    
    # 4. Calculate Euclidean distance squared from every pixel to the agent's center
    dist_squared = (x_coords - agent_x) ** 2 + (y_coords - agent_y) ** 2
    
    # 5. Pixels within the radius are part of the ball
    ball_mask = dist_squared <= (radius ** 2)
    
    return ball_mask