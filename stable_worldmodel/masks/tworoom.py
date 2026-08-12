import torch

def generate_tworoom_circle_mask(
    state: torch.Tensor, 
    radius: float = 10.5, 
    img_shape=(224, 224)
) -> torch.Tensor:
    """
    Generates a boolean mask of shape (H, W) representing a circular entity 
    (either the agent or the target) in the TwoRoom environment.
    
    Args:
        state: Tensor of shape (10,) containing the flat state vector.
               Indices 0-1: Agent position (x, y)
               Indices 2-3: Target position (x, y)
        mode: Either 'agent' or 'target' to specify which mask to build.
        radius: The radius of the circle in pixels. Defaults to 10.5 
                (the midpoint of the environment's Box(7, 14) variation space).
        img_shape: Tuple (height, width) of the image space. Default is (224, 224).
        
    Returns:
        torch.Tensor: Boolean mask of shape (H, W) where True represents the circle.
    """
    h, w = img_shape
    
    # 1. Extract the center coordinates 
    center_x = state[0]
    center_y = state[1]
        
    # 2. Create grid coordinates matching the fixed 224x224 pixel matrix
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), 
        torch.arange(w, dtype=torch.float32), 
        indexing='ij'
    )
    
    # 3. Calculate Euclidean distance squared from every pixel to the center
    dist_squared = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
    
    # 4. Pixels within the radius squared are part of the circle
    circle_mask = dist_squared <= (radius ** 2)
    
    return circle_mask