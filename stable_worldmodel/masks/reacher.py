import torch
import numpy as np

def generate_reacher_masks(
    goal_state: dict, 
    idx: int, 
    ws_min=-0.24, 
    ws_max=0.24,
    img_shape=(224, 224), 
    target_radius=0.009, 
    arm_thickness=0.0075, 
    l1=0.095, 
    l2=0.078, 
    circle_offset_x=0.016,
    circle_offset_y=0.00
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates boolean masks for the Reacher environment's target and arm structure.
    Rotates local offsets dynamically based on the cumulative arm angles.
    """
    h, w = img_shape
    
    # --- 1. Coordinate Mapping Helper ---
    def mj_to_pixel(x, y):
        px = ((x - ws_min) / (ws_max - ws_min)) * w
        py = ((ws_max - y) / (ws_max - ws_min)) * h  # Invert Y for image space grid
        return px, py

    # Create pixel grid matrices
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), 
        torch.arange(w, dtype=torch.float32), 
        indexing='ij'
    )

    # --- 2. Forward Kinematics for the Arm Segments ---
    # Extract joint angles (qpos contains shoulder angle [0] and elbow angle [1])
    theta1, theta2 = goal_state['goal_qpos'][idx]
    
    # Key joint positions in MuJoCo space
    joint0_x, joint0_y = 0.0, 0.0  # Shoulder base center
    
    joint1_x = l1 * np.cos(theta1)
    joint1_y = l1 * np.sin(theta1)  # Elbow joint position
    
    # Total absolute angle of the second link relative to the world frame
    total_angle = theta1 + theta2
    
    joint2_x = joint1_x + l2 * np.cos(total_angle)
    joint2_y = joint1_y + l2 * np.sin(total_angle)  # Fingertip position (End of second segment)

    # --- 3. Rotate Offsets into World Space based on the End-Effector Orientation ---
    # Rotate the local offset vector into MuJoCo world space coordinates
    world_offset_x = circle_offset_x * np.cos(total_angle) - circle_offset_y * np.sin(total_angle)
    world_offset_y = circle_offset_x * np.sin(total_angle) + circle_offset_y * np.cos(total_angle)
    
    # Absolute target position = Fingertip position + rotated world offsets
    t_x = joint2_x + world_offset_x
    t_y = joint2_y + world_offset_y

    # Convert all MuJoCo coordinates to pixel spaces
    pt_x, pt_y = mj_to_pixel(t_x, t_y)
    p0_x, p0_y = mj_to_pixel(joint0_x, joint0_y)
    p1_x, p1_y = mj_to_pixel(joint1_x, joint1_y)
    p2_x, p2_y = mj_to_pixel(joint2_x, joint2_y)

    # --- 4. Generate Target Mask ---
    pixel_radius = (target_radius / (ws_max - ws_min)) * w
    dist_target_sq = (x_coords - pt_x)**2 + (y_coords - pt_y)**2
    target_mask = dist_target_sq <= (pixel_radius ** 2)

    # --- 5. Generate Arm Capsule Masks (Distance to line segments) ---
    pixel_thickness = (arm_thickness / (ws_max - ws_min)) * w

    def segment_mask(ax, ay, bx, by):
        ba_x, ba_y = bx - ax, by - ay
        pa_x, pa_y = x_coords - ax, y_coords - ay
        
        ba_lensq = ba_x**2 + ba_y**2
        t = (pa_x * ba_x + pa_y * ba_y) / ba_lensq
        t = torch.clamp(t, 0.0, 1.0)
        
        closest_x = ax + t * ba_x
        closest_y = ay + t * ba_y
        
        dist_sq = (x_coords - closest_x)**2 + (y_coords - closest_y)**2
        return dist_sq <= (pixel_thickness ** 2)

    # Compute masks for both sequential arm links and combine
    link1_mask = segment_mask(p0_x, p0_y, p1_x, p1_y)
    link2_mask = segment_mask(p1_x, p1_y, p2_x, p2_y)
    arm_mask = link1_mask | link2_mask

    return target_mask, arm_mask