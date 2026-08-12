import torch
import numpy as np
import mujoco

def generate_ogbcube_cube_mask(env, qpos, cube_idx=0, camera_name='front_pixels', height=224, width=224, consider_occluders=False):
    """
    Given an environment instance and a qpos state vector, 
    returns a clean binary mask of the specified cube with proper occlusion handling.
    """
    # 1. Set the simulator state to the target qpos
    qvel = np.zeros(env._model.nv) 
    env.set_state(qpos, qvel)
    mujoco.mj_forward(env._model, env._data)
    
    # 2. Separate into Target Cube, Occluders (Arm/Gripper), and Background Environment
    cube_geom_ids = []
    occluder_geom_ids = []
    env_geom_ids = []
    
    target_cube_name = f'object_{cube_idx}'
    
    for i in range(env._model.ngeom):
        geom_name = env._model.geom(i).name
        if geom_name:
            if target_cube_name in geom_name and 'target' not in geom_name:
                cube_geom_ids.append(i)
            elif consider_occluders and 'ur5e' in geom_name:
                # Keep the entire arm + gripper visible so it blocks the cube mask if it's holding it
                occluder_geom_ids.append(i) 
            else:
                # Floor, walls, and other cubes (if multi-cube env)
                env_geom_ids.append(i)
        else:
            env_geom_ids.append(i)

    # 3. Cache original values and hide background elements
    original_rgbas = {}
    for gid in env_geom_ids:
        original_rgbas[gid] = env._model.geom(gid).rgba.copy()
        env._model.geom(gid).rgba[3] = 0.0 
            
    # 4. Setup MuJoCo renderer for segmentation
    renderer = mujoco.Renderer(env._model, height=height, width=width)
    renderer.enable_segmentation_rendering()
    
    renderer.update_scene(env._data, camera=camera_name)
    seg_image = renderer.render() # Shape: (height, width, 2)
    
    renderer.disable_segmentation_rendering()
    
    # 5. Restore original visibilities
    for gid in env_geom_ids:
        env._model.geom(gid).rgba = original_rgbas[gid]
    
    # 6. Extract clean pixel masks
    geom_ids_in_pixels = seg_image[:, :, 0]
    cube_mask = np.isin(geom_ids_in_pixels, cube_geom_ids).astype(np.uint8)
    
    return torch.tensor(cube_mask)

def generate_ogbcube_gripper_mask(env, qpos, camera_name='front_pixels', height=224, width=224):
    """
    Given an environment instance and a qpos state vector, 
    returns a clean binary mask with proper occlusion handling.
    """
    # 1. Set the simulator state to the target qpos
    qvel = np.zeros(env._model.nv) 
    env.set_state(qpos, qvel)
    mujoco.mj_forward(env._model, env._data)
    
    # 2. Separate into Gripper, Arm (for occlusion), and Environment
    gripper_geom_ids = []
    arm_geom_ids = []
    env_geom_ids = []
    
    for i in range(env._model.ngeom):
        geom_name = env._model.geom(i).name
        if geom_name:
            if 'ur5e/robotiq' in geom_name:
                gripper_geom_ids.append(i)
            elif 'ur5e//' in geom_name:
                arm_geom_ids.append(i) # Keep these visible to occlude the gripper base!
            else:
                env_geom_ids.append(i) # Table, cubes, floor, walls
        else:
            env_geom_ids.append(i)

    # 3. Cache original values and modify visibilities
    original_rgbas = {}
    
    # - Completely hide the external environment (removes floor reflections/cubes)
    for gid in env_geom_ids:
        original_rgbas[gid] = env._model.geom(gid).rgba.copy()
        env._model.geom(gid).rgba[3] = 0.0 
        
    # - Keep the arm visible so it blocks hidden geometries, but it won't be counted in our mask selection
    #   (No changes needed for arm geoms as they just act as masks in the renderer)
            
    # 4. Setup MuJoCo renderer for segmentation
    renderer = mujoco.Renderer(env._model, height=height, width=width)
    renderer.enable_segmentation_rendering()
    
    renderer.update_scene(env._data, camera=camera_name)
    seg_image = renderer.render() # Shape: (height, width, 2)
    
    renderer.disable_segmentation_rendering()
    
    # 5. Restore original visibilities immediately
    for gid in env_geom_ids:
        env._model.geom(gid).rgba = original_rgbas[gid]
    
    # 6. Extract clean pixel masks
    # Only pixels containing a valid gripper geom ID will evaluate to True. 
    # Pixels where the arm blocks the gripper will contain the arm's geom ID instead.
    geom_ids_in_pixels = seg_image[:, :, 0]
    gripper_mask = np.isin(geom_ids_in_pixels, gripper_geom_ids).astype(np.uint8)
    
    return torch.tensor(gripper_mask)