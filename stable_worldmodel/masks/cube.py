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
    
    # 2. Collect the target cube geom IDs.
    cube_geom_ids = []
    
    target_cube_name = f'object_{cube_idx}'
    
    for i in range(env._model.ngeom):
        geom_name = env._model.geom(i).name
        if geom_name:
            if target_cube_name in geom_name and 'target' not in geom_name:
                cube_geom_ids.append(i)
    original_rgbas = [env._model.geom(i).rgba.copy() for i in range(env._model.ngeom)]
    try:
        for i in range(env._model.ngeom):
            geom_name = env._model.geom(i).name or ''
            keep_visible = consider_occluders and 'ur5e' in geom_name
            env._model.geom(i).rgba[3] = 1.0 if i in cube_geom_ids or keep_visible else 0.0
            if i in cube_geom_ids:
                env._model.geom(i).rgba[:3] = [1.0, 0.0, 0.0]

        renderer = mujoco.Renderer(env._model, height=height, width=width)
        renderer.update_scene(env._data, camera=camera_name)
        image = renderer.render()
        cube_mask = ((image[:, :, 0] > 80)
                     & (image[:, :, 0] > image[:, :, 1] * 1.5)
                     & (image[:, :, 0] > image[:, :, 2] * 1.5)).astype(np.uint8)
    finally:
        for i, rgba in enumerate(original_rgbas):
            env._model.geom(i).rgba[:] = rgba
    
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
    
    # 2. Collect gripper geom IDs.
    gripper_geom_ids = []
    
    for i in range(env._model.ngeom):
        geom_name = env._model.geom(i).name
        if geom_name:
            if 'ur5e/robotiq' in geom_name:
                gripper_geom_ids.append(i)
            

    original_rgbas = [env._model.geom(i).rgba.copy() for i in range(env._model.ngeom)]
    try:
        for i in range(env._model.ngeom):
            geom_name = env._model.geom(i).name or ''
            keep_visible = 'ur5e//' in geom_name
            env._model.geom(i).rgba[3] = 1.0 if i in gripper_geom_ids or keep_visible else 0.0
            if i in gripper_geom_ids:
                env._model.geom(i).rgba[:3] = [0.0, 1.0, 0.0]

        renderer = mujoco.Renderer(env._model, height=height, width=width)
        renderer.update_scene(env._data, camera=camera_name)
        image = renderer.render()
        gripper_mask = ((image[:, :, 1] > 80)
                        & (image[:, :, 1] > image[:, :, 0] * 1.5)
                        & (image[:, :, 1] > image[:, :, 2] * 1.5)).astype(np.uint8)
    finally:
        for i, rgba in enumerate(original_rgbas):
            env._model.geom(i).rgba[:] = rgba
    
    return torch.tensor(gripper_mask)