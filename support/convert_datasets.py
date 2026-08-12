import os
import torch
from tqdm import tqdm
import stable_worldmodel as swm
from stable_worldmodel.data import LanceWriter

dataset_dir = '/raid/datasets/world_model_simulations/datasets/point_maze'
output_lance_path = os.path.join(dataset_dir, 'point_maze.lance')

print("Loading raw PyTorch datasets...")
actions = torch.load(f'{dataset_dir}/actions.pth')         
states = torch.load(f'{dataset_dir}/states.pth')           
seq_lengths = torch.load(f'{dataset_dir}/seq_lengths.pth') 

# Gather and explicitly sort all observation files numerically
obs_dir = f'{dataset_dir}/obses'
obs_files = [f for f in os.listdir(obs_dir) if f.endswith('.pth')]
# Sorts naturally: episode_1, episode_2, ..., episode_100, episode_101...
obs_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

print(f"Found {len(obs_files)} observation files on disk.")
print(f"Tensor array contains rows for {actions.shape[0]} episodes.")
print(f"Converting dataset to SWM Lance format at: {output_lance_path}")

# Reset our array pointer
array_idx = 0

with LanceWriter(output_lance_path, mode='overwrite') as writer:
    for file_name in tqdm(obs_files, desc="Processing Episodes"):
        
        # Stop safely if we run out of rows in our actions/states tensors
        if array_idx >= actions.shape[0]:
            print(f"\n[Warning] Reached tensor capacity limit of {actions.shape[0]}. Stopping.")
            break

        actual_length = int(seq_lengths[array_idx].item())
        
        # Load the observation frame safely
        ep_observations = torch.load(os.path.join(obs_dir, file_name))
        ep_observations = ep_observations.permute(0, 3, 1, 2)
        
        # Pull data using our linear array index pointer
        ep_actions = actions[array_idx, :actual_length].cpu().numpy()
        ep_states = states[array_idx, :actual_length].cpu().numpy()
        ep_pixels = ep_observations[:actual_length].cpu().numpy()
        
        episode_dict = {
            "action": [step for step in ep_actions],
            "state":  [step for step in ep_states],
            "pixels": [step for step in ep_pixels], 
        }
        
        writer.write_episode(episode_dict)
        
        # Advance to the next tensor index row for the next file
        array_idx += 1

print(f"\nConversion complete! Total episodes successfully written: {array_idx}")