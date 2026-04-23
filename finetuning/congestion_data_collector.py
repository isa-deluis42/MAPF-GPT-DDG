"""
Congestion Classification Data Collector

Collects raw inputs at every timestep with labels based on whether
the episode will eventually fail (expert - fast makespan > threshold).

This is Phase 1 of the congestion classification pipeline:
- Run episodes with both fast and expert solvers
- Label each timestep based on final makespan diff
- Save raw 256-dim inputs with binary labels
"""

from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

import cppimport.import_hook
from pogema import AnimationMonitor, AnimationConfig
from pogema_toolbox.run_episode import run_episode
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import BaseModel

from lacam.inference import LacamInference, LacamInferenceConfig
from pogema.wrappers.metrics import RuntimeMetricWrapper
from macro_env import PogemaMacroEnvironment, MAPFGPTObservationWrapper
from gpt.observation_generator import ObservationGenerator, InputParameters
from utils.wrappers import UnrollWrapper
from finetuning.congestion_utils import bucket_to_label, diff_to_confidence_bucket


class CongestionDataCollectorConfig(BaseModel):
    """Configuration for congestion data collection."""
    fast_solver_time_limit: int = 2      # Time limit for fast solver
    expert_solver_time_limit: int = 10   # Time limit for expert solver
    diff_threshold: int = 3              # High-confidence positive threshold
    low_diff_threshold: int = 1          # High-confidence negative threshold
    save_raw_inputs: bool = True         # Whether to save raw 256-dim inputs
    save_debug_svg: bool = False          # Save SVG visualizations


def run_episode_with_inputs(
    env, 
    algo, 
    obs_generator: ObservationGenerator,
    max_steps: int = 256
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Run an episode and collect raw inputs at each timestep.
    
    Returns:
        inputs: List of raw 256-dim input tensors, one per agent per timestep
        metrics: Episode-level metrics (makespan, success rate, etc.)
    """
    algo.reset_states()
    obs, _ = env.reset()
    
    all_inputs = []
    all_actions = []
    
    for step in range(max_steps):
        # Generate observations for all agents at current timestep
        raw_inputs = obs_generator.generate_observations()
        # raw_inputs shape: (num_agents, 256)
        
        # Get actions from the algorithm
        actions = algo.act(obs)
        
        # Store inputs and actions
        for agent_idx in range(len(raw_inputs)):
            all_inputs.append(raw_inputs[agent_idx])
            all_actions.append(actions[agent_idx] if actions is not None else 0)
        
        # Step the environment
        obs, rew, terminated, truncated, infos = env.step(actions)
        if all(terminated) or all(truncated):
            break
    
    # Get episode metrics
    metrics = infos[0]['metrics'] if infos else {}
    metrics['ep_length'] = len(all_inputs) // max(env.grid.config.num_agents, 1)
    
    return all_inputs, metrics


def collect_congestion_data(
    envs: List,
    learnable_algo,
    fast_solver,
    expert_solver,
    cfg: CongestionDataCollectorConfig
) -> Dict[str, Any]:
    """
    Collect congestion data from multiple environments.
    
    For each episode:
    1. Run with fast solver (2s) -> makespan_fast
    2. Run with expert solver (10s) -> makespan_expert  
    3. Label = 1 if (makespan_expert - makespan_fast) > threshold
    4. Run again with learnable algo, collect raw inputs at each timestep
    5. Assign label to every timestep in the episode
    
    Returns:
        Dictionary with:
        - inputs: raw 256-dim tensors
        - labels: binary labels (0=pass, 1=fail)
        - episode_info: metadata for each episode
    """
    all_inputs = []
    all_labels = []
    all_episode_ids = []
    all_diffs = []
    all_confidence_buckets = []
    episode_info = []
    
    for env_idx, env in enumerate(envs):
        env = RuntimeMetricWrapper(deepcopy(env))
        obs, _ = env.reset(seed=env.grid_config.seed)
        
        # Create observation generator
        obs_generator = ObservationGenerator(
            obs[0]["global_obstacles"].copy().astype(int).tolist(),
            InputParameters(20, 13, 5, 256, 5, 5, 64, False)
        )
        obs_generator.create_agents(
            [o["global_xy"] for o in obs],
            [o["global_target_xy"] for o in obs]
        )
        
        # Step 1: Run with fast solver to get makespan_fast
        fast_env = UnrollWrapper(deepcopy(env))
        fast_env.set_unroll_steps(256)
        fast_results = run_episode(fast_env, fast_solver)
        makespan_fast = fast_results.get('makespan', 256)
        
        # Step 2: Run with expert solver to get makespan_expert
        expert_env = UnrollWrapper(deepcopy(env))
        expert_env.set_unroll_steps(256)
        expert_results = run_episode(expert_env, expert_solver)
        makespan_expert = expert_results.get('makespan', 256)
        
        # Step 3: Compute label based on diff
        diff = makespan_expert - makespan_fast
        confidence_bucket = diff_to_confidence_bucket(
            diff=diff,
            low_diff_threshold=cfg.low_diff_threshold,
            high_diff_threshold=cfg.diff_threshold,
        )
        label = bucket_to_label(confidence_bucket)
        
        ToolboxRegistry.debug(
            f"Env {env_idx}: fast={makespan_fast}, expert={makespan_expert}, "
            f"diff={diff}, label={label}, bucket={confidence_bucket}"
        )
        
        # Step 4: Run with learnable algo to collect raw inputs
        wrapped_env = UnrollWrapper(deepcopy(env))
        wrapped_env = MAPFGPTObservationWrapper(wrapped_env, obs_generator)
        
        learnable_algo.reset_states()
        obs, _ = wrapped_env.reset()
        
        num_agents = wrapped_env.grid.config.num_agents
        episode_inputs = []
        
        for step in range(wrapped_env.grid.config.max_episode_steps):
            # Generate raw inputs at this timestep
            raw_inputs = obs_generator.generate_observations()
            # raw_inputs: (num_agents, 256)
            
            # Get actions
            actions = learnable_algo.act(obs)
            
            # Store inputs for all agents
            for agent_idx in range(len(raw_inputs)):
                episode_inputs.append(raw_inputs[agent_idx])
            
            # Step environment
            obs, rew, terminated, truncated, infos = wrapped_env.step(actions)
            if all(terminated) or all(truncated):
                break
        
        # Step 5: Assign label to all timesteps in this episode
        num_timesteps = len(episode_inputs) // num_agents if num_agents > 0 else 0
        episode_labels = [label] * len(episode_inputs)
        episode_ids = [env_idx] * len(episode_inputs)
        episode_diffs = [diff] * len(episode_inputs)
        episode_buckets = [confidence_bucket] * len(episode_inputs)

        # Store data
        all_inputs.extend(episode_inputs)
        all_labels.extend(episode_labels)
        all_episode_ids.extend(episode_ids)
        all_diffs.extend(episode_diffs)
        all_confidence_buckets.extend(episode_buckets)

        episode_info.append({
            'env_idx': env_idx,
            'map_name': env.grid.config.map_name,
            'makespan_fast': makespan_fast,
            'makespan_expert': makespan_expert,
            'diff': diff,
            'label': label,
            'confidence_bucket': confidence_bucket,
            'num_timesteps': num_timesteps,
            'num_agents': num_agents
        })
    
    return {
        'inputs': np.array(all_inputs, dtype=np.int8),
        'labels': np.array(all_labels, dtype=np.int8),
        'episode_ids': np.array(all_episode_ids, dtype=np.int32),
        'diffs': np.array(all_diffs, dtype=np.int16),
        'confidence_buckets': np.array(all_confidence_buckets, dtype=object),
        'episode_info': episode_info
    }


def save_congestion_dataset(
    data: Dict[str, Any],
    output_path: str
):
    """Save congestion dataset to disk."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as Arrow format (consistent with existing data)
    table = pa.table({
        'inputs': pa.array(data['inputs'].tolist()),
        'labels': pa.array(data['labels'].tolist()),
        'episode_ids': pa.array(data['episode_ids'].tolist()),
        'diffs': pa.array(data['diffs'].tolist()),
        'confidence_buckets': pa.array(data['confidence_buckets'].tolist()),
    })
    
    # Also save episode info as JSON
    import json
    with open(output_path.parent / 'episode_info.json', 'w') as f:
        json.dump(data['episode_info'], f, indent=2)
    
    pa.ipc.new_file(table, str(output_path)).write()
    ToolboxRegistry.debug(f"Saved congestion dataset to {output_path}")


# ============================================================================
# Example usage
# ============================================================================

def main():
    """Example: Collect congestion data from a few episodes."""
    from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
    from finetuning.scenario_generators import make_pogema_maze_instance
    
    ToolboxRegistry.setup_logger('DEBUG')
    
    # Setup algorithms
    learnable_algo = MAPFGPTInference(
        MAPFGPTInferenceConfig(
            device='cuda',
            path_to_weights='../weights/model-2M.pt'
        )
    )
    
    lacam_lib_path = "../lacam/liblacam.so"
    fast_solver = LacamInference(
        LacamInferenceConfig(
            time_limit=2,
            timeouts=[2],
            lacam_lib_path=lacam_lib_path
        )
    )
    expert_solver = LacamInference(
        LacamInferenceConfig(
            time_limit=10,
            timeouts=[10],
            lacam_lib_path=lacam_lib_path
        )
    )
    
    # Create test environments
    envs = [
        make_pogema_maze_instance(
            num_agents=32,
            max_episode_steps=256,
            map_seed=45,
            scenario_seed=45
        ),
        make_pogema_maze_instance(
            num_agents=32,
            max_episode_steps=256,
            map_seed=46,
            scenario_seed=46
        ),
    ]
    
    # Collect data
    cfg = CongestionDataCollectorConfig(
        fast_solver_time_limit=2,
        expert_solver_time_limit=10,
        diff_threshold=3,
        save_raw_inputs=True
    )
    
    data = collect_congestion_data(
        envs=envs,
        learnable_algo=learnable_algo,
        fast_solver=fast_solver,
        expert_solver=expert_solver,
        cfg=cfg
    )
    
    # Print statistics
    labels = np.array(data['labels'])
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== Congestion Data Collection Summary ===")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} samples ({100*count/len(labels):.1f}%)")
    
    print(f"\nTotal episodes: {len(data['episode_info'])}")
    for info in data['episode_info']:
        print(f"  {info['map_name']}: diff={info['diff']}, label={info['label']}")
    
    # Save dataset
    save_congestion_dataset(data, 'dataset/congestion/train.arrow')


if __name__ == '__main__':
    main()
