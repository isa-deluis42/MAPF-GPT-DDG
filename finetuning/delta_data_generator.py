from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
from gymnasium import Wrapper

from pogema import AnimationMonitor, AnimationConfig

from pogema_toolbox.run_episode import run_episode
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import BaseModel

from finetuning.filter_data import filter_data

from utils.data_collection import fill_actions_with_solver
from finetuning.scenario_generators import make_pogema_maze_instance

from utils.svg_utils import cut_history, create_multi_animation
from utils.wrappers import UnrollWrapper

from multiprocessing import Pool
import cppimport.import_hook
from lacam.inference import LacamInference, LacamInferenceConfig
from pogema.wrappers.metrics import RuntimeMetricWrapper
from macro_env import PogemaMacroEnvironment, MAPFGPTObservationWrapper
from gpt.observation_generator import ObservationGenerator, InputParameters

class FastSolverDeltaConfig(BaseModel):
    steps_delta: int = 16
    steps_saved: int = 32
    save_debug_svg: bool = False
    diff_threshold = 3
    low_diff_threshold = 1
    # When set, the segment classifier picks which segment per env runs the full
    # LaCAM (replacing the fast-LaCAM-diff probe path). expert_top_k caps how
    # many envs per batch get the expert (sorted by their top segment score).
    segment_classifier_path: Optional[str] = None
    expert_top_k: Optional[int] = None
    # Device for the segment classifier (e.g. "cuda:0"). Falls back to cpu if
    # cuda isn't available at runtime.
    segment_classifier_device: str = "cpu"


class _PositionRecorder(Wrapper):
    """Record (T, N, 2) positions in raw global_xy space each step.

    Sits between UnrollWrapper and MAPFGPTObservationWrapper so 'global_xy' is
    still on the per-agent obs dicts. featurize_segment expects the same
    coordinate convention used in collected .npz files (place() reads xy as
    (row, col))."""

    def __init__(self, env):
        super().__init__(env)
        self._positions = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._positions = [np.array([o["global_xy"] for o in obs], dtype=np.int16)]
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._positions.append(np.array([o["global_xy"] for o in obs], dtype=np.int16))
        return obs, reward, term, trunc, info

    def get_recorded_positions(self):
        if not self._positions:
            return np.empty((0, 0, 2), dtype=np.int16)
        return np.stack(self._positions)


# Module-level cache for the segment classifier so repeated fast_solver_delta
# calls (one per NUM_ENVS batch) don't reload from disk every time.
_SEGMENT_CLASSIFIER_CACHE = {}


def _load_segment_classifier(path, device):
    import torch
    from train_segment_classifier import Segment3DCNN

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cache_key = (path, device)
    if cache_key in _SEGMENT_CLASSIFIER_CACHE:
        return _SEGMENT_CLASSIFIER_CACHE[cache_key]

    ckpt = torch.load(path, map_location=device, weights_only=False)
    base_ch = ckpt.get("base_ch", 16)
    in_channels = ckpt.get("in_channels", 4)
    context_segments = ckpt.get("context_segments", 1)
    model = Segment3DCNN(in_channels=in_channels, base_ch=base_ch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    bundle = {"model": model, "context_segments": context_segments, "device": device}
    _SEGMENT_CLASSIFIER_CACHE[cache_key] = bundle
    return bundle


def _score_env_segments_batch(model_bundle, per_env_inputs, steps_delta):
    """Score every segment of every env in a single forward pass.

    per_env_inputs: list of (obstacles, positions, goals) tuples, one per env.
    Returns a list of np.ndarray, one per env, holding that env's segment scores
    (empty array for envs whose ep_length is too short for any segment).
    """
    import torch
    from train_segment_classifier import featurize_segment

    context = model_bundle["context_segments"]

    feats_per_env = []
    counts = []
    for obstacles, positions, goals in per_env_inputs:
        ep_length = positions.shape[0]
        num_segments = max(0, ep_length // steps_delta - (context - 1))
        counts.append(num_segments)
        if num_segments <= 0:
            continue
        feats_per_env.append(np.stack([
            featurize_segment(
                obstacles=obstacles,
                positions=positions,
                goals=goals,
                segment_idx=i,
                context_segments=context,
            )
            for i in range(num_segments)
        ]))

    if not feats_per_env:
        return [np.zeros(0, dtype=np.float32) for _ in counts]

    batch = np.concatenate(feats_per_env, axis=0)  # (sum_S, 4, T, H, W)
    with torch.no_grad():
        x = torch.tensor(batch, dtype=torch.float32, device=model_bundle["device"])
        flat_scores = model_bundle["model"](x).squeeze(-1).cpu().numpy().astype(np.float32)

    out = []
    cursor = 0
    for n in counts:
        if n <= 0:
            out.append(np.zeros(0, dtype=np.float32))
        else:
            out.append(flat_scores[cursor:cursor + n])
            cursor += n
    return out


def run_solver(env, unroll_steps, time_limit):
    env = deepcopy(env)
    solver = LacamInference(LacamInferenceConfig(time_limit=time_limit, timeouts=[time_limit]))
    env.set_unroll_steps(unroll_steps)
    results = run_episode(env, solver)
    results['step'] = unroll_steps
    results['map_name'] = env.grid.config.map_name
    return results

def run_episode_macro(env, algo):
    algo.reset_states()
    obs, _ = env.reset()
    while True:
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        if all(terminated) or all(truncated):
            break
    return [info[0]['metrics'] for info in infos]

def run_expert(env, unroll_steps, steps_saved, chosen_agents, time_limit):
    env = deepcopy(env)
    solver = LacamInference(LacamInferenceConfig(time_limit=time_limit, timeouts=[time_limit]))
    input, gt_action, metrics = fill_actions_with_solver(env, unroll_steps, steps_saved, chosen_agents, solver)
    if metrics is not None:
        metrics['step'] = unroll_steps
        metrics['map_name'] = env.grid.config.map_name
    return input, gt_action, metrics

def fast_solver_delta(envs, learnable_algo, fast_solver, solver, cfg: FastSolverDeltaConfig):

    def create_svg(env, unroll_steps):
        obstacles = env.get_obstacles(ignore_borders=False)
        algo_history = env.get_full_history()
        fast_env = deepcopy(env)
        fast_env.set_unroll_steps(unroll_steps)
        run_episode(fast_env, fast_solver)
        fast_solver_history = fast_env.get_full_history()
        oracle_env = deepcopy(env)
        oracle_env.set_unroll_steps(unroll_steps)
        run_episode(oracle_env, solver)
        oracle_history = oracle_env.get_full_history()
        histories = [algo_history, fast_solver_history, oracle_history]
        ToolboxRegistry.debug('Histories sizes: ' + str([len(x[0]) for x in histories]))
        cut_histories = [cut_history(x, start=unroll_steps, finish=unroll_steps + cfg.steps_saved) for x in histories]
        ToolboxRegistry.debug('Cut histories sizes: ' + str([len(x[0]) for x in cut_histories]))

        svg_path = f'renders/seed-{env.grid.config.map_name}-step-{unroll_steps}.svg'
        Path(svg_path).parent.mkdir(exist_ok=True)
        create_multi_animation(obstacles, cut_histories, env.grid.config, name=svg_path)
        ToolboxRegistry.debug(f'Saved svg to: {svg_path}')

    inputs = []
    gt_actions = []
    gpt_envs = []
    initial_obstacles = []  # one (H, W) int8 array per env, captured at reset
    initial_goals = []      # one (N, 2) int16 array per env, captured at reset
    position_recorders = [] # one _PositionRecorder per env, parallel to gpt_envs
    for env in envs:
        env = RuntimeMetricWrapper(env)
        if cfg.save_debug_svg:
            env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))
        obs, _ = env.reset(seed=env.grid_config.seed)
        obs_generator = ObservationGenerator(obs[0]["global_obstacles"].copy().astype(int).tolist(),
                                             InputParameters(20, 13, 5, 256, 5, 5, 64, False))
        obs_generator.create_agents([o["global_xy"] for o in obs], [o["global_target_xy"] for o in obs])
        initial_obstacles.append(obs[0]["global_obstacles"].copy().astype(np.int8))
        initial_goals.append(np.array([o["global_target_xy"] for o in obs], dtype=np.int16))
        env = UnrollWrapper(env)
        recorder = _PositionRecorder(env)
        position_recorders.append(recorder)
        env = MAPFGPTObservationWrapper(recorder, obs_generator)
        gpt_envs.append(env)
    macro_env = PogemaMacroEnvironment(gpt_envs)
    gpt_results = run_episode_macro(macro_env, learnable_algo)

    envs = [env.get_inner_env() for env in macro_env.environments]

    unroll_steps_lists = []
    for gpt_result in gpt_results:
        unroll_steps_list = range(0, gpt_result['ep_length'], cfg.steps_delta)
        unroll_steps_lists.append(unroll_steps_list)

    fast_solver_results_by_map = {}
    diffs_by_map = {}
    diff_buckets = {}
    scores_by_map = {}  # populated only in segment-classifier mode
    envs_with_positive_diffs = []  # list of (env, unroll_steps) chosen for the expert

    if cfg.segment_classifier_path:
        # Segment-ranker selection: score every segment of every env, pick each
        # env's argmax segment, then top-K envs by their top score get the
        # expert. No fast LaCAM probes needed.
        model_bundle = _load_segment_classifier(cfg.segment_classifier_path, cfg.segment_classifier_device)
        per_env_inputs = [
            (initial_obstacles[ep_idx], recorder.get_recorded_positions(), initial_goals[ep_idx])
            for ep_idx, recorder in enumerate(position_recorders)
        ]
        all_scores = _score_env_segments_batch(model_bundle, per_env_inputs, cfg.steps_delta)
        per_env_top = []  # (env, top_segment_idx, top_score)
        for env, scores in zip(envs, all_scores):
            scores_by_map[env.grid.config.map_name] = scores.tolist()
            if scores.size == 0:
                continue
            top_idx = int(np.argmax(scores))
            per_env_top.append((env, top_idx, float(scores[top_idx])))

        per_env_top.sort(key=lambda t: t[2], reverse=True)
        keep = per_env_top if cfg.expert_top_k is None else per_env_top[:cfg.expert_top_k]
        for env, top_idx, _score in keep:
            unroll_steps = cfg.steps_delta * top_idx
            env.set_unroll_steps(unroll_steps)
            envs_with_positive_diffs.append((env, unroll_steps))
        chosen_agents = list(range(envs[-1].grid.config.num_agents)) if envs else []
        ToolboxRegistry.debug(f'Segment ranker scores: {scores_by_map}')
    else:
        with Pool(processes=8) as pool:
            fast_solver_results = pool.starmap(run_solver,
                [(env, unroll_steps, 2) for env, unroll_steps_list in zip(envs, unroll_steps_lists) for unroll_steps in unroll_steps_list])

        for result in fast_solver_results:
            if result['map_name'] not in fast_solver_results_by_map:
                fast_solver_results_by_map[result['map_name']] = {}
            fast_solver_results_by_map[result['map_name']][result['step']] = result

        for map_name, results in fast_solver_results_by_map.items():
            unroll_steps = sorted(results.keys())
            diffs = []
            for i in range(1, len(unroll_steps)):
                prev_step = unroll_steps[i - 1]
                curr_step = unroll_steps[i]
                diff = results[curr_step]['makespan'] - results[prev_step]['makespan']
                diffs.append(diff)
            diffs_by_map[map_name] = diffs

        max_diff_indices = {map_name: diffs.index(max(diffs)) for map_name, diffs in diffs_by_map.items()}

        for map_name, diffs in diffs_by_map.items():
            max_diff = max(diffs) if diffs else 0
            if max_diff > cfg.diff_threshold:
                bucket = 'auto_expert'
            elif max_diff >= cfg.low_diff_threshold:
                bucket = 'human_midrange'
            else:
                bucket = 'skip'
            diff_buckets[map_name] = {'max_diff': max_diff, 'bucket': bucket}

        for env in envs:
            if diffs_by_map[env.grid.config.map_name][max_diff_indices[env.grid.config.map_name]] > cfg.diff_threshold:
                env.set_unroll_steps(cfg.steps_delta*max_diff_indices[env.grid.config.map_name])
                envs_with_positive_diffs.append((env, cfg.steps_delta*max_diff_indices[env.grid.config.map_name]))
        chosen_agents = list(range(env.grid.config.num_agents))
        ToolboxRegistry.debug(f'Makespan difference: {diffs_by_map}')
    with Pool(processes=8) as pool:
        expert_results = pool.starmap(run_expert, 
            [(env, unroll_steps, cfg.steps_saved, chosen_agents, 10) for env, unroll_steps in envs_with_positive_diffs])
        
    inputs = []
    gt_actions = []
    expert_logs = {}
    for result in expert_results:
        if result[0] is not None:
            filtered_data = filter_data(result[0], result[1])
            inputs.extend(filtered_data['inputs'])
            gt_actions.extend(filtered_data['gt_actions'])
            expert_logs[result[2]['map_name']] = result[2]
        else:
            ToolboxRegistry.debug('No expert results for env')
    if cfg.save_debug_svg:
        for env, unroll_steps in envs_with_positive_diffs:
            create_svg(env, unroll_steps)
    logs = []
    for i in range(len(envs)):
        map_name = envs[i].grid.config.map_name
        entry = {
            'map_name': map_name,
            'gpt_results': gpt_results[i],
            'expert_results': expert_logs.get(map_name, "Not selected for expert"),
        }
        if cfg.segment_classifier_path:
            entry['segment_scores'] = scores_by_map.get(map_name)
            entry['selection_mode'] = 'segment_ranker'
        else:
            entry['fast_expert_results'] = fast_solver_results_by_map[map_name]
            entry['max_diff'] = diff_buckets[map_name]['max_diff']
            entry['bucket'] = diff_buckets[map_name]['bucket']
            entry['selection_mode'] = 'fast_lacam_diff'
        logs.append(entry)
    return {'inputs': inputs, 'gt_actions': gt_actions}, logs


def main():
    ToolboxRegistry.setup_logger('DEBUG')

    learnable_algo = MAPFGPTInference(MAPFGPTInferenceConfig(device='cuda', path_to_weights='../weights/model-2M.pt'))
    fast_time_limit = 2
    slow_time_limit = 10
    lacam_lib_path = "../lacam/liblacam.so"
    fast_solver = LacamInference(
        LacamInferenceConfig(time_limit=fast_time_limit, timeouts=[fast_time_limit], lacam_lib_path=lacam_lib_path), )
    solver = LacamInference(
        LacamInferenceConfig(time_limit=slow_time_limit, timeouts=[slow_time_limit], lacam_lib_path=lacam_lib_path))

    env = make_pogema_maze_instance(num_agents=32,
                                    max_episode_steps=256,
                                    map_seed=45,
                                    scenario_seed=45)

    fast_solver_delta(env=env, learnable_algo=learnable_algo, fast_solver=fast_solver, solver=solver,
                      cfg=FastSolverDeltaConfig(save_debug_svg=True))


if __name__ == '__main__':
    main()
