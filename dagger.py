import argparse
from pogema_toolbox.registry import ToolboxRegistry
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import glob

def run_dagger(dagger_type, num_workers, device_id, seed, file_size, size_min=17, size_max=21, run_infinite=False,
               segment_classifier_path=None, expert_top_k=None, out_dir="out"):
    extra = ""
    if segment_classifier_path:
        extra += f" --segment_classifier_path {segment_classifier_path}"
    if expert_top_k is not None:
        extra += f" --expert_top_k {expert_top_k}"

    def run_worker(worker_id):
        if 'warehouse' in dagger_type:
            subprocess.run(f"python worker.py --worker_id {worker_id} --device_id {device_id} --map_seed {0} --scenario_seed {seed + worker_id * 1000} --dataset_path {path_to_dataset} --dagger_type {dagger_type} --path_to_weights {path_to_weights} --num_agents {','.join(map(str, [32, 64, 96, 128, 160, 192]))} --file_size {file_size}{extra}", shell=True)
        else:
            subprocess.run(f"python worker.py --worker_id {worker_id} --device_id {device_id} --map_seed {seed + worker_id * 1000} --scenario_seed {0} --dataset_path {path_to_dataset} --dagger_type {dagger_type} --path_to_weights {path_to_weights} --file_size {file_size} --size_min {size_min} --size_max {size_max}{extra}", shell=True)

    path_to_weights = os.path.join(out_dir, f"ckpt_{dagger_type}.pt")
    path_to_dataset = f"dataset/{dagger_type}"
    ToolboxRegistry.setup_logger('INFO')
    while True:
        checkpoint_files = glob.glob(os.path.join(out_dir, f"ckpt_{dagger_type}_*0.pt"))
        if dagger_type == 'dagger' or dagger_type == 'ddg':
            checkpoint_files = [f for f in checkpoint_files if 'warehouse' not in f]
        if checkpoint_files:
            path_to_weights = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(run_worker, range(num_workers))
        seed += 1000*num_workers
        if not run_infinite:
            break

def main():
    parser = argparse.ArgumentParser(description='Dagger')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers (default: %(default)d)')
    parser.add_argument('--dagger_type', type=str, choices=['ddg', 'ddg_warehouse', 'dagger', 'dagger_warehouse'], default='ddg', help='Dagger type (default: %(default)s)')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, help='Seed (default: %(default)d)')
    parser.add_argument('--file_size', type=int, default=50 * 2 ** 11, help='File size (default: %(default)d)')
    parser.add_argument('--size_min', type=int, default=17, help='Minimum map size (default: %(default)d)')
    parser.add_argument('--size_max', type=int, default=21, help='Maximum map size (default: %(default)d)')
    parser.add_argument('--run_infinite', action='store_true', help='Run infinite (default: %(default)d)')
    parser.add_argument('--segment_classifier_path', type=str, default=None,
                        help='Path to segment classifier checkpoint. When set, the ranker selects which segment runs the full LaCAM (replacing fast LaCAM probes).')
    parser.add_argument('--expert_top_k', type=int, default=None,
                        help='Top-K env gate for segment-ranker mode: only the K envs per batch with the highest top-segment score get the full LaCAM. None means no gate.')
    parser.add_argument('--out_dir', type=str, default='out',
                        help='Directory holding ckpt_{dagger_type}_*.pt checkpoints (default: %(default)s)')
    args = parser.parse_args()
    run_dagger(args.dagger_type, args.num_workers, args.device_id, args.seed, args.file_size, size_min=args.size_min, size_max=args.size_max, run_infinite=args.run_infinite,
               segment_classifier_path=args.segment_classifier_path, expert_top_k=args.expert_top_k, out_dir=args.out_dir)

if __name__ == '__main__':
    main()
