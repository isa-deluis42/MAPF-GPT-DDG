"""
Fixed held-out seed set for congestion classifier training and validation.

These seeds are never used in DDG training:
- map_seeds are far outside the DDG training range (which starts at 8000+)
- scenario_seeds are never 0 (DDG always uses scenario_seed=0)
"""

MAP_SEEDS = list(range(128, 148))             # 20 procedural maps — above benchmark eval (0-127), below DDG training (8000+)
SCENARIO_SEEDS = [1000, 1001, 1002]           # 3 agent placements per map
AGENT_COUNTS = [16, 32, 48]                   # 3 density levels
MAP_TYPES = ["maze", "random"]

STEPS_DELTA = 16
MAX_EPISODE_STEPS = 256
GRID_PAD_SIZE = 32                            # CNN spatial input size — actual maps are up to 31x31

# Train/val split by map seed — last 4 map seeds reserved for validation
TRAIN_MAP_SEEDS = set(MAP_SEEDS[:16])
VAL_MAP_SEEDS = set(MAP_SEEDS[16:])


def iter_held_out_configs():
    """Yields (map_type, map_seed, scenario_seed, num_agents) for all held-out episodes."""
    for map_type in MAP_TYPES:
        for map_seed in MAP_SEEDS:
            for scenario_seed in SCENARIO_SEEDS:
                for num_agents in AGENT_COUNTS:
                    yield map_type, map_seed, scenario_seed, num_agents
