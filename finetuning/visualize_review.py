"""
Render each human-review sample as an animated SVG of the full episode.

Ground-truth per-step positions, goals, and the obstacle grid are read from
per-episode `.npz` files produced by the patched data collector
(see `collect_congestion_data(..., episodes_output_dir=...)`).

For each review sample we:
  - Map `sample_index` back to `(episode_id, timestep, agent_id)` using
    `episode_info.json` (inputs are stored flat in [episode, step, agent] order).
  - Load that episode's obstacles and (T+1, N, 2) position tensor.
  - Emit an SVG animating every agent across the whole episode, with the
    focal agent marked distinctly and the focal timestep flagged by a
    red stroke that flashes when the animation reaches it.

Usage:
    python -m finetuning.visualize_review \
        --review dataset/congestion/review_samples.json \
        --episode_info dataset/congestion/episode_info.json \
        --episodes_dir dataset/congestion/episodes \
        --output out/review_viz
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


AGENT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#808080", "#000000",
    "#ff6f61", "#6b5b95", "#88b04b", "#f7cac9", "#92a8d1",
    "#955251", "#b565a7", "#009b77", "#dd4124", "#45b8ac",
    "#eeaec2", "#c3447a",
]


def build_sample_lookup(episode_info: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int]]:
    """Map global sample_index to (timestep, agent_id) and record per-episode offsets."""
    offsets: Dict[int, int] = {}
    offset = 0
    for ep in episode_info:
        offsets[int(ep["env_idx"])] = offset
        offset += int(ep["num_timesteps"]) * int(ep["num_agents"])
    return offsets


def sample_coords(
    sample_index: int,
    episode_id: int,
    offsets: Dict[int, int],
    num_agents: int,
) -> Tuple[int, int]:
    local = sample_index - offsets[episode_id]
    if local < 0:
        raise ValueError(f"sample_index {sample_index} precedes episode {episode_id} offset")
    return local // num_agents, local % num_agents


def render_sample_svg(
    sample: Dict[str, Any],
    episode: Dict[str, Any],
    episode_file: Path,
    focal_t: int,
    focal_agent: int,
    cell_size: int = 16,
    frame_duration: float = 0.35,
) -> str:
    data = np.load(episode_file)
    obstacles: np.ndarray = data["obstacles"]    # (H, W)
    positions: np.ndarray = data["positions"]    # (T+1, N, 2)
    goals: np.ndarray = data["goals"]            # (T+1, N, 2)

    # Pogema often pads obstacles with an `obs_radius` border. Trim rows/cols
    # that are entirely obstacle so positions line up with the viewport.
    H, W = obstacles.shape
    r_lo, r_hi = 0, H
    c_lo, c_hi = 0, W
    while r_lo < r_hi and obstacles[r_lo].all():
        r_lo += 1
    while r_hi > r_lo and obstacles[r_hi - 1].all():
        r_hi -= 1
    while c_lo < c_hi and obstacles[:, c_lo].all():
        c_lo += 1
    while c_hi > c_lo and obstacles[:, c_hi - 1].all():
        c_hi -= 1

    # Positions are reported in full-grid coordinates — offset them to the trimmed view.
    def to_local(rc: np.ndarray) -> np.ndarray:
        return rc - np.array([r_lo, c_lo], dtype=rc.dtype)

    obstacles = obstacles[r_lo:r_hi, c_lo:c_hi]
    H, W = obstacles.shape

    num_frames = positions.shape[0]
    num_agents = positions.shape[1]
    total_duration = num_frames * frame_duration

    width = W * cell_size
    height = H * cell_size

    def cx(col: float) -> float:
        return (col + 0.5) * cell_size

    def cy(row: float) -> float:
        return (row + 0.5) * cell_size

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height + 22}" '
        f'viewBox="0 0 {width} {height + 22}">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')

    # Obstacles
    for r in range(H):
        for c in range(W):
            if obstacles[r, c]:
                parts.append(
                    f'<rect x="{c*cell_size}" y="{r*cell_size}" '
                    f'width="{cell_size}" height="{cell_size}" fill="#2a2a2a"/>'
                )

    # Grid lines
    for r in range(H + 1):
        y = r * cell_size
        parts.append(
            f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" '
            f'stroke="#eeeeee" stroke-width="0.5"/>'
        )
    for c in range(W + 1):
        x = c * cell_size
        parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" '
            f'stroke="#eeeeee" stroke-width="0.5"/>'
        )

    positions_local = to_local(positions)
    goals_local = to_local(goals)

    key_times = ";".join(f"{k/(num_frames-1):.4f}" for k in range(num_frames)) if num_frames > 1 else "0"

    # Per-agent goal markers — animate position in case of lifelong goal changes.
    for a in range(num_agents):
        color = AGENT_COLORS[a % len(AGENT_COLORS)]
        gxs = ";".join(f"{cx(goals_local[t, a, 1]) - cell_size*0.3:.2f}" for t in range(num_frames))
        gys = ";".join(f"{cy(goals_local[t, a, 0]) - cell_size*0.3:.2f}" for t in range(num_frames))
        start_x = cx(goals_local[0, a, 1]) - cell_size * 0.3
        start_y = cy(goals_local[0, a, 0]) - cell_size * 0.3
        parts.append(
            f'<rect x="{start_x:.2f}" y="{start_y:.2f}" '
            f'width="{cell_size*0.6:.2f}" height="{cell_size*0.6:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="2,2" opacity="0.55">'
            f'<animate attributeName="x" values="{gxs}" keyTimes="{key_times}" '
            f'dur="{total_duration}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<animate attributeName="y" values="{gys}" keyTimes="{key_times}" '
            f'dur="{total_duration}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</rect>'
        )

    # Agents — one circle per agent with position animated across the episode.
    for a in range(num_agents):
        color = AGENT_COLORS[a % len(AGENT_COLORS)]
        is_focal = a == focal_agent
        stroke = "#000000"
        stroke_w = 2.6 if is_focal else 0.8
        radius = cell_size * (0.42 if is_focal else 0.36)

        xs = ";".join(f"{cx(positions_local[t, a, 1]):.2f}" for t in range(num_frames))
        ys = ";".join(f"{cy(positions_local[t, a, 0]):.2f}" for t in range(num_frames))
        start_x = cx(positions_local[0, a, 1])
        start_y = cy(positions_local[0, a, 0])

        parts.append(
            f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="{radius:.2f}" '
            f'fill="{color}" stroke="{stroke}" stroke-width="{stroke_w}" fill-opacity="0.92">'
            f'<animate attributeName="cx" values="{xs}" keyTimes="{key_times}" '
            f'dur="{total_duration}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="cy" values="{ys}" keyTimes="{key_times}" '
            f'dur="{total_duration}s" repeatCount="indefinite" calcMode="linear"/>'
            f'</circle>'
        )

        # Agent index label (smaller for non-focal).
        font_size = cell_size * (0.6 if is_focal else 0.42)
        parts.append(
            f'<text x="{start_x:.2f}" y="{start_y + font_size*0.35:.2f}" '
            f'font-size="{font_size:.1f}" fill="#000" text-anchor="middle" '
            f'font-family="monospace" font-weight="{"bold" if is_focal else "normal"}">'
            f'<animate attributeName="x" values="{xs}" keyTimes="{key_times}" '
            f'dur="{total_duration}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="y" values="{";".join(f"{cy(positions_local[t, a, 0]) + font_size*0.35:.2f}" for t in range(num_frames))}" '
            f'keyTimes="{key_times}" dur="{total_duration}s" repeatCount="indefinite" calcMode="linear"/>'
            f'{a}</text>'
        )

    # Focal-timestep flasher: a red ring around the focal agent that only
    # appears during the focal frame window.
    if num_frames > 1 and 0 <= focal_t < num_frames:
        pre = max(focal_t - 1, 0) / (num_frames - 1)
        on = focal_t / (num_frames - 1)
        post = min(focal_t + 1, num_frames - 1) / (num_frames - 1)
        fxs = ";".join(f"{cx(positions_local[t, focal_agent, 1]):.2f}" for t in range(num_frames))
        fys = ";".join(f"{cy(positions_local[t, focal_agent, 0]):.2f}" for t in range(num_frames))
        parts.append(
            f'<circle cx="{cx(positions_local[0, focal_agent, 1]):.2f}" '
            f'cy="{cy(positions_local[0, focal_agent, 0]):.2f}" '
            f'r="{cell_size*0.55:.2f}" fill="none" stroke="#ff0000" stroke-width="2.2" opacity="0">'
            f'<animate attributeName="cx" values="{fxs}" keyTimes="{key_times}" '
            f'dur="{total_duration}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="cy" values="{fys}" keyTimes="{key_times}" '
            f'dur="{total_duration}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="opacity" values="0;0;1;0;0" '
            f'keyTimes="0;{pre:.4f};{on:.4f};{post:.4f};1" '
            f'dur="{total_duration}s" repeatCount="indefinite"/>'
            f'</circle>'
        )

    # Footer strip with a live frame counter.
    frame_values = ";".join(str(t) for t in range(num_frames))
    parts.append(
        f'<rect x="0" y="{height}" width="{width}" height="22" fill="#111"/>'
        f'<text x="6" y="{height + 15}" font-size="12" fill="#eee" '
        f'font-family="monospace">'
        f'<tspan>frame </tspan>'
        f'<tspan>0'
        f'<animate attributeName="textContent" values="{frame_values}" '
        f'keyTimes="{key_times}" dur="{total_duration}s" '
        f'repeatCount="indefinite" calcMode="discrete"/>'
        f'</tspan>'
        f'<tspan>/{num_frames - 1} &#183; focal t={focal_t} &#183; '
        f'focal agent #{focal_agent}</tspan>'
        f'</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _label_text(value) -> str:
    if value is None:
        return "?"
    return "fail" if int(value) == 1 else "pass"


def render_index_html(
    samples: List[Dict[str, Any]],
    svg_names: List[str],
    coords: List[Tuple[int, int]],
) -> str:
    rows: List[str] = []
    for s, svg_name, (t, a) in zip(samples, svg_names, coords):
        auto = _label_text(s.get("auto_label"))
        model = _label_text(s.get("model_pred"))
        human = _label_text(s.get("human_label"))
        prob_fail = s.get("model_prob_fail", 0.0)
        rows.append(
            f'<div class="card">'
            f'<div class="meta">'
            f'<div><b>sample</b> {s.get("sample_index")} '
            f'&nbsp;|&nbsp; <b>episode</b> {s.get("episode_id")} '
            f'&nbsp;|&nbsp; <b>t/a</b> {t}/{a} '
            f'&nbsp;|&nbsp; <b>diff</b> {s.get("diff")} '
            f'&nbsp;|&nbsp; <b>bucket</b> {html.escape(str(s.get("review_bucket", "")))}</div>'
            f'<div><b>auto</b> {auto} &nbsp;|&nbsp; '
            f'<b>model</b> {model} (p_fail={prob_fail:.3f}) &nbsp;|&nbsp; '
            f'<b>human</b> {human}</div>'
            f'</div>'
            f'<iframe src="{html.escape(svg_name)}" loading="lazy"></iframe>'
            f'</div>'
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Congestion review</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px}"
        ".meta{font-size:12px;color:#333;margin-bottom:6px;line-height:1.4}"
        "iframe{display:block;width:100%;height:540px;border:none}"
        "</style></head><body>"
        f"<h2>Congestion review · {len(samples)} samples</h2>"
        "<div class='grid'>" + "\n".join(rows) + "</div></body></html>"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True, help="Path to review_samples.json")
    ap.add_argument(
        "--episode_info",
        required=True,
        help="Path to episode_info.json produced by the data collector",
    )
    ap.add_argument(
        "--episodes_dir",
        required=True,
        help="Directory containing per-episode .npz files (ep_XXXX.npz)",
    )
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--limit", type=int, default=None, help="Only render first N samples")
    args = ap.parse_args()

    with open(args.review) as f:
        samples = json.load(f)
    if args.limit is not None:
        samples = samples[: args.limit]

    with open(args.episode_info) as f:
        episode_info = json.load(f)
    episode_by_id = {int(ep["env_idx"]): ep for ep in episode_info}
    offsets = build_sample_lookup(episode_info)

    episodes_dir = Path(args.episodes_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    kept_samples: List[Dict[str, Any]] = []
    svg_names: List[str] = []
    coords: List[Tuple[int, int]] = []
    skipped = 0
    for i, s in enumerate(samples):
        ep_id = int(s["episode_id"])
        ep = episode_by_id.get(ep_id)
        if ep is None:
            print(f"[skip] sample {i}: no episode_info for episode {ep_id}")
            skipped += 1
            continue

        episode_file = episodes_dir / (ep.get("episode_file") or f"ep_{ep_id:04d}.npz")
        if not episode_file.exists():
            print(f"[skip] sample {i}: missing episode file {episode_file}")
            skipped += 1
            continue

        try:
            t, a = sample_coords(
                int(s["sample_index"]),
                ep_id,
                offsets,
                int(ep["num_agents"]),
            )
        except ValueError as exc:
            print(f"[skip] sample {i}: {exc}")
            skipped += 1
            continue

        svg = render_sample_svg(s, ep, episode_file, focal_t=t, focal_agent=a)
        name = f"sample_{i:03d}_ep{ep_id}_t{t}_a{a}.svg"
        (out / name).write_text(svg)
        kept_samples.append(s)
        svg_names.append(name)
        coords.append((t, a))

    (out / "index.html").write_text(render_index_html(kept_samples, svg_names, coords))
    print(
        f"Wrote {len(svg_names)} SVGs + index.html to {out}"
        + (f" (skipped {skipped})" if skipped else "")
    )


if __name__ == "__main__":
    main()
