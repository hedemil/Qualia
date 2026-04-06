"""Validate noise augmentation on real LeRobot dataset data.

Runs three checks:
1. Statistical verification — mean ≈ 0, std ≈ noise_std
2. Boundary check — flag values that exceed dataset min/max
3. Trajectory visualization — plot original vs augmented per joint
"""

import argparse

import numpy as np
import torch

from augmentations.action_noise import ActionNoiseAugmentation


def load_episode_data(dataset_name, episode_idx):
    """Load all action/state frames for one episode from a LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading dataset: {dataset_name}")
    ds = LeRobotDataset(dataset_name)

    ep = ds.meta.episodes[episode_idx]
    from_idx, to_idx = ep["dataset_from_index"], ep["dataset_to_index"]
    num_frames = to_idx - from_idx
    print(f"Episode {episode_idx}: {num_frames} frames")

    actions = []
    states = []
    for i in range(from_idx, to_idx):
        frame = ds[i]
        if "action" in frame:
            actions.append(frame["action"].numpy())
        if "observation.state" in frame:
            states.append(frame["observation.state"].numpy())

    # Extract stats
    stats = {}
    for key in ["action", "observation.state"]:
        if key in ds.meta.stats:
            to_np = lambda v: v.numpy() if hasattr(v, "numpy") else np.asarray(v)
            stats[key] = {
                "min": to_np(ds.meta.stats[key]["min"]),
                "max": to_np(ds.meta.stats[key]["max"]),
                "std": to_np(ds.meta.stats[key]["std"]),
            }

    return np.array(actions), np.array(states), stats


def verify_statistics(original, augmented, expected_std, label):
    """Check that noise delta has mean ≈ 0 and std ≈ expected_std."""
    delta = augmented - original
    mean = np.mean(delta)
    std = np.std(delta)
    print(f"\n  [{label}] Statistical Verification:")
    print(f"    Mean of noise:  {mean:.6f}  (expected ≈ 0)")
    print(f"    Std of noise:   {std:.6f}  (expected ≈ {expected_std})")
    mean_ok = abs(mean) < 3 * expected_std / np.sqrt(delta.size)
    std_ok = abs(std - expected_std) < 0.3 * expected_std
    print(f"    Mean check: {'PASS' if mean_ok else 'FAIL'}")
    print(f"    Std check:  {'PASS' if std_ok else 'FAIL'}")
    return mean_ok and std_ok


def verify_boundaries(augmented, stats, label):
    """Check how many augmented values fall outside dataset min/max."""
    mins = stats["min"]
    maxs = stats["max"]
    below = augmented < mins
    above = augmented > maxs
    oob = below | above
    total = oob.size
    n_oob = np.sum(oob)
    pct = 100 * n_oob / total
    print(f"\n  [{label}] Boundary Check:")
    print(f"    Total values: {total}")
    print(f"    Out of bounds: {n_oob} ({pct:.2f}%)")
    if n_oob > 0:
        below_dims = np.any(below, axis=0)
        above_dims = np.any(above, axis=0)
        for d in np.where(below_dims)[0]:
            worst = np.min(augmented[:, d])
            print(f"    Dim {d}: min violation (value {worst:.6f} < limit {mins[d]:.6f})")
        for d in np.where(above_dims)[0]:
            worst = np.max(augmented[:, d])
            print(f"    Dim {d}: max violation (value {worst:.6f} > limit {maxs[d]:.6f})")
    return n_oob == 0


def plot_trajectories(original, augmented, label, filename):
    """Plot original vs augmented trajectory for each joint dimension."""
    import matplotlib.pyplot as plt

    n_dims = original.shape[1]
    cols = min(n_dims, 4)
    rows = (n_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    fig.suptitle(f"{label} — Original vs Augmented Trajectory", fontsize=14)

    for d in range(n_dims):
        ax = axes[d // cols][d % cols]
        ax.plot(original[:, d], label="Original", linewidth=1.5)
        ax.plot(augmented[:, d], label="Augmented", alpha=0.7, linestyle="--", linewidth=1)
        ax.set_title(f"Dim {d}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        if d == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for d in range(n_dims, rows * cols):
        axes[d // cols][d % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\n  Plot saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Validate noise augmentation on real data")
    parser.add_argument("--dataset", default="lerobot/aloha_static_cups_open", help="LeRobot dataset repo_id")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to validate")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Noise standard deviation")
    args = parser.parse_args()

    actions, states, stats = load_episode_data(args.dataset, args.episode)

    # Apply noise WITHOUT clipping (to see raw boundary violations)
    aug_no_clip = ActionNoiseAugmentation(noise_std=args.noise_std)
    aug_actions_raw = np.array([
        aug_no_clip.apply_frame(
            {"action": a.copy(), "observation.state": s.copy()}, {}
        )["action"]
        for a, s in zip(actions, states)
    ])
    aug_states_raw = np.array([
        aug_no_clip.apply_frame(
            {"action": a.copy(), "observation.state": s.copy()}, {}
        )["observation.state"]
        for a, s in zip(actions, states)
    ])

    # Apply noise WITH clipping
    aug_clipped = ActionNoiseAugmentation(noise_std=args.noise_std)
    aug_clipped.prepare([], stats=stats)
    aug_actions_clipped = np.array([
        aug_clipped.apply_frame(
            {"action": a.copy(), "observation.state": s.copy()}, {}
        )["action"]
        for a, s in zip(actions, states)
    ])
    aug_states_clipped = np.array([
        aug_clipped.apply_frame(
            {"action": a.copy(), "observation.state": s.copy()}, {}
        )["observation.state"]
        for a, s in zip(actions, states)
    ])

    print("=" * 60)
    print("NOISE AUGMENTATION VALIDATION")
    print(f"Dataset: {args.dataset}, Episode: {args.episode}, noise_std: {args.noise_std}")
    print("=" * 60)

    all_pass = True

    # 1. Statistical verification (use raw/unclipped for accurate stats)
    all_pass &= verify_statistics(actions, aug_actions_raw, args.noise_std, "action")
    if states.size > 0:
        all_pass &= verify_statistics(states, aug_states_raw, args.noise_std, "observation.state")

    # 2. Boundary check — raw (no clipping)
    print("\n--- Without clipping ---")
    if "action" in stats:
        verify_boundaries(aug_actions_raw, stats["action"], "action (raw)")
    if "observation.state" in stats:
        verify_boundaries(aug_states_raw, stats["observation.state"], "observation.state (raw)")

    # 3. Boundary check — clipped
    print("\n--- With clipping ---")
    if "action" in stats:
        all_pass &= verify_boundaries(aug_actions_clipped, stats["action"], "action (clipped)")
    if "observation.state" in stats:
        all_pass &= verify_boundaries(aug_states_clipped, stats["observation.state"], "observation.state (clipped)")

    # 4. Trajectory plots
    if actions.size > 0:
        plot_trajectories(actions, aug_actions_clipped, "Action", "noise_validation_action.png")
    if states.size > 0:
        plot_trajectories(states, aug_states_clipped, "Observation State", "noise_validation_state.png")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
