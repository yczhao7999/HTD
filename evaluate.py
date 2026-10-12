"""Evaluate previously frozen detection scores against ground truth."""

import csv
import os

import numpy as np

from datasets import scene_array
from utils import detection_metrics


DATASETS = [
    ("cri", "data/cri/cri.mat", "data/cri/groundtruth.mat"),
    ("Sandiego", "data/Sandiego/sandiego.mat", "data/Sandiego/groundtruth.mat"),
    (
        "segundo",
        "data/segundo/data_segundo.mat",
        "data/segundo/groundtruth_segundo.mat",
    ),
    (
        "abu-urban-1",
        "data/abu-urban-1/data_abu-urban-1.mat",
        "data/abu-urban-1/groundtruth_abu-urban-1.mat",
    ),
    (
        "abu-urban-2",
        "data/abu-urban-2/data_abu-urban-2.mat",
        "data/abu-urban-2/groundtruth_abu-urban-2.mat",
    ),
    (
        "abu-urban-3",
        "data/abu-urban-3/data_abu-urban-3.mat",
        "data/abu-urban-3/groundtruth_abu-urban-3.mat",
    ),
    (
        "abu-urban-4",
        "data/abu-urban-4/data_abu-urban-4.mat",
        "data/abu-urban-4/groundtruth_abu-urban-4.mat",
    ),
    (
        "changan-3-t3",
        "data/changan-3-t3/data_changan-3-t3.mat",
        "data/changan-3-t3/groundtruth_changan-3-t3.mat",
    ),
    (
        "changan-4-t4",
        "data/changan-4-t4/data_changan-4-t4.mat",
        "data/changan-4-t4/groundtruth_changan-4-t4.mat",
    ),
    (
        "salinas-t1",
        "data/salinas-t1/data_salinas-t1.mat",
        "data/salinas-t1/groundtruth_salinas-t1.mat",
    ),
    (
        "sandiego-2",
        "data/sandiego-2/data_sandiego-2.mat",
        "data/sandiego-2/groundtruth_sandiego-2.mat",
    ),
]


def main():
    method_name = os.getenv(
        "METHOD_NAME", "hfpn_v2_cwcaglobal_trainaug_lite_nodrop_shot1_priorraw_normcube_seed3327"
    )
    mode = "frozen"
    seed = int(os.getenv("SEED", "3327"))
    selected_datasets = set(filter(None, os.getenv("DETECT_DATASETS", "").split(",")))
    rows = []
    for dataset_name, data_path, ground_truth_path in DATASETS:
        if selected_datasets and dataset_name not in selected_datasets:
            continue
        stem = f"{dataset_name}_{method_name}_{mode}_seed{seed}"
        score_path = f"result/{stem}_score.npy"
        if not all(
            os.path.exists(path) for path in (data_path, ground_truth_path, score_path)
        ):
            continue
        _, ground_truth = scene_array(data_path, ground_truth_path)
        score = np.load(score_path, allow_pickle=False)
        if score.shape != ground_truth.shape:
            raise ValueError(
                f"shape mismatch for {dataset_name}: score={score.shape}, "
                f"ground_truth={ground_truth.shape}"
            )
        branches = {"fusion": score}
        component_path = f"result/{stem}_components.npz"
        if os.path.exists(component_path):
            with np.load(component_path, allow_pickle=False) as components:
                if "target_only" in components and bool(components["target_only"]):
                    branches["target"] = components["target_similarity"]
                elif (
                    "target_distance" in components
                    and "background_distance" in components
                ):
                    branches["target"] = -components["target_distance"]
                    branches["background"] = components["background_distance"]
                elif (
                    "target_similarity" in components
                    and "background_similarity" in components
                ):
                    branches["target"] = components["target_similarity"]
                    branches["background"] = -components["background_similarity"]
        for branch, branch_score in branches.items():
            metrics = detection_metrics(ground_truth, branch_score)
            row = {
                "dataset": dataset_name,
                "method": method_name,
                "mode": mode,
                "branch": branch,
                "seed": seed,
                **metrics,
            }
            rows.append(row)
            print(row)
    if not rows:
        raise FileNotFoundError("no matching frozen score files were found")
    os.makedirs("result", exist_ok=True)
    metrics_path = os.getenv(
        "METRICS_PATH", f"result/{method_name}_{mode}_seed{seed}_metrics.txt"
    )
    with open(metrics_path, "w", newline="", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={metrics_path}")


if __name__ == "__main__":
    main()
