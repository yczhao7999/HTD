"""Frozen downstream dual-prototype score generation."""

import json
import os
import random

import numpy as np
import torch
from scipy import io

from datasets import (
    load_prior,
    normalize_scene,
    prepare_spectra,
    scene_array,
)
from model import build_encoder
from prototype_scoring import (
    normalized_mean,
    robust_dual_evidence_score,
    spherical_kmeans_background,
    target_prototype,
)


DATASETS = [
    ("cri", "data/cri/cri.mat", "data/cri/tgt_cri.mat"),
    ("Sandiego", "data/Sandiego/sandiego.mat", "data/Sandiego/tgt_sandiego_2.mat"),
    ("segundo", "data/segundo/data_segundo.mat", "data/segundo/tgt_segundo.mat"),
    (
        "abu-urban-1",
        "data/abu-urban-1/data_abu-urban-1.mat",
        "data/abu-urban-1/tgt_abu-urban-1.mat",
    ),
    (
        "abu-urban-2",
        "data/abu-urban-2/data_abu-urban-2.mat",
        "data/abu-urban-2/tgt_abu-urban-2.mat",
    ),
    (
        "abu-urban-3",
        "data/abu-urban-3/data_abu-urban-3.mat",
        "data/abu-urban-3/tgt_abu-urban-3.mat",
    ),
    (
        "abu-urban-4",
        "data/abu-urban-4/data_abu-urban-4.mat",
        "data/abu-urban-4/tgt_abu-urban-4.mat",
    ),
    (
        "changan-3-t3",
        "data/changan-3-t3/data_changan-3-t3.mat",
        "data/changan-3-t3/tgt_changan-3-t3.mat",
    ),
    (
        "changan-4-t4",
        "data/changan-4-t4/data_changan-4-t4.mat",
        "data/changan-4-t4/tgt_changan-4-t4.mat",
    ),
    (
        "salinas-t1",
        "data/salinas-t1/data_salinas-t1.mat",
        "data/salinas-t1/tgt_salinas-t1.mat",
    ),
    (
        "sandiego-2",
        "data/sandiego-2/data_sandiego-2.mat",
        "data/sandiego-2/tgt_sandiego-2.mat",
    ),
]

LEGACY_SCORE = "scaled_angular_relative_dual_evidence_v3"
NORMALIZED_SQUARE_SCORE = "normalized_square_a6_v1"
NORMALIZED_CUBE_SCORE = "normalized_cube_a6_v1"
SCORE_FORMULA = os.getenv("SCORE_FORMULA", NORMALIZED_CUBE_SCORE)
if SCORE_FORMULA not in {
    LEGACY_SCORE,
    NORMALIZED_SQUARE_SCORE,
    NORMALIZED_CUBE_SCORE,
}:
    raise ValueError(f"unknown SCORE_FORMULA: {SCORE_FORMULA}")
METHOD_NAME = os.getenv(
    "METHOD_NAME", "hfpn_v2_cwcaglobal_trainaug_lite_nodrop_shot1_priorraw_normcube_seed3327"
)
DEFAULT_CHECKPOINT = "checkpoints/hfpn_v2_cwcaglobal_trainaug_lite_nodrop_shot1_seed3327_fixedbudget_ema.pt"
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "4096"))
BACKGROUND_FRACTION = float(os.getenv("BACKGROUND_FRACTION", "0.90"))
BACKGROUND_TRIM_FRACTION = float(os.getenv("BACKGROUND_TRIM_FRACTION", "0.10"))
BACKGROUND_PROTOTYPES = int(os.getenv("BACKGROUND_PROTOTYPES", "6"))
BACKGROUND_CLUSTER_ITERATIONS = int(os.getenv("BACKGROUND_CLUSTER_ITERATIONS", "10"))
PROTOTYPE_MIN_SCALE = float(os.getenv("PROTOTYPE_MIN_SCALE", "0.05"))
PROTOTYPE_SCALE_QUANTILE = float(os.getenv("PROTOTYPE_SCALE_QUANTILE", "0.5"))
SOFTMIN_TEMPERATURE = float(os.getenv("SOFTMIN_TEMPERATURE", "0.1"))
HARDNESS_TEMPERATURE = float(os.getenv("HARDNESS_TEMPERATURE", "0.2"))
MODE = "frozen"

TARGET_ONLY = os.getenv("TARGET_ONLY", "0") == "1"


def embed_with_frequency_in_batches(model, spectra, device):
    main_embeddings = []
    frequency_embeddings = []
    gate_weights = []
    with torch.no_grad():
        for start in range(0, len(spectra), EMBED_BATCH_SIZE):
            batch = spectra[start : start + EMBED_BATCH_SIZE].to(device)
            main_embed, freq_embed, gate_weight = model.embed_with_frequency(batch)
            main_embeddings.append(main_embed)
            frequency_embeddings.append(freq_embed)
            gate_weights.append(gate_weight)
    return (
        torch.cat(main_embeddings, dim=0),
        torch.cat(frequency_embeddings, dim=0),
        torch.cat(gate_weights, dim=0),
    )


def background_prototype_set(embeddings):
    return spherical_kmeans_background(
        embeddings,
        prototype_count=BACKGROUND_PROTOTYPES,
        iterations=BACKGROUND_CLUSTER_ITERATIONS,
        trim_fraction=BACKGROUND_TRIM_FRACTION,
        minimum_scale=PROTOTYPE_MIN_SCALE,
        scale_quantile=PROTOTYPE_SCALE_QUANTILE,
    )


def score_prototypes(
    embeddings,
    target_center,
    target_scale,
    background_centers,
    background_scales,
):
    return robust_dual_evidence_score(
        embeddings,
        target_center,
        target_scale,
        background_centers,
        background_scales,
        SOFTMIN_TEMPERATURE,
        HARDNESS_TEMPERATURE,
    )


def finalize_detection_score(score, score_formula):
    """Apply the selected post-processing to the raw A6 detection score."""
    score = np.asarray(score, dtype=np.float32)
    if score_formula == LEGACY_SCORE:
        return score
    if score_formula == NORMALIZED_SQUARE_SCORE:
        minimum = float(np.min(score))
        maximum = float(np.max(score))
        normalized = (score - minimum) / (maximum - minimum + 1e-12)
        return np.square(normalized).astype(np.float32)
    if score_formula == NORMALIZED_CUBE_SCORE:
        minimum = float(np.min(score))
        maximum = float(np.max(score))
        normalized = (score - minimum) / (maximum - minimum + 1e-12)
        return np.power(normalized, 3).astype(np.float32)
    raise ValueError(f"unknown score formula: {score_formula}")


def apply_checkpoint_scoring_config(checkpoint):
    """Use training-time score parameters unless the environment overrides them."""
    train_config = checkpoint.get("train_config", {})
    mappings = (
        ("PROTOTYPE_MIN_SCALE", "prototype_min_scale", "PROTOTYPE_MIN_SCALE"),
        (
            "PROTOTYPE_SCALE_QUANTILE",
            "prototype_scale_quantile",
            "PROTOTYPE_SCALE_QUANTILE",
        ),
        ("SOFTMIN_TEMPERATURE", "softmin_temperature", "SOFTMIN_TEMPERATURE"),
        ("HARDNESS_TEMPERATURE", "hardness_temperature", "HARDNESS_TEMPERATURE"),
    )
    for environment_name, config_name, global_name in mappings:
        if environment_name not in os.environ and config_name in train_config:
            globals()[global_name] = float(train_config[config_name])


def save_detection_artifacts(
    dataset_name,
    score,
    target_similarity,
    background_similarity,
    target_distance,
    background_distance,
    background_centers,
    background_scales,
    background_counts,
    background_weights,
    reliable_indices,
    height,
    width,
    method_name,
    mode,
    seed,
    metadata,
    score_logit=None,
    base_score=None,
    frequency_outputs=None,
    gate_weights=None,
    target_only=False,
):
    os.makedirs("result", exist_ok=True)
    stem = f"{dataset_name}_{method_name}_{mode}_seed{seed}"
    score_map = score.reshape(height, width).astype(np.float32)
    np.save(f"result/{stem}_score.npy", score_map)
    components = {
        "target_similarity": target_similarity.reshape(height, width).astype(
            np.float32
        ),
        "background_similarity": background_similarity.reshape(height, width).astype(
            np.float32
        ),
        "target_distance": target_distance.reshape(height, width).astype(np.float32),
        "background_distance": background_distance.reshape(height, width).astype(
            np.float32
        ),
        "background_prototypes": np.asarray(background_centers, dtype=np.float32),
        "background_scales": np.asarray(background_scales, dtype=np.float32),
        "background_counts": np.asarray(background_counts, dtype=np.float32),
        "background_weights": np.asarray(background_weights, dtype=np.float32),
        "reliable_background_indices": np.asarray(reliable_indices, dtype=np.int64),
        "target_only": bool(target_only),
    }
    if score_logit is not None:
        components["score_logit"] = score_logit.reshape(height, width).astype(
            np.float32
        )
    if frequency_outputs and "score" in frequency_outputs:
        components["base_hfpn_score"] = (
            (score if base_score is None else base_score)
            .reshape(height, width)
            .astype(np.float32)
        )
        components["frequency_branch_score"] = (
            frequency_outputs["score"].reshape(height, width).astype(np.float32)
        )
        components["gate_weight"] = gate_weights.reshape(height, width).astype(
            np.float32
        )
    np.savez_compressed(f"result/{stem}_components.npz", **components)
    io.savemat(f"result/{stem}_score.mat", {"detect": score_map})
    with open(f"result/{stem}_config.json", "w", encoding="utf8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=True)
    try:
        import matplotlib.pyplot as plt

        minimum = float(np.min(score_map))
        maximum = float(np.max(score_map))
        display_map = (score_map - minimum) / (maximum - minimum + 1e-12)
        plt.imsave(
            f"result/{stem}_map.png", display_map, cmap="gray", vmin=0.0, vmax=1.0
        )
    except ImportError:
        print("  warning=matplotlib_unavailable_png_not_saved")
    return stem


def main():
    seed = int(os.getenv("SEED", "3327"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(DEVICE)
    checkpoint_path = os.getenv("CKPT_PATH", DEFAULT_CHECKPOINT)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    apply_checkpoint_scoring_config(checkpoint)
    score_formula = SCORE_FORMULA
    method_name = os.getenv("METHOD_NAME", METHOD_NAME)
    selected_datasets = set(filter(None, os.getenv("DETECT_DATASETS", "").split(",")))
    print(
        f"device={device} checkpoint={checkpoint_path} mode={MODE} "
        f"score_formula={score_formula} "
        f"ground_truth_access=disabled"
    )

    # Build model once before loop
    model = build_encoder(
        checkpoint["architecture"],
        checkpoint.get("bands") or 0,
        checkpoint["embedding_dim"],
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    for dataset_offset, (dataset_name, data_path, prior_path) in enumerate(DATASETS):
        if selected_datasets and dataset_name not in selected_datasets:
            continue
        if not all(os.path.exists(path) for path in (data_path, prior_path)):
            continue
        dataset_seed = seed + dataset_offset * 1009
        torch.manual_seed(dataset_seed)

        raw_cube, _ = scene_array(data_path)
        native_bands = int(raw_cube.shape[-1])
        cube, normalization_stats = normalize_scene(raw_cube, return_stats=True)
        cube = prepare_spectra(cube, checkpoint.get("bands"))
        height, width = cube.shape[:2]
        scene = torch.from_numpy(cube.reshape(-1, cube.shape[-1]))
        prior = torch.from_numpy(
            load_prior(
                prior_path,
                checkpoint.get("bands"),
                normalization_stats=normalization_stats,
            )
        )
        if prior.shape[-1] != scene.shape[-1]:
            raise ValueError(
                f"{dataset_name}: scene/prior band mismatch: "
                f"{scene.shape[-1]} != {prior.shape[-1]}"
            )
        with torch.no_grad():
            scene_embeddings, scene_frequency_embeddings, scene_gate_weights = (
                embed_with_frequency_in_batches(model, scene, device)
            )
            target_embeddings, target_frequency_embeddings, _ = (
                embed_with_frequency_in_batches(model, prior.to(device), device)
            )
        model.eval()
        with torch.no_grad():
            target_center, target_scale = target_prototype(
                target_embeddings,
                minimum_scale=PROTOTYPE_MIN_SCALE,
                quantile=PROTOTYPE_SCALE_QUANTILE,
            )
            if TARGET_ONLY:
                target_similarity = scene_embeddings @ target_center
                target_distance = (1.0 - target_similarity).clamp_min(
                    0.0
                ) / target_scale.clamp_min(1e-6)
                background_similarity = torch.zeros_like(target_similarity)
                background_distance = torch.zeros_like(target_distance)
                background_centers = torch.empty(
                    (0, target_center.shape[0]), device=device
                )
                background_scales = torch.empty((0,), device=device)
                background_counts = torch.empty((0,), device=device)
                background_weights = torch.empty((0,), device=device)
                raw_score = target_similarity.cpu().numpy()
                score = finalize_detection_score(raw_score, score_formula)
                reliable_indices = np.array([], dtype=np.int64)
                frequency_outputs = {}
            else:
                initial_target_prototype = normalized_mean(target_embeddings)
                initial_similarities = scene_embeddings @ initial_target_prototype
                candidate_count = max(
                    1,
                    min(
                        len(initial_similarities),
                        int(round(len(initial_similarities) * BACKGROUND_FRACTION)),
                    ),
                )
                reliable_indices = (
                    torch.argsort(initial_similarities)[:candidate_count].cpu().numpy()
                )
                reliable_tensor = torch.as_tensor(
                    reliable_indices, dtype=torch.long, device=device
                )
                background_centers, background_scales, background_counts = (
                    background_prototype_set(scene_embeddings[reliable_tensor])
                )
                outputs = score_prototypes(
                    scene_embeddings,
                    target_center,
                    target_scale,
                    background_centers,
                    background_scales,
                )
                raw_score = outputs["score"].cpu().numpy()
                score = finalize_detection_score(raw_score, score_formula)
                target_similarity = outputs["target_similarity"]
                background_similarity = outputs["background_similarity"]
                target_distance = outputs["target_distance"]
                background_distance = outputs["background_distance"]
                background_weights = outputs["background_weights"]

                frequency_target_center, frequency_target_scale = target_prototype(
                    target_frequency_embeddings,
                    minimum_scale=PROTOTYPE_MIN_SCALE,
                    quantile=PROTOTYPE_SCALE_QUANTILE,
                )
                frequency_background_centers, frequency_background_scales, _ = (
                    background_prototype_set(
                        scene_frequency_embeddings[reliable_tensor]
                    )
                )
                frequency_outputs = score_prototypes(
                    scene_frequency_embeddings,
                    frequency_target_center,
                    frequency_target_scale,
                    frequency_background_centers,
                    frequency_background_scales,
                )
        target_similarity_np = target_similarity.cpu().numpy()
        background_similarity_np = background_similarity.cpu().numpy()
        target_distance_np = target_distance.cpu().numpy()
        background_distance_np = background_distance.cpu().numpy()
        background_weights_np = background_weights.cpu().numpy()
        background_centers_np = background_centers.cpu().numpy()
        background_scales_np = background_scales.cpu().numpy()
        background_counts_np = background_counts.cpu().numpy()

        metadata = {
            "dataset": dataset_name,
            "source_checkpoint": checkpoint_path,
            "method_name": method_name,
            "source_method_name": checkpoint.get("method_name"),
            "seed": seed,
            "dataset_seed": dataset_seed,
            "setting": "unlabeled_transductive",
            "ground_truth_used": False,
            "mode": MODE,
            "target_only": bool(TARGET_ONLY),
            "target_prior_count": int(prior.shape[0]),
            "target_prior_augmentation": "disabled",
            "native_bands": native_bands,
            "model_input_bands": int(scene.shape[-1]),
            "background_fraction": BACKGROUND_FRACTION,
            "background_selection": "global_lowest_target_similarity",
            "background_trim_fraction": BACKGROUND_TRIM_FRACTION,
            "background_prototypes": len(background_centers),
            "background_cluster_iterations": BACKGROUND_CLUSTER_ITERATIONS,
            "prototype_min_scale": PROTOTYPE_MIN_SCALE,
            "prototype_scale_quantile": PROTOTYPE_SCALE_QUANTILE,
            "softmin_temperature": SOFTMIN_TEMPERATURE,
            "hardness_temperature": HARDNESS_TEMPERATURE,
            "score_formula": score_formula,
            "prior_protocol": "provided_prior_may_be_ground_truth_derived",
            "normalization": "scene_global_minmax_shared_with_prior",
            "band_adaptation": (
                "native_length"
                if checkpoint.get("bands") is None
                else f"fixed_{checkpoint['bands']}_interpolation"
            ),
        }
        stem = save_detection_artifacts(
            dataset_name,
            score,
            target_similarity_np,
            background_similarity_np,
            target_distance_np,
            background_distance_np,
            background_centers_np,
            background_scales_np,
            background_counts_np,
            background_weights_np,
            reliable_indices,
            height,
            width,
            method_name,
            MODE,
            seed,
            metadata,
            raw_score,
            base_score=raw_score,
            frequency_outputs={
                key: value.cpu().numpy() for key, value in frequency_outputs.items()
            },
            gate_weights=scene_gate_weights.cpu().numpy(),
            target_only=bool(TARGET_ONLY),
        )
        print(f"  saved=result/{stem}_score.npy")


if __name__ == "__main__":
    main()
