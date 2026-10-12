"""Detection-aligned episodic pre-training for dual-prototype transfer."""

import copy

import os

import random


import numpy as np

import torch

import torch.nn.functional as F


from datasets import load_xuzhou

from model import build_encoder

from prototype_scoring import (
    BackgroundMemoryQueue,
    angular_scale,
    labeled_background_prototypes,
    normalized_mean,
    robust_dual_evidence_score,
    upper_tail,
)


LEGACY_SCORE = "scaled_angular_relative_dual_evidence_v3"

SCORE_FORMULA = os.getenv("SCORE_FORMULA", LEGACY_SCORE)

if SCORE_FORMULA != LEGACY_SCORE:

    raise ValueError(f"unknown SCORE_FORMULA: {SCORE_FORMULA}")

METHOD_NAME = os.getenv(
    "METHOD_NAME", "hfpn_v2_cwcaglobal_trainaug_lite_nodrop_shot1_seed3327"
)

SCNP_ROUTE = os.getenv("SCNP_ROUTE", "none")  # none | a | b

SCNP_K = int(os.getenv("SCNP_K", "3"))

SCNP_WEIGHT = float(os.getenv("SCNP_WEIGHT", "0.1"))

SCNP_MARGIN = float(os.getenv("SCNP_MARGIN", "0.05"))

TARGET_ONLY = os.getenv("TARGET_ONLY", "0") == "1"

AUGMENT_CONFIG = {
    "gain_std": 0.03,
    "smooth_noise_std": 0.01,
}


def spectral_augment(x, config=AUGMENT_CONFIG):
    """Apply the default gain and locally smoothed noise augmentation."""
    batch_size, bands = x.shape
    gain = 1.0 + torch.randn(batch_size, 1, device=x.device) * config["gain_std"]
    augmented = x * gain

    noise = torch.randn_like(augmented).unsqueeze(1)
    kernel_size = min(7, bands if bands % 2 else bands - 1)
    if kernel_size >= 3:
        noise = F.avg_pool1d(
            noise, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        ).squeeze(1)
        augmented = augmented + config["smooth_noise_std"] * noise

    return augmented.clamp(0.0, 1.0)


class EpisodicSampler:

    def __init__(self, spectra, labels, seed):

        self.spectra = spectra

        self.labels = labels

        self.rng = np.random.default_rng(seed)

        self.classes = np.unique(labels)

        self.class_indices = {
            class_id: np.flatnonzero(labels == class_id) for class_id in self.classes
        }

    def _sample(self, indices, count):

        if len(indices) < count:

            raise ValueError(
                f"strict support/query separation requires {count} unique samples, "
                f"but only {len(indices)} are available"
            )

        return self.rng.choice(indices, size=count, replace=False)

    def sample(
        self,
        target_class,
        target_shots,
        target_queries,
        background_support,
        background_candidates,
        background_classes,
    ):

        target_pool = self.class_indices[target_class]

        target_indices = self._sample(target_pool, target_shots + target_queries)

        target_support = target_indices[:target_shots]

        target_query = target_indices[target_shots:]

        available = self.classes[self.classes != target_class]

        class_count = min(background_classes, len(available))

        selected = self.rng.choice(available, size=class_count, replace=False)

        support_parts = []

        candidate_parts = []

        support_each = max(1, int(np.ceil(background_support / class_count)))

        candidate_each = max(1, int(np.ceil(background_candidates / class_count)))

        for class_id in selected:

            pool = self.class_indices[class_id]

            chosen = self._sample(pool, support_each + candidate_each)

            support_parts.append(chosen[:support_each])

            candidate_parts.append(chosen[support_each:])

        support_indices = np.concatenate(support_parts)[:background_support]

        candidate_indices = np.concatenate(candidate_parts)[:background_candidates]

        return {
            "target_class": int(target_class),
            "target_support": self.spectra[target_support],
            "target_query": self.spectra[target_query],
            "background_support": self.spectra[support_indices],
            "background_support_labels": self.labels[support_indices],
            "background_candidates": self.spectra[candidate_indices],
            "background_candidate_labels": self.labels[candidate_indices],
        }


def select_background_queries(
    candidate_embeddings, target_prototype, count, hard_ratio
):

    count = min(count, len(candidate_embeddings))

    hard_count = min(count, int(round(count * hard_ratio)))

    similarities = candidate_embeddings @ target_prototype

    hard_indices = torch.argsort(similarities, descending=True)[:hard_count]

    available = torch.ones(
        len(candidate_embeddings), dtype=torch.bool, device=similarities.device
    )

    available[hard_indices] = False

    random_count = count - hard_count

    random_pool = torch.nonzero(available, as_tuple=False).flatten()

    random_indices = random_pool[
        torch.randperm(len(random_pool), device=similarities.device)[:random_count]
    ]

    return candidate_embeddings[torch.cat((hard_indices, random_indices))]


def _same_class_knn_indices(embeddings, labels, k):
    """Return [N, k] indices of k nearest same-class embeddings (self included).



    embeddings: [N, D] L2-normalized.

    labels: [N] binary 0/1.

    """

    n = embeddings.shape[0]

    sim = embeddings @ embeddings.T  # [N, N]

    same = labels[:, None] == labels[None, :]

    sim = torch.where(same, sim, torch.full_like(sim, -1e9))

    k_eff = min(k, int(same.sum(dim=1).min().item()))

    _, indices = torch.topk(sim, k_eff, dim=1)

    return indices


def scnp_propagate_scores(
    target_scores,
    background_scores,
    target_embeddings,
    background_embeddings,
    k,
):
    """Direct 1-D adaptation of Same Class Neighbor Penalization.



    Each target query is replaced by the minimum score among its k nearest

    target queries; each background query is replaced by the maximum score

    among its k nearest background queries. Only affects training logits.

    """

    target_labels = torch.ones(
        target_embeddings.shape[0], device=target_embeddings.device
    )

    background_labels = torch.zeros(
        background_embeddings.shape[0], device=background_embeddings.device
    )

    target_indices = _same_class_knn_indices(target_embeddings, target_labels, k)

    background_indices = _same_class_knn_indices(
        background_embeddings, background_labels, k
    )

    target_local = target_scores[target_indices].min(dim=1).values

    background_local = background_scores[background_indices].max(dim=1).values

    return target_local, background_local


def dual_topology_worst_neighbor_loss(
    target_scores,
    background_scores,
    target_embeddings,
    background_embeddings,
    k,
    margin,
):
    """Task-specific SCNP-inspired local worst-neighbor ranking term.



    For every target query, compare its worst local target score against the

    hardest nearby background score; symmetrically for background queries.

    This preserves SCNP's local topology pressure while acting on the

    dual-prototype detection score used downstream.

    """

    target_labels = torch.ones(
        target_embeddings.shape[0], device=target_embeddings.device
    )

    background_labels = torch.zeros(
        background_embeddings.shape[0], device=background_embeddings.device
    )

    target_indices = _same_class_knn_indices(target_embeddings, target_labels, k)

    background_indices = _same_class_knn_indices(
        background_embeddings, background_labels, k
    )

    target_worst = target_scores[target_indices].min(dim=1).values

    background_worst = background_scores[background_indices].max(dim=1).values

    cross_sim = target_embeddings @ background_embeddings.T  # [T, B]

    nearest_background = background_scores[torch.argmax(cross_sim, dim=1)]

    nearest_target = target_scores[torch.argmax(cross_sim, dim=0)]

    target_margin_loss = F.relu(margin - target_worst + nearest_background).mean()

    background_margin_loss = F.relu(margin + background_worst - nearest_target).mean()

    return target_margin_loss + background_margin_loss


def episodic_objective(
    model,
    episode,
    device,
    hard_ratio,
    background_queries,
    temperature,
    ranking_margin,
    ranking_weight,
    separation_similarity,
    separation_weight,
    consistency_weight,
    memory_queue,
    minimum_scale,
    scale_quantile,
    softmin_temperature,
    hardness_temperature,
    tail_fraction,
    tail_minimum,
):

    tensors = {
        name: torch.from_numpy(values).to(device)
        for name, values in episode.items()
        if name
        not in {
            "target_class",
            "background_support_labels",
            "background_candidate_labels",
        }
    }

    support_labels = torch.from_numpy(episode["background_support_labels"]).to(device)

    candidate_labels = torch.from_numpy(episode["background_candidate_labels"]).to(
        device
    )

    target_support = model.embed(tensors["target_support"])

    target_query = model.embed(tensors["target_query"])

    background_support = model.embed(tensors["background_support"])

    background_candidates = model.embed(tensors["background_candidates"])

    target_prototype = normalized_mean(target_support)

    augmented_support = model.embed(spectral_augment(tensors["target_support"]))

    target_scale = angular_scale(
        torch.cat((target_support, augmented_support), dim=0),
        target_prototype,
        minimum_scale,
        scale_quantile,
    )

    background_prototypes, background_scales = labeled_background_prototypes(
        background_support, support_labels, minimum_scale, scale_quantile
    )

    background_query = select_background_queries(
        background_candidates, target_prototype, background_queries, hard_ratio
    )

    def score_embeddings(embeddings):

        return robust_dual_evidence_score(
            embeddings,
            target_prototype,
            target_scale,
            background_prototypes,
            background_scales,
            softmin_temperature,
            hardness_temperature,
        )

    target_outputs = score_embeddings(target_query)

    background_outputs = score_embeddings(background_query)

    target_scores = target_outputs["score"]

    background_scores = background_outputs["score"]

    if SCNP_ROUTE == "a":

        target_scores, background_scores = scnp_propagate_scores(
            target_scores,
            background_scores,
            target_query,
            background_query,
            SCNP_K,
        )

    target_ranking_scores = target_scores

    background_ranking_scores = background_scores

    logits = torch.cat((target_scores, background_scores)) / temperature

    labels = torch.cat(
        (torch.ones_like(target_scores), torch.zeros_like(background_scores))
    )

    detection_loss = F.binary_cross_entropy_with_logits(logits, labels)

    tail_candidates = [background_ranking_scores]

    memory_embeddings = memory_queue.negatives_for(episode["target_class"])

    if memory_embeddings is not None and len(memory_embeddings):

        memory_outputs = score_embeddings(memory_embeddings)

        tail_candidates.append(memory_outputs["score"])

    hard_negatives = upper_tail(torch.cat(tail_candidates), tail_fraction, tail_minimum)

    ranking_loss = F.softplus(
        ranking_margin - target_ranking_scores[:, None] + hard_negatives[None, :]
    ).mean()

    prototype_similarity = (background_prototypes @ target_prototype).max()

    separation_loss = F.relu(prototype_similarity - separation_similarity)

    consistency_loss = 1.0 - (target_support * augmented_support).sum(dim=-1).mean()

    scnp_loss = torch.tensor(0.0, device=device)

    if SCNP_ROUTE == "b":

        scnp_loss = dual_topology_worst_neighbor_loss(
            target_scores,
            background_scores,
            target_query,
            background_query,
            SCNP_K,
            SCNP_MARGIN,
        )

    total = (
        detection_loss
        + ranking_weight * ranking_loss
        + separation_weight * separation_loss
        + consistency_weight * consistency_loss
        + SCNP_WEIGHT * scnp_loss
    )

    diagnostics = {
        "det": detection_loss.detach(),
        "rank": ranking_loss.detach(),
        "sep": separation_loss.detach(),
        "cons": consistency_loss.detach(),
        "target_score": target_scores.mean().detach(),
        "background_score": background_scores.mean().detach(),
        "prototype_similarity": prototype_similarity.detach(),
        "target_scale": target_scale.detach(),
        "background_scale": background_scales.mean().detach(),
        "tail_score": hard_negatives.mean().detach(),
        "memory_size": torch.as_tensor(float(len(memory_queue)), device=device),
    }

    if SCNP_ROUTE == "b":

        diagnostics["scnp"] = scnp_loss.detach()

    memory_queue.enqueue(background_candidates, candidate_labels)

    return total, diagnostics


@torch.no_grad()
def update_ema(ema_state, model, decay):

    for name, value in model.state_dict().items():

        if value.is_floating_point():

            ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)

        else:

            ema_state[name].copy_(value)


def main():

    seed = int(os.getenv("SEED", "3327"))

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )

    bands_value = os.getenv("BANDS", "256")

    bands = None if bands_value.lower() == "native" else int(bands_value)

    embedding_dim = int(os.getenv("EMBEDDING_DIM", "128"))

    epochs = int(os.getenv("EPOCHS", "20"))

    episodes_per_epoch = int(os.getenv("EPISODES_PER_EPOCH", "100"))

    target_queries = int(os.getenv("TARGET_QUERIES", "24"))

    background_support = int(os.getenv("BACKGROUND_SUPPORT", "32"))

    background_queries = int(os.getenv("BACKGROUND_QUERIES", "48"))

    candidate_multiplier = int(os.getenv("BACKGROUND_CANDIDATE_MULTIPLIER", "3"))

    background_classes = int(os.getenv("BACKGROUND_CLASSES", "6"))

    shots = tuple(
        sorted(
            {
                int(value.strip())
                for value in os.getenv("TARGET_SHOTS", "1").split(",")
                if value.strip()
            }
        )
    )
    if not shots or any(value < 1 for value in shots):
        raise ValueError("TARGET_SHOTS must contain positive integers")

    temperature = float(os.getenv("TEMPERATURE", "0.1"))

    ranking_margin = float(os.getenv("RANKING_MARGIN", "0.1"))

    ranking_weight = float(os.getenv("RANKING_WEIGHT", "0.0"))

    separation_similarity = float(os.getenv("SEPARATION_SIMILARITY", "0.5"))

    separation_weight = float(os.getenv("SEPARATION_WEIGHT", "0.0"))

    consistency_weight = float(os.getenv("CONSISTENCY_WEIGHT", "0.0"))

    hard_background_ratio = float(os.getenv("HARD_BACKGROUND_RATIO", "0.4"))

    hard_warmup_epochs = int(os.getenv("HARD_WARMUP_EPOCHS", "10"))

    minimum_scale = float(os.getenv("PROTOTYPE_MIN_SCALE", "0.05"))

    scale_quantile = float(os.getenv("PROTOTYPE_SCALE_QUANTILE", "0.5"))

    softmin_temperature = float(os.getenv("SOFTMIN_TEMPERATURE", "0.1"))

    hardness_temperature = float(os.getenv("HARDNESS_TEMPERATURE", "0.2"))

    memory_capacity = int(os.getenv("BACKGROUND_MEMORY_CAPACITY", "4096"))

    tail_fraction = float(os.getenv("TAIL_BACKGROUND_FRACTION", "0.01"))

    tail_minimum = int(os.getenv("TAIL_BACKGROUND_MINIMUM", "16"))

    ema_decay = float(os.getenv("EMA_DECAY", "0.995"))

    architecture = os.getenv("ENCODER_ARCH", "hfpn_v2_cwcaglobal")

    target_only = os.getenv("TARGET_ONLY", "0") == "1"

    checkpoint_path = os.getenv(
        "CHECKPOINT_PATH", f"checkpoints/{METHOD_NAME}_fixedbudget_ema.pt"
    )

    spectra, labels = load_xuzhou(bands=bands)

    num_classes = int(labels.max() + 1)

    sampler = EpisodicSampler(spectra, labels, seed)

    model = build_encoder(architecture, bands, embedding_dim, num_classes).to(device)

    optimized_parameters = list(model.parameters())

    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=float(os.getenv("LEARNING_RATE", "0.001")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0001")),
    )

    ema_state = copy.deepcopy(model.state_dict())

    memory_queue = BackgroundMemoryQueue(memory_capacity)

    print(
        f"device={device} method={METHOD_NAME} architecture={architecture} "
        f"score_formula={SCORE_FORMULA} "
        f"epochs={epochs} episodes_per_epoch={episodes_per_epoch} checkpoint={checkpoint_path}"
    )

    model.train()

    class_schedule = []

    schedule_position = 0

    for epoch in range(epochs):

        hard_progress = min(1.0, epoch / max(1, hard_warmup_epochs))

        current_hard_ratio = hard_background_ratio * hard_progress

        totals = {"loss": 0.0}

        if SCNP_ROUTE == "b":

            totals["scnp"] = 0.0

        for _ in range(episodes_per_epoch):

            if schedule_position >= len(class_schedule):

                class_schedule = sampler.rng.permutation(sampler.classes).tolist()

                schedule_position = 0

            target_class = class_schedule[schedule_position]

            schedule_position += 1

            episode = sampler.sample(
                target_class=target_class,
                target_shots=int(sampler.rng.choice(shots)),
                target_queries=target_queries,
                background_support=background_support,
                background_candidates=background_queries * candidate_multiplier,
                background_classes=background_classes,
            )

            optimizer.zero_grad(set_to_none=True)

            if target_only:
                loss, diagnostics = episodic_objective_target_only(
                    model=model,
                    episode=episode,
                    device=device,
                    hard_ratio=current_hard_ratio,
                    background_queries=background_queries,
                    temperature=temperature,
                    consistency_weight=consistency_weight,
                    minimum_scale=minimum_scale,
                    scale_quantile=scale_quantile,
                )
            else:
                loss, diagnostics = episodic_objective(
                    model=model,
                    episode=episode,
                    device=device,
                    hard_ratio=current_hard_ratio,
                    background_queries=background_queries,
                    temperature=temperature,
                    ranking_margin=ranking_margin,
                    ranking_weight=ranking_weight,
                    separation_similarity=separation_similarity,
                    separation_weight=separation_weight,
                    consistency_weight=consistency_weight,
                    memory_queue=memory_queue,
                    minimum_scale=minimum_scale,
                    scale_quantile=scale_quantile,
                    softmin_temperature=softmin_temperature,
                    hardness_temperature=hardness_temperature,
                    tail_fraction=tail_fraction,
                    tail_minimum=tail_minimum,
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(optimized_parameters, max_norm=5.0)

            optimizer.step()

            update_ema(ema_state, model, ema_decay)

            totals["loss"] += float(loss.detach())

            for name, value in diagnostics.items():

                totals[name] = totals.get(name, 0.0) + float(value)

        summary = " ".join(
            f"{name}={value / episodes_per_epoch:.5f}" for name, value in totals.items()
        )

        print(
            f"epoch={epoch + 1}/{epochs} hard_ratio={current_hard_ratio:.3f} {summary}"
        )

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    torch.save(
        {
            "model": ema_state,
            "raw_model": model.state_dict(),
            "architecture": architecture,
            "method_name": METHOD_NAME,
            "bands": bands,
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
            "selection_rule": "fixed_budget_final_ema",
            "train_config": {
                "seed": seed,
                "target_only": target_only,
                "epochs": epochs,
                "episodes_per_epoch": episodes_per_epoch,
                "target_shots": shots,
                "target_queries": target_queries,
                "background_support": background_support,
                "background_queries": background_queries,
                "background_classes": background_classes,
                "temperature": temperature,
                "ranking_margin": ranking_margin,
                "ranking_weight": ranking_weight,
                "separation_similarity": separation_similarity,
                "separation_weight": separation_weight,
                "consistency_weight": consistency_weight,
                "hard_background_ratio": hard_background_ratio,
                "hard_warmup_epochs": hard_warmup_epochs,
                "prototype_min_scale": minimum_scale,
                "prototype_scale_quantile": scale_quantile,
                "softmin_temperature": softmin_temperature,
                "hardness_temperature": hardness_temperature,
                "background_memory_capacity": memory_capacity,
                "tail_background_fraction": tail_fraction,
                "tail_background_minimum": tail_minimum,
                "ema_decay": ema_decay,
                "augment_config": AUGMENT_CONFIG,
                "augmentation_profile": "lite_nodrop_gain_smooth",
            },
        },
        checkpoint_path,
    )

    print(f"saved={checkpoint_path}")


if __name__ == "__main__":

    main()
