"""Shared prototype construction and robust dual-evidence scoring."""

import math

import torch
import torch.nn.functional as F


def normalized_mean(values):
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("values must be a non-empty [samples, features] tensor")
    return F.normalize(values.mean(dim=0), p=2, dim=0)


def angular_scale(values, center, minimum=0.05, quantile=0.5):
    """Estimate a robust within-prototype angular radius."""
    distances = (1.0 - values @ center).clamp_min(0.0)
    if len(distances) == 1:
        scale = distances[0]
    else:
        scale = torch.quantile(distances, quantile)
    return scale.clamp_min(minimum)


def target_prototype(values, minimum_scale=0.05, quantile=0.5):
    center = normalized_mean(values)
    scale = angular_scale(values, center, minimum_scale, quantile)
    return center, scale


def labeled_background_prototypes(
    embeddings,
    labels,
    minimum_scale=0.05,
    scale_quantile=0.5,
    return_counts=False,
):
    """Build one equally represented spherical prototype per background class."""
    centers = []
    scales = []
    counts = []
    for label in torch.unique(labels, sorted=True):
        members = embeddings[labels == label]
        center = normalized_mean(members)
        centers.append(center)
        scales.append(angular_scale(members, center, minimum_scale, scale_quantile))
        counts.append(len(members))
    if not centers:
        raise ValueError("at least one background class is required")
    centers = torch.stack(centers)
    scales = torch.stack(scales)
    if not return_counts:
        return centers, scales
    counts = torch.as_tensor(counts, device=embeddings.device, dtype=embeddings.dtype)
    return centers, scales, counts


def _farthest_point_initialization(embeddings, count):
    global_center = normalized_mean(embeddings)
    first = torch.argmin(embeddings @ global_center)
    selected = [first]
    while len(selected) < count:
        centers = embeddings[torch.stack(selected)]
        nearest_similarity = (embeddings @ centers.T).max(dim=1).values
        nearest_similarity[torch.stack(selected)] = float("inf")
        selected.append(torch.argmin(nearest_similarity))
    return embeddings[torch.stack(selected)]


@torch.no_grad()
def spherical_kmeans_background(
    embeddings,
    prototype_count=6,
    iterations=10,
    trim_fraction=0.1,
    minimum_scale=0.05,
    scale_quantile=0.5,
):
    """Deterministic robust spherical k-means for unlabeled scene background."""
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must be non-empty [samples, features]")
    count = max(1, min(int(prototype_count), len(embeddings)))
    centers = _farthest_point_initialization(embeddings, count)
    assignments = torch.zeros(
        len(embeddings), dtype=torch.long, device=embeddings.device
    )

    for _ in range(max(1, int(iterations))):
        assignments = torch.argmax(embeddings @ centers.T, dim=1)
        updated = []
        for index in range(count):
            members = embeddings[assignments == index]
            updated.append(
                centers[index] if len(members) == 0 else normalized_mean(members)
            )
        new_centers = torch.stack(updated)
        if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    assignments = torch.argmax(embeddings @ centers.T, dim=1)
    robust_centers = []
    scales = []
    counts = []
    for index in range(count):
        members = embeddings[assignments == index]
        if len(members) == 0:
            continue
        center = normalized_mean(members)
        if trim_fraction > 0.0 and len(members) >= 4:
            keep_count = max(2, int(round(len(members) * (1.0 - trim_fraction))))
            keep = torch.topk(members @ center, k=keep_count).indices
            members = members[keep]
            center = normalized_mean(members)
        robust_centers.append(center)
        scales.append(angular_scale(members, center, minimum_scale, scale_quantile))
        counts.append(len(members))
    return (
        torch.stack(robust_centers),
        torch.stack(scales),
        torch.as_tensor(counts, device=embeddings.device, dtype=embeddings.dtype),
    )


def robust_dual_evidence_score(
    embeddings,
    target_center,
    target_scale,
    background_centers,
    background_scales,
    softmin_temperature=0.1,
    hardness_temperature=0.2,
    eps=1e-6,
):
    """Score target proximity against a target-conditioned background mixture."""
    if background_centers.ndim != 2 or len(background_centers) == 0:
        raise ValueError("at least one background prototype is required")
    softmin_temperature = max(float(softmin_temperature), eps)
    hardness_temperature = max(float(hardness_temperature), eps)
    target_similarity = embeddings @ target_center
    background_similarities = embeddings @ background_centers.T
    target_distance = (1.0 - target_similarity).clamp_min(0.0) / target_scale.clamp_min(
        eps
    )
    component_distances = (1.0 - background_similarities).clamp_min(0.0)
    component_distances = (
        component_distances / background_scales.clamp_min(eps)[None, :]
    )

    confusion_logits = (background_centers @ target_center) / hardness_temperature
    log_weights = F.log_softmax(confusion_logits, dim=0)
    background_distance = -softmin_temperature * torch.logsumexp(
        log_weights[None, :] - component_distances / softmin_temperature,
        dim=1,
    )
    background_distance = background_distance.clamp_min(0.0)
    denominator = (background_distance + target_distance).clamp_min(eps)
    score = (background_distance - target_distance) / denominator
    outputs = {
        "score": score,
        "target_similarity": target_similarity,
        "background_similarity": background_similarities.max(dim=1).values,
        "target_distance": target_distance,
        "background_distance": background_distance,
        "background_component_similarities": background_similarities,
        "background_weights": log_weights.exp(),
    }
    return outputs


class BackgroundMemoryQueue:
    """Detached background embeddings with labels for false-negative filtering."""

    def __init__(self, capacity):
        self.capacity = max(0, int(capacity))
        self.embeddings = None
        self.labels = None

    def __len__(self):
        return 0 if self.embeddings is None else len(self.embeddings)

    @torch.no_grad()
    def enqueue(self, embeddings, labels):
        if self.capacity == 0 or len(embeddings) == 0:
            return
        embeddings = embeddings.detach()
        labels = labels.detach()
        if self.embeddings is None:
            self.embeddings = embeddings[-self.capacity :].clone()
            self.labels = labels[-self.capacity :].clone()
            return
        self.embeddings = torch.cat((self.embeddings, embeddings), dim=0)[
            -self.capacity :
        ]
        self.labels = torch.cat((self.labels, labels), dim=0)[-self.capacity :]

    def negatives_for(self, target_label):
        if self.embeddings is None:
            return None
        keep = self.labels != target_label
        return self.embeddings[keep]


def upper_tail(values, fraction=0.01, minimum_count=16):
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty vector")
    count = max(int(minimum_count), int(math.ceil(len(values) * float(fraction))))
    return torch.topk(values, k=min(len(values), count)).values
