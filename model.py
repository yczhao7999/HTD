import torch


import torch.nn.functional as F


from torch import nn


class SpectralResidualBlock(nn.Module):

    def __init__(self, channels, expansion=2, dropout=0.05):

        super().__init__()

        hidden = channels * expansion

        groups = min(8, channels)

        while channels % groups:

            groups -= 1

        self.norm = nn.GroupNorm(groups, channels)

        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.depthwise(x)

        return residual + self.pointwise(x)


class HaarWaveletDecomposition(nn.Module):
    """Anti-aliased 1D Haar wavelet decomposition for spectral signals.































































    Extracts low-frequency (LL) and mid-frequency (LH) subbands from































    relative spectral axis without learnable parameters.































    """

    def __init__(self, gaussian_sigma=1.5):

        super().__init__()

        self.gaussian_sigma = gaussian_sigma

        # Create fixed Gaussian kernel for anti-aliasing

        kernel_size = int(6 * gaussian_sigma + 1)

        if kernel_size % 2 == 0:

            kernel_size += 1

        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2

        gaussian = torch.exp(-0.5 * (x / gaussian_sigma) ** 2)

        gaussian = gaussian / gaussian.sum()

        self.register_buffer("gaussian_kernel", gaussian.view(1, 1, -1))

    def forward(self, x):
        """































        Args:































            x: [B, 1, 256] spectral signal































        Returns:































            ll: [B, 1, 128] low-frequency subband































            lh: [B, 1, 128] mid-frequency subband































        """

        # Anti-aliasing low-pass filter

        pad = self.gaussian_kernel.shape[-1] // 2

        x_smooth = F.conv1d(
            F.pad(x, (pad, pad), mode="reflect"), self.gaussian_kernel, padding=0
        )

        # 1D Haar wavelet decomposition

        # Low-pass: (x[2i] + x[2i+1]) / sqrt(2)

        # High-pass: (x[2i] - x[2i+1]) / sqrt(2)

        ll = (x_smooth[:, :, 0::2] + x_smooth[:, :, 1::2]) / (2**0.5)

        lh = (x_smooth[:, :, 0::2] - x_smooth[:, :, 1::2]) / (2**0.5)

        return ll, lh


class PoolingSubbandAttention(nn.Module):
    """Pooling-guided subband collaborative attention.































































    Uses stable pooling queries from LL to guide unstable LH subband,































    with SNR-based reliability gating.































    """

    def __init__(self, ll_dim=64, lh_dim=64, dropout=0.05):

        super().__init__()

        self.ll_dim = ll_dim

        self.lh_dim = lh_dim

        # Pooling branches for stable queries

        self.avg_query = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(ll_dim, ll_dim, 1),
        )

        self.max_key = nn.Sequential(
            nn.AdaptiveMaxPool1d(16),
            nn.Conv1d(ll_dim, ll_dim, 1),
        )

        # Cross-attention to LH subband

        self.value_proj = nn.Conv1d(lh_dim, lh_dim, 1)

        self.out_proj = nn.Sequential(
            nn.Conv1d(lh_dim, lh_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # SNR-based reliability gate

        self.snr_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(lh_dim, lh_dim // 4, 1),
            nn.GELU(),
            nn.Conv1d(lh_dim // 4, lh_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, ll_feat, lh_feat):
        """































        Args:































            ll_feat: [B, ll_dim, 128] low-frequency features































            lh_feat: [B, lh_dim, 128] mid-frequency features































        Returns:































            enhanced: [B, lh_dim, 128] attention-enhanced LH features































        """

        B = ll_feat.shape[0]

        # Stable pooling queries and keys from LL

        Q = self.avg_query(ll_feat)  # [B, ll_dim, 16]

        K = self.max_key(ll_feat)  # [B, ll_dim, 16]

        # LH as values

        V = self.value_proj(lh_feat)  # [B, lh_dim, 128]

        # Cross-attention: pool(LL) attends to LH

        # Reshape V for attention

        V_pool = F.adaptive_avg_pool1d(V, 16)  # [B, lh_dim, 16]

        # Compute attention scores

        attn_scores = torch.bmm(Q.transpose(1, 2), K) / (
            self.ll_dim**0.5
        )  # [B, 16, 16]

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to LH values

        attn_out = torch.bmm(V_pool, attn_weights.transpose(1, 2))  # [B, lh_dim, 16]

        # Upsample back to original resolution

        attn_out = F.interpolate(attn_out, size=128, mode="linear", align_corners=False)

        # Output projection

        enhanced = self.out_proj(attn_out)

        # SNR-based reliability gating

        reliability = self.snr_gate(lh_feat)

        enhanced = lh_feat + enhanced * reliability

        return enhanced


class HierarchicalFrequencyPrototypeNetworkV2(nn.Module):
    """HFPN v2: Frequency-aware encoder with enhanced robustness.































































    Improvements over v1:































    1. L2 normalization of frequency features before fusion































    2. Frequency regularization loss to prevent domain overfitting































    3. Branch consistency constraint for cross-domain alignment































    """

    def __init__(
        self,
        input_bands=256,
        embedding_dim=128,
        num_classes=15,
        branch_width=32,
        dropout=0.05,
        freq_dim_ll=64,
        freq_dim_lh=64,
    ):

        super().__init__()

        self.input_bands = input_bands

        self.embedding_dim = embedding_dim

        self.num_classes = num_classes

        # Module 1: Haar wavelet decomposition (same as v1)

        self.wavelet = HaarWaveletDecomposition(gaussian_sigma=1.5)

        # Module 2: Frequency branch encoders (same as v1)

        self.ll_encoder = nn.Sequential(
            nn.Conv1d(1, freq_dim_ll, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(min(8, freq_dim_ll), freq_dim_ll),
            nn.GELU(),
            nn.Conv1d(
                freq_dim_ll,
                freq_dim_ll,
                kernel_size=3,
                padding=1,
                groups=freq_dim_ll,
                bias=False,
            ),
        )

        self.lh_encoder = nn.Sequential(
            nn.Conv1d(1, freq_dim_lh, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(min(8, freq_dim_lh), freq_dim_lh),
            nn.GELU(),
            nn.Conv1d(
                freq_dim_lh,
                freq_dim_lh,
                kernel_size=3,
                padding=1,
                groups=freq_dim_lh,
                bias=False,
            ),
        )

        # Module 3: Pooling-subband attention (same as v1)

        self.subband_attention = PoolingSubbandAttention(
            freq_dim_ll, freq_dim_lh, dropout
        )

        # === NEW: Frequency L2 normalization layers ===

        self.freq_norm_ll = nn.LayerNorm(freq_dim_ll)

        self.freq_norm_lh = nn.LayerNorm(freq_dim_lh)

        # Module 4: Spatial multi-scale branch (same as v1)

        self.band_adapter = nn.Conv1d(1, 1, 3, padding=1, bias=True)

        nn.init.zeros_(self.band_adapter.weight)

        nn.init.zeros_(self.band_adapter.bias)

        self.spatial_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, branch_width, k, padding=k // 2, bias=False),
                    nn.GroupNorm(min(8, branch_width), branch_width),
                    nn.GELU(),
                )
                for k in (3, 7, 15)
            ]
        )

        spatial_dim = branch_width * 3

        self.spatial_fuse = nn.Sequential(
            nn.Conv1d(spatial_dim, branch_width * 2, 1, bias=False),
            nn.GroupNorm(min(8, branch_width * 2), branch_width * 2),
            nn.GELU(),
        )

        # Module 5: Dual-branch fusion with enhanced control

        freq_fused_dim = freq_dim_ll + freq_dim_lh

        spatial_fused_dim = branch_width * 2

        total_dim = freq_fused_dim + spatial_fused_dim

        # === MODIFIED: More conservative branch gate ===

        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(total_dim, total_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),  # NEW: Add dropout for robustness
            nn.Linear(total_dim // 4, 2),
            nn.Softmax(dim=-1),
        )

        self.final_fuse = nn.Sequential(
            nn.Conv1d(total_dim, embedding_dim * 2, 1, bias=False),
            nn.GroupNorm(min(8, embedding_dim * 2), embedding_dim * 2),
            nn.GELU(),
            SpectralResidualBlock(embedding_dim * 2, expansion=2, dropout=dropout),
        )

        # === NEW: Separate projection heads for consistency loss ===

        self.freq_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(freq_fused_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.spatial_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spatial_fused_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_features(self, x, return_auxiliary=False):
        """































        Args:































            x: [B, 256] spectral input































            return_auxiliary: if True, return freq/spatial features for consistency loss































        Returns:































            features: [B, embedding_dim] unnormalized features































            (optional) auxiliary: dict with freq_embed, spatial_embed, freq_raw, spatial_raw































        """

        if x.ndim != 2:

            raise ValueError(f"expected [batch, bands], got {tuple(x.shape)}")

        x_1d = x.unsqueeze(1)  # [B, 1, 256]

        if hasattr(self, "input_extra_blocks"):
            for block in self.input_extra_blocks:
                x_1d = block(x_1d)

        if hasattr(self, "dc_gate"):

            x_1d = self.dc_gate(x_1d)

        # === Frequency branch ===

        ll, lh = self.wavelet(x_1d)  # [B, 1, 128] each

        ll_feat = self.ll_encoder(ll)  # [B, freq_dim_ll, 128]

        lh_feat = self.lh_encoder(lh)  # [B, freq_dim_lh, 128]

        lh_enhanced = self.subband_attention(ll_feat, lh_feat)  # [B, freq_dim_lh, 128]

        # === NEW: L2 normalization of frequency features ===

        # Transpose to [B, 128, freq_dim] for LayerNorm, then back

        ll_feat_norm = self.freq_norm_ll(ll_feat.transpose(1, 2)).transpose(1, 2)

        lh_enhanced_norm = self.freq_norm_lh(lh_enhanced.transpose(1, 2)).transpose(
            1, 2
        )

        freq_feat = torch.cat(
            [ll_feat_norm, lh_enhanced_norm], dim=1
        )  # [B, freq_dim_ll+lh, 128]

        if hasattr(self, "dcmm_freq"):

            freq_feat = freq_feat + self.dcmm_scale * self.dcmm_freq(freq_feat)

        if hasattr(self, "gkan_block"):

            freq_feat = freq_feat + self.gkan_scale * self.gkan_block(freq_feat)

        if hasattr(self, "infsa"):

            freq_feat = freq_feat + self.infsa_scale * self.infsa(freq_feat)

        if hasattr(self, "sagb"):

            freq_feat = freq_feat + self.sagb_scale * self.sagb(freq_feat)

        if hasattr(self, "sagbwave"):

            freq_feat = freq_feat + self.sagbwave_scale * self.sagbwave(freq_feat)

        if hasattr(self, "freq_extra_blocks"):
            for block, scale in self.freq_extra_blocks:
                freq_feat = freq_feat + scale * block(freq_feat)

        # === Spatial branch ===

        x_adapted = x_1d + self.band_adapter(x_1d)

        spatial_parts = [branch(x_adapted) for branch in self.spatial_branches]

        if hasattr(self, "dcmm"):

            spatial_parts.append(self.dcmm(x_adapted))

        spatial_multi = torch.cat(spatial_parts, dim=1)

        spatial_feat = self.spatial_fuse(spatial_multi)  # [B, branch_width*2, 256]

        spatial_feat = F.adaptive_avg_pool1d(
            spatial_feat, 128
        )  # [B, branch_width*2, 128]

        if hasattr(self, "spatial_extra_blocks"):
            for block, scale in self.spatial_extra_blocks:
                spatial_feat = spatial_feat + scale * block(spatial_feat)

        # === Dual-branch fusion ===

        if hasattr(self, "cwca"):

            fused_input, branch_weights = self.cwca(freq_feat, spatial_feat)

            if hasattr(self, "cwca_res_scale"):

                combined = torch.cat(
                    [freq_feat, spatial_feat], dim=1
                )  # [B, total_dim, 128]

                fused_input = combined + self.cwca_res_scale * (fused_input - combined)

            freq_weight = branch_weights[:, 0].view(-1, 1, 1)

            spatial_weight = branch_weights[:, 1].view(-1, 1, 1)

            fused = self.final_fuse(fused_input)  # [B, embedding_dim*2, 128]

        else:

            combined = torch.cat(
                [freq_feat, spatial_feat], dim=1
            )  # [B, total_dim, 128]

            branch_weights = self.branch_gate(combined)  # [B, 2]

            freq_weight = branch_weights[:, 0].view(-1, 1, 1)

            spatial_weight = branch_weights[:, 1].view(-1, 1, 1)

            weighted = torch.cat(
                [freq_feat * freq_weight, spatial_feat * spatial_weight], dim=1
            )

            fused = self.final_fuse(weighted)  # [B, embedding_dim*2, 128]

        final_embed = self.projection(fused)  # [B, embedding_dim]

        if return_auxiliary:

            # Project frequency and spatial features separately for consistency loss

            freq_embed = self.freq_projection(freq_feat)  # [B, embedding_dim]

            spatial_embed = self.spatial_projection(spatial_feat)  # [B, embedding_dim]

            auxiliary = {
                "freq_embed": F.normalize(freq_embed, p=2, dim=-1),
                "spatial_embed": F.normalize(spatial_embed, p=2, dim=-1),
                "freq_weight": freq_weight.squeeze(-1).squeeze(-1),  # [B]
                "spatial_weight": spatial_weight.squeeze(-1).squeeze(-1),  # [B]
                "freq_raw": freq_feat,  # For regularization loss
            }

            return final_embed, auxiliary

        return final_embed

    def forward(self, x, return_auxiliary=False):

        if return_auxiliary:

            features, aux = self.forward_features(x, return_auxiliary=True)

            logits = self.classifier(F.normalize(features, p=2, dim=-1))

            return logits, aux

        else:

            features = self.forward_features(x, return_auxiliary=False)

            return self.classifier(F.normalize(features, p=2, dim=-1))

    def embed(self, x):

        return F.normalize(
            self.forward_features(x, return_auxiliary=False), p=2, dim=-1
        )

    def embed_with_frequency(self, x):
        """Return main fusion embedding, frequency embedding, and gate weight.































































        This avoids recomputing the backbone when downstream code needs































        both the main and frequency-branch scores for ablation artifacts.































        """

        features, auxiliary = self.forward_features(x, return_auxiliary=True)

        main_embed = F.normalize(features, p=2, dim=-1)

        return main_embed, auxiliary["freq_embed"], auxiliary["freq_weight"]


class ChannelAttention1D(nn.Module):

    def __init__(self, channels, reduction=4):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        hidden = max(4, channels // reduction)

        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.fc(self.avg_pool(x))

        max_v = self.fc(self.max_pool(x))

        return self.sigmoid(avg + max_v)


class SELayer1D(nn.Module):

    def __init__(self, channels, reduction=4):

        super().__init__()

        hidden = max(4, channels // reduction)

        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):

        return x * self.conv(nn.AdaptiveAvgPool1d(1)(x))


class SpectralCWCAGlobal(nn.Module):
    """Transfer-oriented CWCA variant: global branch competition + channel recalibration."""

    def __init__(self, freq_channels, spatial_channels, common_channels, reduction=4):

        super().__init__()

        self.proj_freq = nn.Conv1d(freq_channels, common_channels, 1, bias=False)

        self.proj_spatial = nn.Conv1d(spatial_channels, common_channels, 1, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv1d(common_channels * 2, common_channels, 1, bias=False),
            nn.GroupNorm(min(8, common_channels), common_channels),
            nn.GELU(),
        )

        self.weight_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(common_channels, 2),
            nn.Softmax(dim=1),
        )

        self.channel_attention = ChannelAttention1D(common_channels * 2, reduction)

        self.se = SELayer1D(common_channels * 2, reduction)

    def forward(self, freq_feat, spatial_feat):

        a = self.proj_freq(freq_feat)

        b = self.proj_spatial(spatial_feat)

        x = self.fuse(torch.cat([a, b], dim=1))

        weights = self.weight_mlp(x).unsqueeze(-1)  # [B, 2, 1]

        weighted = torch.cat([a * weights[:, 0:1, :], b * weights[:, 1:2, :]], dim=1)

        weighted = weighted * self.channel_attention(weighted)

        out = self.se(weighted)

        return out, weights.squeeze(-1)


class HierarchicalFrequencyPrototypeNetworkV2CWCAGlobal(
    HierarchicalFrequencyPrototypeNetworkV2
):
    """HFPN v2 + global-branch CWCA fusion variant."""

    def __init__(
        self,
        input_bands=256,
        embedding_dim=128,
        num_classes=15,
        branch_width=32,
        dropout=0.05,
        freq_dim_ll=64,
        freq_dim_lh=64,
    ):

        super().__init__(
            input_bands,
            embedding_dim,
            num_classes,
            branch_width,
            dropout,
            freq_dim_ll,
            freq_dim_lh,
        )

        freq_channels = freq_dim_ll + freq_dim_lh

        spatial_channels = branch_width * 2

        common_channels = (freq_channels + spatial_channels) // 2

        self.cwca = SpectralCWCAGlobal(freq_channels, spatial_channels, common_channels)


def build_encoder(architecture, input_bands, embedding_dim, num_classes):
    if architecture == "hfpn_v2_cwcaglobal":
        return HierarchicalFrequencyPrototypeNetworkV2CWCAGlobal(
            input_bands, embedding_dim, num_classes
        )
    raise ValueError(f"unknown encoder architecture: {architecture}")
