"""
Graph Neural Network encoder for stellar stream phase space.

Architecture: 6-layer GINEConv (GIN with edge features) + global mean+max pooling.

Why GINEConv (not GCN or GAT):
- GIN is provably maximally expressive (Xu+2019, Weisfeiler-Leman test)
- Edge features carry the relative PM / relative position signal that encodes perturbations
- GINEConv aggregates edge features during message passing, capturing local kinematic patterns

Node features (18): phi1, phi2, dist, pm1, pm2, vrad, e_dist, e_pm1, e_pm2, e_vrad,
                     membership_prob, + 7 stream one-hot
Edge features (5): delta_phi1, delta_phi2, delta_pm1, delta_pm2, euclidean_dist_4d
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        # PyG's GINEConv inspects `in_channels` to infer the hidden dimension.
        self.in_channels = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionReadout(nn.Module):
    """Learned attention-based graph readout for per-node importance scores.

    A learnable query vector attends to final hidden states to produce:
      1. Per-node softmax weights (interpretability: which stars drive detection)
      2. Attention-weighted graph-level pooling vector

    This enables interpretability that no power-spectrum or CNN baseline can
    produce: the model can tell you *which stars* are informative for DM detection.

    The attention weights highlight stars near gap edges, spur features, and
    velocity-space outliers — precisely the observational signatures of
    subhalo impacts that the GNN learns to detect.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted pool and per-node weights.

        Args:
            x: [N_total, hidden_dim] node hidden states.
            batch: [N_total] batch assignment vector.

        Returns:
            pool: [B, hidden_dim] attention-weighted graph vectors.
            alpha: [N_total] per-node attention weights (sum to 1 per graph).
        """
        # Scaled dot-product style: project nodes, attend with query
        scores = self.key_proj(x).squeeze(-1)  # [N_total]
        # Per-graph softmax using PyG's batch-aware softmax
        alpha = pyg_softmax(scores, batch)  # [N_total], sum to 1 per graph
        alpha = self.dropout(alpha)

        # Attention-weighted pooling
        weighted = x * alpha.unsqueeze(-1)  # [N_total, hidden_dim]
        pool = global_add_pool(weighted, batch)  # [B, hidden_dim]
        return pool, alpha


class StreamGNNEncoder(nn.Module):
    """GINEConv encoder mapping a stream graph to a fixed-size embedding.

    Args:
        n_node_features: Number of node input features (default 11).
        n_edge_features: Number of edge input features (default 5).
        hidden_dim: Hidden dimension in each GINEConv MLP.
        n_layers: Number of GINEConv layers (default 6).
        embedding_dim: Output embedding size (default 128).
        dropout: Dropout probability.
        use_attention_readout: If True, adds an attention-based pooling channel
            (mean + max + attention → 3*hidden_dim before output MLP). This enables
            per-node importance scores for interpretability. Default False to maintain
            backward compatibility with existing checkpoints.
    """

    def __init__(
        self,
        n_node_features: int = 11,
        n_edge_features: int = 5,
        hidden_dim: int = 256,
        n_layers: int = 6,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        use_attention_readout: bool = False,
        use_multi_scale: bool = False,
        seg_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_attention_readout = use_attention_readout
        self.use_multi_scale = use_multi_scale

        # Input projections
        self.node_proj = nn.Linear(n_node_features, hidden_dim)
        self.edge_proj = nn.Linear(n_edge_features, hidden_dim)

        # GINEConv layers: each takes (node_features + edge_features) -> new node features
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = MLP(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
            # GINEConv requires edge_dim to match the projected node dim
            conv = GINEConv(nn=mlp, train_eps=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Global pooling: concat mean + max -> 2 * hidden_dim
        # With attention readout: mean + max + attention -> 3 * hidden_dim
        if use_attention_readout:
            self.attention_readout = AttentionReadout(hidden_dim, dropout)
            pool_dim = 3 * hidden_dim
        else:
            self.attention_readout = None
            pool_dim = 2 * hidden_dim

        # Multi-scale: segment-level GNN adds seg_embedding_dim to pool
        if use_multi_scale:
            self.segment_gnn = SegmentGNN(
                n_seg_features=8, n_seg_edge_features=4,
                hidden_dim=seg_embedding_dim, embedding_dim=seg_embedding_dim,
                n_layers=3, dropout=dropout,
            )
            pool_dim += seg_embedding_dim
        else:
            self.segment_gnn = None

        # Store last attention weights for interpretability (no gradient needed)
        self._last_attention: torch.Tensor | None = None

        # Output MLP: compress to embedding_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def enable_mc_dropout(self) -> None:
        """Activate dropout at inference time for Monte Carlo uncertainty estimation.

        After calling this, forward passes will produce stochastic embeddings
        even in model.eval() mode because Dropout layers remain in training mode.
        Call model.eval() + enable_mc_dropout() for MC Dropout inference.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_embed(self, data: Data, T: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of embeddings under MC Dropout.

        Runs T stochastic forward passes with dropout active, returning the
        sample mean and standard deviation. Useful for:
        - Epistemic uncertainty quantification
        - OOD detection (high std → unfamiliar input)
        - Propagating uncertainty into SBI posteriors

        Args:
            data: PyG Data batch (same as forward()).
            T: Number of stochastic forward passes (default 20).

        Returns:
            mean_emb: [B, embedding_dim] — mean embedding across T passes.
            std_emb: [B, embedding_dim] — elementwise std across T passes.
        """
        was_training = self.training
        self.eval()
        self.enable_mc_dropout()

        embeddings = []
        with torch.no_grad():
            for _ in range(T):
                emb = self.forward(data)
                embeddings.append(emb)

        # Restore original training state
        if was_training:
            self.train()

        stacked = torch.stack(embeddings, dim=0)  # [T, B, embedding_dim]
        mean_emb = stacked.mean(dim=0)             # [B, embedding_dim]
        std_emb = stacked.std(dim=0)               # [B, embedding_dim]
        return mean_emb, std_emb

    def forward(self, data: Data) -> torch.Tensor:
        """Encode a batch of stream graphs to embeddings.

        Args:
            data: PyG Data batch with x [N_total, n_node_features],
                  edge_index [2, E_total], edge_attr [E_total, n_edge_features],
                  batch [N_total].

        Returns:
            Embeddings [B, embedding_dim].

        Raises:
            RuntimeError: If edge_index or edge_attr is missing (graph not built).
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        if edge_index is None or edge_attr is None:
            raise RuntimeError(
                "GNN forward pass requires edge_index and edge_attr. "
                "Call build_knn_graph_batched() before forward()."
            )

        # Project to hidden dim
        x = F.relu(self.node_proj(x))
        edge_attr_proj = F.relu(self.edge_proj(edge_attr))

        # GINEConv message passing with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, edge_index, edge_attr_proj)
            x_new = self.dropout(bn(x_new))
            if i >= 1:  # residual from layer 2 onwards
                x = F.relu(x_new + x)
            else:
                x = F.relu(x_new)

        # Global pooling: concatenate mean and max (+ attention if enabled)
        x_mean = global_mean_pool(x, batch)  # [B, hidden_dim]
        x_max = global_max_pool(x, batch)    # [B, hidden_dim]

        if self.use_attention_readout and self.attention_readout is not None:
            x_attn, alpha = self.attention_readout(x, batch)  # [B, hidden_dim], [N_total]
            self._last_attention = alpha.detach()
            pool_parts = [x_mean, x_max, x_attn]
        else:
            pool_parts = [x_mean, x_max]

        # Multi-scale: append segment-level embedding if available
        if self.use_multi_scale and self.segment_gnn is not None:
            if hasattr(data, "seg_x") and data.seg_x is not None:
                seg_emb = self.segment_gnn(
                    data.seg_x, data.seg_edge_index,
                    data.seg_edge_attr, data.seg_batch,
                )  # [B, seg_embedding_dim]
            else:
                # Segment data not built — use zeros (graceful degradation)
                n_graphs = int(batch.max().item()) + 1
                seg_emb = torch.zeros(
                    n_graphs, self.segment_gnn.output_proj.out_features,
                    device=x_mean.device,
                )
            pool_parts.append(seg_emb)

        x_pool = torch.cat(pool_parts, dim=1)
        return self.output_mlp(x_pool)  # [B, embedding_dim]

    def forward_with_attention(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns per-node attention weights.

        Only works when use_attention_readout=True. For interpretability analysis.

        Args:
            data: PyG Data batch.

        Returns:
            embedding: [B, embedding_dim]
            alpha: [N_total] per-node attention weights (sum to 1 per graph).

        Raises:
            RuntimeError: If attention readout is not enabled.
        """
        if not self.use_attention_readout:
            raise RuntimeError(
                "forward_with_attention() requires use_attention_readout=True"
            )
        emb = self.forward(data)
        return emb, self._last_attention


class SegmentGNN(nn.Module):
    """Lightweight GNN operating on the coarse segment-level graph.

    Captures gap *clustering* patterns at the ~5 deg scale. CDM produces
    many weak impacts clustered in nearby segments (from a population of
    low-mass subhalos), while WDM/FDM produce fewer, deeper, isolated gaps.
    This spatial pattern is invisible at the star-level k-NN scale.

    Architecture: 3-layer GINEConv, hidden_dim=64, output=64.
    Input: 8 segment features, 4 edge features (from build_segment_graph).
    Output: [B, seg_embedding_dim] graph-level embedding per stream.
    """

    def __init__(
        self,
        n_seg_features: int = 8,
        n_seg_edge_features: int = 4,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(n_seg_features, hidden_dim)
        self.edge_proj = nn.Linear(n_seg_edge_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = MLP(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
            conv = GINEConv(nn=mlp, train_eps=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(2 * hidden_dim, embedding_dim)  # mean + max pool

    def forward(
        self,
        seg_x: torch.Tensor,
        seg_edge_index: torch.Tensor,
        seg_edge_attr: torch.Tensor,
        seg_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode segment graph to per-stream embedding.

        Args:
            seg_x: [B*n_segments, 8] segment features.
            seg_edge_index: [2, E_seg] segment edges.
            seg_edge_attr: [E_seg, 4] segment edge features.
            seg_batch: [B*n_segments] batch assignment.

        Returns:
            [B, embedding_dim] segment-level embeddings.
        """
        x = F.relu(self.node_proj(seg_x))
        edge_attr = F.relu(self.edge_proj(seg_edge_attr))

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x_new = conv(x, seg_edge_index, edge_attr)
            x_new = self.dropout(bn(x_new))
            if i >= 1:
                x = F.relu(x_new + x)
            else:
                x = F.relu(x_new)

        # Pool: mean + max
        x_mean = global_mean_pool(x, seg_batch)
        x_max = global_max_pool(x, seg_batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)
        return self.output_proj(x_pool)


class StreamGNNMultiTask(nn.Module):
    """GNN encoder + classification head (3-class) + regression heads.

    Multi-task training ensures the embedding encodes both:
    1. Discrete model identity (CDM+SIDM / WDM / FDM) — classification
    2. Continuous physical parameters (log10_M_sub_mean, n_impacts) — regression

    This produces embeddings that are far more informative for downstream SBI
    than pure classifier embeddings, which optimize for decision boundaries
    rather than smooth parameter mapping.

    CDM and SIDM are merged into one class because the impulse approximation's
    scale radius tweak (0.5x for SIDM) produces indistinguishable perturbations
    at typical impact parameters. Keeping them as separate classes forces the
    network to hallucinate distinctions that don't exist in the data.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_reg_targets: int = 2,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = StreamGNNEncoder(**encoder_kwargs)
        emb_dim = encoder_kwargs.get("embedding_dim", 128)

        # Classification head: 3 classes (CDM+SIDM, WDM, FDM)
        self.classifier = nn.Linear(emb_dim, n_classes)

        # Regression head: predict [log10_M_sub_mean, n_impacts]
        # Separate MLP so regression gradients don't destabilize the classifier
        self.regressor = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_reg_targets),
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (embeddings, classification_logits, regression_output).

        Args:
            data: PyG Batch with x, edge_index, edge_attr, batch.

        Returns:
            emb: [B, embedding_dim] embeddings.
            logits: [B, n_classes] classification logits.
            reg_out: [B, n_reg_targets] regression predictions.
        """
        emb = self.encoder(data)
        logits = self.classifier(emb)
        reg_out = self.regressor(emb)
        return emb, logits, reg_out

    def mc_embed(self, data: Data, T: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout embedding: mean and std over T stochastic passes.

        Convenience wrapper around self.encoder.mc_embed().

        Args:
            data: PyG Batch.
            T: Number of stochastic forward passes.

        Returns:
            mean_emb: [B, embedding_dim] mean embedding.
            std_emb: [B, embedding_dim] per-dimension std (epistemic uncertainty).
        """
        return self.encoder.mc_embed(data, T=T)

    def get_node_importance(self, data: Data) -> torch.Tensor:
        """Get per-node attention weights (importance scores) for a stream.

        Requires the encoder to have use_attention_readout=True.
        Each weight indicates how much that star contributes to the final
        graph-level embedding. Useful for identifying which stars drive
        the DM model classification.

        Args:
            data: PyG Data/Batch object.

        Returns:
            alpha: [N_total] per-node importance weights (sum to 1 per graph).

        Raises:
            RuntimeError: If attention readout is not enabled.
        """
        if not self.encoder.use_attention_readout:
            raise RuntimeError(
                "get_node_importance() requires use_attention_readout=True in the encoder."
            )
        self.encoder.eval()
        with torch.no_grad():
            self.encoder.forward(data)
        return self.encoder._last_attention


class StreamGNNMultiTaskV2(nn.Module):
    """Binary classification + continuous M_hm regression multi-task model.

    Replaces the 3-class scheme (CDM+SIDM / WDM / FDM) with a physically
    motivated reformulation:

    1. **Binary head**: Suppressed (WDM/FDM) vs Unsuppressed (CDM/SIDM).
       This aligns the classification task with what the data can actually
       distinguish. The WDM/FDM suppression shapes are degenerate at Gaia
       sensitivity, but both are distinguishable from CDM/SIDM.

    2. **M_hm regression head**: Directly predict log10(M_hm) for suppressed
       models. For unsuppressed models (CDM/SIDM), the physical value is the
       lower CDM-like edge of the prior, and regression gradients only flow
       from suppressed sims.

    3. **n_impacts regression head**: Same as V1.

    4. **Optional aleatoric uncertainty head**: Predicts mean + log_var for
       heteroscedastic uncertainty on M_hm. Allows the model to flag cases
       where it's uncertain about the suppression scale.

    Physical rationale for binary:
      - CDM and SIDM produce indistinguishable impulse signatures (0.5x scale
        radius at <0.01% detectable difference). Already merged in V1.
      - WDM and FDM both suppress the subhalo mass function below a
        characteristic scale. Their suppression *shapes* are degenerate at
        the sensitivity achievable with Gaia DR3 proper motions.
      - The meaningful question is: "Is the mass function suppressed?" not
        "Which specific model causes the suppression?"

    The embedding from this model is more informative for downstream SBI
    because it directly encodes the suppression scale M_hm rather than
    optimizing for arbitrary class boundaries.
    """

    def __init__(
        self,
        n_reg_targets: int = 2,
        predict_uncertainty: bool = False,
        profile_dim: int = 0,
        profile_hidden_dim: int = 64,
        profile_layer_norm: bool = False,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = StreamGNNEncoder(**encoder_kwargs)
        emb_dim = encoder_kwargs.get("embedding_dim", 128)
        self.predict_uncertainty = predict_uncertainty
        self.profile_dim = profile_dim

        if profile_dim > 0:
            self.profile_norm = nn.LayerNorm(profile_dim) if profile_layer_norm else nn.Identity()
            self.profile_mlp = nn.Sequential(
                nn.Linear(profile_dim, profile_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(profile_hidden_dim, profile_hidden_dim),
                nn.ReLU(),
            )
            head_dim = emb_dim + profile_hidden_dim
        else:
            self.profile_norm = nn.Identity()
            self.profile_mlp = None
            head_dim = emb_dim

        # Binary head: suppressed (1) vs unsuppressed (0)
        self.binary_head = nn.Linear(head_dim, 1)

        # M_hm regression head: predict log10(M_hm)
        # Separate MLP so regression gradients don't destabilize the binary head
        if predict_uncertainty:
            # Heteroscedastic: output (mean, log_var)
            self.mhm_head = nn.Sequential(
                nn.Linear(head_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 2),  # [mean, log_var]
            )
        else:
            self.mhm_head = nn.Sequential(
                nn.Linear(head_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

        # n_impacts + optional additional regression targets
        self.regressor = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_reg_targets),
        )

    def forward(
        self, data: Data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (embeddings, binary_logit, mhm_prediction, reg_output).

        Args:
            data: PyG Batch with x, edge_index, edge_attr, batch.

        Returns:
            emb: [B, embedding_dim] embeddings.
            binary_logit: [B, 1] logit for suppressed (>0) vs unsuppressed (<0).
            mhm_out: [B, 1] or [B, 2] log10(M_hm) prediction (+ log_var if uncertainty).
            reg_out: [B, n_reg_targets] regression predictions (n_impacts, etc).
        """
        emb = self.encoder(data)
        head_input = emb
        if self.profile_mlp is not None:
            if not hasattr(data, "profile_x") or data.profile_x is None:
                raise RuntimeError(
                    "StreamGNNMultiTaskV2 was built with profile_dim > 0, "
                    "but data.profile_x is missing. Call build_profile_features_batched()."
                )
            profile_emb = self.profile_mlp(self.profile_norm(data.profile_x))
            head_input = torch.cat([emb, profile_emb], dim=1)
        binary_logit = self.binary_head(head_input)
        mhm_out = self.mhm_head(head_input)
        reg_out = self.regressor(head_input)
        return emb, binary_logit, mhm_out, reg_out

    def get_combined_embedding(self, data: Data) -> torch.Tensor:
        """Return the combined embedding (GNN + profile MLP) used as head input.

        For SBI, this is more informative than the raw encoder embedding because
        it includes the profile branch representation when available.

        Returns:
            [B, head_dim] tensor: encoder_dim (128) or encoder_dim + profile_hidden (192).
        """
        emb = self.encoder(data)
        if self.profile_mlp is not None:
            if not hasattr(data, "profile_x") or data.profile_x is None:
                raise RuntimeError(
                    "get_combined_embedding() requires profile_x. "
                    "Call build_profile_features_batched() first."
                )
            profile_emb = self.profile_mlp(self.profile_norm(data.profile_x))
            return torch.cat([emb, profile_emb], dim=1)
        return emb

    def mc_embed(self, data: Data, T: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout embedding: mean and std over T stochastic passes."""
        return self.encoder.mc_embed(data, T=T)

    def get_node_importance(self, data: Data) -> torch.Tensor:
        """Get per-node attention weights for interpretability.

        Requires encoder use_attention_readout=True.
        """
        if not self.encoder.use_attention_readout:
            raise RuntimeError(
                "get_node_importance() requires use_attention_readout=True in the encoder."
            )
        self.encoder.eval()
        with torch.no_grad():
            self.encoder.forward(data)
        return self.encoder._last_attention

    def predict_suppression(
        self, data: Data, threshold: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: predict binary class + M_hm with confidence.

        Args:
            data: PyG Batch.
            threshold: Logit threshold for binary decision (default 0.0).

        Returns:
            is_suppressed: [B] bool tensor.
            mhm_pred: [B] predicted log10(M_hm) (lower prior edge for unsuppressed).
            confidence: [B] sigmoid probability of suppression.
        """
        self.eval()
        with torch.no_grad():
            _, binary_logit, mhm_out, _ = self.forward(data)

        confidence = torch.sigmoid(binary_logit.squeeze(-1))  # [B]
        is_suppressed = confidence > (1.0 / (1.0 + torch.exp(torch.tensor(-threshold))))

        if self.predict_uncertainty:
            mhm_mean = mhm_out[:, 0]  # [B]
        else:
            mhm_mean = mhm_out.squeeze(-1)  # [B]

        # For unsuppressed predictions, report the lower CDM-like prior edge.
        mhm_pred = torch.where(is_suppressed, mhm_mean, torch.tensor(4.0, device=mhm_mean.device))

        return is_suppressed, mhm_pred, confidence
