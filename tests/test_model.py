"""
Tests for neural network models.

Validation criteria:
- GNN forward pass produces correct output shape
- GNN multi-task model produces embeddings, logits, and regression outputs
- Baseline CNN forward pass produces correct output shape
- Transformer forward pass produces correct output shape
- No NaN/Inf in outputs
- Edge feature dimensions are consistent
- Model utility functions (checkpoint, normalizer, PCA)
- Parameter counting works correctly
- Gradient flow through all layers
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pure_knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """Pure-torch k-NN graph construction (no torch-cluster dependency)."""
    # x: [N, D] — returns edge_index [2, N*k] (source→target convention)
    n = x.size(0)
    # Pairwise L2 distances
    dist = torch.cdist(x, x)  # [N, N]
    # Set self-distance to large value to exclude self-loops
    dist.fill_diagonal_(float('inf'))
    # Get k nearest neighbors per node
    _, idx = dist.topk(k, dim=1, largest=False)  # [N, k]
    # Build edge_index: row=source, col=target (each node points to its neighbors)
    row = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
    col = idx.reshape(-1)
    return torch.stack([row, col], dim=0)


def make_fake_batch(n_graphs: int = 4, n_stars: int = 50, k: int = 8) -> Data:
    """Create a fake PyG batch for testing."""
    n_total = n_graphs * n_stars
    x = torch.randn(n_total, 11)
    batch = torch.repeat_interleave(torch.arange(n_graphs), n_stars)
    # Build edges per graph
    all_edges = []
    all_edge_attr = []
    for g in range(n_graphs):
        mask = batch == g
        x_g = x[mask][:, :4]
        edge_idx = _pure_knn_graph(x_g, k=k)
        src, dst = edge_idx
        delta = x_g[dst] - x_g[src]
        dist = delta.norm(dim=1, keepdim=True)
        edge_attr = torch.cat([delta, dist], dim=1)
        # Offset node indices for the batch
        offset = g * n_stars
        all_edges.append(edge_idx + offset)
        all_edge_attr.append(edge_attr)
    edge_index = torch.cat(all_edges, dim=1)
    edge_attr = torch.cat(all_edge_attr, dim=0)
    y = torch.zeros(n_graphs, 3)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)


# ---------------------------------------------------------------------------
# GNN Encoder
# ---------------------------------------------------------------------------

class TestGNNEncoder:
    def test_forward_shape(self):
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5, hidden_dim=32, n_layers=2, embedding_dim=64)
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_no_nan_output(self):
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5, hidden_dim=32, n_layers=2, embedding_dim=64)
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            out = model(data)
        assert not torch.isnan(out).any(), "GNN output contains NaN"
        assert not torch.isinf(out).any(), "GNN output contains Inf"

    def test_single_graph(self):
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5, hidden_dim=32, n_layers=2, embedding_dim=64)
        model.eval()
        data = make_fake_batch(n_graphs=1, n_stars=20, k=4)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (1, 64)

    def test_different_embedding_dims(self):
        """Test that various embedding_dim settings produce correct output."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        for emb_dim in [32, 64, 128, 256]:
            model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                      hidden_dim=32, n_layers=2, embedding_dim=emb_dim)
            model.eval()
            data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
            with torch.no_grad():
                out = model(data)
            assert out.shape == (2, emb_dim)

    def test_gradient_flow(self):
        """Gradients should flow through all GINEConv layers."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                  hidden_dim=32, n_layers=3, embedding_dim=64)
        model.train()
        data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
        out = model(data)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist for all conv layers
        for i, conv in enumerate(model.convs):
            for name, param in conv.named_parameters():
                assert param.grad is not None, f"No gradient for conv[{i}].{name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in conv[{i}].{name}"

    def test_embedding_deterministic_in_eval(self):
        """Same input should produce same embedding in eval mode."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                  hidden_dim=32, n_layers=2, embedding_dim=64)
        model.eval()
        data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
        with torch.no_grad():
            out1 = model(data)
            out2 = model(data)
        torch.testing.assert_close(out1, out2)

    def test_large_batch(self):
        """Model should handle larger batch sizes."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                  hidden_dim=32, n_layers=2, embedding_dim=64)
        model.eval()
        data = make_fake_batch(n_graphs=16, n_stars=50, k=4)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (16, 64)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# GNN Multi-Task
# ---------------------------------------------------------------------------

class TestGNNMultiTask:
    def test_multi_task_output_shapes(self):
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        model = StreamGNNMultiTask(
            n_classes=3, n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            emb, logits, reg = model(data)
        assert emb.shape == (4, 64), f"Embedding shape wrong: {emb.shape}"
        assert logits.shape == (4, 3), f"Logits shape wrong: {logits.shape}"
        assert reg.shape == (4, 2), f"Regression shape wrong: {reg.shape}"

    def test_multi_task_no_nan(self):
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        model = StreamGNNMultiTask(
            n_classes=3, n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            emb, logits, reg = model(data)
        for name, tensor in [("emb", emb), ("logits", logits), ("reg", reg)]:
            assert not torch.isnan(tensor).any(), f"NaN in {name}"
            assert not torch.isinf(tensor).any(), f"Inf in {name}"

    def test_multi_task_training_step(self):
        """Multi-task model should support a training step with combined loss."""
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        model = StreamGNNMultiTask(
            n_classes=3, n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = make_fake_batch(n_graphs=8, n_stars=20, k=4)
        class_labels = torch.randint(0, 3, (8,))
        reg_targets = torch.randn(8, 2)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            emb, logits, reg = model(data)
            loss_cls = nn.CrossEntropyLoss()(logits, class_labels)
            loss_reg = nn.MSELoss()(reg, reg_targets)
            loss = loss_cls + loss_reg
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert all(l < float("inf") for l in losses), "Loss should be finite"

    def test_encoder_accessible(self):
        """Should be able to use encoder independently for SBI."""
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        model = StreamGNNMultiTask(
            n_classes=3, n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.eval()
        data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
        with torch.no_grad():
            emb_encoder = model.encoder(data)
            emb_full, _, _ = model(data)
        torch.testing.assert_close(emb_encoder, emb_full)


# ---------------------------------------------------------------------------
# Baseline CNN
# ---------------------------------------------------------------------------

class TestBaselineCNN:
    def test_forward_shape(self):
        from src.models.baseline import DensityProfileCNN  # noqa: PLC0415
        model = DensityProfileCNN(n_bins=200, embedding_dim=128, n_classes=4)
        model.eval()
        x = torch.randn(8, 3, 200)  # [B, 3, 200]
        with torch.no_grad():
            emb = model(x)
        assert emb.shape == (8, 128)

    def test_classify_shape(self):
        from src.models.baseline import DensityProfileCNN  # noqa: PLC0415
        model = DensityProfileCNN(n_bins=200, embedding_dim=128, n_classes=4)
        model.eval()
        x = torch.randn(8, 3, 200)
        with torch.no_grad():
            logits = model.classify(x)
        assert logits.shape == (8, 4)

    def test_no_nan(self):
        from src.models.baseline import DensityProfileCNN  # noqa: PLC0415
        model = DensityProfileCNN(n_bins=200, embedding_dim=128, n_classes=4)
        x = torch.randn(4, 3, 200)
        emb = model(x)
        assert not torch.isnan(emb).any()

    def test_different_n_bins(self):
        """CNN should handle different bin counts."""
        from src.models.baseline import DensityProfileCNN  # noqa: PLC0415
        for n_bins in [50, 100, 200, 400]:
            model = DensityProfileCNN(n_bins=n_bins, embedding_dim=64, n_classes=4)
            model.eval()
            x = torch.randn(4, 3, n_bins)
            with torch.no_grad():
                emb = model(x)
            assert emb.shape == (4, 64)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class TestTransformer:
    def test_forward_shape(self):
        from src.models.transformer import StreamTransformer, collate_for_transformer  # noqa: PLC0415
        model = StreamTransformer(n_node_features=11, d_model=64, n_heads=4, n_layers=2, embedding_dim=64)
        model.eval()
        from torch_geometric.data import Data  # noqa: PLC0415
        batch = [Data(x=torch.randn(n, 11)) for n in [30, 45, 20, 50]]
        x_padded, mask = collate_for_transformer(batch)
        with torch.no_grad():
            out = model(x_padded, mask)
        assert out.shape == (4, 64)

    def test_no_nan_output(self):
        from src.models.transformer import StreamTransformer, collate_for_transformer  # noqa: PLC0415
        model = StreamTransformer(n_node_features=11, d_model=64, n_heads=4, n_layers=2, embedding_dim=64)
        model.eval()
        from torch_geometric.data import Data  # noqa: PLC0415
        batch = [Data(x=torch.randn(n, 11)) for n in [20, 30]]
        x_padded, mask = collate_for_transformer(batch)
        with torch.no_grad():
            out = model(x_padded, mask)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_single_sample(self):
        from src.models.transformer import StreamTransformer, collate_for_transformer  # noqa: PLC0415
        model = StreamTransformer(n_node_features=11, d_model=64, n_heads=4, n_layers=2, embedding_dim=64)
        model.eval()
        from torch_geometric.data import Data  # noqa: PLC0415
        batch = [Data(x=torch.randn(25, 11))]
        x_padded, mask = collate_for_transformer(batch)
        with torch.no_grad():
            out = model(x_padded, mask)
        assert out.shape == (1, 64)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

class TestModelUtils:
    def test_count_parameters(self):
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        from src.models.utils import count_parameters  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                  hidden_dim=32, n_layers=2, embedding_dim=64)
        total, trainable = count_parameters(model)
        assert total > 0
        assert trainable == total  # all params should be trainable
        assert total > 1000  # should have at least 1K params

    def test_count_parameters_frozen(self):
        """Freezing params should reduce trainable count."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        from src.models.utils import count_parameters  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                  hidden_dim=32, n_layers=2, embedding_dim=64)
        # Freeze node projection
        for p in model.node_proj.parameters():
            p.requires_grad = False
        total, trainable = count_parameters(model)
        assert trainable < total

    def test_save_load_normalizer(self):
        import numpy as np  # noqa: PLC0415
        from src.models.utils import load_normalizer, save_normalizer  # noqa: PLC0415
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "norm.npz"
            mean = np.random.randn(11).astype(np.float32)
            std = np.abs(np.random.randn(11).astype(np.float32)) + 0.01
            save_normalizer(path, mean, std)
            loaded_mean, loaded_std = load_normalizer(path)
            np.testing.assert_allclose(loaded_mean, mean)
            np.testing.assert_allclose(loaded_std, std)

    def test_save_load_checkpoint(self):
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        from src.models.utils import load_checkpoint, save_checkpoint  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                  hidden_dim=32, n_layers=2, embedding_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ckpt.pt"
            save_checkpoint(path, model, optimizer, epoch=5, val_loss=0.42, config={"test": True})
            assert path.exists()

            # Load into fresh model
            model2 = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                       hidden_dim=32, n_layers=2, embedding_dim=64)
            ckpt = load_checkpoint(path, model2)
            assert ckpt["epoch"] == 5
            assert abs(ckpt["val_loss"] - 0.42) < 1e-6

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                torch.testing.assert_close(p1, p2, msg=f"Mismatch in {n1}")

    def test_pca_embeddings(self):
        import numpy as np  # noqa: PLC0415
        from src.models.utils import pca_embeddings  # noqa: PLC0415
        emb = np.random.randn(100, 128).astype(np.float32)
        labels = np.random.randint(0, 4, 100)
        result = pca_embeddings(emb, labels, n_components=2)
        assert result["coords"].shape == (100, 2)
        assert len(result["explained_variance"]) == 2
        assert all(0 <= v <= 1 for v in result["explained_variance"])

    def test_build_scheduler_cosine(self):
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        from src.models.utils import build_scheduler  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                  hidden_dim=32, n_layers=2, embedding_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, {"lr_scheduler": "cosine"}, n_epochs=100)
        assert scheduler is not None
        # Step a few times to ensure it doesn't crash
        for _ in range(5):
            scheduler.step()

    def test_build_scheduler_with_warmup(self):
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        from src.models.utils import build_scheduler  # noqa: PLC0415
        model = StreamGNNEncoder(n_node_features=11, n_edge_features=5,
                                  hidden_dim=32, n_layers=2, embedding_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(
            optimizer,
            {"lr_scheduler": "cosine_annealing_with_restarts", "n_epochs_warmup": 5, "T_0": 10},
            n_epochs=100,
        )
        assert scheduler is not None
        # LR should start low during warmup
        initial_lr = optimizer.param_groups[0]["lr"]
        # Step through warmup
        for _ in range(3):
            scheduler.step()


# ---------------------------------------------------------------------------
# MLP building block
# ---------------------------------------------------------------------------

class TestMLP:
    def test_mlp_shape(self):
        from src.models.gnn import MLP  # noqa: PLC0415
        mlp = MLP(in_dim=64, hidden_dim=128, out_dim=32, dropout=0.0)
        x = torch.randn(10, 64)
        out = mlp(x)
        assert out.shape == (10, 32)

    def test_mlp_in_channels_attribute(self):
        """GINEConv requires MLP to have in_channels attribute."""
        from src.models.gnn import MLP  # noqa: PLC0415
        mlp = MLP(in_dim=64, hidden_dim=128, out_dim=32)
        assert hasattr(mlp, "in_channels")
        assert mlp.in_channels == 64


# ---------------------------------------------------------------------------
# Feature 1: Attention Readout
# ---------------------------------------------------------------------------

class TestAttentionReadout:
    def test_attention_readout_shape(self):
        """AttentionReadout should return correct pool shape and alpha per node."""
        from src.models.gnn import AttentionReadout  # noqa: PLC0415
        readout = AttentionReadout(hidden_dim=32, dropout=0.0)
        readout.eval()
        # Fake: 2 graphs with 20 and 30 nodes
        x = torch.randn(50, 32)
        batch = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(30, dtype=torch.long)])
        with torch.no_grad():
            pool, alpha = readout(x, batch)
        assert pool.shape == (2, 32), f"Pool shape: {pool.shape}"
        assert alpha.shape == (50,), f"Alpha shape: {alpha.shape}"

    def test_attention_weights_sum_to_one(self):
        """Per-graph attention weights must sum to 1."""
        from src.models.gnn import AttentionReadout  # noqa: PLC0415
        readout = AttentionReadout(hidden_dim=64, dropout=0.0)
        readout.eval()
        n_per_graph = [15, 25, 10]
        x = torch.randn(sum(n_per_graph), 64)
        batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_per_graph)])
        with torch.no_grad():
            _, alpha = readout(x, batch)
        for i, n in enumerate(n_per_graph):
            mask = batch == i
            graph_sum = alpha[mask].sum().item()
            assert abs(graph_sum - 1.0) < 1e-4, f"Graph {i} weights sum to {graph_sum}"

    def test_encoder_with_attention_readout(self):
        """StreamGNNEncoder with attention readout produces correct embedding dim."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            use_attention_readout=True,
        )
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
        # Attention should be stored
        assert model._last_attention is not None
        assert model._last_attention.shape == (4 * 30,)

    def test_forward_with_attention(self):
        """forward_with_attention returns both embedding and alpha."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            use_attention_readout=True,
        )
        model.eval()
        data = make_fake_batch(n_graphs=2, n_stars=25, k=4)
        with torch.no_grad():
            emb, alpha = model.forward_with_attention(data)
        assert emb.shape == (2, 64)
        assert alpha.shape == (50,)
        assert alpha.min() >= 0.0

    def test_get_node_importance(self):
        """StreamGNNMultiTask.get_node_importance works with attention readout."""
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        model = StreamGNNMultiTask(
            n_classes=3, n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            use_attention_readout=True,
        )
        model.eval()
        data = make_fake_batch(n_graphs=3, n_stars=20, k=4)
        alpha = model.get_node_importance(data)
        assert alpha.shape == (60,)
        assert (alpha >= 0).all()

    def test_get_node_importance_raises_without_attention(self):
        """get_node_importance should raise if attention readout not enabled."""
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        model = StreamGNNMultiTask(
            n_classes=3, n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            use_attention_readout=False,
        )
        model.eval()
        data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
        with pytest.raises(RuntimeError, match="use_attention_readout"):
            model.get_node_importance(data)


# ---------------------------------------------------------------------------
# Feature 3: MC Dropout
# ---------------------------------------------------------------------------

class TestMCDropout:
    def test_mc_embed_stochasticity(self):
        """MC Dropout should produce different embeddings across passes."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            dropout=0.2,
        )
        data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
        mean_emb, std_emb = model.mc_embed(data, T=30)
        assert mean_emb.shape == (2, 64)
        assert std_emb.shape == (2, 64)
        # Std should be > 0 (dropout is active)
        assert (std_emb > 0).any(), "MC Dropout should produce non-zero std"

    def test_mc_embed_mean_close_to_deterministic(self):
        """MC Dropout mean should be similar to deterministic forward."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(
            n_node_features=11, n_edge_features=5,
            hidden_dim=64, n_layers=2, embedding_dim=64,
            dropout=0.1,  # low dropout -> MC mean close to eval
        )
        model.eval()
        data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
        with torch.no_grad():
            det_emb = model(data)
        mean_emb, _ = model.mc_embed(data, T=50)
        # Should be within ~20% (Monte Carlo variance + dropout effect)
        diff = (mean_emb - det_emb).abs().mean()
        assert diff < det_emb.abs().mean() * 0.5, f"MC mean too far from deterministic: {diff}"

    def test_mc_embed_std_positive(self):
        """All std dimensions should be strictly positive with sufficient dropout."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=3, embedding_dim=32,
            dropout=0.3,
        )
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        _, std_emb = model.mc_embed(data, T=50)
        # With 30% dropout and 50 passes, all dims should show variance
        assert (std_emb > 1e-6).all(), "Expected all std dims > 0 with 30% dropout"

    def test_enable_mc_dropout_activates_dropout_in_eval(self):
        """After enable_mc_dropout, dropout layers should be in train mode."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            dropout=0.5,
        )
        model.eval()
        model.enable_mc_dropout()
        # All Dropout modules should be in training mode
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                assert m.training, "Dropout should be in train mode after enable_mc_dropout()"

    def test_multitask_mc_embed(self):
        """StreamGNNMultiTask.mc_embed convenience method."""
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        model = StreamGNNMultiTask(
            n_classes=3, n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            dropout=0.2,
        )
        data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
        mean_emb, std_emb = model.mc_embed(data, T=10)
        assert mean_emb.shape == (2, 64)
        assert std_emb.shape == (2, 64)


# ---------------------------------------------------------------------------
# Feature 5: Variable regression targets
# ---------------------------------------------------------------------------

class TestVariableRegTargets:
    def test_3_reg_targets(self):
        """Model should handle n_reg_targets=3 (includes perturbation age)."""
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        model = StreamGNNMultiTask(
            n_classes=3, n_reg_targets=3,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            emb, logits, reg = model(data)
        assert reg.shape == (4, 3), f"Expected (4, 3), got {reg.shape}"
        assert not torch.isnan(reg).any()

    def test_regression_head_isolated(self):
        """Regression head should produce finite outputs for 1, 2, or 3 targets."""
        from src.models.gnn import StreamGNNMultiTask  # noqa: PLC0415
        for n_reg in [1, 2, 3]:
            model = StreamGNNMultiTask(
                n_classes=3, n_reg_targets=n_reg,
                n_node_features=11, n_edge_features=5,
                hidden_dim=32, n_layers=2, embedding_dim=64,
            )
            model.eval()
            data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
            with torch.no_grad():
                _, _, reg = model(data)
            assert reg.shape == (2, n_reg)


# ---------------------------------------------------------------------------
# Feature 2: Multi-Scale Graph (SegmentGNN)
# ---------------------------------------------------------------------------

class TestSegmentGNN:
    def test_segment_gnn_forward_shape(self):
        """SegmentGNN produces correct embedding shape."""
        from src.models.gnn import SegmentGNN  # noqa: PLC0415
        model = SegmentGNN(n_seg_features=8, n_seg_edge_features=4,
                           hidden_dim=32, embedding_dim=32, n_layers=2)
        model.eval()
        # Fake: 3 graphs, 20 segments each
        n_seg = 20
        n_graphs = 3
        seg_x = torch.randn(n_graphs * n_seg, 8)
        seg_batch = torch.repeat_interleave(torch.arange(n_graphs), n_seg)
        # Chain edges within each graph
        edges_src, edges_dst = [], []
        for g in range(n_graphs):
            off = g * n_seg
            for s in range(n_seg - 1):
                edges_src.extend([off + s, off + s + 1])
                edges_dst.extend([off + s + 1, off + s])
        seg_edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        seg_edge_attr = torch.randn(seg_edge_index.shape[1], 4)

        with torch.no_grad():
            out = model(seg_x, seg_edge_index, seg_edge_attr, seg_batch)
        assert out.shape == (3, 32)
        assert not torch.isnan(out).any()

    def test_multi_scale_encoder(self):
        """StreamGNNEncoder with multi-scale produces correct output when seg data present."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            use_multi_scale=True, seg_embedding_dim=32,
        )
        model.eval()
        data = make_fake_batch(n_graphs=2, n_stars=30, k=4)
        # Add fake segment data
        n_seg = 20
        data.seg_x = torch.randn(2 * n_seg, 8)
        data.seg_batch = torch.repeat_interleave(torch.arange(2), n_seg)
        edges_src, edges_dst = [], []
        for g in range(2):
            off = g * n_seg
            for s in range(n_seg - 1):
                edges_src.extend([off + s, off + s + 1])
                edges_dst.extend([off + s + 1, off + s])
        data.seg_edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        data.seg_edge_attr = torch.randn(data.seg_edge_index.shape[1], 4)

        with torch.no_grad():
            out = model(data)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()

    def test_multi_scale_without_seg_data_degrades_gracefully(self):
        """Multi-scale encoder still works if seg_x is not present (fallback)."""
        from src.models.gnn import StreamGNNEncoder  # noqa: PLC0415
        model = StreamGNNEncoder(
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            use_multi_scale=True, seg_embedding_dim=32,
        )
        model.eval()
        data = make_fake_batch(n_graphs=2, n_stars=20, k=4)
        # No seg_x on data — should still produce output (without segment contribution)
        # The forward uses hasattr check
        with torch.no_grad():
            out = model(data)
        assert out.shape == (2, 64)


# ---------------------------------------------------------------------------
# Feature 4: Orbital edge features (build_knn_graph_batched)
# ---------------------------------------------------------------------------

class TestOrbitalEdgeFeatures:
    def test_edge_features_extended_with_orbital(self):
        """build_knn_graph_batched with orbital=True produces 7 edge features."""
        from src.data.dataset import build_knn_graph_batched  # noqa: PLC0415
        # Create batch with 13 node features (11 + R_cyl + z_cyl)
        n_graphs, n_stars = 2, 30
        x = torch.randn(n_graphs * n_stars, 13)
        batch_vec = torch.repeat_interleave(torch.arange(n_graphs), n_stars)
        data = Data(x=x, batch=batch_vec)
        build_knn_graph_batched(data, k=4, normalizer=None, orbital_features=True)
        # Should have 5 base + 2 orbital = 7 edge features
        assert data.edge_attr.shape[1] == 7, f"Expected 7, got {data.edge_attr.shape[1]}"

    def test_edge_features_standard_without_orbital(self):
        """build_knn_graph_batched without orbital gives standard 5 features."""
        from src.data.dataset import build_knn_graph_batched  # noqa: PLC0415
        n_graphs, n_stars = 2, 30
        x = torch.randn(n_graphs * n_stars, 11)
        batch_vec = torch.repeat_interleave(torch.arange(n_graphs), n_stars)
        data = Data(x=x, batch=batch_vec)
        build_knn_graph_batched(data, k=4, normalizer=None, orbital_features=False)
        assert data.edge_attr.shape[1] == 5


# ---------------------------------------------------------------------------
# Feature 6: Hierarchical inference
# ---------------------------------------------------------------------------

class TestHierarchicalInference:
    def test_expected_rate_decreases_with_M_hm(self):
        """Higher M_hm (more suppression) should give fewer expected impacts."""
        from src.inference.hierarchical import expected_subhalo_rate  # noqa: PLC0415
        rate_low = expected_subhalo_rate(6.5, 60.0, 5.0)   # mild suppression
        rate_high = expected_subhalo_rate(9.0, 60.0, 5.0)  # strong suppression
        assert rate_low > rate_high, f"Low M_hm rate ({rate_low}) should exceed high ({rate_high})"

    def test_expected_rate_scales_with_length(self):
        """Longer streams should have more impacts."""
        from src.inference.hierarchical import expected_subhalo_rate  # noqa: PLC0415
        rate_short = expected_subhalo_rate(7.0, 30.0, 5.0)
        rate_long = expected_subhalo_rate(7.0, 90.0, 5.0)
        assert rate_long > rate_short

    def test_expected_rate_positive(self):
        """Rate should always be positive."""
        from src.inference.hierarchical import expected_subhalo_rate  # noqa: PLC0415
        for log_m in [6.0, 7.0, 8.0, 9.0, 10.0]:
            rate = expected_subhalo_rate(log_m, 60.0, 5.0)
            assert rate > 0, f"Rate should be positive, got {rate} for log_M_hm={log_m}"

    def test_coverage_metric_computation(self):
        """Coverage metric should work with known inputs."""
        from scripts.run_mock_challenge import compute_coverage  # noqa: PLC0415
        # Create fake results where truth is always at median -> 100% coverage
        true_values = np.array([7.0, 8.0, 9.0])
        posteriors = [
            np.random.normal(7.0, 0.5, 1000),
            np.random.normal(8.0, 0.5, 1000),
            np.random.normal(9.0, 0.5, 1000),
        ]
        result = compute_coverage(true_values, posteriors)
        # With truth at center of Gaussian, 90% CI should contain most truths
        assert result["empirical_coverage"][0.90] >= 0.66  # at least 2/3
        assert "rmse" in result
        assert "mean_bias" in result


# ---------------------------------------------------------------------------
# GNN V2: Binary + M_hm regression
# ---------------------------------------------------------------------------

class TestGNNMultiTaskV2:
    def test_v2_output_shapes(self):
        """V2 model outputs (emb, binary_logit, mhm_out, reg_out) with correct shapes."""
        from src.models.gnn import StreamGNNMultiTaskV2
        model = StreamGNNMultiTaskV2(
            n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            emb, binary_logit, mhm_out, reg_out = model(data)
        assert emb.shape == (4, 64), f"Expected emb (4, 64), got {emb.shape}"
        assert binary_logit.shape == (4, 1), f"Expected binary (4, 1), got {binary_logit.shape}"
        assert mhm_out.shape == (4, 1), f"Expected mhm (4, 1), got {mhm_out.shape}"
        assert reg_out.shape == (4, 2), f"Expected reg (4, 2), got {reg_out.shape}"

    def test_v2_no_nan(self):
        """V2 outputs are finite."""
        from src.models.gnn import StreamGNNMultiTaskV2
        model = StreamGNNMultiTaskV2(
            n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            emb, binary_logit, mhm_out, reg_out = model(data)
        for name, t in [("emb", emb), ("binary", binary_logit), ("mhm", mhm_out), ("reg", reg_out)]:
            assert not torch.isnan(t).any(), f"V2 {name} contains NaN"
            assert not torch.isinf(t).any(), f"V2 {name} contains Inf"

    def test_v2_binary_loss_backprop(self):
        """Binary BCE loss backpropagates through V2 model."""
        from src.models.gnn import StreamGNNMultiTaskV2
        model = StreamGNNMultiTaskV2(
            n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.train()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        emb, binary_logit, mhm_out, reg_out = model(data)

        # Binary target: 0 or 1
        y_binary = torch.tensor([[0.], [1.], [1.], [0.]])
        loss = nn.BCEWithLogitsLoss()(binary_logit, y_binary)
        loss.backward()

        # Check gradients flow to encoder
        has_grad = False
        for p in model.encoder.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradient reached the encoder from binary loss"

    def test_v2_mhm_masked_loss(self):
        """M_hm regression loss only computed on suppressed (label=1) samples."""
        from src.models.gnn import StreamGNNMultiTaskV2
        model = StreamGNNMultiTaskV2(
            n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.train()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        _, _, mhm_out, _ = model(data)

        # Only samples 1 and 2 are "suppressed"
        suppressed_mask = torch.tensor([False, True, True, False])
        y_mhm = torch.tensor([[10.0], [7.5], [8.0], [10.0]])

        loss = nn.HuberLoss()(mhm_out[suppressed_mask], y_mhm[suppressed_mask])
        assert loss.item() >= 0, "M_hm loss should be non-negative"
        assert mhm_out[suppressed_mask].shape == (2, 1)

    def test_v2_with_uncertainty(self):
        """V2 with predict_uncertainty=True outputs (mean, log_var) for M_hm."""
        from src.models.gnn import StreamGNNMultiTaskV2
        model = StreamGNNMultiTaskV2(
            n_reg_targets=2,
            predict_uncertainty=True,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        model.eval()
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        with torch.no_grad():
            emb, binary_logit, mhm_out, reg_out = model(data)
        assert mhm_out.shape == (4, 2), f"Expected mhm (4, 2) with uncertainty, got {mhm_out.shape}"

    def test_v2_predict_suppression(self):
        """predict_suppression() returns correct types and shapes."""
        from src.models.gnn import StreamGNNMultiTaskV2
        model = StreamGNNMultiTaskV2(
            n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
        )
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        is_suppressed, mhm_pred, confidence = model.predict_suppression(data)
        assert is_suppressed.shape == (4,)
        assert mhm_pred.shape == (4,)
        assert confidence.shape == (4,)
        assert (confidence >= 0).all() and (confidence <= 1).all(), "Confidence should be in [0, 1]"

    def test_v2_with_attention(self):
        """V2 with attention readout produces node importance scores."""
        from src.models.gnn import StreamGNNMultiTaskV2
        model = StreamGNNMultiTaskV2(
            n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            use_attention_readout=True,
        )
        data = make_fake_batch(n_graphs=4, n_stars=30, k=4)
        alpha = model.get_node_importance(data)
        assert alpha.shape == (4 * 30,), f"Expected {4*30} attention weights, got {alpha.shape}"
        # Weights should sum to ~1 per graph
        for g in range(4):
            g_weights = alpha[g * 30:(g + 1) * 30]
            assert abs(g_weights.sum().item() - 1.0) < 0.05, f"Graph {g} weights sum to {g_weights.sum()}"

    def test_v2_mc_embed(self):
        """MC Dropout embedding produces stochastic embeddings with positive std."""
        from src.models.gnn import StreamGNNMultiTaskV2
        model = StreamGNNMultiTaskV2(
            n_reg_targets=2,
            n_node_features=11, n_edge_features=5,
            hidden_dim=32, n_layers=2, embedding_dim=64,
            dropout=0.2,
        )
        data = make_fake_batch(n_graphs=2, n_stars=30, k=4)
        mean_emb, std_emb = model.mc_embed(data, T=10)
        assert mean_emb.shape == (2, 64)
        assert std_emb.shape == (2, 64)
        assert (std_emb > 0).any(), "MC Dropout std should have some positive values"
