"""Tests for v2 simulation labels and membership probabilities."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.dataset import StreamSimDataset


def _write_minimal_sim(path: Path) -> None:
    n = 8
    with h5py.File(path, "w") as f:
        grp = f.create_group("simulations/run0001")
        grp.attrs["stream_name"] = np.bytes_("GD1")
        sd = grp.create_group("stream_data")
        base = np.linspace(0.0, 1.0, n, dtype=np.float32)
        for name in ["phi1", "phi2", "dist", "pm1", "pm2", "vrad"]:
            sd.create_dataset(name, data=base)
        for name in ["e_dist", "e_pm1", "e_pm2", "e_vrad"]:
            sd.create_dataset(name, data=np.ones(n, dtype=np.float32))
        membership = np.linspace(1.0, 0.0, n, dtype=np.float32)
        sd.create_dataset("membership_prob", data=membership)

        lab = grp.create_group("labels")
        lab.create_dataset("dm_model_idx", data=1)
        lab.create_dataset("log_m_sub_mean", data=7.1)
        lab.create_dataset("n_subhalos", data=3)
        lab.create_dataset("log_t_since_last_impact_gyr", data=0.2)
        lab.create_dataset("log10_M_hm", data=7.6)


def test_dataset_reads_membership_prob_and_mhm_label(tmp_path):
    h5_path = tmp_path / "chunk_00000.h5"
    _write_minimal_sim(h5_path)

    dataset = StreamSimDataset(tmp_path, preload_ram=False, max_stars=100)
    data = dataset[0]

    assert data.x.shape[1] == 18
    np.testing.assert_allclose(data.x[:, 10].numpy(), np.linspace(1.0, 0.0, 8))
    assert data.y.shape == (6,)
    assert data.y[0].item() == 1
    assert data.y[4].item() == pytest.approx(7.6)
    assert data.y[5].item() == pytest.approx(0.0)  # impact_strength defaults to 0


def test_dataset_reads_dm_labels_without_feature_loading(tmp_path):
    h5_path = tmp_path / "chunk_00000.h5"
    _write_minimal_sim(h5_path)

    dataset = StreamSimDataset(tmp_path, preload_ram=False, max_stars=100)

    labels = dataset.get_dm_model_labels()
    assert labels.tolist() == [1]


def test_dataset_reads_timeline_label_schema(tmp_path):
    h5_path = tmp_path / "chunk_00000.h5"
    _write_minimal_sim(h5_path)

    with h5py.File(h5_path, "r+") as f:
        labels = f["simulations/run0001/labels"]
        labels.attrs["impact_strength"] = 1.2
        labels.attrs["n_detectable_impacts"] = 2
        labels.attrs["effective_n_impacts"] = 1.75
        labels.attrs["log_t_since_strongest_impact_gyr"] = 0.6
        labels.attrs["strongest_impact_t_since_gyr"] = 4.0
        labels.attrs["timeline_effective_impacts"] = np.asarray([0.0, 0.25, 1.0, 0.5, 0.0], dtype=np.float32)

    dataset = StreamSimDataset(tmp_path, preload_ram=False, max_stars=100, label_schema="timeline")
    data = dataset[0]
    labels = dataset.get_label_matrix()

    assert data.y.shape == (15,)
    assert data.y[5].item() == pytest.approx(1.2)
    assert data.y[6].item() == pytest.approx(2.0)
    assert data.y[7].item() == pytest.approx(1.75)
    assert data.y[8].item() == pytest.approx(0.6)
    np.testing.assert_allclose(data.y[10:].numpy(), [0.0, 0.25, 1.0, 0.5, 0.0])
    assert labels.shape == (1, 15)
