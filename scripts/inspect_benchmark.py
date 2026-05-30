"""Quick inspection of the benchmark HDF5 output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
from collections import Counter

path = Path("data/simulations/chunk_00000.h5")
with h5py.File(str(path), "r") as f:
    sims = f["simulations"]
    run_ids = list(sims.keys())
    print(f"Total simulations: {len(run_ids)}")

    dm_models = []
    streams = []
    n_particles = []
    n_subhalos = []

    for rid in run_ids:
        g = sims[rid]
        dm = g.attrs["dm_model"]
        if isinstance(dm, bytes):
            dm = dm.decode()
        dm_models.append(dm)

        sn = g.attrs["stream_name"]
        if isinstance(sn, bytes):
            sn = sn.decode()
        streams.append(sn)

        n_particles.append(len(g["stream_data/phi1"]))
        n_subhalos.append(int(g["labels"].attrs["n_subhalos"]))

    print(f"\nDM model distribution:")
    for model, count in sorted(Counter(dm_models).items()):
        print(f"  {model}: {count}")

    print(f"\nStream distribution:")
    for stream, count in sorted(Counter(streams).items()):
        print(f"  {stream}: {count}")

    print(f"\nParticle counts: min={min(n_particles)}, max={max(n_particles)}, "
          f"mean={np.mean(n_particles):.0f}, median={np.median(n_particles):.0f}")

    print(f"\nSubhalo encounters: min={min(n_subhalos)}, max={max(n_subhalos)}, "
          f"mean={np.mean(n_subhalos):.1f}")

    # Check one sim in detail
    rid0 = run_ids[0]
    g0 = sims[rid0]
    print(f"\n--- Example sim: {rid0} ---")
    print(f"  DM model: {g0.attrs['dm_model']}")
    print(f"  Stream: {g0.attrs['stream_name']}")
    print(f"  Datasets in stream_data/:")
    for k in g0["stream_data"].keys():
        ds = g0["stream_data"][k]
        print(f"    {k}: shape={ds.shape}, dtype={ds.dtype}, "
              f"range=[{ds[:].min():.3f}, {ds[:].max():.3f}]")
    print(f"  Labels:")
    for k, v in g0["labels"].attrs.items():
        print(f"    {k}: {v}")
    if len(g0["subhalos/mass"]) > 0:
        print(f"  Subhalos: {len(g0['subhalos/mass'])} encounters")
        print(f"    masses: {g0['subhalos/mass'][:]}")
    else:
        print(f"  Subhalos: 0 encounters")
