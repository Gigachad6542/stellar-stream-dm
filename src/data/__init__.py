"""Data acquisition, processing, and PyTorch datasets.

Modules:
    gaia_query     — ADQL queries to Gaia DR3 via astroquery
    stream_process — Stream frame transformation, dust correction, HDF5 output
    dataset        — PyG datasets for simulated and real streams, k-NN graph builder
    splits         — Reproducible stratified train/val/test split management
"""
