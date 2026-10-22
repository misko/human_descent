"""Public next-batch/dimension requests must release old tensor allocations."""

import weakref

import torch

from hudes.model_data_and_subspace import DatasetBatcher, ModelDataAndSubspace


def test_advancing_batches_releases_old_buffers_without_changing_data():
    dataset = [(torch.tensor([float(i)]), i % 2) for i in range(16)]
    batcher = DatasetBatcher(dataset)
    first = batcher.get_batch(1, 0)
    expected = first["data"].clone()
    retained = weakref.ref(first["data"])
    del first
    for index in range(1, 1024):
        batcher.get_batch(1, index)
    assert retained() is None
    assert torch.equal(batcher.get_batch(1, 0)["data"], expected)


def test_advancing_dimensions_releases_old_buffers_and_preserves_seeded_vectors():
    model = ModelDataAndSubspace(torch.nn.Linear(2, 2), None, None, None)
    model.fuse()
    first = model.get_dim_vec(0, torch.float32)
    expected = first.clone()
    retained = weakref.ref(first)
    del first
    for dimension in range(1, 1024):
        model.get_dim_vec(dimension, torch.float32)
    assert retained() is None
    assert torch.equal(model.get_dim_vec(0, torch.float32), expected)
