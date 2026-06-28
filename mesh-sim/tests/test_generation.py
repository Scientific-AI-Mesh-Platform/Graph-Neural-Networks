import pytest
from mesh_sim.generation import generate_samples


def test_generate_samples() -> None:
    samples = generate_samples(num_samples=2, noise_level=0.1)
    assert len(samples) == 2
