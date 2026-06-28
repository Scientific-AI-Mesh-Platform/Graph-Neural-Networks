"""Basic benchmarking script for mesh_sim."""

import time
import logging
import mesh_sim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark() -> None:
    logger.info("Starting benchmark for %s", "mesh_sim")
    start = time.time()
    # TODO: Add specific performance tests
    end = time.time()
    logger.info("Benchmark completed in %.4f seconds", end - start)


if __name__ == "__main__":
    run_benchmark()
