"""Basic benchmarking script for mesh_serve."""

import time
import logging
import mesh_serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark() -> None:
    logger.info("Starting benchmark for %s", "mesh_serve")
    start = time.time()
    # TODO: Add specific performance tests
    end = time.time()
    logger.info("Benchmark completed in %.4f seconds", end - start)


if __name__ == "__main__":
    run_benchmark()
