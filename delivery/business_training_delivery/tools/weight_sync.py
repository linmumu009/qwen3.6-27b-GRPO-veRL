"""Replica-local rank contract for the delivery's colocated TP4 rollout."""


def colocated_local_rank(worker_local_rank, parallel_config):
    # Ray exposes a different physical device slice to each replica. The
    # socket name already contains the replica id, so its rank is local.
    rank = int(worker_local_rank)
    tp = int(getattr(parallel_config, "tensor_parallel_size", 1))
    pp = int(getattr(parallel_config, "pipeline_parallel_size", 1))
    dp = int(getattr(parallel_config, "data_parallel_size", 1))
    if dp != 1 or pp != 1 or not 0 <= rank < tp:
        raise ValueError("Delivery weight sync expects one TP-only engine per replica")
    return rank
