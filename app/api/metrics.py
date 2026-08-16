import time
from collections import defaultdict


request_count = 0
error_count = 0
total_latency = 0.0


def record_request(latency: float, error: bool = False):

    global request_count
    global error_count
    global total_latency

    request_count += 1
    total_latency += latency

    if error:
        error_count += 1


def get_metrics():

    avg_latency = (
        total_latency / request_count
        if request_count > 0
        else 0
    )

    error_rate = (
        error_count / request_count
        if request_count > 0
        else 0
    )

    return {
        "total_requests": request_count,
        "errors": error_count,
        "error_rate": round(error_rate, 4),
        "average_latency_seconds": round(avg_latency, 4),
    }