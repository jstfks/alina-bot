"""
Prometheus metrics for Alina Bot.
"""

import os
import threading
import time
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, push_to_gateway

# ---- Registry ----
REGISTRY = CollectorRegistry()

# ---- Counters ----
MESSAGES_TOTAL = Counter(
    "alina_messages_total",
    "Total messages processed",
    ["type", "status"],  # type: text/photo, status: success/fallback/error
    registry=REGISTRY,
)

LLM_TOKENS = Counter(
    "alina_llm_tokens_total",
    "LLM tokens consumed",
    ["model", "direction"],  # direction: input/output
    registry=REGISTRY,
)

LLM_ERRORS = Counter(
    "alina_llm_errors_total",
    "LLM errors by provider and error type",
    ["provider", "error_type"],
    registry=REGISTRY,
)

# ---- Histograms ----
MESSAGE_LATENCY = Histogram(
    "alina_message_latency_seconds",
    "Message processing latency",
    ["type"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60),
    registry=REGISTRY,
)

DB_LATENCY = Histogram(
    "alina_db_latency_seconds",
    "Database query latency",
    ["operation"],  # get_user_context, save_message, etc.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    registry=REGISTRY,
)

# ---- Gauges ----
ACTIVE_USERS = Gauge("alina_active_users", "Currently active users", registry=REGISTRY)
BG_QUEUE_SIZE = Gauge("alina_bg_queue_size", "Background task queue size", registry=REGISTRY)


def record_message(type_: str, status: str, latency: float, model: str = "", tokens_in: int = 0, tokens_out: int = 0):
    """Record message processing metrics."""
    MESSAGES_TOTAL.labels(type=type_, status=status).inc()
    MESSAGE_LATENCY.labels(type=type_).observe(latency)
    if tokens_in:
        LLM_TOKENS.labels(model=model, direction="input").inc(tokens_in)
    if tokens_out:
        LLM_TOKENS.labels(model=model, direction="output").inc(tokens_out)


def record_db_latency(operation: str, latency: float):
    """Record database query latency."""
    DB_LATENCY.labels(operation=operation).observe(latency)


def record_llm_error(provider: str, error_type: str):
    """Record LLM error."""
    LLM_ERRORS.labels(provider=provider, error_type=error_type).inc()


# ---- Grafana Cloud Push ----
GRAFANA_CLOUD_PROMETHEUS_URL = os.getenv("GRAFANA_CLOUD_PROMETHEUS_URL")
GRAFANA_CLOUD_USER = os.getenv("GRAFANA_CLOUD_USER")
GRAFANA_CLOUD_API_KEY = os.getenv("GRAFANA_CLOUD_API_KEY")

# Import log from logging_config to avoid circular imports
from logging_config import log

def _push_metrics_periodically():
    """Background task to push metrics to Grafana Cloud."""
    if not (GRAFANA_CLOUD_PROMETHEUS_URL and GRAFANA_CLOUD_USER and GRAFANA_CLOUD_API_KEY):
        return

    # Extract host from URL for push_to_gateway
    # URL format: https://prometheus-prod-XX.grafana.net/api/prom/push
    # push_to_gateway expects host:port without scheme and path
    url = GRAFANA_CLOUD_PROMETHEUS_URL
    if url.startswith("https://"):
        url = url[8:]  # remove https://
    if url.startswith("http://"):
        url = url[7:]  # remove http://
    # Remove /api/prom/push path
    if "/api/prom/push" in url:
        url = url.replace("/api/prom/push", "")

    while True:
        try:
            # push_to_gateway doesn't accept username/password kwargs
            # Use basic auth via the URL or use the auth parameter
            # Format: push_to_gateway(gateway, job, registry, auth=(username, password))
            push_to_gateway(
                url,
                job="alina-bot",
                registry=REGISTRY,
                auth=(GRAFANA_CLOUD_USER, GRAFANA_CLOUD_API_KEY),
            )
            log.info("Metrics pushed to Grafana Cloud")
        except Exception as e:
            log.error("Failed to push metrics to Grafana Cloud", error=str(e))
        time.sleep(60)  # push every minute


# Start background pusher if configured
if os.getenv("GRAFANA_CLOUD_PROMETHEUS_URL") and os.getenv("GRAFANA_CLOUD_USER") and os.getenv("GRAFANA_CLOUD_API_KEY"):
    import threading
    push_thread = threading.Thread(target=_push_metrics_periodically, daemon=True)
    push_thread.start()
    log.info("Grafana Cloud metrics pusher started")