"""
Prometheus metrics for Alina Bot.
"""

from prometheus_client import Counter, Histogram, Gauge

# ---- Counters ----
MESSAGES_TOTAL = Counter(
    "alina_messages_total",
    "Total messages processed",
    ["type", "status"],  # type: text/photo, status: success/fallback/error
)

LLM_TOKENS = Counter(
    "alina_llm_tokens_total",
    "LLM tokens consumed",
    ["model", "direction"],  # direction: input/output
)

LLM_ERRORS = Counter(
    "alina_llm_errors_total",
    "LLM errors by provider and error type",
    ["provider", "error_type"],
)

# ---- Histograms ----
MESSAGE_LATENCY = Histogram(
    "alina_message_latency_seconds",
    "Message processing latency",
    ["type"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60),
)

DB_LATENCY = Histogram(
    "alina_db_latency_seconds",
    "Database query latency",
    ["operation"],  # get_user_context, save_message, etc.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

# ---- Gauges ----
ACTIVE_USERS = Gauge("alina_active_users", "Currently active users")
BG_QUEUE_SIZE = Gauge("alina_bg_queue_size", "Background task queue size")


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