import logging
import re
import threading


# ─── Redaction patterns ───────────────────────────────────────────────
# These run in order over any line that survives the drop list.

# IPv4 with optional :port
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")

# IPv6 — require the compressed `::` form, which is what every modern
# stringifier produces. Avoids false-positives on `HH:MM:SS` timestamps.
# Full-form IPv6 (8 groups, no `::`) is vanishingly rare in bittensor logs
# and would be caught by the URL regex anyway when inside an http(s)://.
_IPV6_RE = re.compile(r"\[?[A-Fa-f0-9:]*::[A-Fa-f0-9:]+\]?(?::\d{1,5})?")

# Any URL: scheme://anything (whitespace-terminated). Catches host:port
# embedded in axon URLs, including weird schemes the bittensor client uses.
_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+")

# host:port where host is a dotted hostname (e.g. miner.example.com:8080).
_HOSTPORT_RE = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[A-Za-z]{2,}:\d{1,5}\b"
)

# Bare host:port with a numeric port — covers `myhost:8080` even without dots.
_BAREHOSTPORT_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9.-]{1,253}:\d{2,5}\b")


# ─── Drop list ────────────────────────────────────────────────────────
# Substrings that mean the message body almost certainly contains miner
# endpoint info. We drop the entire log line rather than rely on regex
# redaction — defense in depth.
_DROP_SUBSTRINGS = (
    "ClientConnectorError",
    "ContentTypeError",
    "ServerDisconnectedError",
    "ServerTimeoutError",
    "ClientPayloadError",
    "ClientResponseError",
    "ConnectionRefusedError",
    "Cannot connect to host",
    "Connect call failed",
    "/CgSynapse",
    "axon=",
    "endpoint=",
    "ip=",
    "Dendrite ",
    "dendrite=",
)

# Any record originating from these bittensor source modules is dropped
# wholesale — they exclusively handle endpoint/network info and have no
# safe-by-default messages we care about in W&B analytics.
_DROP_SOURCE_KEYWORDS = (
    "dendrite",
    "axon",
    "synapse",
    "subtensor",  # query logs that include node ws:// urls
    "websocket",
    "metagraph",  # has occasional endpoint blobs in debug
)


def _scrub(message: str) -> str:
    """Redact anything endpoint-shaped from a log message.

    Order matters: URLs first (so we replace `http://1.2.3.4:5/CgSynapse`
    as one token), then narrower IP/host patterns to mop up leftovers.
    """
    message = _URL_RE.sub("[REDACTED_URL]", message)
    message = _IPV4_RE.sub("[REDACTED_ENDPOINT]", message)
    message = _IPV6_RE.sub("[REDACTED_ENDPOINT]", message)
    message = _HOSTPORT_RE.sub("[REDACTED_ENDPOINT]", message)
    message = _BAREHOSTPORT_RE.sub("[REDACTED_ENDPOINT]", message)
    return message


def _is_from_network_module(record: logging.LogRecord, formatted: str) -> bool:
    """Detect bittensor network-layer log records by both record metadata
    and the formatted message (bittensor's custom formatter encodes the
    source file in the message text, e.g. `bittensor:dendrite.py:262`).
    """
    pathname = (getattr(record, "pathname", "") or "").lower()
    module = (getattr(record, "module", "") or "").lower()
    lowered = formatted.lower()
    for kw in _DROP_SOURCE_KEYWORDS:
        if kw in pathname or kw in module:
            return True
        # Match `bittensor:dendrite.py` style source tag in the rendered text.
        if f":{kw}.py" in lowered or f"{kw}.py:" in lowered:
            return True
    return False


class WandbCountingHandler(logging.Handler):
    def __init__(self, wandb_lib_instance):
        super().__init__()
        self.wandb_lib = wandb_lib_instance
        # Reentrancy guard: if wandb_lib.log() raises and that raise gets
        # logged through bittensor, the handler would recurse and blow the
        # stack (we saw this during shutdown). Per-thread flag stops it.
        self._in_emit = threading.local()

    def emit(self, record):
        if getattr(self._in_emit, "active", False):
            return
        self._in_emit.active = True
        try:
            log_entry = self.format(record)

            # Drop bittensor network-layer log records wholesale — they're
            # the dominant source of endpoint leaks (dendrite errors,
            # subtensor RPC URLs, axon registration blobs).
            if _is_from_network_module(record, log_entry):
                return

            # Drop any message containing known endpoint-leaking substrings.
            if any(s in log_entry for s in _DROP_SUBSTRINGS):
                return

            # Belt-and-suspenders: redact remaining IP / URL / host:port
            # patterns before forwarding.
            log_entry = _scrub(log_entry)

            self.wandb_lib.log({"bt_log": log_entry})
        except Exception as e:
            # NB: use print, not bt.logging — otherwise this re-enters emit().
            print(f"Logging handler error: {e}")
        finally:
            self._in_emit.active = False
