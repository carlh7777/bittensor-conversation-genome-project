"""Shared text scrubbing for outbound log surfaces.

Single source of truth so the W&B handler series feed, the W&B console
capture (via stdout/stderr), and pm2's view of stdout all apply identical
drop/redact rules. Keeps decrypted miner endpoints — the whole reason
commitments exist — out of every observability surface.
"""

import re
import sys
import threading


# ─── Redaction patterns ──────────────────────────────────────────────

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b")

# IPv6 — require the compressed `::` form, which every modern stringifier
# produces. Avoids false-positives on HH:MM:SS timestamps.
_IPV6_RE = re.compile(r"\[?[A-Fa-f0-9:]*::[A-Fa-f0-9:]+\]?(?::\d{1,5})?")

# Any URL: scheme://host... whitespace-terminated.
_URL_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+")

# Hostnames that are safe to leave un-redacted. These never carry miner
# endpoint information — they're infra/docs/UI links we actively want to
# remain clickable in logs (e.g. the wandb run URL printed at init).
_URL_HOST_ALLOWLIST = (
    "wandb.ai",
    "wandb.com",
    "wandb.io",
    "github.com",
    "anthropic.com",
    "opentensor.ai",
    "bittensor.com",
    "conversations.xyz",   # CGP backend; not a miner
)


def _redact_url(match: re.Match) -> str:
    url = match.group(0)
    # Cheap allowlist check — any allowed host substring inside the URL means
    # it's safe to keep verbatim.
    if any(host in url for host in _URL_HOST_ALLOWLIST):
        return url
    return "[REDACTED_URL]"

# Dotted hostname:port (e.g. miner.example.com:8080).
_HOSTPORT_RE = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[A-Za-z]{2,}:\d{1,5}\b"
)

# Bare host:port — covers `myhost:8080` even without dots.
_BAREHOSTPORT_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9.-]{1,253}:\d{2,5}\b")


# ─── Drop list ────────────────────────────────────────────────────────
#
# Substrings that almost certainly mean a log line contains miner endpoint
# data we're trying to keep private. Drop entire matching lines rather than
# rely on regex redaction — defense in depth.

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
    "Dendrite ",
    "dendrite=",
)


def should_drop(line: str) -> bool:
    return any(s in line for s in _DROP_SUBSTRINGS)


def scrub(message: str) -> str:
    """Redact endpoint-shaped tokens. Order matters: URLs first so a full
    `http://1.2.3.4:5/CgSynapse` becomes one redaction token, then narrower
    IP/host patterns mop up bare cases.
    """
    message = _URL_RE.sub(_redact_url, message)
    message = _IPV4_RE.sub("[REDACTED_ENDPOINT]", message)
    message = _IPV6_RE.sub("[REDACTED_ENDPOINT]", message)
    message = _HOSTPORT_RE.sub("[REDACTED_ENDPOINT]", message)
    message = _BAREHOSTPORT_RE.sub("[REDACTED_ENDPOINT]", message)
    return message


# ─── stdout / stderr scrubbing wrapper ───────────────────────────────


class _ScrubbingStream:
    """File-like wrapper that drops sensitive lines and redacts the rest
    before forwarding to the real stream.

    Reentrancy is guarded so a write() that itself triggers logging (rare,
    but possible if the underlying stream raises) doesn't loop.
    """

    def __init__(self, real_stream):
        self._real = real_stream
        self._reentrant = threading.local()

    def write(self, data):
        if not isinstance(data, str):
            # Some loggers pass bytes; pass through unmodified rather than
            # making assumptions about encoding.
            return self._real.write(data)
        if getattr(self._reentrant, "active", False):
            return self._real.write(data)
        if not data:
            return 0
        self._reentrant.active = True
        try:
            # splitlines(keepends=True) keeps separators so a write of
            # "a\nb" preserves boundaries exactly.
            lines = data.splitlines(keepends=True)
            out = []
            for line in lines:
                if should_drop(line):
                    continue
                out.append(scrub(line))
            return self._real.write("".join(out))
        finally:
            self._reentrant.active = False

    def flush(self):
        return self._real.flush()

    def __getattr__(self, name):
        # Proxy everything else (isatty, fileno, etc.) to the real stream.
        return getattr(self._real, name)


def install_stdio_scrubbers() -> None:
    """Replace sys.stdout and sys.stderr with scrubbing wrappers.

    Idempotent: subsequent calls are no-ops. Safe to call before or after
    wandb.init — the wrappers stay attached for the process lifetime.
    """
    if not isinstance(sys.stdout, _ScrubbingStream):
        sys.stdout = _ScrubbingStream(sys.stdout)
    if not isinstance(sys.stderr, _ScrubbingStream):
        sys.stderr = _ScrubbingStream(sys.stderr)
