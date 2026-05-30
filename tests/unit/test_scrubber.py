"""Tests for the stdio scrubbing wrapper used to filter W&B console capture."""

import io
import sys

from conversationgenome.analytics import _scrubber
from conversationgenome.analytics._scrubber import (
    _ScrubbingStream,
    install_stdio_scrubbers,
)


class FakeStream:
    """Stand-in for sys.stdout that records every write call."""
    def __init__(self):
        self.buf = io.StringIO()
    def write(self, data):
        return self.buf.write(data)
    def flush(self):
        pass
    def getvalue(self):
        return self.buf.getvalue()


class TestUrlAllowlist:
    """Wandb's own infra URLs and other safe domains must survive scrubbing,
    so the wandb init banner stays clickable in pm2 logs."""

    def test_wandb_run_url_passes(self):
        from conversationgenome.analytics._scrubber import scrub
        out = scrub("View run at https://wandb.ai/cgp/validator-78-2.32.68")
        assert "wandb.ai/cgp/validator-78-2.32.68" in out
        assert "REDACTED" not in out

    def test_wandb_docs_url_passes(self):
        from conversationgenome.analytics._scrubber import scrub
        out = scrub("see https://docs.wandb.ai/guides/track for info")
        assert "docs.wandb.ai" in out

    def test_github_url_passes(self):
        from conversationgenome.analytics._scrubber import scrub
        out = scrub("file a bug at https://github.com/afterpartyai/bittensor-conversation-genome-project/issues")
        assert "github.com" in out

    def test_unknown_url_is_redacted(self):
        from conversationgenome.analytics._scrubber import scrub
        out = scrub("calling http://18.119.135.29:8210/CgSynapse to score miner")
        assert "18.119.135.29" not in out
        assert "REDACTED_URL" in out

    def test_unknown_https_url_is_redacted(self):
        from conversationgenome.analytics._scrubber import scrub
        out = scrub("reaching https://miner.somewhere.io:9000/foo")
        assert "miner.somewhere.io" not in out
        assert "REDACTED" in out


class TestScrubbingStream:
    def test_drops_line_matching_drop_pattern(self):
        real = FakeStream()
        s = _ScrubbingStream(real)
        s.write("ClientConnectorError: cannot reach 18.119.135.29:8210\n")
        assert real.getvalue() == "", "drop-pattern line must not reach real stream"

    def test_redacts_ip_in_non_drop_line(self):
        real = FakeStream()
        s = _ScrubbingStream(real)
        s.write("Heartbeat from 10.0.0.5:7000\n")
        out = real.getvalue()
        assert "10.0.0.5" not in out
        assert "7000" not in out
        assert "REDACTED" in out

    def test_passes_clean_text(self):
        real = FakeStream()
        s = _ScrubbingStream(real)
        s.write("Validator step 1234 complete\n")
        assert real.getvalue() == "Validator step 1234 complete\n"

    def test_mixed_multiline_write(self):
        real = FakeStream()
        s = _ScrubbingStream(real)
        s.write(
            "Validator step 1234 complete\n"
            "ClientConnectorError: 1.2.3.4:5000\n"
            "Heartbeat from 10.0.0.5\n"
        )
        out = real.getvalue()
        assert "Validator step 1234 complete" in out
        assert "ClientConnectorError" not in out
        assert "1.2.3.4" not in out
        assert "10.0.0.5" not in out
        assert "REDACTED" in out

    def test_empty_write_is_noop(self):
        real = FakeStream()
        s = _ScrubbingStream(real)
        assert s.write("") == 0
        assert real.getvalue() == ""

    def test_bytes_pass_through_unmodified(self):
        """Some loggers write bytes; we forward without trying to decode."""
        captured = []
        class ByteStream:
            def write(self, data):
                captured.append(data)
                return len(data) if hasattr(data, "__len__") else 0
            def flush(self): pass
        s = _ScrubbingStream(ByteStream())
        s.write(b"raw bytes 1.2.3.4")
        assert captured == [b"raw bytes 1.2.3.4"]

    def test_attribute_proxy(self):
        """isatty, fileno, etc. must proxy to the real stream."""
        class TtyStream:
            def isatty(self): return True
            def fileno(self): return 99
            def write(self, _): return 0
            def flush(self): pass
        s = _ScrubbingStream(TtyStream())
        assert s.isatty() is True
        assert s.fileno() == 99


class TestInstallStdioScrubbers:
    def test_installs_on_stdout_and_stderr(self, monkeypatch):
        # Use fresh fakes so we don't affect the real test runner streams.
        fake_out, fake_err = FakeStream(), FakeStream()
        monkeypatch.setattr(sys, "stdout", fake_out)
        monkeypatch.setattr(sys, "stderr", fake_err)

        install_stdio_scrubbers()

        assert isinstance(sys.stdout, _ScrubbingStream)
        assert isinstance(sys.stderr, _ScrubbingStream)

    def test_install_is_idempotent(self, monkeypatch):
        fake_out = FakeStream()
        monkeypatch.setattr(sys, "stdout", fake_out)

        install_stdio_scrubbers()
        first_wrapper = sys.stdout
        install_stdio_scrubbers()
        # Second call must not wrap again, otherwise nested wrappers stack.
        assert sys.stdout is first_wrapper

    def test_installed_stdout_actually_scrubs(self, monkeypatch):
        fake_out = FakeStream()
        monkeypatch.setattr(sys, "stdout", fake_out)
        install_stdio_scrubbers()

        sys.stdout.write("Reaching out to 18.119.135.29:8210\n")

        out = fake_out.getvalue()
        assert "18.119.135.29" not in out
        assert "REDACTED" in out
