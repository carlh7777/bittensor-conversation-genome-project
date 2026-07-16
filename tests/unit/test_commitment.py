"""Tests for the encrypted commitment system."""

from unittest.mock import MagicMock, patch

import pytest
from nacl.public import PrivateKey

from conversationgenome.commitment.commitment import (
    decrypt_endpoint,
    encrypt_endpoint,
    publish_commitment,
    read_all_commitments,
    read_commitment,
)

# publish_commitment imports publish_metadata_extrinsic on bittensor 10.x and
# falls back to publish_metadata on 9.x. Patch whichever the installed version
# actually exposes so these tests pass on either (matches the code's fallback).
import bittensor.core.extrinsics.serving as _bt_serving

_PUBLISH_TARGET = "bittensor.core.extrinsics.serving." + (
    "publish_metadata_extrinsic"
    if hasattr(_bt_serving, "publish_metadata_extrinsic")
    else "publish_metadata"
)


# ── helpers ──────────────────────────────────────────────────────────

def _generate_keypair():
    """Generate a fresh NaCl keypair and return (public_bytes, private_bytes)."""
    private = PrivateKey.generate()
    return bytes(private.public_key), bytes(private)


# ── encrypt / decrypt round-trip ─────────────────────────────────────

class TestEncryptDecrypt:
    def test_round_trip_basic(self):
        pub, priv = _generate_keypair()
        ct = encrypt_endpoint("192.168.1.100", 8091, pub, hotkey="5FakeHotkey")
        ip, port = decrypt_endpoint(ct, priv, expected_hotkey="5FakeHotkey")
        assert ip == "192.168.1.100"
        assert port == 8091

    def test_round_trip_worst_case_ipv4(self):
        pub, priv = _generate_keypair()
        ct = encrypt_endpoint("255.255.255.255", 65535, pub, hotkey="5FakeHotkey")
        ip, port = decrypt_endpoint(ct, priv, expected_hotkey="5FakeHotkey")
        assert ip == "255.255.255.255"
        assert port == 65535

    def test_round_trip_localhost(self):
        pub, priv = _generate_keypair()
        ct = encrypt_endpoint("127.0.0.1", 1, pub, hotkey="5FakeHotkey")
        ip, port = decrypt_endpoint(ct, priv, expected_hotkey="5FakeHotkey")
        assert ip == "127.0.0.1"
        assert port == 1

    def test_ciphertext_differs_each_time(self):
        """Sealed boxes use ephemeral keys, so encrypting the same plaintext twice produces different ciphertext."""
        pub, _ = _generate_keypair()
        ct1 = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey="5FakeHotkey")
        ct2 = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey="5FakeHotkey")
        assert ct1 != ct2

    def test_ciphertext_size_within_limit(self):
        """Worst-case: 48-char hotkey + | + 21-char ip:port = 70 bytes plaintext + 48 overhead = 118 bytes."""
        pub, _ = _generate_keypair()
        worst_case_hotkey = "5" * 48  # SS58 hotkeys are up to 48 chars
        ct = encrypt_endpoint("255.255.255.255", 65535, pub, hotkey=worst_case_hotkey)
        assert len(ct) <= 128

    def test_wrong_key_fails(self):
        pub1, _ = _generate_keypair()
        _, priv2 = _generate_keypair()
        ct = encrypt_endpoint("10.0.0.1", 8080, pub1, hotkey="5FakeHotkey")
        with pytest.raises(Exception):
            decrypt_endpoint(ct, priv2)

    def test_hotkey_mismatch_rejected(self):
        """Copying another miner's commitment should be rejected due to hotkey mismatch."""
        pub, priv = _generate_keypair()
        ct = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey="5MinerA")
        with pytest.raises(ValueError, match="mismatch"):
            decrypt_endpoint(ct, priv, expected_hotkey="5MinerB")

    def test_old_format_without_hotkey_rejected(self):
        """Old format commitments (ip:port without hotkey) should be rejected."""
        pub, priv = _generate_keypair()
        # Simulate old format by manually encrypting just "ip:port" without pipe
        from nacl.public import PublicKey, SealedBox
        box = SealedBox(PublicKey(pub))
        ct = box.encrypt(b"10.0.0.1:8080")
        with pytest.raises(ValueError, match="old format no longer supported"):
            decrypt_endpoint(ct, priv)


# ── publish_commitment ───────────────────────────────────────────────

class TestPublishCommitment:
    def test_success(self):
        with patch(_PUBLISH_TARGET) as mock_pub:
            result = publish_commitment(
                subtensor=MagicMock(),
                wallet=MagicMock(),
                netuid=138,
                ciphertext=b"\x01" * 69,
            )
            assert result is True
            mock_pub.assert_called_once()
            call_kwargs = mock_pub.call_args.kwargs
            assert call_kwargs["data_type"] == "Raw69"
            assert call_kwargs["data"] == b"\x01" * 69

    def test_rate_limit_returns_false(self):
        with patch(_PUBLISH_TARGET, side_effect=Exception("rate limit exceeded")):
            result = publish_commitment(
                subtensor=MagicMock(),
                wallet=MagicMock(),
                netuid=138,
                ciphertext=b"\x01" * 69,
            )
            assert result is False

    def test_generic_error_returns_false(self):
        with patch(_PUBLISH_TARGET, side_effect=Exception("something broke")):
            result = publish_commitment(
                subtensor=MagicMock(),
                wallet=MagicMock(),
                netuid=138,
                ciphertext=b"\x01" * 69,
            )
            assert result is False


# ── read_commitment ──────────────────────────────────────────────────

class TestReadCommitment:
    def test_reads_ciphertext_from_metadata(self):
        pub, _ = _generate_keypair()
        ct = encrypt_endpoint("10.0.0.1", 9000, pub)
        metadata = {"info": {"fields": [[{"Raw69": [list(ct)]}]]}}

        subtensor = MagicMock()
        subtensor.substrate.query.return_value = metadata
        result = read_commitment(subtensor, 138, "5FakeHotkey")
        assert result == ct

    def test_reads_ciphertext_from_bt10_hex_metadata(self):
        # bittensor 10.x / async-substrate-interface 2.x returns fields as
        # [{RawN: '0x...'}] (single dict, hex string) rather than [[{RawN: [[...]]}]]
        pub, _ = _generate_keypair()
        ct = encrypt_endpoint("10.0.0.1", 9000, pub)
        metadata = {"info": {"fields": [{f"Raw{len(ct)}": "0x" + ct.hex()}]}}

        subtensor = MagicMock()
        subtensor.substrate.query.return_value = metadata
        result = read_commitment(subtensor, 138, "5FakeHotkey")
        assert result == ct

    def test_returns_none_when_no_metadata(self):
        subtensor = MagicMock()
        subtensor.substrate.query.return_value = None
        result = read_commitment(subtensor, 138, "5FakeHotkey")
        assert result is None

    def test_returns_none_on_exception(self):
        subtensor = MagicMock()
        subtensor.substrate.query.side_effect = Exception("network error")
        result = read_commitment(subtensor, 138, "5FakeHotkey")
        assert result is None


# ── read_all_commitments ─────────────────────────────────────────────

class TestReadAllCommitments:
    @staticmethod
    def _make_commitment_data(ciphertext, block=100):
        return {"block": block, "info": {"fields": [[{f"Raw{len(ciphertext)}": [list(ciphertext)]}]]}}

    @staticmethod
    def _make_bt10_commitment_data(ciphertext, block=100):
        # bittensor 10.x shape: fields -> [{RawN: '0x...'}]
        return {"block": block, "info": {"fields": [{f"Raw{len(ciphertext)}": "0x" + ciphertext.hex()}]}}

    def test_decrypts_bt10_hex_format(self):
        pub, priv = _generate_keypair()
        ct = encrypt_endpoint("10.0.0.5", 8091, pub, hotkey="hk0")

        subtensor = MagicMock()
        subtensor.query_map.return_value = [("hk0", self._make_bt10_commitment_data(ct, block=42))]

        endpoints, cache = read_all_commitments(subtensor, 138, ["hk0"], priv)

        assert endpoints["hk0"] == ("10.0.0.5", 8091)
        assert cache["hk0"][0] == 42

    def test_decrypts_all_available(self):
        pub, priv = _generate_keypair()
        ct1 = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey="hk0")
        ct2 = encrypt_endpoint("10.0.0.2", 9090, pub, hotkey="hk1")

        query_map_result = [
            ("hk0", self._make_commitment_data(ct1, block=10)),
            ("hk1", self._make_commitment_data(ct2, block=20)),
        ]

        subtensor = MagicMock()
        subtensor.query_map.return_value = query_map_result

        endpoints, cache = read_all_commitments(
            subtensor, 138, ["hk0", "hk1", "hk2"], priv
        )

        assert endpoints["hk0"] == ("10.0.0.1", 8080)
        assert endpoints["hk1"] == ("10.0.0.2", 9090)
        assert "hk2" not in endpoints
        assert "hk0" in cache
        assert "hk1" in cache

    def test_skips_undecryptable(self):
        pub1, _ = _generate_keypair()
        _, priv2 = _generate_keypair()
        ct = encrypt_endpoint("10.0.0.1", 8080, pub1, hotkey="hk0")

        query_map_result = [
            ("hk0", self._make_commitment_data(ct, block=10)),
        ]

        subtensor = MagicMock()
        subtensor.query_map.return_value = query_map_result

        # priv2 can't decrypt ct encrypted with pub1
        endpoints, cache = read_all_commitments(subtensor, 138, ["hk0"], priv2)

        assert endpoints == {}
        assert cache == {}

    def test_rejects_copied_commitment(self):
        """A miner copying another miner's commitment should be rejected."""
        pub, priv = _generate_keypair()
        # hk0 encrypts with its own hotkey
        ct = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey="hk0")

        # hk1 copies hk0's ciphertext
        query_map_result = [
            ("hk0", self._make_commitment_data(ct, block=10)),
            ("hk1", self._make_commitment_data(ct, block=10)),  # copied!
        ]

        subtensor = MagicMock()
        subtensor.query_map.return_value = query_map_result

        endpoints, cache = read_all_commitments(subtensor, 138, ["hk0", "hk1"], priv)

        assert endpoints["hk0"] == ("10.0.0.1", 8080)
        assert "hk1" not in endpoints  # rejected due to hotkey mismatch

    def test_reuses_cache_when_block_unchanged(self):
        pub, priv = _generate_keypair()
        ct = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey="hk0")

        query_map_result = [
            ("hk0", self._make_commitment_data(ct, block=10)),
        ]

        subtensor = MagicMock()
        subtensor.query_map.return_value = query_map_result

        # First call populates cache
        endpoints, cache = read_all_commitments(subtensor, 138, ["hk0"], priv)
        assert endpoints["hk0"] == ("10.0.0.1", 8080)

        # Second call with same block — should reuse cache, not re-decrypt
        endpoints2, cache2 = read_all_commitments(subtensor, 138, ["hk0"], priv, cache=cache)
        assert endpoints2["hk0"] == ("10.0.0.1", 8080)
        assert cache2["hk0"] == cache["hk0"]

    def test_re_decrypts_when_block_changes(self):
        pub, priv = _generate_keypair()
        ct1 = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey="hk0")
        ct2 = encrypt_endpoint("10.0.0.2", 9090, pub, hotkey="hk0")

        subtensor = MagicMock()

        # First call
        subtensor.query_map.return_value = [("hk0", self._make_commitment_data(ct1, block=10))]
        endpoints, cache = read_all_commitments(subtensor, 138, ["hk0"], priv)
        assert endpoints["hk0"] == ("10.0.0.1", 8080)

        # Second call with new block and new ciphertext
        subtensor.query_map.return_value = [("hk0", self._make_commitment_data(ct2, block=20))]
        endpoints2, cache2 = read_all_commitments(subtensor, 138, ["hk0"], priv, cache=cache)
        assert endpoints2["hk0"] == ("10.0.0.2", 9090)


# ── real query_map key shape ─────────────────────────────────────────
#
# Regression guard for the bug where read_all_commitments filtered every row
# because real Substrate query_map returns AccountId32 as ((b0..b31),) tuples,
# not SS58 strings. Previous tests mocked the keys as plain strings ("hk0"),
# which doesn't reflect chain reality, and the bug shipped to production.
#
# These tests reproduce the real shape so the filter is exercised correctly.

try:
    from async_substrate_interface.utils.ss58 import ss58_encode
except ImportError:
    try:
        from substrateinterface.utils.ss58 import ss58_encode
    except ImportError:
        from scalecodec.utils.ss58 import ss58_encode


class TestReadAllCommitmentsChainKeyShape:
    @staticmethod
    def _make_commitment_data(ciphertext, block=100):
        return {"block": block, "info": {"fields": [[{f"Raw{len(ciphertext)}": [list(ciphertext)]}]]}}

    @staticmethod
    def _account_id_bytes_to_ss58(raw: bytes) -> str:
        return ss58_encode(raw, ss58_format=42)

    def _make_chain_key(self, raw: bytes):
        """Return the ((b0..b31),) tuple shape that real query_map yields."""
        return (tuple(raw),)

    def test_decodes_account_id_tuple_key(self):
        """query_map keys arriving as ((b0..b31),) must decode to SS58 and match."""
        pub, priv = _generate_keypair()
        raw = bytes(range(32))
        ss58 = self._account_id_bytes_to_ss58(raw)
        ct = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey=ss58)

        subtensor = MagicMock()
        subtensor.query_map.return_value = [
            (self._make_chain_key(raw), self._make_commitment_data(ct, block=10)),
        ]

        endpoints, cache = read_all_commitments(subtensor, 138, [ss58], priv)

        assert endpoints[ss58] == ("10.0.0.1", 8080), (
            "AccountId tuple key was not decoded to SS58 — read_all_commitments "
            "would silently filter every row in production."
        )
        assert ss58 in cache

    def test_decodes_account_id_bytes_key(self):
        """Some substrate layers return raw bytes instead of a tuple-wrapped tuple."""
        pub, priv = _generate_keypair()
        raw = bytes(range(32, 64))
        ss58 = self._account_id_bytes_to_ss58(raw)
        ct = encrypt_endpoint("10.0.0.2", 9090, pub, hotkey=ss58)

        subtensor = MagicMock()
        subtensor.query_map.return_value = [
            (raw, self._make_commitment_data(ct, block=10)),
        ]

        endpoints, _ = read_all_commitments(subtensor, 138, [ss58], priv)
        assert endpoints[ss58] == ("10.0.0.2", 9090)

    def test_stale_chain_keys_are_filtered(self):
        """Commitments on chain for hotkeys no longer in the metagraph must be filtered."""
        pub, priv = _generate_keypair()
        live_raw = bytes(range(0, 32))
        stale_raw = bytes(range(64, 96))
        live_ss58 = self._account_id_bytes_to_ss58(live_raw)
        stale_ss58 = self._account_id_bytes_to_ss58(stale_raw)
        ct_live = encrypt_endpoint("10.0.0.1", 8080, pub, hotkey=live_ss58)
        ct_stale = encrypt_endpoint("10.0.0.2", 9090, pub, hotkey=stale_ss58)

        subtensor = MagicMock()
        subtensor.query_map.return_value = [
            (self._make_chain_key(live_raw), self._make_commitment_data(ct_live, block=10)),
            (self._make_chain_key(stale_raw), self._make_commitment_data(ct_stale, block=10)),
        ]

        # Only live_ss58 is in the metagraph
        endpoints, _ = read_all_commitments(subtensor, 138, [live_ss58], priv)

        assert live_ss58 in endpoints
        assert stale_ss58 not in endpoints


# ── validator integration ────────────────────────────────────────────

class TestValidatorIntegration:
    @staticmethod
    def _add_ip_port_to_axons(validator):
        """Add ip/port attrs to the mock axon objects so _get_axons_for_uids can log them."""
        for i, axon in enumerate(validator.metagraph.axons):
            axon.ip = f"10.0.0.{i}"
            axon.port = 8000 + i

    def test_get_axons_uses_committed_endpoint(self, bare_validator):
        """When a committed endpoint exists, _get_axons_for_uids should override the axon ip/port."""
        self._add_ip_port_to_axons(bare_validator)
        bare_validator.committed_endpoints = {"hk1": ("10.99.99.99", 1234)}
        axons = bare_validator._get_axons_for_uids([0, 1, 2])

        assert axons[1].ip == "10.99.99.99"
        assert axons[1].port == 1234
        assert axons[0].ip == "10.0.0.0"
        assert len(axons) == 3

    def test_get_axons_no_committed_endpoints(self, bare_validator):
        """When no commitments exist, axons come straight from metagraph."""
        self._add_ip_port_to_axons(bare_validator)
        bare_validator.committed_endpoints = {}
        axons = bare_validator._get_axons_for_uids([0, 1, 2])
        # All 3 axons returned unchanged
        assert len(axons) == 3
        assert axons[0] is bare_validator.metagraph.axons[0]

    def test_committed_endpoint_does_not_mutate_metagraph(self, bare_validator):
        """The override should copy the axon, not mutate the metagraph's version."""
        self._add_ip_port_to_axons(bare_validator)
        bare_validator.metagraph.axons[1].ip = "original"
        bare_validator.metagraph.axons[1].port = 5555
        bare_validator.committed_endpoints = {"hk1": ("10.99.99.99", 1234)}

        axons = bare_validator._get_axons_for_uids([1])
        assert axons[0].ip == "10.99.99.99"
        # Original metagraph axon should be untouched
        assert bare_validator.metagraph.axons[1].ip == "original"
        assert bare_validator.metagraph.axons[1].port == 5555
