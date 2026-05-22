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

    def test_backwards_compatible_no_hotkey(self):
        """Old format commitments (ip:port without hotkey) should still decrypt."""
        pub, priv = _generate_keypair()
        # Simulate old format by encrypting without hotkey
        ct = encrypt_endpoint("10.0.0.1", 8080, pub)
        ip, port = decrypt_endpoint(ct, priv)
        assert ip == "10.0.0.1"
        assert port == 8080


# ── publish_commitment ───────────────────────────────────────────────

class TestPublishCommitment:
    def test_success(self):
        with patch(
            "bittensor.core.extrinsics.serving.publish_metadata"
        ) as mock_pub:
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
        with patch(
            "bittensor.core.extrinsics.serving.publish_metadata",
            side_effect=Exception("rate limit exceeded"),
        ):
            result = publish_commitment(
                subtensor=MagicMock(),
                wallet=MagicMock(),
                netuid=138,
                ciphertext=b"\x01" * 69,
            )
            assert result is False

    def test_generic_error_returns_false(self):
        with patch(
            "bittensor.core.extrinsics.serving.publish_metadata",
            side_effect=Exception("something broke"),
        ):
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

        with patch(
            "bittensor.core.extrinsics.serving.get_metadata",
            return_value=metadata,
        ):
            result = read_commitment(MagicMock(), 138, "5FakeHotkey")
            assert result == ct

    def test_returns_none_when_no_metadata(self):
        with patch(
            "bittensor.core.extrinsics.serving.get_metadata",
            return_value=None,
        ):
            result = read_commitment(MagicMock(), 138, "5FakeHotkey")
            assert result is None

    def test_returns_none_on_exception(self):
        with patch(
            "bittensor.core.extrinsics.serving.get_metadata",
            side_effect=Exception("network error"),
        ):
            result = read_commitment(MagicMock(), 138, "5FakeHotkey")
            assert result is None


# ── read_all_commitments ─────────────────────────────────────────────

class TestReadAllCommitments:
    @staticmethod
    def _make_commitment_data(ciphertext, block=100):
        return {"block": block, "info": {"fields": [[{f"Raw{len(ciphertext)}": [list(ciphertext)]}]]}}

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
