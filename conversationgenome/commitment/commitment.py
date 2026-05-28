from typing import Optional

import bittensor as bt
import nacl.exceptions
from nacl.public import PrivateKey, PublicKey, SealedBox


def encrypt_endpoint(ip: str, port: int, public_key_bytes: bytes, hotkey: str = "") -> bytes:
    """Encrypt hotkey|ip:port using a NaCl sealed box.

    The hotkey is embedded so the validator can verify the commitment
    belongs to the miner that published it (prevents replay attacks).
    """
    plaintext = f"{hotkey}|{ip}:{port}".encode()
    box = SealedBox(PublicKey(public_key_bytes))
    return box.encrypt(plaintext)


def decrypt_endpoint(ciphertext: bytes, private_key_bytes: bytes, expected_hotkey: str = "") -> tuple:
    """Decrypt ciphertext to recover (ip, port).

    If expected_hotkey is provided, verifies the embedded hotkey matches.
    Raises ValueError on mismatch (commitment was copied from another miner).
    """
    box = SealedBox(PrivateKey(private_key_bytes))
    plaintext = box.decrypt(ciphertext).decode()

    if "|" not in plaintext:
        raise ValueError("Invalid commitment format: missing hotkey (old format no longer supported)")

    hotkey_part, endpoint = plaintext.split("|", 1)
    if expected_hotkey and hotkey_part != expected_hotkey:
        raise ValueError(f"Commitment hotkey mismatch: expected {expected_hotkey[:8]}..., got {hotkey_part[:8]}...")

    ip, port_str = endpoint.rsplit(":", 1)
    return ip, int(port_str)


def publish_commitment(subtensor, wallet, netuid: int, ciphertext: bytes) -> bool:
    """Publish encrypted endpoint ciphertext on-chain via publish_metadata."""
    from bittensor.core.extrinsics.serving import publish_metadata

    try:
        publish_metadata(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            data_type=f"Raw{len(ciphertext)}",
            data=ciphertext,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if any(kw in error_msg for kw in ("rate", "cooldown", "limit")):
            bt.logging.warning(f"Commitment rate-limited, will retry after cooldown: {e}")
        else:
            bt.logging.error(f"Failed to publish commitment: {e}")
        return False


def read_commitment(subtensor, netuid: int, hotkey_ss58: str) -> Optional[bytes]:
    """Read a single miner's encrypted commitment from chain."""
    from bittensor.core.extrinsics.serving import get_metadata

    try:
        metadata = get_metadata(subtensor, netuid, hotkey_ss58)
        if metadata is None:
            return None
        commitment = metadata["info"]["fields"][0][0]
        raw_key = next(iter(commitment.keys()))
        return bytes(commitment[raw_key][0])
    except Exception as e:
        bt.logging.debug(f"Could not read commitment for {hotkey_ss58}: {e}")
        return None


def _extract_ciphertext(commitment_data) -> Optional[bytes]:
    """Extract ciphertext bytes from a commitment data structure."""
    try:
        commitment = commitment_data["info"]["fields"][0][0]
        raw_key = next(iter(commitment.keys()))
        return bytes(commitment[raw_key][0])
    except Exception:
        return None


def read_all_commitments(
    subtensor, netuid: int, hotkeys: list, private_key_bytes: bytes,
    cache: dict = None,
) -> dict:
    """Read and decrypt all commitments for a subnet in a single RPC call.

    Uses query_map to fetch every commitment on the subnet at once (~0.6s),
    then only decrypts entries whose block number changed since the last call.

    Args:
        cache: Dict of {hotkey: (block, ip, port)} from previous call.
               Used to skip re-decrypting unchanged commitments.

    Returns:
        (endpoints, new_cache) tuple:
            endpoints: {hotkey: (ip, port)} for use by the validator
            new_cache: {hotkey: (block, ip, port)} to pass back on next call
    """
    if cache is None:
        cache = {}

    hotkey_set = set(hotkeys)

    bt.logging.info(f"Fetching all commitments for subnet via query_map...")
    try:
        result = subtensor.query_map(
            module="Commitments",
            name="CommitmentOf",
            params=[netuid],
        )
    except Exception as e:
        bt.logging.error(f"query_map failed: {e}")
        # Fall back to cached data
        return {hk: (ip, port) for hk, (_, ip, port) in cache.items() if hk in hotkey_set}, cache

    new_cache = {}
    endpoints = {}
    found = 0
    reused = 0

    for hotkey, commitment_data in result:
        hotkey_str = str(hotkey)
        if hotkey_str not in hotkey_set:
            continue

        block = commitment_data.get("block", 0) if hasattr(commitment_data, "get") else 0

        # If block hasn't changed, reuse cached decryption
        if hotkey_str in cache and cache[hotkey_str][0] == block:
            _, cached_ip, cached_port = cache[hotkey_str]
            new_cache[hotkey_str] = (block, cached_ip, cached_port)
            endpoints[hotkey_str] = (cached_ip, cached_port)
            reused += 1
            continue

        ciphertext = _extract_ciphertext(commitment_data)
        if ciphertext is None:
            continue

        try:
            ip, port = decrypt_endpoint(ciphertext, private_key_bytes, expected_hotkey=hotkey_str)
            new_cache[hotkey_str] = (block, ip, port)
            endpoints[hotkey_str] = (ip, port)
            found += 1
        except ValueError as e:
            # Hotkey mismatch — commitment was likely copied from another miner
            bt.logging.warning(f"Rejected commitment for {hotkey_str}: {e}")
        except nacl.exceptions.CryptoError:
            # Wrong public key or corrupted/random data
            bt.logging.debug(f"Could not decrypt commitment for {hotkey_str}: invalid ciphertext (wrong key or garbage data)")
        except UnicodeDecodeError:
            bt.logging.debug(f"Could not decrypt commitment for {hotkey_str}: decrypted bytes are not valid UTF-8")
        except Exception as e:
            bt.logging.debug(f"Could not decrypt commitment for {hotkey_str}: {e}")

    bt.logging.info(f"Commitments: {found} new, {reused} cached, {found + reused} total.")
    return endpoints, new_cache
