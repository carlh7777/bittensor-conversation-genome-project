from typing import Optional

import bittensor as bt
import nacl.exceptions
from nacl.public import PrivateKey, PublicKey, SealedBox

try:
    from async_substrate_interface.utils.ss58 import ss58_encode  # type: ignore
except ImportError:
    try:
        from substrateinterface.utils.ss58 import ss58_encode  # type: ignore
    except ImportError:
        from scalecodec.utils.ss58 import ss58_encode  # type: ignore


_SS58_FORMAT = 42


def _key_to_ss58(hotkey) -> str:
    """Normalize a query_map key (AccountId32) to an SS58 string.

    query_map returns the AccountId as ((b0, ..., b31),) on newer substrate
    layers; older versions return an SS58 string directly. Both shapes flow
    through this helper.
    """
    if isinstance(hotkey, str):
        return hotkey
    if isinstance(hotkey, (bytes, bytearray)):
        return ss58_encode(bytes(hotkey), ss58_format=_SS58_FORMAT)
    if isinstance(hotkey, tuple):
        inner = hotkey[0] if len(hotkey) == 1 and isinstance(hotkey[0], (tuple, list, bytes, bytearray)) else hotkey
        return ss58_encode(bytes(inner), ss58_format=_SS58_FORMAT)
    return ss58_encode(bytes(hotkey), ss58_format=_SS58_FORMAT)


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
    """Publish encrypted endpoint ciphertext on-chain via publish_metadata.

    bittensor 10.x renamed publish_metadata → publish_metadata_extrinsic; the
    call signature is unchanged, so we import whichever the installed version
    exposes (10.x first, 9.x fallback).
    """
    try:
        from bittensor.core.extrinsics.serving import publish_metadata_extrinsic as publish_metadata
    except ImportError:
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
    """Read a single miner's encrypted commitment from chain.

    Queries the Commitments.CommitmentOf storage directly. This replaces the
    old bittensor.core.extrinsics.serving.get_metadata helper, which was
    removed in bittensor 10.x; the direct query works on both 9.x and 10.x.
    """
    try:
        metadata = subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey_ss58],
        )
        if metadata is None:
            return None
        return _extract_ciphertext(metadata)
    except Exception as e:
        bt.logging.debug(f"Could not read commitment for {hotkey_ss58}: {e}")
        return None


def _raw_field_to_bytes(value) -> Optional[bytes]:
    """Normalize a Commitments Raw{N} field value to bytes.

    The on-chain value decodes differently across substrate stacks:
      - bittensor 10.x (async-substrate-interface 2.x): a hex string '0x...'
      - bittensor 9.x: a nested byte list, e.g. [[b0, b1, ...]]
      - occasionally already bytes
    """
    if isinstance(value, str):
        return bytes.fromhex(value[2:] if value.startswith("0x") else value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, (list, tuple)):
        inner = value
        if len(value) == 1 and isinstance(value[0], (list, tuple, bytes, bytearray, str)):
            inner = value[0]
        if isinstance(inner, str):
            return bytes.fromhex(inner[2:] if inner.startswith("0x") else inner)
        return bytes(inner)
    return None


def _extract_ciphertext(commitment_data) -> Optional[bytes]:
    """Extract ciphertext bytes from a commitment data structure.

    Handles both the bittensor 9.x shape (fields -> [[{RawN: [[...]]}]]) and the
    10.x shape (fields -> [{RawN: '0x...'}]), and unwraps ScaleObj via .value.
    """
    try:
        data = commitment_data.value if hasattr(commitment_data, "value") else commitment_data
        entry = data["info"]["fields"][0]
        if isinstance(entry, (list, tuple)):
            entry = entry[0]
        raw_key = next(iter(entry.keys()))
        return _raw_field_to_bytes(entry[raw_key])
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
        hotkey_str = _key_to_ss58(hotkey)
        if hotkey_str not in hotkey_set:
            continue

        data = commitment_data.value if hasattr(commitment_data, "value") else commitment_data
        block = data.get("block", 0) if hasattr(data, "get") else 0

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
