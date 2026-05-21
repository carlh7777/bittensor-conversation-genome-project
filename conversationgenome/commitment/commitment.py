from typing import Optional

import bittensor as bt
from nacl.public import PrivateKey, PublicKey, SealedBox


def encrypt_endpoint(ip: str, port: int, public_key_bytes: bytes) -> bytes:
    """Encrypt an ip:port string using a NaCl sealed box."""
    plaintext = f"{ip}:{port}".encode()
    box = SealedBox(PublicKey(public_key_bytes))
    return box.encrypt(plaintext)


def decrypt_endpoint(ciphertext: bytes, private_key_bytes: bytes) -> tuple:
    """Decrypt ciphertext to recover (ip, port)."""
    box = SealedBox(PrivateKey(private_key_bytes))
    plaintext = box.decrypt(ciphertext).decode()
    ip, port_str = plaintext.rsplit(":", 1)
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


def read_all_commitments(
    subtensor, netuid: int, hotkeys: list, private_key_bytes: bytes
) -> dict:
    """Read and decrypt commitments for all hotkeys. Returns {hotkey: (ip, port)}."""
    endpoints = {}
    for hotkey in hotkeys:
        ciphertext = read_commitment(subtensor, netuid, hotkey)
        if ciphertext is None:
            continue
        try:
            ip, port = decrypt_endpoint(ciphertext, private_key_bytes)
            endpoints[hotkey] = (ip, port)
        except Exception as e:
            bt.logging.debug(f"Could not decrypt commitment for {hotkey}: {e}")
    return endpoints
