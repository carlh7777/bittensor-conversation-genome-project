# The MIT License (MIT)
# Copyright © 2024 Afterparty, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import torch
import asyncio
import threading
import argparse
import traceback

import bittensor as bt

from conversationgenome.base.neuron import BaseNeuron
from conversationgenome.utils.config import add_miner_args


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_miner_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, config=self.config)

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Publish encrypted endpoint commitment if configured.
        # When active, the real ip:port goes into the encrypted commitment only,
        # and the metagraph gets a dummy address so the real endpoint stays hidden.
        # Shared public key for encrypting endpoint commitments.
        # Mainnet (netuid 33) and testnet use different keypairs.
        # Can be overridden via COMMITMENT_PUBLIC_KEY env var.
        _COMMITMENT_PUBLIC_KEYS = {
            33: "aadbfa93972378fbc1bd8e854bc6fb915bb57506f56c17f0531647a127a0bd69",  # mainnet
            138: "2c068a9b7c3480225ab56888227218228d803208f26bfbd8c875a919467a7516",  # testnet
        }
        default_key = _COMMITMENT_PUBLIC_KEYS.get(self.config.netuid, "")
        commitment_pub_key_hex = os.environ.get("COMMITMENT_PUBLIC_KEY", default_key).strip()
        if commitment_pub_key_hex:
            try:
                from conversationgenome.commitment.commitment import encrypt_endpoint, publish_commitment

                # --axon.ip and --axon.port hold the user-specified values;
                # external_ip/external_port may be auto-detected or None.
                real_ip = self.axon.ip
                real_port = self.axon.port
                bt.logging.info(f"Real endpoint: {real_ip}:{real_port} — will be encrypted in commitment.")

                public_key_bytes = bytes.fromhex(commitment_pub_key_hex)
                ciphertext = encrypt_endpoint(real_ip, real_port, public_key_bytes)
                success = publish_commitment(self.subtensor, self.wallet, self.config.netuid, ciphertext)
                if success:
                    bt.logging.info(f"Encrypted endpoint commitment published successfully.")
                else:
                    bt.logging.warning(f"Encrypted endpoint commitment failed — will retry on next restart.")

                # Serve blackhole address to metagraph so real endpoint is not visible
                # 192.0.2.0/24 is TEST-NET-1 (RFC 5737), reserved and non-routable
                self.axon.external_ip = "192.0.2.1"
                self.axon.external_port = 1234
                bt.logging.info(f"Serving dummy endpoint to metagraph.")
            except Exception as e:
                bt.logging.error(f"Error publishing encrypted commitment: {e}")

        # Serve passes the axon information to the network + netuid we are hosting on.
        bt.logging.info(
            f"Serving miner axon (external: {self.axon.external_ip}:{self.axon.external_port}) on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        bt.logging.info(f"Miner starting at block: {self.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while (
                    self.block - self.metagraph.last_update[self.uid]
                    < self.config.neuron.epoch_length
                ):
                    # Wait before checking again.
                    import time
                    time.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                try:
                    self.sync()
                except Exception as e:
                    print("Miner sync error. Pausing for 10 seconds to reconnect.", e)
                    import time
                    time.sleep(10)

                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        #bt.logging.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
