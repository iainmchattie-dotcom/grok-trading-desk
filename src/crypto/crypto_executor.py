"""Solana execution — DELIBERATE STUB.

Every method raises NotImplementedError. The code that signs transactions with
real keys is written by the desk owner, not generated. What follows is the
contract the rest of the system already depends on, plus notes on what each
method has to do.

Implementation notes (Jito bundle path):

  buy()
    1. Load the keypair from config["solana"]["wallet_key"] (base58 secret) via
       solders.keypair.Keypair.from_base58_string. Never log it.
    2. Quote the swap. On pump.fun a token that has not bonded is bought through
       the bonding curve program; once bonded it routes through Raydium/Jupiter.
       Check the curve's `complete` flag and pick the venue accordingly.
    3. Build the swap instruction with slippage_bps from config, plus a
       ComputeBudget SetComputeUnitPrice using priority_fee_microlamports.
    4. If jito.enabled: append a tip transfer of tip_lamports to a Jito tip
       account and submit the signed transaction as a single-transaction bundle
       to `{block_engine_url}/api/v1/bundles`. Otherwise send through the plain
       RPC with skip_preflight=False.
    5. Poll for confirmation with a deadline (a bundle that never lands must
       surface as a failure, not a silent no-op). Return the signature as tx_id
       and the filled token amount.

  sell()
    Same path in reverse, with `fraction` of the held balance. Amounts are raw
    u64 in token decimals — read the mint's decimals, do not assume 6 or 9.

  get_positions()
    Read the wallet's SPL token accounts, drop dust and anything not opened by
    this desk (match against the event log), and price each holding off the
    current curve/pool state to fill current_price.

  A partial fill or a failed bundle must never be reported as success: the risk
  manager sizes the next trade off what it believes is deployed.
"""

from __future__ import annotations

from typing import Any

from ..models import Market, Position

_MESSAGE = (
    "crypto execution is intentionally not implemented — "
    "wire your own signing path in src/crypto/crypto_executor.py"
)


class CryptoExecutor:
    """Stub. Mirrors StockExecutor's interface so desk.py can treat both alike."""

    market = Market.CRYPTO

    def __init__(self, config: dict[str, Any]):
        self.config = config or {}
        solana = self.config.get("solana", {}) or {}
        self.rpc_url = solana.get("rpc_url", "")
        self.jito = solana.get("jito", {}) or {}
        self.slippage_bps = int(solana.get("slippage_bps", 500))
        self.priority_fee = int(solana.get("priority_fee_microlamports", 0))

    async def buy(self, mint: str, amount_usd: float, **kwargs: Any) -> dict[str, Any]:
        """Swap SOL for `amount_usd` worth of `mint`. Returns {tx_id, quantity, price}."""
        raise NotImplementedError(_MESSAGE)

    async def sell(self, mint: str, fraction: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Swap `fraction` of the held balance back to SOL."""
        raise NotImplementedError(_MESSAGE)

    async def close_position(self, mint: str) -> dict[str, Any]:
        """Full exit. Equivalent to sell(mint, 1.0)."""
        raise NotImplementedError(_MESSAGE)

    async def tighten_stop(self, mint: str, new_stop_price: float) -> dict[str, Any]:
        """No on-chain stop exists on pump.fun.

        A stop here means a local price watcher that fires sell() when breached.
        Implement it as desk-side state, not as an exchange order.
        """
        raise NotImplementedError(_MESSAGE)

    async def get_positions(self) -> list[Position]:
        """Wallet holdings priced at current curve/pool state."""
        raise NotImplementedError(_MESSAGE)
