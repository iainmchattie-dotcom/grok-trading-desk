"""Solana execution — pump.fun bonding curve or Jupiter, then Jito or RPC.

Mirrors StockExecutor: paper unless `mode: live` AND `--i-understand-the-risk`.
Paper never broadcasts. Live signs with the configured key (never logged),
submits, and treats anything short of a confirmed full fill as failure.

Network I/O is injected (`rpc`, `http`) so the test suite stays offline.

Remaining follow-ups (not blockers for the contract):
  - Jupiter routes that require address-lookup tables may fail to compile;
    we request `asLegacyTransaction` and treat a compile miss as failure.
  - pump.fun `buy_v2`/`sell_v2` account lists track the 2026 public docs;
    another program upgrade would need the same kind of IDL bump.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..models import Market, Position
from . import onchain

log = logging.getLogger(__name__)

_PLACEHOLDER_KEYS = frozenset(
    {"", "REPLACE_ME", "REPLACE_ME_BASE58_SECRET_KEY", "xai-REPLACE_ME"}
)


class ExecutionFailed(Exception):
    """A swap that must not be treated as a fill.

    `reason` is a stable slug so the event log can group failures. Partial fills,
    dropped Jito bundles and unconfirmed RPC sends all land here — never as
    success. The risk manager sizes the next trade off what it believes is
    deployed; a silent no-op would lie to it.

    When a transaction confirms but the wallet delta is short, `stranded`
    carries the inventory that landed so the desk can see it. Failure stays
    a failure; the tokens must not vanish from the book.
    """

    def __init__(self, reason: str, detail: str = "", stranded: dict[str, Any] | None = None):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.stranded = stranded or {}


@dataclass
class _Quote:
    venue: str
    in_amount: int
    out_amount: int
    min_out: int
    decimals: int
    price_usd: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Lot:
    mint: str
    raw_amount: int
    decimals: int
    entry_price: float
    amount_usd: float
    symbol: str = ""


class CryptoExecutor:
    """Swap SOL for pump.fun (and graduated) tokens. Same verbs as StockExecutor."""

    market = Market.CRYPTO

    def __init__(
        self,
        config: dict[str, Any],
        event_log: Any = None,
        live_ack: bool = False,
        client: Any = None,
        rpc: Any = None,
        http: Any = None,
        keypair: Any = None,
        monotonic: Any = None,
        sleep: Any = None,
    ):
        self.config = config or {}
        solana = self.config.get("solana", {}) or {}
        pump = self.config.get("pump_fun", {}) or {}
        jito = solana.get("jito", {}) or {}

        wants_live = str(self.config.get("mode", "paper")).lower() == "live"
        self.paper = not (wants_live and live_ack)
        if wants_live and not live_ack:
            log.warning(
                "config asks for live crypto trading but --i-understand-the-risk "
                "was not passed; staying on paper"
            )

        self.rpc_url = str(solana.get("rpc_url", "") or "")
        self.wallet_key = str(solana.get("wallet_key", "") or "")
        self.slippage_bps = int(solana.get("slippage_bps", 500))
        self.priority_fee = int(solana.get("priority_fee_microlamports", 0))
        self.compute_unit_limit = int(solana.get("compute_unit_limit", 400_000))
        self.confirm_timeout = float(solana.get("confirm_timeout_seconds", 30))
        self.confirm_poll = float(solana.get("confirm_poll_seconds", 0.4))
        self.dust_usd = float(solana.get("dust_usd", 0.5))
        self.fee_bps = int(solana.get("curve_fee_bps", 100))
        self.jupiter_base_url = str(
            solana.get("jupiter_base_url", "https://quote-api.jup.ag/v6")
        ).rstrip("/")
        self.sol_usd = float(
            solana.get("sol_price_usd") or pump.get("sol_price_usd") or 0
        )

        self.jito_enabled = bool(jito.get("enabled", False))
        self.jito_url = str(jito.get("block_engine_url", "") or "").rstrip("/")
        self.tip_lamports = int(jito.get("tip_lamports", 0) or 0)

        pump_exec = solana.get("pump", {}) or {}
        self.fee_recipient_override = str(pump_exec.get("fee_recipient") or "") or None
        self.buyback_fee_override = str(pump_exec.get("buyback_fee_recipient") or "") or None

        self.event_log = event_log
        self._rpc = rpc if rpc is not None else client
        self._http = http
        self._keypair = keypair
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._http_owned: Any = None

        self._stops: dict[str, float] = {}
        self._lots: dict[str, _Lot] = {}
        self._opened: set[str] = set()

    async def aclose(self) -> None:
        """Close the httpx client we created. Injected clients are left alone."""
        owned = self._http_owned
        self._http_owned = None
        if owned is not None and hasattr(owned, "aclose"):
            await owned.aclose()

    async def __aenter__(self) -> "CryptoExecutor":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    # -- SDK plumbing ---------------------------------------------------------------

    def _http_client(self) -> Any:
        if self._http is not None:
            return self._http
        if self._http_owned is None:
            import httpx

            self._http_owned = httpx.AsyncClient(timeout=20.0)
        return self._http_owned

    async def _rpc_call(self, method: str, params: list[Any]) -> Any:
        if self._rpc is not None:
            return await self._rpc.call(method, params)
        if not self.rpc_url:
            raise ExecutionFailed("rpc_error", "solana.rpc_url is empty")
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        response = await self._http_client().post(self.rpc_url, json=payload)
        body = _as_json(response)
        if isinstance(body, dict) and body.get("error"):
            raise ExecutionFailed("rpc_error", str(body["error"]))
        return body.get("result") if isinstance(body, dict) else body

    async def _http_get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        client = self._http_client()
        if not hasattr(client, "get"):
            raise ExecutionFailed("quote_failed", "http client cannot GET")
        return _as_json(await client.get(url, params=params))

    async def _http_post(self, url: str, payload: dict[str, Any]) -> Any:
        client = self._http_client()
        if not hasattr(client, "post"):
            raise ExecutionFailed("submit_failed", "http client cannot POST")
        return _as_json(await client.post(url, json=payload))

    def _load_keypair(self) -> Any:
        """Load the configured secret. The secret itself is never logged."""
        if self._keypair is not None:
            return self._keypair
        raw = self.wallet_key.strip()
        if not raw or raw in _PLACEHOLDER_KEYS or raw.startswith("REPLACE"):
            if self.paper:
                from solders.keypair import Keypair

                self._keypair = Keypair()
                return self._keypair
            raise ExecutionFailed(
                "missing_wallet_key", "solana.wallet_key is unset or a placeholder"
            )
        try:
            from solders.keypair import Keypair

            self._keypair = Keypair.from_base58_string(raw)
        except Exception as exc:  # noqa: BLE001 - do not echo the secret
            raise ExecutionFailed(
                "invalid_wallet_key", "wallet_key is not a valid base58 secret"
            ) from exc
        return self._keypair

    def _pubkey(self) -> str:
        return str(self._load_keypair().pubkey())

    def _require_sol_usd(self) -> float:
        if self.sol_usd <= 0:
            raise ExecutionFailed("no_sol_price", "pump_fun.sol_price_usd is missing")
        return self.sol_usd

    def _usd_to_lamports(self, amount_usd: float) -> int:
        sol_usd = self._require_sol_usd()
        if amount_usd <= 0:
            raise ExecutionFailed("amount_too_small", f"amount_usd={amount_usd}")
        lamports = int(amount_usd / sol_usd * onchain.LAMPORTS_PER_SOL)
        if lamports <= 0:
            raise ExecutionFailed("amount_too_small", "converts to 0 lamports")
        return lamports

    # -- venue / quote --------------------------------------------------------------

    async def _account_data(self, address: str) -> bytes | None:
        result = await self._rpc_call(
            "getAccountInfo", [address, {"encoding": "base64"}]
        )
        value = (result or {}).get("value") if isinstance(result, dict) else None
        if not value:
            return None
        blob = value.get("data")
        if isinstance(blob, list) and blob:
            return base64.b64decode(blob[0])
        if isinstance(blob, str):
            return base64.b64decode(blob)
        return None

    async def _account_owner(self, address: str) -> str:
        result = await self._rpc_call(
            "getAccountInfo", [address, {"encoding": "base64"}]
        )
        value = (result or {}).get("value") if isinstance(result, dict) else None
        return str((value or {}).get("owner") or onchain.TOKEN_PROGRAM)

    async def _fetch_curve(self, mint: str) -> onchain.BondingCurve | None:
        try:
            data = await self._account_data(str(onchain.bonding_curve_pda(mint)))
        except ExecutionFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionFailed("rpc_error", f"bonding curve fetch failed: {exc}") from exc
        if data is None:
            return None
        try:
            return onchain.parse_bonding_curve(data)
        except ValueError as exc:
            raise ExecutionFailed("quote_failed", str(exc)) from exc

    async def _mint_decimals(self, mint: str) -> int:
        data = await self._account_data(mint)
        if data is None:
            raise ExecutionFailed("mint_decimals_unknown", f"no mint account for {mint}")
        try:
            return onchain.mint_decimals(data)
        except ValueError as exc:
            raise ExecutionFailed("mint_decimals_unknown", str(exc)) from exc

    async def _venue(self, mint: str) -> tuple[str, onchain.BondingCurve | None]:
        curve = await self._fetch_curve(mint)
        if curve is None or curve.complete:
            return "jupiter", curve
        return "pump", curve

    async def _quote_buy(self, mint: str, lamports: int) -> _Quote:
        venue, curve = await self._venue(mint)
        decimals = await self._mint_decimals(mint)
        if venue == "pump":
            if curve is None:
                raise ExecutionFailed("quote_failed", "bonding curve missing")
            net = lamports - onchain.apply_bps(lamports, self.fee_bps)
            out = onchain.tokens_out_for_quote_in(
                curve.virtual_quote_reserves, curve.virtual_token_reserves, net
            )
            if out <= 0:
                raise ExecutionFailed("quote_empty", "bonding curve quoted 0 tokens")
            min_out = out  # buy_v2 is exact-out; the slippage cap is on SOL in
            price = self._price_from_curve(curve, decimals)
            return _Quote("pump", lamports, out, min_out, decimals, price, {"curve": curve})
        return await self._jupiter_quote(
            onchain.WSOL_MINT, mint, lamports, decimals, side="buy"
        )

    async def _quote_sell(self, mint: str, raw: int, decimals: int) -> _Quote:
        venue, curve = await self._venue(mint)
        if venue == "pump":
            if curve is None:
                raise ExecutionFailed("quote_failed", "bonding curve missing")
            out = onchain.quote_out_for_tokens_in(
                curve.virtual_quote_reserves, curve.virtual_token_reserves, raw
            )
            out = out - onchain.apply_bps(out, self.fee_bps)
            if out <= 0:
                raise ExecutionFailed("quote_empty", "bonding curve quoted 0 SOL")
            min_out = out - onchain.apply_bps(out, self.slippage_bps)
            price = self._price_from_curve(curve, decimals)
            return _Quote("pump", raw, out, min_out, decimals, price, {"curve": curve})
        return await self._jupiter_quote(mint, onchain.WSOL_MINT, raw, decimals, side="sell")

    def _price_from_curve(self, curve: onchain.BondingCurve, decimals: int) -> float:
        if curve.virtual_token_reserves <= 0:
            return 0.0
        sol_per_token = (
            curve.virtual_quote_reserves
            / curve.virtual_token_reserves
            * (10 ** decimals)
            / onchain.LAMPORTS_PER_SOL
        )
        return sol_per_token * self.sol_usd

    async def _jupiter_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        decimals: int,
        side: str,
    ) -> _Quote:
        try:
            body = await self._http_get(
                f"{self.jupiter_base_url}/quote",
                {
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": str(amount),
                    "slippageBps": str(self.slippage_bps),
                },
            )
        except ExecutionFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionFailed("quote_failed", f"jupiter: {exc}") from exc
        if not isinstance(body, dict):
            raise ExecutionFailed("quote_failed", "jupiter quote was not an object")
        try:
            out_amount = int(body.get("outAmount") or 0)
            in_amount = int(body.get("inAmount") or amount)
            min_out = int(body.get("otherAmountThreshold") or 0)
        except (TypeError, ValueError) as exc:
            raise ExecutionFailed("quote_failed", "jupiter quote amounts unreadable") from exc
        if out_amount <= 0:
            raise ExecutionFailed("quote_empty", "jupiter quoted 0")
        if min_out <= 0:
            min_out = out_amount - onchain.apply_bps(out_amount, self.slippage_bps)
        price = 0.0
        sol_usd = self.sol_usd
        if side == "buy" and out_amount > 0 and sol_usd > 0:
            price = (in_amount / onchain.LAMPORTS_PER_SOL) * sol_usd / onchain.ui_amount(
                out_amount, decimals
            )
        elif side == "sell" and in_amount > 0 and sol_usd > 0:
            price = (out_amount / onchain.LAMPORTS_PER_SOL) * sol_usd / onchain.ui_amount(
                in_amount, decimals
            )
        return _Quote("jupiter", in_amount, out_amount, min_out, decimals, price, {"quote": body})

    async def _mark_price(self, mint: str, decimals: int) -> float:
        """USD per whole token from current curve / pool state."""
        try:
            venue, curve = await self._venue(mint)
        except ExecutionFailed:
            return 0.0
        if venue == "pump" and curve is not None:
            return self._price_from_curve(curve, decimals)
        sample = 10 ** max(decimals, 0)
        try:
            quote = await self._jupiter_quote(
                mint, onchain.WSOL_MINT, sample, decimals, side="sell"
            )
        except ExecutionFailed:
            return 0.0
        return quote.price_usd

    # -- orders --------------------------------------------------------------------

    async def buy(self, mint: str, amount_usd: float, **kwargs: Any) -> dict[str, Any]:
        """Swap SOL for `amount_usd` worth of `mint`. Returns {tx_id, quantity, price}."""
        if not mint:
            raise ExecutionFailed("invalid_mint", "mint is required")
        lamports = self._usd_to_lamports(float(amount_usd))
        quote = await self._quote_buy(mint, lamports)
        quantity = onchain.ui_amount(quote.out_amount, quote.decimals)
        if quantity <= 0:
            raise ExecutionFailed("quote_empty", "quoted quantity is 0")
        price = quote.price_usd or (float(amount_usd) / quantity)

        if self.paper:
            tx_id = f"PAPER-{uuid.uuid4().hex[:16]}"
            self._credit(mint, quote.out_amount, quote.decimals, price, float(amount_usd))
            log.info("paper buy %s qty=%.6f @ %.8f", mint, quantity, price)
            return {
                "tx_id": tx_id,
                "quantity": quantity,
                "price": price,
                "filled": True,
                "paper": True,
                "venue": quote.venue,
                "amount_usd": float(amount_usd),
            }

        before = await self._token_balance_raw(mint)
        tx_id = await self._execute(mint, "buy", quote)
        after = await self._token_balance_raw(mint)
        filled_raw = after - before
        self._reconcile_fill(
            mint=mint,
            side="buy",
            filled_raw=filled_raw,
            expected_raw=quote.min_out,
            decimals=quote.decimals,
            signature=tx_id,
            amount_usd=float(amount_usd),
        )
        filled_qty = onchain.ui_amount(filled_raw, quote.decimals)
        fill_price = float(amount_usd) / filled_qty
        self._credit(mint, filled_raw, quote.decimals, fill_price, float(amount_usd))
        log.info("live buy %s qty=%.6f sig=%s venue=%s", mint, filled_qty, tx_id, quote.venue)
        return {
            "tx_id": tx_id,
            "quantity": filled_qty,
            "price": fill_price,
            "filled": True,
            "paper": False,
            "venue": quote.venue,
            "amount_usd": float(amount_usd),
        }

    async def sell(self, mint: str, fraction: float = 1.0, **kwargs: Any) -> dict[str, Any]:
        """Swap `fraction` of the held balance back to SOL."""
        if not mint:
            raise ExecutionFailed("invalid_mint", "mint is required")
        try:
            fraction = float(fraction)
        except (TypeError, ValueError) as exc:
            raise ExecutionFailed("invalid_fraction", str(fraction)) from exc
        if fraction <= 0 or fraction > 1.0:
            raise ExecutionFailed("invalid_fraction", str(fraction))

        if self.paper and mint in self._lots:
            decimals = self._lots[mint].decimals
            balance_raw = self._lots[mint].raw_amount
        else:
            decimals = await self._mint_decimals(mint)
            balance_raw = await self._token_balance_raw(mint)
        raw = onchain.fraction_raw(balance_raw, fraction)
        if raw <= 0:
            raise ExecutionFailed("insufficient_balance", "nothing to sell after decimals")

        quote = await self._quote_sell(mint, raw, decimals)
        quantity = onchain.ui_amount(raw, decimals)
        sol_usd = self._require_sol_usd()
        proceeds = quote.out_amount / onchain.LAMPORTS_PER_SOL * sol_usd
        price = quote.price_usd or (proceeds / quantity if quantity else 0.0)

        if self.paper:
            tx_id = f"PAPER-{uuid.uuid4().hex[:16]}"
            self._debit(mint, raw)
            log.info("paper sell %s fraction=%.3f qty=%.6f", mint, fraction, quantity)
            return {
                "tx_id": tx_id,
                "quantity": quantity,
                "price": price,
                "filled": True,
                "paper": True,
                "venue": quote.venue,
                "amount_usd": proceeds,
            }

        before = await self._token_balance_raw(mint)
        tx_id = await self._execute(mint, "sell", quote)
        after = await self._token_balance_raw(mint)
        sold_raw = before - after
        self._reconcile_fill(
            mint=mint,
            side="sell",
            filled_raw=sold_raw,
            expected_raw=raw,
            decimals=decimals,
            signature=tx_id,
            amount_usd=proceeds,
        )
        self._debit(mint, sold_raw)
        sold_qty = onchain.ui_amount(sold_raw, decimals)
        log.info("live sell %s qty=%.6f sig=%s venue=%s", mint, sold_qty, tx_id, quote.venue)
        return {
            "tx_id": tx_id,
            "quantity": sold_qty,
            "price": price,
            "filled": True,
            "paper": False,
            "venue": quote.venue,
            "amount_usd": proceeds,
        }

    async def close_position(self, mint: str) -> dict[str, Any]:
        """Full exit. Equivalent to sell(mint, 1.0)."""
        return await self.sell(mint, 1.0)

    async def tighten_stop(self, mint: str, new_stop_price: float) -> dict[str, Any]:
        """Desk-side stop. pump.fun has no on-chain stop order.

        The desk already writes `position.stop_price` after this returns. We
        keep a local copy so `get_positions()` can reattach it. A non-positive
        price is a deny — failure must not look like a tighter stop.
        """
        try:
            price = float(new_stop_price)
        except (TypeError, ValueError) as exc:
            raise ExecutionFailed("invalid_stop", "stop price is not a number") from exc
        if not mint or price <= 0:
            raise ExecutionFailed(
                "invalid_stop", "desk-side stop requires a mint and a positive price"
            )
        self._stops[mint] = price
        if mint in self._lots:
            # lots do not carry stop; get_positions reads `_stops`
            pass
        log.info("local stop %s -> %.8f (not on-chain)", mint, price)
        return {"mint": mint, "stop_price": price, "on_chain": False}

    async def get_positions(self) -> list[Position]:
        """Wallet holdings priced at current curve/pool state.

        Paper returns the in-memory book. Live reads SPL token accounts, drops
        dust and anything this desk did not open (event log + local fills).
        """
        if self.paper:
            return await self._positions_from_lots()
        try:
            return await self._positions_from_chain()
        except ExecutionFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionFailed("rpc_error", f"get_positions: {exc}") from exc

    # -- live submit ---------------------------------------------------------------

    async def _execute(self, mint: str, side: str, quote: _Quote) -> str:
        instructions = await self._build_instructions(mint, side, quote)
        from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
        from solders.hash import Hash
        from solders.message import MessageV0
        from solders.transaction import VersionedTransaction

        keyed = self._load_keypair()
        head = [set_compute_unit_limit(self.compute_unit_limit)]
        if self.priority_fee > 0:
            head.append(set_compute_unit_price(self.priority_fee))
        if self.jito_enabled:
            if self.tip_lamports <= 0:
                raise ExecutionFailed("jito_tip_missing", "jito.enabled requires tip_lamports")
            head.append(self._tip_instruction(str(keyed.pubkey())))
        ixs = head + instructions

        blockhash = await self._recent_blockhash()
        try:
            message = MessageV0.try_compile(
                keyed.pubkey(), ixs, quote.extra.get("alts") or [], Hash.from_string(blockhash)
            )
            tx = VersionedTransaction(message, [keyed])
        except Exception as exc:  # noqa: BLE001 - compile errors are not fills
            raise ExecutionFailed("submit_failed", f"transaction compile failed: {exc}") from exc

        signature = str(tx.signatures[0])
        raw = bytes(tx)
        if self.jito_enabled:
            await self._submit_bundle(raw)
        else:
            await self._submit_rpc(raw)
        await self._confirm(signature)
        return signature

    def _tip_instruction(self, payer: str):
        from solders.pubkey import Pubkey
        from solders.system_program import TransferParams, transfer

        tip_to = onchain.JITO_TIP_ACCOUNTS[hash(payer) % len(onchain.JITO_TIP_ACCOUNTS)]
        try:
            return transfer(
                TransferParams(
                    from_pubkey=Pubkey.from_string(payer),
                    to_pubkey=Pubkey.from_string(tip_to),
                    lamports=self.tip_lamports,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise ExecutionFailed("jito_tip_missing", "invalid Jito tip account") from exc

    async def _build_instructions(self, mint: str, side: str, quote: _Quote) -> list[Any]:
        if quote.venue == "jupiter":
            return await self._jupiter_instructions(quote)
        return await self._pump_instructions(mint, side, quote)

    async def _pump_instructions(self, mint: str, side: str, quote: _Quote) -> list[Any]:
        curve: onchain.BondingCurve | None = quote.extra.get("curve")
        if curve is None:
            curve = await self._fetch_curve(mint)
        if curve is None or not curve.creator:
            raise ExecutionFailed("quote_failed", "cannot build pump ix without curve creator")
        user = self._pubkey()
        owner = await self._account_owner(mint)
        token_program = (
            onchain.TOKEN_2022_PROGRAM
            if owner == onchain.TOKEN_2022_PROGRAM
            else onchain.TOKEN_PROGRAM
        )
        ixs = [onchain.create_ata_idempotent(user, user, mint, token_program)]
        fee_recipient = self.fee_recipient_override
        buyback = self.buyback_fee_override
        if side == "buy":
            max_cost = quote.in_amount + onchain.apply_bps(quote.in_amount, self.slippage_bps)
            # buy_v2 reads associated_quote_user (the WSOL ATA). Native SOL in
            # the wallet does not fund that account — wrap first.
            ixs.extend(onchain.wrap_sol_instructions(user, max_cost))
            ixs.append(
                onchain.pump_buy_v2_instruction(
                    mint=mint,
                    user=user,
                    amount=quote.out_amount,
                    max_quote_cost=max_cost,
                    creator=curve.creator,
                    base_token_program=token_program,
                    fee_recipient=fee_recipient,
                    buyback_fee_recipient=buyback,
                    is_mayhem_mode=curve.is_mayhem_mode,
                )
            )
            ixs.append(onchain.unwrap_wsol_instruction(user))
        else:
            # Ensure a WSOL ATA exists to receive quote, then unwrap leftovers.
            ixs.append(onchain.create_ata_idempotent(user, user, onchain.WSOL_MINT))
            ixs.append(
                onchain.pump_sell_v2_instruction(
                    mint=mint,
                    user=user,
                    amount=quote.in_amount,
                    min_quote_out=quote.min_out,
                    creator=curve.creator,
                    base_token_program=token_program,
                    fee_recipient=fee_recipient,
                    buyback_fee_recipient=buyback,
                    is_mayhem_mode=curve.is_mayhem_mode,
                )
            )
            ixs.append(onchain.unwrap_wsol_instruction(user))
        return ixs

    async def _jupiter_instructions(self, quote: _Quote) -> list[Any]:
        payload = {
            "quoteResponse": quote.extra.get("quote") or {},
            "userPublicKey": self._pubkey(),
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            # Prefer a self-contained message; ALT fetch is a follow-up if a
            # route refuses to compile without lookup tables.
            "asLegacyTransaction": True,
        }
        try:
            body = await self._http_post(f"{self.jupiter_base_url}/swap-instructions", payload)
        except ExecutionFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionFailed("submit_failed", f"jupiter swap-instructions: {exc}") from exc
        if not isinstance(body, dict) or body.get("error"):
            raise ExecutionFailed("submit_failed", f"jupiter swap-instructions: {body}")
        raw_ixs: list[Any] = []
        for key in ("setupInstructions", "otherInstructions"):
            raw_ixs.extend(body.get(key) or [])
        swap = body.get("swapInstruction")
        if swap:
            raw_ixs.append(swap)
        cleanup = body.get("cleanupInstruction")
        if cleanup:
            raw_ixs.append(cleanup)
        if not raw_ixs:
            raise ExecutionFailed("submit_failed", "jupiter returned no instructions")
        try:
            return [onchain.jupiter_instruction(item) for item in raw_ixs]
        except Exception as exc:  # noqa: BLE001
            raise ExecutionFailed("submit_failed", f"jupiter ix decode: {exc}") from exc

    async def _recent_blockhash(self) -> str:
        result = await self._rpc_call("getLatestBlockhash", [{"commitment": "confirmed"}])
        value = (result or {}).get("value") if isinstance(result, dict) else None
        blockhash = (value or {}).get("blockhash") if isinstance(value, dict) else None
        if not blockhash:
            raise ExecutionFailed("submit_failed", "no recent blockhash")
        return str(blockhash)

    async def _submit_rpc(self, raw: bytes) -> None:
        encoded = base64.b64encode(raw).decode("ascii")
        result = await self._rpc_call(
            "sendTransaction",
            [encoded, {"encoding": "base64", "skipPreflight": False, "preflightCommitment": "confirmed"}],
        )
        if not result:
            raise ExecutionFailed("submit_failed", "rpc sendTransaction returned empty")

    async def _submit_bundle(self, raw: bytes) -> None:
        if not self.jito_url:
            raise ExecutionFailed("bundle_rejected", "jito.block_engine_url is empty")
        url = f"{self.jito_url}/api/v1/bundles"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [[onchain.b58encode(raw)]],
        }
        try:
            body = await self._http_post(url, payload)
        except ExecutionFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionFailed("bundle_rejected", str(exc)) from exc
        if isinstance(body, dict) and body.get("error"):
            raise ExecutionFailed("bundle_rejected", str(body["error"]))
        if not (isinstance(body, dict) and body.get("result")):
            raise ExecutionFailed("bundle_rejected", "block engine returned no bundle id")

    async def _confirm(self, signature: str) -> None:
        deadline = self._monotonic() + self.confirm_timeout
        last_err = "not landed"
        while self._monotonic() < deadline:
            try:
                result = await self._rpc_call(
                    "getSignatureStatuses", [[signature], {"searchTransactionHistory": True}]
                )
            except ExecutionFailed as exc:
                last_err = exc.detail or exc.reason
                await self._sleep(self.confirm_poll)
                continue
            value = (result or {}).get("value") if isinstance(result, dict) else None
            status = (value or [None])[0] if isinstance(value, list) else None
            if isinstance(status, dict):
                if status.get("err"):
                    raise ExecutionFailed("submit_failed", f"on-chain error: {status['err']}")
                confirmation = str(status.get("confirmationStatus") or "")
                if confirmation in {"confirmed", "finalized"}:
                    return
                if status.get("confirmations") is not None or status.get("slot"):
                    # processed-or-better without an err is not enough; keep polling
                    last_err = f"status={confirmation or 'processed'}"
            await self._sleep(self.confirm_poll)
        raise ExecutionFailed(
            "unconfirmed",
            f"signature {signature} did not confirm within {self.confirm_timeout:.1f}s ({last_err})",
        )

    def _record_stranded(
        self,
        *,
        mint: str,
        side: str,
        signature: str,
        raw_amount: int,
        decimals: int,
        expected_raw: int,
        reason: str,
    ) -> dict[str, Any]:
        payload = {
            "market": Market.CRYPTO.value,
            "mint": mint,
            "side": side,
            "signature": signature,
            "raw_amount": int(raw_amount),
            "decimals": int(decimals),
            "expected_raw": int(expected_raw),
            "quantity": onchain.ui_amount(raw_amount, decimals) if raw_amount > 0 else 0.0,
            "reason": reason,
        }
        if self.event_log is not None and hasattr(self.event_log, "write"):
            self.event_log.write("stranded", **payload)
        log.warning(
            "stranded inventory mint=%s side=%s raw=%s expected=%s sig=%s",
            mint,
            side,
            raw_amount,
            expected_raw,
            signature,
        )
        return payload

    def _reconcile_fill(
        self,
        *,
        mint: str,
        side: str,
        filled_raw: int,
        expected_raw: int,
        decimals: int,
        signature: str,
        amount_usd: float,
    ) -> None:
        """After confirm: credit whatever landed, then fail if it was not full.

        Fail-closed for risk (the exception still denies a success fill) but
        never leave wallet tokens invisible to `get_positions`.
        """
        if filled_raw >= expected_raw and filled_raw > 0:
            return
        reason = "zero_fill" if filled_raw <= 0 else "partial_fill"
        if filled_raw > 0:
            qty = onchain.ui_amount(filled_raw, decimals)
            price = (amount_usd / qty) if qty else 0.0
            if side == "buy":
                # Assume the quoted USD left the wallet even if tokens fell short.
                self._credit(mint, filled_raw, decimals, price, amount_usd)
            else:
                self._debit(mint, filled_raw)
        stranded = self._record_stranded(
            mint=mint,
            side=side,
            signature=signature,
            raw_amount=max(0, filled_raw),
            decimals=decimals,
            expected_raw=expected_raw,
            reason=reason,
        )
        raise ExecutionFailed(
            reason,
            f"{side} filled {filled_raw} raw, expected at least {expected_raw}",
            stranded=stranded,
        )

    # -- balances / book -----------------------------------------------------------

    async def _token_balance_raw(self, mint: str) -> int:
        if self.paper and mint in self._lots:
            return self._lots[mint].raw_amount
        owner = self._pubkey()
        for program in (onchain.TOKEN_PROGRAM, onchain.TOKEN_2022_PROGRAM):
            result = await self._rpc_call(
                "getTokenAccountsByOwner",
                [owner, {"mint": mint}, {"encoding": "jsonParsed"}],
            )
            value = (result or {}).get("value") if isinstance(result, dict) else None
            for item in value or []:
                amount = _parsed_token_amount(item)
                if amount is not None:
                    return amount
        return 0

    def _credit(self, mint: str, raw: int, decimals: int, price: float, amount_usd: float) -> None:
        lot = self._lots.get(mint)
        if lot is None:
            self._lots[mint] = _Lot(mint, raw, decimals, price, amount_usd)
        else:
            new_raw = lot.raw_amount + raw
            new_usd = lot.amount_usd + amount_usd
            lot.raw_amount = new_raw
            lot.amount_usd = new_usd
            if new_raw:
                lot.entry_price = new_usd / onchain.ui_amount(new_raw, decimals)
        self._opened.add(mint)

    def _debit(self, mint: str, raw: int) -> None:
        lot = self._lots.get(mint)
        if lot is None:
            self._opened.discard(mint)
            return
        lot.raw_amount = max(0, lot.raw_amount - raw)
        if lot.raw_amount <= 0:
            self._lots.pop(mint, None)
            self._opened.discard(mint)
            self._stops.pop(mint, None)
        else:
            qty = onchain.ui_amount(lot.raw_amount, lot.decimals)
            lot.amount_usd = qty * lot.entry_price

    async def _positions_from_lots(self) -> list[Position]:
        positions: list[Position] = []
        for mint, lot in list(self._lots.items()):
            qty = onchain.ui_amount(lot.raw_amount, lot.decimals)
            if qty <= 0:
                continue
            price = lot.entry_price
            try:
                marked = await self._mark_price(mint, lot.decimals)
                if marked > 0:
                    price = marked
            except Exception:  # noqa: BLE001 - a mark failure must not hide the lot
                pass
            if price > 0 and qty * price < self.dust_usd:
                continue
            positions.append(
                Position(
                    market=Market.CRYPTO,
                    symbol=lot.symbol or mint,
                    quantity=qty,
                    entry_price=lot.entry_price,
                    current_price=price,
                    amount_usd=lot.amount_usd,
                    stop_price=self._stops.get(mint),
                    meta={"mint": mint},
                )
            )
        return positions

    async def _positions_from_chain(self) -> list[Position]:
        opened = self._desk_opened_mints()
        owner = self._pubkey()
        holdings: list[tuple[str, int, int]] = []
        for program in (onchain.TOKEN_PROGRAM, onchain.TOKEN_2022_PROGRAM):
            result = await self._rpc_call(
                "getTokenAccountsByOwner",
                [owner, {"programId": program}, {"encoding": "jsonParsed"}],
            )
            value = (result or {}).get("value") if isinstance(result, dict) else None
            for item in value or []:
                parsed = _parsed_holding(item)
                if parsed is None:
                    continue
                mint, raw, decimals = parsed
                if raw <= 0:
                    continue
                if opened and mint not in opened:
                    continue
                holdings.append((mint, raw, decimals))

        # If the desk has no recorded opens, do not invent positions from the wallet.
        if not opened:
            return []

        positions: list[Position] = []
        for mint, raw, decimals in holdings:
            qty = onchain.ui_amount(raw, decimals)
            price = 0.0
            try:
                price = await self._mark_price(mint, decimals)
            except Exception:  # noqa: BLE001
                price = 0.0
            if price > 0 and qty * price < self.dust_usd:
                continue
            lot = self._lots.get(mint)
            entry = lot.entry_price if lot else price
            amount = lot.amount_usd if lot else (qty * entry if entry else 0.0)
            positions.append(
                Position(
                    market=Market.CRYPTO,
                    symbol=(lot.symbol if lot else "") or mint,
                    quantity=qty,
                    entry_price=entry,
                    current_price=price,
                    amount_usd=amount,
                    stop_price=self._stops.get(mint),
                    meta={"mint": mint},
                )
            )
        return positions

    def _desk_opened_mints(self) -> set[str]:
        opened = set(self._opened)
        if self.event_log is None:
            return opened
        try:
            records = list(self.event_log.read())
        except Exception:  # noqa: BLE001 - a broken log must not invent positions
            return opened

        still: dict[str, bool] = {}
        for record in records:
            if record.get("market") != Market.CRYPTO.value:
                continue
            key = _mint_from_record(record)
            if not key:
                continue
            if record.get("type") == "buy":
                still[key] = True
            elif record.get("type") == "stranded" and int(record.get("raw_amount") or 0) > 0:
                still[key] = True
            elif record.get("type") == "close":
                still.pop(key, None)
        opened.update(still)
        return opened


def _as_json(response: Any) -> Any:
    if hasattr(response, "raise_for_status"):
        try:
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise ExecutionFailed("rpc_error", str(exc)) from exc
    if hasattr(response, "json"):
        body = response.json()
        return body() if callable(body) else body
    return response


def _parsed_token_amount(item: dict[str, Any]) -> int | None:
    try:
        info = item["account"]["data"]["parsed"]["info"]
        return int(info["tokenAmount"]["amount"])
    except (KeyError, TypeError, ValueError):
        return None


def _parsed_holding(item: dict[str, Any]) -> tuple[str, int, int] | None:
    try:
        info = item["account"]["data"]["parsed"]["info"]
        mint = str(info["mint"])
        amount = info["tokenAmount"]
        return mint, int(amount["amount"]), int(amount["decimals"])
    except (KeyError, TypeError, ValueError):
        return None


def _mint_from_record(record: dict[str, Any]) -> str:
    if record.get("mint"):
        return str(record["mint"])
    scores = record.get("all_agent_scores") or {}
    token = scores.get("token") if isinstance(scores, dict) else None
    if isinstance(token, dict) and token.get("mint"):
        return str(token["mint"])
    symbol = str(record.get("symbol") or "")
    if onchain.looks_like_pubkey(symbol):
        return symbol
    return ""
