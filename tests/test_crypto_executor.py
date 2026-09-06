"""CryptoExecutor: paper fills, live submit/confirm, pessimistic failures.

Every test injects RPC/HTTP fakes. Nothing here talks to mainnet.
"""

from __future__ import annotations

import base64
import logging

import pytest
from solders.hash import Hash
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from src.crypto.crypto_executor import CryptoExecutor, ExecutionFailed
from src.crypto import onchain
from src.models import Market
from src.shared.log import EventLog


SECRET_MARKER = "SUPER_SECRET_TEST_MATERIAL_NOT_A_REAL_KEY"


class DenyHttp:
    """Any unexpected network call fails the test."""

    async def get(self, *args, **kwargs):
        raise AssertionError(f"unexpected HTTP GET: {args} {kwargs}")

    async def post(self, *args, **kwargs):
        raise AssertionError(f"unexpected HTTP POST: {args} {kwargs}")


class Json:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self):
        return self._body


class FakeRpc:
    def __init__(self, accounts=None, blockhash=None):
        self.accounts = accounts or {}
        self.blockhash = blockhash or str(Hash.default())
        self.token_raw = 0
        self.token_decimals = 6
        self.mint = ""
        self.statuses = [{"confirmationStatus": "confirmed", "err": None}]
        self.wallet_holdings: list[dict] = []
        self.sent = False
        self.balance_reads = 0
        self.calls: list[tuple[str, list]] = []

    async def call(self, method, params):
        self.calls.append((method, params))
        if method == "getAccountInfo":
            return {"value": self.accounts.get(params[0])}
        if method == "getLatestBlockhash":
            return {"value": {"blockhash": self.blockhash}}
        if method == "sendTransaction":
            opts = params[1] if len(params) > 1 else {}
            assert opts.get("skipPreflight") is False
            self.sent = True
            return "ok"
        if method == "getSignatureStatuses":
            return {"value": list(self.statuses)}
        if method == "getTokenAccountsByOwner":
            if self.wallet_holdings:
                return {"value": list(self.wallet_holdings)}
            self.balance_reads += 1
            amount = self.token_raw if (self.sent or self.balance_reads > 1) else 0
            if not self.mint:
                return {"value": []}
            return {
                "value": [
                    _token_account(self.mint, amount, self.token_decimals)
                ]
            }
        raise AssertionError(f"unexpected RPC {method}")


class FakeHttp:
    def __init__(self, quote=None, swap_ixs=None, bundle=None):
        self.quote = quote
        self.swap_ixs = swap_ixs
        self.bundle = bundle if bundle is not None else {"result": "bundle-1"}
        self.posts: list[dict] = []
        self.gets: list[dict] = []

    async def get(self, url, params=None, **kwargs):
        self.gets.append({"url": url, "params": params})
        if "/quote" in url:
            return Json(self.quote)
        raise AssertionError(f"unexpected GET {url}")

    async def post(self, url, json=None, **kwargs):
        self.posts.append({"url": url, "json": json})
        if url.endswith("/api/v1/bundles"):
            return Json(self.bundle)
        if url.endswith("/swap-instructions"):
            return Json(self.swap_ixs)
        raise AssertionError(f"unexpected POST {url}")


class Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _token_account(mint: str, amount: int, decimals: int) -> dict:
    return {
        "account": {
            "data": {
                "parsed": {
                    "info": {
                        "mint": mint,
                        "tokenAmount": {
                            "amount": str(amount),
                            "decimals": decimals,
                            "uiAmount": amount / (10 ** decimals),
                        },
                    }
                }
            }
        }
    }


def _account(data: bytes, owner: str = onchain.TOKEN_PROGRAM) -> dict:
    return {"data": [base64.b64encode(data).decode(), "base64"], "owner": owner}


def _pump_world(decimals: int = 8, complete: bool = False):
    mint = str(Keypair().pubkey())
    creator = str(Keypair().pubkey())
    curve = onchain.pack_bonding_curve(
        virtual_token=1_000_000_000_000,
        virtual_quote=30_000_000_000,
        real_token=800_000_000_000,
        real_quote=0,
        supply=1_000_000_000_000,
        complete=complete,
        creator=bytes(Pubkey.from_string(creator)),
    )
    accounts = {
        mint: _account(onchain.pack_mint(decimals), onchain.TOKEN_PROGRAM),
        str(onchain.bonding_curve_pda(mint)): _account(curve, onchain.PUMP_PROGRAM),
    }
    return mint, creator, accounts


def _config(**overrides) -> dict:
    cfg = {
        "mode": "paper",
        "solana": {
            "rpc_url": "https://rpc.test",
            "wallet_key": SECRET_MARKER,
            "slippage_bps": 500,
            "priority_fee_microlamports": 1000,
            "compute_unit_limit": 400000,
            "confirm_timeout_seconds": 5,
            "confirm_poll_seconds": 0.1,
            "dust_usd": 0.5,
            "jito": {
                "enabled": False,
                "block_engine_url": "https://jito.test",
                "tip_lamports": 100000,
            },
        },
        "pump_fun": {"sol_price_usd": 200.0},
    }
    cfg.update(overrides)
    return cfg


def _executor(rpc=None, http=None, live=False, event_log=None, **kwargs) -> CryptoExecutor:
    config = kwargs.pop("config", None) or _config()
    if live:
        config["mode"] = "live"
    return CryptoExecutor(
        config,
        event_log=event_log,
        live_ack=live,
        rpc=rpc,
        http=http if http is not None else DenyHttp(),
        keypair=kwargs.pop("keypair", Keypair()),
        **kwargs,
    )


# -- on-chain helpers --------------------------------------------------------------

def test_curve_math_and_decimals_are_not_assumed():
    out = onchain.tokens_out_for_quote_in(30_000_000_000, 1_000_000_000_000, 1_000_000_000)
    assert out > 0
    back = onchain.quote_out_for_tokens_in(30_000_000_000, 1_000_000_000_000, out)
    assert 0 < back <= 1_000_000_000

    packed = onchain.pack_mint(8)
    assert onchain.mint_decimals(packed) == 8
    packed6 = onchain.pack_mint(6)
    assert onchain.mint_decimals(packed6) == 6
    assert onchain.raw_amount(1.5, 8) == 150_000_000
    assert onchain.raw_amount(1.5, 6) == 1_500_000
    assert onchain.fraction_raw(1_000_000, 0.5) == 500_000
    assert onchain.fraction_raw(7, 1.0) == 7


def test_bonding_curve_complete_flag_is_the_venue_switch():
    open_curve = onchain.parse_bonding_curve(
        onchain.pack_bonding_curve(1, 2, 3, 4, 5, complete=False)
    )
    done = onchain.parse_bonding_curve(
        onchain.pack_bonding_curve(1, 2, 3, 4, 5, complete=True)
    )
    assert open_curve.complete is False
    assert done.complete is True


def test_pump_instructions_encode_raw_u64_and_known_program():
    mint = str(Keypair().pubkey())
    user = str(Keypair().pubkey())
    creator = str(Keypair().pubkey())
    buy = onchain.pump_buy_v2_instruction(
        mint=mint, user=user, amount=123, max_quote_cost=456, creator=creator
    )
    sell = onchain.pump_sell_v2_instruction(
        mint=mint, user=user, amount=789, min_quote_out=10, creator=creator
    )
    assert str(buy.program_id) == onchain.PUMP_PROGRAM
    assert str(sell.program_id) == onchain.PUMP_PROGRAM
    assert buy.data[:8] == onchain.anchor_discriminator("buy_v2")
    assert sell.data[:8] == onchain.anchor_discriminator("sell_v2")
    assert int.from_bytes(buy.data[8:16], "little") == 123
    assert int.from_bytes(sell.data[8:16], "little") == 789
    assert len(buy.accounts) == 27
    assert len(sell.accounts) == 26


def test_b58encode_matches_solders_pubkey():
    pubkey = Keypair().pubkey()
    assert onchain.b58encode(bytes(pubkey)) == str(pubkey)


def test_jito_tip_accounts_are_valid_pubkeys():
    for address in onchain.JITO_TIP_ACCOUNTS:
        Pubkey.from_string(address)


# -- paper path --------------------------------------------------------------------

async def test_paper_buy_quotes_the_bonding_curve_and_never_sends():
    mint, _creator, accounts = _pump_world(decimals=8)
    rpc = FakeRpc(accounts)
    executor = _executor(rpc=rpc)

    fill = await executor.buy(mint, 20.0)

    assert fill["filled"] is True
    assert fill["paper"] is True
    assert fill["venue"] == "pump"
    assert fill["tx_id"].startswith("PAPER-")
    assert fill["quantity"] > 0
    assert fill["price"] > 0
    assert rpc.sent is False
    assert not any(method == "sendTransaction" for method, _ in rpc.calls)


async def test_paper_sell_uses_the_mint_decimals_not_six_or_nine():
    mint, _creator, accounts = _pump_world(decimals=8)
    rpc = FakeRpc(accounts)
    executor = _executor(rpc=rpc)
    await executor.buy(mint, 20.0)
    raw_before = executor._lots[mint].raw_amount
    assert executor._lots[mint].decimals == 8

    fill = await executor.sell(mint, 0.5)

    assert fill["filled"] is True
    assert executor._lots[mint].raw_amount == raw_before - raw_before // 2
    # 8-decimal half-lot would be wrong if we assumed 6 (100x) or 9 (1/10).
    assert executor._lots[mint].decimals == 8


async def test_close_position_is_sell_all():
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    executor = _executor(rpc=rpc)
    await executor.buy(mint, 20.0)
    fill = await executor.close_position(mint)
    assert fill["filled"] is True
    assert mint not in executor._lots


async def test_paper_buy_fails_without_a_sol_price():
    mint, _creator, accounts = _pump_world()
    config = _config()
    config["pump_fun"]["sol_price_usd"] = 0
    executor = _executor(rpc=FakeRpc(accounts), config=config)
    with pytest.raises(ExecutionFailed) as err:
        await executor.buy(mint, 20.0)
    assert err.value.reason == "no_sol_price"


async def test_paper_buy_fails_on_an_empty_quote():
    mint, _creator, accounts = _pump_world()
    # empty reserves → 0 tokens out
    accounts[str(onchain.bonding_curve_pda(mint))] = _account(
        onchain.pack_bonding_curve(0, 0, 0, 0, 0, complete=False),
        onchain.PUMP_PROGRAM,
    )
    executor = _executor(rpc=FakeRpc(accounts))
    with pytest.raises(ExecutionFailed) as err:
        await executor.buy(mint, 20.0)
    assert err.value.reason in {"quote_empty", "quote_failed"}


async def test_bonded_token_routes_through_jupiter_on_paper():
    mint, _creator, accounts = _pump_world(decimals=6, complete=True)
    rpc = FakeRpc(accounts)
    http = FakeHttp(
        quote={
            "inAmount": "100000000",
            "outAmount": "5000000",
            "otherAmountThreshold": "4750000",
        }
    )
    executor = _executor(rpc=rpc, http=http)
    fill = await executor.buy(mint, 20.0)
    assert fill["venue"] == "jupiter"
    assert fill["quantity"] == pytest.approx(5.0)
    assert http.gets and "/quote" in http.gets[0]["url"]
    assert rpc.sent is False


# -- tighten_stop ------------------------------------------------------------------

async def test_tighten_stop_is_local_state():
    executor = _executor(rpc=FakeRpc())
    result = await executor.tighten_stop("So11111111111111111111111111111111111111112", 0.004)
    assert result["on_chain"] is False
    assert result["stop_price"] == 0.004
    assert executor._stops[result["mint"]] == 0.004


async def test_tighten_stop_denies_a_non_positive_price():
    executor = _executor(rpc=FakeRpc())
    with pytest.raises(ExecutionFailed) as err:
        await executor.tighten_stop("So11111111111111111111111111111111111111112", 0)
    assert err.value.reason == "invalid_stop"


# -- get_positions -----------------------------------------------------------------

async def test_get_positions_returns_paper_lots_and_drops_dust():
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    executor = _executor(rpc=rpc)
    executor.dust_usd = 0.5
    await executor.buy(mint, 20.0)
    positions = await executor.get_positions()
    assert len(positions) == 1
    assert positions[0].market == Market.CRYPTO
    assert positions[0].meta["mint"] == mint

    # Force the lot under the dust floor.
    executor._lots[mint].raw_amount = 1
    executor._lots[mint].entry_price = 0.0001
    executor._lots[mint].amount_usd = 0.0001
    dusted = await executor.get_positions()
    assert dusted == []


async def test_get_positions_live_drops_holdings_the_desk_did_not_open(tmp_path):
    mint, _creator, accounts = _pump_world(decimals=6)
    stranger = str(Keypair().pubkey())
    rpc = FakeRpc(accounts)
    rpc.wallet_holdings = [
        _token_account(mint, 1_000_000, 6),
        _token_account(stranger, 9_000_000, 6),
    ]
    log = EventLog({"logging": {"path": str(tmp_path / "desk.jsonl"), "echo_stdout": False}})
    log.buy("crypto", "WIF2", 0.7, {"token": {"mint": mint}}, 20.0, tx_id="x", mint=mint)
    executor = _executor(rpc=rpc, live=True, event_log=log)
    executor.dust_usd = 0
    executor._opened.add(mint)
    positions = await executor.get_positions()
    mints = {p.meta["mint"] for p in positions}
    assert mint in mints
    assert stranger not in mints


# -- live submit -------------------------------------------------------------------

async def test_live_rpc_buy_confirms_and_checks_the_fill():
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    rpc.mint = mint
    rpc.token_decimals = 6
    executor = _executor(rpc=rpc, live=True)

    # Prime the expected post-send balance from the same quote the executor will use.
    quote = await executor._quote_buy(mint, executor._usd_to_lamports(20.0))
    rpc.token_raw = quote.out_amount

    fill = await executor.buy(mint, 20.0)
    assert fill["paper"] is False
    assert fill["filled"] is True
    assert fill["tx_id"]
    assert not fill["tx_id"].startswith("PAPER-")
    assert rpc.sent is True
    assert any(method == "sendTransaction" for method, _ in rpc.calls)
    assert any(method == "getSignatureStatuses" for method, _ in rpc.calls)


async def test_live_jito_bundle_path_posts_to_the_block_engine():
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    rpc.mint = mint
    http = FakeHttp(bundle={"result": "bundle-abc"})
    config = _config()
    config["mode"] = "live"
    config["solana"]["jito"]["enabled"] = True
    executor = _executor(rpc=rpc, http=http, live=True, config=config)
    quote = await executor._quote_buy(mint, executor._usd_to_lamports(20.0))
    rpc.token_raw = quote.out_amount

    fill = await executor.buy(mint, 20.0)
    assert fill["filled"] is True
    assert http.posts and http.posts[0]["url"].endswith("/api/v1/bundles")
    assert http.posts[0]["json"]["method"] == "sendBundle"
    assert rpc.sent is False  # Jito path must not also hit sendTransaction


async def test_unconfirmed_bundle_is_a_failure_not_a_fill():
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    rpc.mint = mint
    rpc.statuses = [None]
    clock = Clock()

    async def sleep(dt):
        clock.advance(dt)

    http = FakeHttp(bundle={"result": "bundle-abc"})
    config = _config()
    config["mode"] = "live"
    config["solana"]["jito"]["enabled"] = True
    config["solana"]["confirm_timeout_seconds"] = 1
    executor = _executor(
        rpc=rpc, http=http, live=True, config=config, monotonic=clock.monotonic, sleep=sleep
    )
    with pytest.raises(ExecutionFailed) as err:
        await executor.buy(mint, 20.0)
    assert err.value.reason == "unconfirmed"
    assert mint not in executor._lots


async def test_partial_fill_is_a_failure():
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    rpc.mint = mint
    executor = _executor(rpc=rpc, live=True)
    quote = await executor._quote_buy(mint, executor._usd_to_lamports(20.0))
    rpc.token_raw = max(1, quote.min_out // 2)  # confirmed, but short

    with pytest.raises(ExecutionFailed) as err:
        await executor.buy(mint, 20.0)
    assert err.value.reason == "partial_fill"
    assert mint not in executor._lots


async def test_zero_fill_after_confirm_is_a_failure():
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    rpc.mint = mint
    rpc.token_raw = 0
    executor = _executor(rpc=rpc, live=True)
    with pytest.raises(ExecutionFailed) as err:
        await executor.buy(mint, 20.0)
    assert err.value.reason == "zero_fill"


async def test_live_without_a_wallet_key_fails():
    mint, _creator, accounts = _pump_world()
    config = _config()
    config["mode"] = "live"
    config["solana"]["wallet_key"] = "REPLACE_ME_BASE58_SECRET_KEY"
    executor = CryptoExecutor(
        config, live_ack=True, rpc=FakeRpc(accounts), http=DenyHttp(), keypair=None
    )
    with pytest.raises(ExecutionFailed) as err:
        await executor.buy(mint, 20.0)
    assert err.value.reason == "missing_wallet_key"


async def test_rejected_jito_bundle_never_looks_like_success():
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    http = FakeHttp(bundle={"error": {"message": "bundle dropped"}})
    config = _config()
    config["mode"] = "live"
    config["solana"]["jito"]["enabled"] = True
    executor = _executor(rpc=rpc, http=http, live=True, config=config)
    with pytest.raises(ExecutionFailed) as err:
        await executor.buy(mint, 20.0)
    assert err.value.reason == "bundle_rejected"


async def test_wallet_key_is_never_logged(caplog):
    mint, _creator, accounts = _pump_world(decimals=6)
    rpc = FakeRpc(accounts)
    caplog.set_level(logging.DEBUG)
    executor = _executor(rpc=rpc)
    await executor.buy(mint, 20.0)
    await executor.tighten_stop(mint, 0.001)
    blob = "\n".join(record.getMessage() for record in caplog.records)
    assert SECRET_MARKER not in blob
    # solders Keypair stringifies to the secret; that must not appear either
    assert str(executor._keypair) not in blob


async def test_invalid_fraction_and_empty_mint_are_denies():
    mint, _creator, accounts = _pump_world()
    executor = _executor(rpc=FakeRpc(accounts))
    with pytest.raises(ExecutionFailed) as err:
        await executor.sell(mint, 0)
    assert err.value.reason == "invalid_fraction"
    with pytest.raises(ExecutionFailed) as err:
        await executor.buy("", 10)
    assert err.value.reason == "invalid_mint"


# -- desk wiring -------------------------------------------------------------------

async def test_desk_records_a_paper_crypto_fill(tmp_path):
    from tests.test_desk import TOKEN, build

    desk = build(tmp_path, dry_run=False)
    mint, _creator, accounts = _pump_world(decimals=6)
    token = TOKEN.model_copy(update={"mint": mint})
    desk.crypto_executor = _executor(rpc=FakeRpc(accounts))

    result = await desk.evaluate_token(token)
    assert result["bought"] is True
    assert result["tx_id"].startswith("PAPER-")
    assert desk.positions[0].meta["mint"] == mint


async def test_desk_skips_when_the_executor_fails(tmp_path):
    from tests.test_desk import TOKEN, build

    desk = build(tmp_path, dry_run=False)

    class Boom:
        async def buy(self, *a, **k):
            raise ExecutionFailed("bundle_rejected", "dropped")

    desk.crypto_executor = Boom()
    result = await desk.evaluate_token(TOKEN)
    assert result["bought"] is False
    assert result["reason"] == "bundle_rejected"
    assert desk.positions == []
