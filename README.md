# grok-trading-desk

A two-market trading system: twelve bots covering Solana memecoins (pump.fun) and
US equities (Alpaca), orchestrated by Grok. Two of the twelve are pure code; the
other ten are LLM agents with a strict JSON contract and a pessimistic fallback.

The design principle throughout: **a model that fails is a model that says no.**
An unparseable audit vetoes the buy. An unreachable checker rejects. A broken exit
manager holds. A dead allocator returns 50/50. Nothing about a failure looks like
permission.

Every agent answers under a strict JSON schema, reads live web/X/news data
through xAI's `search_parameters`, and reports what it cost. Decisions that
close are fed back into later prompts. See [RESEARCH.md](RESEARCH.md) for what
the upstream docs actually say and why several defaults here are what they are.

---

## Architecture

```
                          ┌─────────────────────────────┐
                          │          desk.py            │
                          │   4 concurrent asyncio loops│
                          └──────────────┬──────────────┘
        ┌────────────────────┬───────────┴───────┬────────────────────┐
        │                    │                   │                    │
 ╔══════▼══════╗      ╔══════▼══════╗     ╔══════▼══════╗     ╔═══════▼══════╗
 ║ crypto_loop ║      ║ stock_loop  ║     ║  exit_loop  ║     ║allocator_loop║
 ║  continuous ║      ║ 1/min, RTH  ║     ║   every 4h  ║     ║   every 24h  ║
 ╚══════╤══════╝      ╚══════╤══════╝     ╚══════╤══════╝     ╚═══════╤══════╝
        │                    │                   │                    │
   ┌────▼─────┐         ┌────▼─────┐             │               ┌────▼─────┐
   │ 1 scout  │         │4 screener│             │               │10 alloc. │
   │  (code)  │         │  (code)  │             │               │  (fast)  │
   └────┬─────┘         └────┬─────┘             │               └────┬─────┘
        │                    │                   │                    │
   ┌────▼─────┬────────┐ ┌───▼──────┬─────────┬──────────┐            │
   │2 auditor │3 narr. │ │5 analyst │6 radar  │7 insider │            │
   │  (fast)  │ (fast) │ │  (fast)  │ (fast)  │  (fast)  │            │
   └────┬─────┴────┬───┘ └────┬─────┴────┬────┴────┬─────┘            │
        │          │          │          │         │                  │
   ┌────▼──────────▼───┐ ┌────▼──────────▼─────────▼───┐              │
   │ 8 crypto_pulse    │ │ 9 market_pulse              │              │
   │ (fast, 15m cache) │ │ (fast, 30m cache)           │              │
   └────┬──────────────┘ └────┬────────────────────────┘              │
        │                     │                                       │
   ┌────▼──────────────┐ ┌────▼────────────────────────┐              │
   │ crypto_scoring    │ │ stock_scoring               │              │
   │ hard vetoes, code │ │ hard vetoes, code           │              │
   └────┬──────────────┘ └────┬────────────────────────┘              │
        │                     │                                       │
   ┌────▼──────────────┐ ┌────▼────────────────────────┐              │
   │11 crypto_checker  │ │12 stock_checker             │              │
   │ (grok-4, adversar)│ │ (grok-4, adversarial)       │              │
   └────┬──────────────┘ └────┬────────────────────────┘              │
        │                     │                                       │
   ┌────▼──────────────┐ ┌────▼────────────────────────┐  ┌───────────▼────────┐
   │ crypto_executor   │ │ stock_executor              │  │  13 exit_manager   │
   │ STUB (you wire it)│ │ Alpaca bracket orders       │  │  (fast) HOLD/      │
   └───────────────────┘ └─────────────────────────────┘  │  TIGHTEN/TRIM/CLOSE│
                                                          └────────────────────┘
        └──────────── shared/risk.py — one portfolio, both markets ──────────────┘
        └──────────── shared/log.py — append-only JSONL, everything ─────────────┘
```

---

## The bots

**1 · scout** (`crypto/scout.py`, code) — Subscribes to the pump.fun WebSocket and
runs in two stages, because a `subscribeNewToken` event does not contain the facts
worth filtering on: it carries curve reserves, market cap in SOL, the deployer's
opening buy and a metadata URI, and nothing else. There are no holders, no
buy/sell counts and no age — at creation, age is zero and the holder count is one.
So stage one screens the create event on what it really has, and stage two
subscribes to that mint's trades and accumulates genuine buy/sell counts, unique
traders and price action across an observation window. Only a token that survives
the window costs a model call. Trade subscriptions are metered, so the watchlist
is capacity-bounded, and reconnects back off with jitter because PumpPortal bans
clients that hammer it.

**2 · auditor** (`crypto/auditor.py`) — Audits the wallet graph behind
a launch: coordinated buy rings funded from a common source, wash trading cycling
the same capital, bundled supply sniped by the deployer, sniper and insider share.
Two of its outputs are hard vetoes, so its failure mode returns `true` for both.

**3 · narrative** (`crypto/narrative.py`) — Rates meme potential: is
the reference current, is the ticker memorable, does the branding read as effort,
is there a community already, is this the original or the fourth copy of a running
meme. Derivative names are discounted 30% in the score.

**4 · screener** (`stocks/screener.py`, code) — Once per trading day. No single
Alpaca endpoint has what the filter needs, so it composes three: movers and
most-actives for the candidate set, snapshots for a real previous close and
same-day volume, and 20 daily bars for a true average volume. Filters on price
band, average volume, relative volume and gap size (absolute — a gap down is as
tradeable as a gap up). Market cap and sector have **no** Alpaca source, so those
thresholds are skipped when the datum is missing rather than rejecting the
universe. Survivors are ranked by relative volume.

**5 · analyst** (`stocks/analyst.py`) — Both halves of the equity
picture in one call: fundamentals (growth, margins, balance sheet, valuation
against its own history) and technicals (trend across timeframes, moving averages,
volume confirmation, support and resistance, whether today's move is extended).

**6 · radar** (`stocks/radar.py`) — Two weeks of news and sentiment:
earnings, analyst actions, product news, regulatory exposure, short reports,
executive departures, retail and professional tone. Its `controversy` output above
0.7 is a hard veto, so the fallback sets it to 1.0.

**7 · insider** (`stocks/insider.py`) — Form 4 and 13F flow.
Open-market officer purchases count heavily; scheduled 10b5-1 sales, option
exercises and tax withholding are discounted. Cluster buying — several officers at
once — is the strongest single signal it can report and earns a bonus in scoring.

**8 · crypto_pulse** (`crypto/crypto_pulse.py`, 15-min cache) — The
memecoin regime: SOL trend, launch volume and survival rate, whether fresh capital
is rotating in. Emits `go_signal`; below 0.3 the entire crypto side pauses. Cached
because the regime moves far slower than the launch feed.

**9 · market_pulse** (`stocks/market_pulse.py`, 30-min cache) — The
equity regime: index trend and breadth, VIX, rates and dollar, sector rotation,
the 48-hour macro calendar. Same `go_signal` contract, same pause threshold.

**10 · allocator** (`shared/allocator.py`) — Once a day, splits the
budget between the two markets given both pulses and the trailing week's realised
PnL per market. Its answer is clamped by `crypto_max_pct` / `stock_max_pct` in
code, so a runaway model cannot put the whole book on one side. Failure returns
50/50 — no tilt.

**11 · crypto_checker** (`crypto/crypto_checker.py`, **deep model**) — The adversarial
gate before money moves. Told explicitly to argue the other side and find the way
this loses: the rug the audit missed, concentration that dumps on the buy, an
exhausted narrative, liquidity too thin to exit. Approves only when it cannot
construct a plausible loss. Runs the stronger model, because check quality *is*
the safety — and it runs its own searches rather than trusting the generators'
summary of the evidence.

**12 · stock_checker** (`stocks/stock_checker.py`, **deep model**) — Same job on the
equity side: the move is already exhausted, the catalyst is priced in, the gap
fills by lunch, an event lands inside the holding window. It also reviews the
proposed stop and target — a stop too tight to survive normal noise is a reason to
reject the entry, not just to widen it.

**13 · exit_manager** (`shared/exit_manager.py`) — Every 4 hours over
every open position on both books. Four verbs: HOLD, TIGHTEN (raise the stop),
TRIM (sell a fraction, let the rest run), CLOSE. Failure returns HOLD, always: a
model that cannot answer must never be the reason a position gets touched.

*(Thirteen bots for twelve slots — the exit manager works both markets, so it's
counted once in the desk's twelve-bot roster and listed separately here.)*

---

## Models, live data and cost

`grok-4-fast` and `grok-4` were **retired on 2026-05-15**. Requests to them still
work — they auto-redirect to `grok-4.3` and bill at its rates — which is the
quiet part: both tiers landed on the same model, so "the checker runs a stronger
model" had stopped being true. Current defaults:

| tier | model | who |
|---|---|---|
| `fast` | `grok-4.3`, `reasoning_effort: none` | the ten generators |
| `deep` | `grok-4.6` | both adversarial checkers |

`reasoning_effort` is a **grok-4.3-only** parameter, so it is attached by model
slug rather than sent blindly.

**Live search is not optional here.** The spec is explicit that without
`search_parameters`, "no data will be acquired by the model" — and Grok 4.6's
knowledge cutoff is 2026-02-01. An agent asked for "the last two weeks of news"
with no retrieval will answer anyway, from a months-old prior. Each agent
declares its own policy:

| bot | sources | window |
|---|---|---|
| insider | `sec.gov`, `secform4.com`, `openinsider.com` | 95 days |
| analyst | SEC + financial press, news | — |
| radar | news, X (≥1k views), web | 14 days |
| narrative | X only (≥1k views) | — |
| crypto_pulse | X (≥2k views), news, web | 1 day |
| market_pulse | news, web, X (≥5k views) | 2 days |
| allocator | none — both pulses are already in its payload | — |

Citations come back with every answer and are stored on the buy record.

Spend is read from `usage.cost_in_usd_ticks`, which is exact, rather than
estimated from a price table that goes stale. `replay.py` reports PnL **net of
inference**; the dashboard shows both. Static prompt blocks are sent first and
keyed with `prompt_cache_key` so the cacheable prefix actually caches — the
`cache_hit_rate` in the cost record tells you whether it is working.

---

## Quick start

```bash
git clone https://github.com/zostaff/grok-trading-desk.git
cd grok-trading-desk

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
$EDITOR config.yaml          # xAI key, Alpaca keys, risk limits

# decide and log everything, execute nothing
python -m src.desk --config config.yaml --dry-run

# what happened
python scripts/dashboard.py --log logs/desk.jsonl
python scripts/replay.py    --log logs/desk.jsonl --days 7
```

Run the tests — they never touch the network:

```bash
pytest -v
```

### Going live

Paper trading is the default and stays the default unless **both** conditions hold:

```yaml
mode: "live"      # in config.yaml
```

```bash
python -m src.desk --config config.yaml --i-understand-the-risk
```

Either one alone leaves you on paper, and the desk logs a warning saying so.

Crypto execution raises `NotImplementedError` by design — `crypto_executor.py`
carries detailed notes on what to implement (Jito bundle path, bonding-curve vs.
Raydium routing, decimals, confirmation deadlines), but the code that signs
transactions with your keys is yours to write.

---

## Configuration

`config.yaml` is gitignored; only `config.example.yaml` is in the repo.

| Section | What it controls |
|---|---|
| `mode` | `paper` or `live` (live also needs the CLI flag) |
| `grok` | API key, `models.fast` / `models.deep`, `reasoning_effort`, `structured_outputs`, `live_search`, timeout, retries |
| `solana` | RPC, wallet key, Jito settings, priority fee, slippage |
| `alpaca` | Key, secret, paper flag |
| `risk` | Total budget, daily loss limit, open-position caps, sector cap, per-market ceilings, position sizing |
| `pump_fun` | WebSocket URL, SOL price fallback, watch-window settings |
| `crypto_launch_filter` | Stage-one thresholds (what a create event carries) |
| `crypto_filter` | Stage-two thresholds (what the watch window measures) |
| `stock_filter` | Screener thresholds |
| `scoring_weights` | Both matrices, including `min_score_to_buy` |
| `pulse` | Cache windows, `min_go_signal` |
| `exits` | Loop interval, default stop/target, trim fraction |
| `market_hours` | Timezone and the RTH window |
| `debate` | Bull/bear stage before the checkers (off by default) |
| `memory` | Outcome recall: lookback window and example count |
| `logging` | JSONL path, stdout echo, cost-report interval |

### Risk, in one place

`shared/risk.py` governs both markets from a single set of counters. A memecoin
loss consumes the same daily allowance an equity loss does, and the open-position
cap counts both books. A position is sized by the tightest of three bounds: 15% of
that market's budget, 25% of what is left of today's loss allowance, and whatever
budget is actually free — then scaled between half and full size by the checker's
confidence.

### Outcome memory

`shared/memory.py` joins every `buy` record to its `close` and injects comparable
past trades into the two checkers, the exit manager and the allocator — matched
on market, symbol, sector and the theme the narrative bot assigned, ranked by how
specifically they match. It is the mechanism common to FinMem, TradingGPT, FinCon
and TradingAgents, and it is the one thing the desk was missing: every ingredient
was already in the log, and nothing ever read it back.

The analysts deliberately get no memory. They describe what is in front of them;
old trades would only anchor them. On a cold desk the block is empty rather than
reporting a misleading 0% win rate.

### Hard vetoes live in code, not in prompts

A prompt can be argued with. These cannot:

- **Crypto** — `coordinated_buys` or `wash_trading` → skip, before any scoring.
- **Stocks** — `controversy > 0.7` → skip. `insider_selling > 0.8` **and**
  `insider_buying < 0.2` → skip.
- **Both** — `pulse.go_signal < 0.3` → that entire market pauses.

---

## Logging

One JSONL line per event, append-only, never rewritten. Five record types:

| type | fields |
|---|---|
| `buy` | market, symbol, score, all_agent_scores, amount, tx_id |
| `skip` | market, symbol, reason, detail |
| `close` | market, symbol, pnl, hold_time |
| `action` | symbol, action (HOLD/TIGHTEN/TRIM/CLOSE), reason |
| `allocation` | crypto_pct, stocks_pct, reason |
| `cost` | cumulative calls, spend, cache hit rate, sources, per-agent |

Broker refusals get their own skip reasons rather than a generic failure:
`pdt_blocked` (a 403 is almost always Alpaca protecting an under-$25k account
from a Pattern Day Trader flag), `wash_trade_blocked`,
`insufficient_buying_power`, `asset_not_tradable`, `broker_rate_limited`.

`all_agent_scores` carries every sub-bot's full output for the trade plus the
citations behind it, which is what makes a bad entry diagnosable after the fact:
you can see which bot was wrong and what it was reading.

---

## Disclaimer

This is experimental software that decides how to spend money, driven by language
models that are wrong on a regular basis. Memecoin trading in particular loses most
participants most of their capital, and no amount of adversarial checking changes
that base rate.

Nothing here is financial advice. It ships paper-first for a reason: run it on
paper long enough to see how it actually behaves before you consider anything else.
Understand every line before you point it at real money — especially the executor
you have to write yourself. You are responsible for your own losses, and for
whatever your jurisdiction has to say about automated trading.
