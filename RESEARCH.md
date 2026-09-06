# Research notes

What the upstream docs and prior art actually say, and what each finding changed
in this repo. Dated 2026-08-30.

## xAI API

Source of truth: `https://docs.x.ai/openapi.json` (the rendered docs lag it).

### Model slugs retired 2026-05-15

`grok-4-fast-reasoning`, `grok-4-fast-non-reasoning`, `grok-4-0709`, `grok-3`,
`grok-code-fast-1` and three others were retired. Requests to a retired slug are
**auto-redirected and billed at `grok-4.3` rates** ($1.25/1M in, $2.50/1M out).

This mattered more than a rename. The desk asked for `grok-4-fast` for the ten
generators and `grok-4` for the two adversarial checkers, on the theory that
check quality is safety. Post-retirement **both slugs land on the same model**,
so the checkers had silently stopped being stronger than what they were checking.

Current lineup and pricing (per 1M tokens, <200k context tier):

| model | context | in | cached in | out |
|---|---|---|---|---|
| grok-4.6 | 500k | $2.00 | $0.50 | $6.00 |
| grok-4.5 | 500k | $2.00 | $0.30 | $6.00 |
| grok-4.3 | 1M | $1.25 | $0.20 | $2.50 |

Now: generators run `grok-4.3` at `reasoning_effort: none`, checkers run
`grok-4.6`. `reasoning_effort` is **only supported by `grok-4.3`** (values
`none`/`low`/`medium`/`high`, default `low`) — sending it to 4.6 is an error, so
`base_agent` only attaches it when the slug matches.

### Live Search retired — HTTP 410 Gone (2026-09)

`ChatRequest.search_parameters` on `/v1/chat/completions` is gone. xAI returns
HTTP 410 with a body pointing at the Agent Tools API
(`https://docs.x.ai/docs/guides/tools/overview`). Confirmed on a Mac dry-run:
`crypto_pulse` and `market_pulse` 410 immediately (not retryable), while
`allocator` — same base URL and key, no `search_parameters` — returns 200.

Replacement: `tools: [{type: web_search}, {type: x_search}]` on
`/v1/responses`. `news` has no dedicated tool and is folded into `web_search`.

What the current Agent Tools docs / xAI Python SDK actually accept (verified
2026-09-06 against `docs.x.ai/developers/tools/web-search`,
`docs.x.ai/developers/tools/x-search`, and `xai_sdk.tools`):

| want | on the wire? | why |
|---|---|---|
| `from_date` / `to_date` | **`x_search` only** | documented x_search params; `web_search` has none |
| `max_search_results` | **no** | neither tool has a result-count / max-results field (`collections_search` has `limit`; web/x do not) |
| `post_view_count` | **no** | not an `x_search` param; restored as a system-prompt engagement floor |
| domain / handle lists | yes | `web_search.filters.allowed_domains`, `x_search.allowed_x_handles` |

So date windows are X-only; web/news are uncapped by date. The result cap stays
in the in-process `SEARCH` dict and is not sent (unknown tool fields 400).
Engagement floors stay in `SEARCH` as `post_view_count` and are copied into the
system prompt so the model still prefers higher-view posts.

Agents that need no retrieval stay on chat/completions.

Degrade path: `grok.live_search: false` omits tools, stays on chat/completions,
and the model answers from the training cutoff. Pulse fallbacks remain
pessimistic (gate shut).

The older Live Search notes below are kept as the policy shape agents still
declare in-process; `base_agent` maps that dict onto tools and never puts
`search_parameters` on the wire.

### Live search — the agents had no data

`ChatRequest.search_parameters`, verbatim from the (pre-retirement) spec:

> Set the parameters to be used for searched data. **If not set, no data will be
> acquired by the model.**

Grok 4.6's knowledge cutoff is 2026-02-01. So radar ("scan the last two weeks of
news"), insider ("recent Form 4 filings"), and both pulse bots were answering
from a six-month-old prior with no retrieval. Confidently. That is the single
largest correctness defect research turned up.

`search_parameters` shape (retired; mapped to Agent Tools):

- `mode`: `off` | `on` | `auto` (default `auto`)
- `sources`: list of `{type: web|news|x|rss, ...}` — defaults to web+X if omitted
- `from_date` / `to_date`: ISO `YYYY-MM-DD`
- `max_search_results`: 1..30, default 15
- `return_citations`: default true

Per-source options that matter here:

- `web` / `news`: `allowed_websites` (max 5, whitelist), `excluded_websites`
  (max 5, mutually exclusive with the whitelist), `country`, `safe_search`
- `x`: `included_x_handles` / `excluded_x_handles` (max 20, mutually exclusive),
  `post_favorite_count`, `post_view_count` — the last two are the useful ones for
  filtering memecoin noise
- `rss`: `links`

Each agent now declares its own `SEARCH` policy: insider whitelists `sec.gov`,
radar reads news + X with an engagement floor, narrative reads X only, the pulse
bots read a short recent window. Citations come back in `response.citations` and
are logged with the decision.

### Structured outputs

`response_format: {"type": "json_schema", "json_schema": {"name", "schema",
"strict": true}}` is supported on `/v1/chat/completions` **only**. A live
dry-run (2026-09-06) showed `/v1/responses` rejecting it with HTTP 400:

> `'response_format' is not supported on /v1/responses — use 'text.format'`

The Responses shape, from the official structured-outputs guide (OpenAI SDK
example with `web_search` tools), is flattened:

```
text: { format: { type: "json_schema", name, schema, strict: true } }
```

`json_object` becomes `text: { format: { type: "json_object" } }`. Retrieval
agents use that; the allocator (no tools) keeps `response_format`.

Notes from the docs:

- `additionalProperties` must be explicitly `false`
- Draft 2020-12 preferred; `minLength`/`maxLength` enforced up to 2048,
  `minItems`/`maxItems` up to 256
- no circular refs; regex is an ECMAScript subset

Every agent now ships a strict schema. The fence-stripping/prose-scraping parser
stays as a fallback for the `json_object` path but is no longer load-bearing.

### Endpoint choice

`/v1/responses` is the recommended API and `/v1/chat/completions` is labelled
legacy. After Live Search's retirement, retrieval agents have to use Responses
anyway: built-in `web_search` / `x_search` tools are documented there, and
`search_parameters` on chat/completions is 410. No-retrieval agents (allocator)
stay on chat/completions. `base_agent` derives `/v1/responses` from
`grok.base_url` (override with `grok.responses_url`).

### Cost and cache accounting

Chat/completions `usage` carries `cost_in_usd_ticks` (exact;
`TICKS_IN_USD_CENT = 100_000_000`, so USD = ticks / 1e10), `num_sources_used`,
`prompt_tokens` / `completion_tokens`, `prompt_tokens_details.cached_tokens` and
`completion_tokens_details.reasoning_tokens`. Responses uses `input_tokens` /
`output_tokens` (and often `server_side_tool_usage` instead of
`num_sources_used`). `normalize_usage` folds both shapes before
`CostTracker.record` so retrieval agents on `/v1/responses` are not
undercounted.

`prompt_cache_key` gives sticky routing for cache hits. Prompts are now built
static-prefix-first so the constant PROMPT block is cacheable, and every agent
sends a stable cache key.

## PumpPortal

`wss://pumpportal.fun/api/data`. Rules that bit us:

- **One connection, many subscriptions.** The docs warn that opening a socket per
  subscription risks an hourly ban. The reconnect path now uses exponential
  backoff with jitter and a cap instead of a flat 5s retry.
- `subscribeNewToken` is free; `subscribeTokenTrade` is metered at 0.01 SOL per
  10k messages.

A `subscribeNewToken` event contains exactly: `signature`, `mint`,
`traderPublicKey`, `txType`, `name`, `symbol`, `uri`, `initialBuy`, `solAmount`,
`bondingCurveKey`, `vTokensInBondingCurve`, `vSolInBondingCurve`, `marketCapSol`,
`pool`.

It does **not** contain holders, top-10 concentration, dev holding, buy/sell
counts or age — the scout filtered on all six. At creation, age is 0 and holders
is 1 by construction, so those thresholds could never be satisfied by a create
event. The filter was unreachable in production and only ever passed in tests,
where the fixtures supplied fields the real feed never sends.

Fix: the scout became two-stage. Stage one accepts a create event on the few
facts it really carries (curve liquidity, market cap, dev's share of the initial
buy, metadata quality). Stage two subscribes to that mint's trades and
accumulates real buy/sell counts, unique traders and price action over an
observation window; only a token that survives the window is scored. `marketCapSol`
and `vSolInBondingCurve` are denominated in SOL, so a SOL/USD rate is required to
compare against USD thresholds.

## Alpaca

The movers endpoint returns `Mover{symbol, percent_change, change, price}` and
most-actives returns `ActiveStock{symbol, volume, trade_count}`. **Neither carries
sector, market cap, or average volume** — three of the screener's filters. The
old `_fetch_alpaca` also reconstructed `prev_close` arithmetically from
`percent_change`, which is lossy.

Fix: the screener now composes most-actives + movers for the candidate set, the
**snapshot** endpoint for real `previous_daily_bar.close` and same-day volume, and
20 daily bars for a true `avg_volume`. Market cap and sector have no Alpaca
source, so their filters are skipped when the datum is absent instead of
rejecting every candidate.

Execution constraints confirmed:

- bracket orders **cannot** be fractional and do not support extended hours;
  `time_in_force` must be DAY or GTC. Integer-share sizing was already right.
- HTTP **403** on submit usually means the order was blocked to avoid flagging
  Pattern Day Trader on an account under $25k equity. Now caught and logged as a
  distinct skip reason rather than a generic failure.

## Prior art

- **TradingAgents** (arXiv 2412.20138, TauricResearch) — analyst team, bull/bear
  researcher debate, trader, then a risk team that can override. Explicitly
  "combines structured outputs for control, clarity, and reasoning with natural
  language dialogue" — the same split adopted here.
- **FinMem** (2311.13743) and **TradingGPT** (2309.03736) — layered memory over
  past trades.
- **FinCon** (2407.06567) — "conceptual verbal reinforcement": outcomes of past
  decisions are fed back as text.

The common thread across all four is the one thing this desk lacked: realized
outcomes never reached the agents. Every decision was made from a cold start,
and the JSONL log — which already holds every past entry, its full agent
breakdown, and its eventual PnL — was written and never read back. `shared/memory.py`
now retrieves comparable past trades and injects their outcomes into the prompt.

A bull/bear debate stage before each checker is the other borrowed mechanism; it
is off by default (`debate.enabled`) because it doubles generator calls.

## Sources

- https://docs.x.ai/openapi.json
- https://docs.x.ai/developers/migration/may-15-retirement
- https://docs.x.ai/docs/guides/structured-outputs
- https://docs.x.ai/docs/guides/tools/overview
- https://docs.x.ai/developers/tools/overview
- https://docs.x.ai/developers/tools/x-search
- https://docs.x.ai/developers/models
- https://pumpportal.fun/data-api/real-time/
- https://alpaca.markets/sdks/python/api_reference/data/models.html
- https://alpaca.markets/sdks/python/api_reference/data/stock/screener.html
- https://docs.alpaca.markets/us/docs/fractional-trading
- https://alpaca.markets/learn/how-to-fix-common-trading-api-errors-at-alpaca
- https://arxiv.org/abs/2412.20138
