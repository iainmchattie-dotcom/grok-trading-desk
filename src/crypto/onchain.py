"""Pure Solana / pump.fun helpers used by CryptoExecutor.

Nothing here talks to the network. Tests can exercise curve math, mint
decimals, raw-amount conversion and instruction encoding without RPC.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

# -- well-known program IDs -------------------------------------------------------

PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_GLOBAL = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
PUMP_FEE_PROGRAM = "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ"
WSOL_MINT = "So11111111111111111111111111111111111111112"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ASSOCIATED_TOKEN_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
LAMPORTS_PER_SOL = 1_000_000_000
MINT_DECIMALS_OFFSET = 44

# Official Jito tip accounts. Bundles without a tip are ignored.
JITO_TIP_ACCOUNTS = (
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
)

# pump.fun fee recipients from pump-public-docs (FEE_RECIPIENTS.md).
PUMP_FEE_RECIPIENTS = (
    "62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV",
    "7VtfL8fvgNfhz17qKRMjzQEXgbdpnHHHQRh54R9jP2RJ",
    "7hTckgnGnLQR6sdH7YkqFTAA7VwTfYFaZ6EhEsU3saCX",
    "9rPYyANsfQZw3DnDmKE3YCQF5E8oD89UXoHn9JFEhJUz",
    "AVmoTthdrX6tKt4nDjco2D775W2YK3sDhxPcMmzUAmTY",
    "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM",
    "FWsW1xNtWscwNmKv6wVsU1iTzRN6wmmk3MjxRP5tT7hz",
    "G5UZAVbAf46s7cKWoyKu8kYTip9DGTpbLZ2qa9Aq69dP",
)
PUMP_MAYHEM_FEE_RECIPIENTS = (
    "GesfTA3X2arioaHp8bbKdjG9vJtskViWACZoYvxp4twS",
    "4budycTjhs9fD6xw62VBducVTNgMgJJ5BgtKq7mAZwn6",
    "8SBKzEQU4nLSzcwF4a74F2iaUDQyTfjGndn6qUWBnrpR",
    "4UQeTP1T39KZ9Sfxzo3WR5skgsaP6NZa87BAkuazLEKH",
    "8sNeir4QsLsJdYpc9RZacohhK1Y5FLU3nC5LXgYB4aa6",
    "Fh9HmeLNUMVCvejxCtCL2DbYaRyBFVJ5xrWkLnMH6fdk",
    "463MEnMeGyJekNZFQSTUABBEbLnvMTALbT6ZmsxAbAdq",
    "6AUH3WEHucYZyC61hqpqYUWVto5qA5hjHuNQ32GNnNxA",
)
PUMP_BUYBACK_FEE_RECIPIENTS = (
    "5YxQFdt3Tr9zJLvkFccqXVUwhdTWJQc1fFg2YPbxvxeD",
    "9M4giFFMxmFGXtc3feFzRai56WbBqehoSeRE5GK7gf7",
    "GXPFM2caqTtQYC2cJ5yJRi9VDkpsYZXzYdwYpGnLmtDL",
    "3BpXnfJaUTiwXnJNe7Ej1rcbzqTTQUvLShZaWazebsVR",
    "5cjcW9wExnJJiqgLjq7DEG75Pm6JBgE1hNv4B2vHXUW6",
    "EHAAiTxcdDwQ3U4bU6YcMsQGaekdzLS3B5SmYo46kJtL",
    "5eHhjP8JaYkz83CWwvGU2uMUXefd3AazWGx4gpcuEEYD",
    "A7hAgCzFw14fejgCp387JUJRMNyz4j89JKnhtKU8piqW",
)

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58encode(data: bytes) -> str:
    """Bitcoin-alphabet base58. Used for Jito bundle payloads."""
    zeros = 0
    for byte in data:
        if byte == 0:
            zeros += 1
        else:
            break
    number = int.from_bytes(data, "big")
    chars: list[str] = []
    while number > 0:
        number, remainder = divmod(number, 58)
        chars.append(_B58_ALPHABET[remainder])
    return ("1" * zeros) + ("".join(reversed(chars)) if chars else "")


def anchor_discriminator(name: str) -> bytes:
    return hashlib.sha256(f"global:{name}".encode()).digest()[:8]


def u64_le(value: int) -> bytes:
    if value < 0:
        raise ValueError("u64 cannot be negative")
    return int(value).to_bytes(8, "little", signed=False)


def apply_bps(amount: int, bps: int) -> int:
    """`amount * bps / 10_000` rounded down, clamped at zero."""
    if amount <= 0 or bps <= 0:
        return 0
    return amount * bps // 10_000


def tokens_out_for_quote_in(virtual_quote: int, virtual_token: int, quote_in: int) -> int:
    """Constant-product tokens received for `quote_in` (already net of fees)."""
    if virtual_quote <= 0 or virtual_token <= 0 or quote_in <= 0:
        return 0
    new_quote = virtual_quote + quote_in
    if new_quote <= 0:
        return 0
    new_token = (virtual_quote * virtual_token) // new_quote
    return max(0, virtual_token - new_token)


def quote_out_for_tokens_in(virtual_quote: int, virtual_token: int, token_in: int) -> int:
    """Constant-product quote received for `token_in` (before output fees)."""
    if virtual_quote <= 0 or virtual_token <= 0 or token_in <= 0:
        return 0
    new_token = virtual_token + token_in
    if new_token <= 0:
        return 0
    new_quote = (virtual_quote * virtual_token) // new_token
    return max(0, virtual_quote - new_quote)


def mint_decimals(data: bytes) -> int:
    """SPL / Token-2022 mint layout: decimals is a u8 at offset 44."""
    if len(data) <= MINT_DECIMALS_OFFSET:
        raise ValueError("mint account too short to contain decimals")
    return int(data[MINT_DECIMALS_OFFSET])


def raw_amount(ui_amount: float, decimals: int) -> int:
    """Convert a UI token amount to raw u64 using the mint's real decimals."""
    if decimals < 0 or decimals > 12:
        raise ValueError(f"implausible mint decimals: {decimals}")
    if ui_amount <= 0:
        return 0
    return int(ui_amount * (10 ** decimals))


def ui_amount(raw: int, decimals: int) -> float:
    if decimals < 0:
        raise ValueError(f"implausible mint decimals: {decimals}")
    if raw <= 0:
        return 0.0
    return raw / float(10 ** decimals)


def fraction_raw(balance_raw: int, fraction: float) -> int:
    """`fraction` of a raw balance, rounded down. Never assumes 6 or 9."""
    if balance_raw <= 0 or fraction <= 0:
        return 0
    if fraction >= 1.0:
        return int(balance_raw)
    return int(balance_raw * fraction)


@dataclass(frozen=True)
class BondingCurve:
    virtual_token_reserves: int
    virtual_quote_reserves: int
    real_token_reserves: int
    real_quote_reserves: int
    token_total_supply: int
    complete: bool
    creator: str | None = None
    is_mayhem_mode: bool = False

    @property
    def bonded(self) -> bool:
        return bool(self.complete)


def parse_bonding_curve(data: bytes) -> BondingCurve:
    """Borsh layout after the 8-byte Anchor discriminator.

    `complete` is the venue switch: False → bonding curve, True → Raydium/Jupiter.
    """
    if len(data) < 49:
        raise ValueError("bonding curve account too short")
    body = data[8:]
    virtual_token, virtual_quote, real_token, real_quote, supply = struct.unpack_from(
        "<5Q", body, 0
    )
    complete = bool(body[40])
    creator = None
    mayhem = False
    if len(body) >= 73:
        raw_creator = body[41:73]
        if any(raw_creator):
            from solders.pubkey import Pubkey

            creator = str(Pubkey.from_bytes(bytes(raw_creator)))
    if len(body) >= 74:
        mayhem = bool(body[73])
    return BondingCurve(
        virtual_token_reserves=virtual_token,
        virtual_quote_reserves=virtual_quote,
        real_token_reserves=real_token,
        real_quote_reserves=real_quote,
        token_total_supply=supply,
        complete=complete,
        creator=creator,
        is_mayhem_mode=mayhem,
    )


def pack_bonding_curve(
    virtual_token: int,
    virtual_quote: int,
    real_token: int,
    real_quote: int,
    supply: int,
    complete: bool,
    creator: bytes | None = None,
    is_mayhem_mode: bool = False,
) -> bytes:
    """Test helper: encode a bonding-curve account the same way we parse it."""
    body = struct.pack("<5Q", virtual_token, virtual_quote, real_token, real_quote, supply)
    body += bytes([1 if complete else 0])
    body += (creator if creator is not None else bytes(32))
    body += bytes([1 if is_mayhem_mode else 0])
    return bytes(8) + body


def pack_mint(decimals: int, supply: int = 0) -> bytes:
    """Test helper: minimal SPL mint account with `decimals` at offset 44."""
    data = bytearray(82)
    struct.pack_into("<Q", data, 36, supply)
    data[MINT_DECIMALS_OFFSET] = decimals
    return bytes(data)


def looks_like_pubkey(value: str) -> bool:
    if not value or not (32 <= len(value) <= 44):
        return False
    try:
        from solders.pubkey import Pubkey

        Pubkey.from_string(value)
        return True
    except Exception:  # noqa: BLE001 - invalid base58 is a no, not a crash
        return False


def pda(seeds: list[bytes], program: str) -> Any:
    from solders.pubkey import Pubkey

    address, _bump = Pubkey.find_program_address(seeds, Pubkey.from_string(program))
    return address


def bonding_curve_pda(mint: str) -> Any:
    from solders.pubkey import Pubkey

    return pda([b"bonding-curve", bytes(Pubkey.from_string(mint))], PUMP_PROGRAM)


def creator_vault_pda(creator: str) -> Any:
    from solders.pubkey import Pubkey

    return pda([b"creator-vault", bytes(Pubkey.from_string(creator))], PUMP_PROGRAM)


def event_authority_pda() -> Any:
    return pda([b"__event_authority"], PUMP_PROGRAM)


def global_volume_accumulator_pda() -> Any:
    return pda([b"global_volume_accumulator"], PUMP_PROGRAM)


def user_volume_accumulator_pda(user: str) -> Any:
    from solders.pubkey import Pubkey

    return pda([b"user_volume_accumulator", bytes(Pubkey.from_string(user))], PUMP_PROGRAM)


def fee_config_pda() -> Any:
    from solders.pubkey import Pubkey

    return pda([b"fee_config", bytes(Pubkey.from_string(PUMP_PROGRAM))], PUMP_FEE_PROGRAM)


def sharing_config_pda(mint: str) -> Any:
    from solders.pubkey import Pubkey

    return pda([b"sharing-config", bytes(Pubkey.from_string(mint))], PUMP_FEE_PROGRAM)


def associated_token_address(owner: str, mint: str, token_program: str = TOKEN_PROGRAM) -> Any:
    from solders.pubkey import Pubkey
    from solders.token.associated import get_associated_token_address

    return get_associated_token_address(
        Pubkey.from_string(owner),
        Pubkey.from_string(mint),
        Pubkey.from_string(token_program),
    )


def create_ata_idempotent(payer: str, owner: str, mint: str, token_program: str = TOKEN_PROGRAM):
    from solders.instruction import AccountMeta, Instruction
    from solders.pubkey import Pubkey

    ata = associated_token_address(owner, mint, token_program)
    return Instruction(
        Pubkey.from_string(ASSOCIATED_TOKEN_PROGRAM),
        bytes([1]),
        [
            AccountMeta(Pubkey.from_string(payer), True, True),
            AccountMeta(ata, False, True),
            AccountMeta(Pubkey.from_string(owner), False, False),
            AccountMeta(Pubkey.from_string(mint), False, False),
            AccountMeta(Pubkey.from_string(SYSTEM_PROGRAM), False, False),
            AccountMeta(Pubkey.from_string(token_program), False, False),
        ],
    )


# SPL Token instruction tags. SyncNative is how wrapped-SOL balance tracks lamports.
_TOKEN_SYNC_NATIVE = 17
_TOKEN_CLOSE_ACCOUNT = 9


def select_recipient(mint: str, pool: tuple[str, ...] | list[str], salt: str = "") -> str:
    """Stable pick from a recipient pool, keyed by mint (and optional salt).

    Rotates load across the documented pump.fun fee accounts instead of always
    hitting index 0. The same mint always maps to the same recipient so a
    retry does not hop accounts mid-flight. An empty pool is a programming error.
    """
    if not pool:
        raise ValueError("fee recipient pool is empty")
    digest = hashlib.sha256(f"{mint}:{salt}".encode()).digest()
    return pool[int.from_bytes(digest[:8], "little") % len(pool)]


def select_pump_fee_recipients(
    mint: str,
    *,
    is_mayhem_mode: bool = False,
    fee_recipient: str | None = None,
    buyback_fee_recipient: str | None = None,
) -> tuple[str, str]:
    """Fee + buyback recipients: config override, else mint-hash rotation."""
    fee_pool = PUMP_MAYHEM_FEE_RECIPIENTS if is_mayhem_mode else PUMP_FEE_RECIPIENTS
    fee = fee_recipient or select_recipient(mint, fee_pool, salt="fee")
    buyback = buyback_fee_recipient or select_recipient(
        mint, PUMP_BUYBACK_FEE_RECIPIENTS, salt="buyback"
    )
    return fee, buyback


def sync_native(account: str, token_program: str = TOKEN_PROGRAM):
    """SPL `SyncNative` — refresh a WSOL ATA after a lamport transfer."""
    from solders.instruction import AccountMeta, Instruction
    from solders.pubkey import Pubkey

    return Instruction(
        Pubkey.from_string(token_program),
        bytes([_TOKEN_SYNC_NATIVE]),
        [AccountMeta(Pubkey.from_string(str(account)), False, True)],
    )


def close_token_account(
    account: str,
    destination: str,
    owner: str,
    token_program: str = TOKEN_PROGRAM,
):
    """Close a token account, sending leftover lamports to `destination`.

    For the native mint this unwraps remaining WSOL. Regular SPL accounts
    must already be empty.
    """
    from solders.instruction import AccountMeta, Instruction
    from solders.pubkey import Pubkey

    return Instruction(
        Pubkey.from_string(token_program),
        bytes([_TOKEN_CLOSE_ACCOUNT]),
        [
            AccountMeta(Pubkey.from_string(str(account)), False, True),
            AccountMeta(Pubkey.from_string(destination), False, True),
            AccountMeta(Pubkey.from_string(owner), True, False),
        ],
    )


def wrap_sol_instructions(owner: str, lamports: int) -> list:
    """Create the user's WSOL ATA if needed, fund it, then SyncNative.

    buy_v2's `associated_quote_user` is the WSOL ATA. Native SOL sitting in
    the wallet does not fund that account; without this wrap the buy ix
    sees an empty quote ATA and fails or takes nothing.
    """
    from solders.pubkey import Pubkey
    from solders.system_program import TransferParams, transfer

    if lamports <= 0:
        raise ValueError("wrap requires a positive lamport amount")
    wsol_ata = associated_token_address(owner, WSOL_MINT, TOKEN_PROGRAM)
    return [
        create_ata_idempotent(owner, owner, WSOL_MINT, TOKEN_PROGRAM),
        transfer(
            TransferParams(
                from_pubkey=Pubkey.from_string(owner),
                to_pubkey=wsol_ata,
                lamports=int(lamports),
            )
        ),
        sync_native(str(wsol_ata)),
    ]


def unwrap_wsol_instruction(owner: str):
    """Close the user's WSOL ATA, returning leftover SOL to the wallet."""
    wsol_ata = associated_token_address(owner, WSOL_MINT, TOKEN_PROGRAM)
    return close_token_account(str(wsol_ata), owner, owner)


def _meta(address: str, signer: bool = False, writable: bool = False):
    from solders.instruction import AccountMeta
    from solders.pubkey import Pubkey

    return AccountMeta(Pubkey.from_string(str(address)), signer, writable)


def pump_buy_v2_instruction(
    *,
    mint: str,
    user: str,
    amount: int,
    max_quote_cost: int,
    creator: str,
    base_token_program: str = TOKEN_PROGRAM,
    quote_mint: str = WSOL_MINT,
    quote_token_program: str = TOKEN_PROGRAM,
    fee_recipient: str | None = None,
    buyback_fee_recipient: str | None = None,
    is_mayhem_mode: bool = False,
):
    """Build a `buy_v2` instruction (pump-public-docs, 2026 account list)."""
    from solders.instruction import Instruction
    from solders.pubkey import Pubkey

    if amount <= 0 or max_quote_cost <= 0:
        raise ValueError("buy_v2 requires positive amount and max_quote_cost")

    fee_recipient, buyback_fee_recipient = select_pump_fee_recipients(
        mint,
        is_mayhem_mode=is_mayhem_mode,
        fee_recipient=fee_recipient,
        buyback_fee_recipient=buyback_fee_recipient,
    )

    curve = bonding_curve_pda(mint)
    creator_vault = creator_vault_pda(creator)
    accounts = [
        _meta(PUMP_GLOBAL),
        _meta(mint),
        _meta(quote_mint),
        _meta(base_token_program),
        _meta(quote_token_program),
        _meta(ASSOCIATED_TOKEN_PROGRAM),
        _meta(fee_recipient, writable=True),
        _meta(associated_token_address(fee_recipient, quote_mint, quote_token_program), writable=True),
        _meta(buyback_fee_recipient, writable=True),
        _meta(
            associated_token_address(buyback_fee_recipient, quote_mint, quote_token_program),
            writable=True,
        ),
        _meta(str(curve), writable=True),
        _meta(associated_token_address(str(curve), mint, base_token_program), writable=True),
        _meta(associated_token_address(str(curve), quote_mint, quote_token_program), writable=True),
        _meta(user, signer=True, writable=True),
        _meta(associated_token_address(user, mint, base_token_program), writable=True),
        _meta(associated_token_address(user, quote_mint, quote_token_program), writable=True),
        _meta(str(creator_vault), writable=True),
        _meta(associated_token_address(str(creator_vault), quote_mint, quote_token_program), writable=True),
        _meta(str(sharing_config_pda(mint))),
        _meta(str(global_volume_accumulator_pda())),
        _meta(str(user_volume_accumulator_pda(user)), writable=True),
        _meta(
            associated_token_address(str(user_volume_accumulator_pda(user)), quote_mint, quote_token_program),
            writable=True,
        ),
        _meta(str(fee_config_pda())),
        _meta(PUMP_FEE_PROGRAM),
        _meta(SYSTEM_PROGRAM),
        _meta(str(event_authority_pda())),
        _meta(PUMP_PROGRAM),
    ]
    data = anchor_discriminator("buy_v2") + u64_le(amount) + u64_le(max_quote_cost)
    return Instruction(Pubkey.from_string(PUMP_PROGRAM), data, accounts)


def pump_sell_v2_instruction(
    *,
    mint: str,
    user: str,
    amount: int,
    min_quote_out: int,
    creator: str,
    base_token_program: str = TOKEN_PROGRAM,
    quote_mint: str = WSOL_MINT,
    quote_token_program: str = TOKEN_PROGRAM,
    fee_recipient: str | None = None,
    buyback_fee_recipient: str | None = None,
    is_mayhem_mode: bool = False,
):
    """Build a `sell_v2` instruction. `amount` is raw token units (mint decimals)."""
    from solders.instruction import Instruction
    from solders.pubkey import Pubkey

    if amount <= 0:
        raise ValueError("sell_v2 requires a positive raw token amount")

    fee_recipient, buyback_fee_recipient = select_pump_fee_recipients(
        mint,
        is_mayhem_mode=is_mayhem_mode,
        fee_recipient=fee_recipient,
        buyback_fee_recipient=buyback_fee_recipient,
    )

    curve = bonding_curve_pda(mint)
    creator_vault = creator_vault_pda(creator)
    accounts = [
        _meta(PUMP_GLOBAL),
        _meta(mint),
        _meta(quote_mint),
        _meta(base_token_program),
        _meta(quote_token_program),
        _meta(ASSOCIATED_TOKEN_PROGRAM),
        _meta(fee_recipient, writable=True),
        _meta(associated_token_address(fee_recipient, quote_mint, quote_token_program), writable=True),
        _meta(buyback_fee_recipient, writable=True),
        _meta(
            associated_token_address(buyback_fee_recipient, quote_mint, quote_token_program),
            writable=True,
        ),
        _meta(str(curve), writable=True),
        _meta(associated_token_address(str(curve), mint, base_token_program), writable=True),
        _meta(associated_token_address(str(curve), quote_mint, quote_token_program), writable=True),
        _meta(user, signer=True, writable=True),
        _meta(associated_token_address(user, mint, base_token_program), writable=True),
        _meta(associated_token_address(user, quote_mint, quote_token_program), writable=True),
        _meta(str(creator_vault), writable=True),
        _meta(associated_token_address(str(creator_vault), quote_mint, quote_token_program), writable=True),
        _meta(str(sharing_config_pda(mint))),
        _meta(str(user_volume_accumulator_pda(user)), writable=True),
        _meta(
            associated_token_address(str(user_volume_accumulator_pda(user)), quote_mint, quote_token_program),
            writable=True,
        ),
        _meta(str(fee_config_pda())),
        _meta(PUMP_FEE_PROGRAM),
        _meta(SYSTEM_PROGRAM),
        _meta(str(event_authority_pda())),
        _meta(PUMP_PROGRAM),
    ]
    data = anchor_discriminator("sell_v2") + u64_le(amount) + u64_le(max(0, min_quote_out))
    return Instruction(Pubkey.from_string(PUMP_PROGRAM), data, accounts)


def jupiter_instruction(raw: dict[str, Any]):
    """Convert a Jupiter `/swap-instructions` instruction object to solders."""
    from solders.instruction import AccountMeta, Instruction
    from solders.pubkey import Pubkey
    import base64

    program_id = raw.get("programId") or raw.get("program_id")
    if not program_id:
        raise ValueError("jupiter instruction missing programId")
    accounts = []
    for item in raw.get("accounts") or []:
        pubkey = item.get("pubkey") or item.get("publicKey")
        accounts.append(
            AccountMeta(
                Pubkey.from_string(str(pubkey)),
                bool(item.get("isSigner", item.get("is_signer", False))),
                bool(item.get("isWritable", item.get("is_writable", False))),
            )
        )
    data = raw.get("data") or ""
    decoded = base64.b64decode(data) if data else b""
    return Instruction(Pubkey.from_string(str(program_id)), decoded, accounts)
