"""
Bybit demo account.

Replaces Binance Futures Testnet: in September 2026 Binance merged its old
testnet into the main account, so testnet.binancefuture.com now redirects
to the regular accounts.binance.com and requires KYC on a live account.
On Bybit, demo mode is enabled from a regular account WITHOUT verification:
the Demo Trading tab in the profile creates a separate demo API key and the
balance (50,000 USDT, 1 BTC, 1 ETH, etc.) is credited immediately.

Leverage is hard-set to 1x and there are no shorts, so a futures position
behaves like spot and metrics stay comparable with the paper account.
"""

from __future__ import annotations

import numpy as np

from ...config import Config, env
from .base import Adapter


class BybitDemoAdapter(Adapter):
    name = "bybit_demo"

    def __init__(self, cfg: Config) -> None:
        import ccxt  # noqa: PLC0415

        self.cfg = cfg
        self.coins = [s.split("/")[0] for s in cfg.market.symbols]
        self.symbols = [f"{c}/USDT:USDT" for c in self.coins]

        key, secret = env("BYBIT_DEMO_KEY"), env("BYBIT_DEMO_SECRET")
        if not key or not secret:
            raise RuntimeError(
                "Bybit demo API keys are missing. Log in to bybit.com with a "
                "regular account (no verification needed for demo), open the "
                "Demo Trading tab in your profile, create an API key there and "
                "put it in .env as BYBIT_DEMO_KEY / BYBIT_DEMO_SECRET"
            )

        self.ex = ccxt.bybit({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        # official ccxt way to enable Bybit demo mode (not sandbox: Bybit's
        # sandbox does not support demo trading, it is a separate flag)
        self.ex.enable_demo_trading(True)
        self.ex.load_markets()
        self._set_leverage()

    def _set_leverage(self) -> None:
        for sym in self.symbols:
            try:
                self.ex.set_leverage(int(self.cfg.trading.leverage), sym)
            except Exception as e:  # noqa: BLE001
                print(f"  could not set leverage for {sym}: {type(e).__name__}")

    def _balance_usdt(self) -> float:
        bal = self.ex.fetch_balance()
        return float(bal["total"].get("USDT", 0.0))

    def _positions(self) -> dict[str, float]:
        out = {c: 0.0 for c in self.coins}
        for p in self.ex.fetch_positions(self.symbols):
            sym = p.get("symbol", "")
            coin = sym.split("/")[0]
            if coin in out:
                amt = float(p.get("contracts") or 0.0)
                side = (p.get("side") or "long").lower()
                out[coin] = amt if side == "long" else -amt
        return out

    def equity(self, prices: np.ndarray) -> float:
        return self._balance_usdt()

    def weights(self, prices: np.ndarray) -> np.ndarray:
        eq = self.equity(prices)
        if eq <= 0:
            return np.zeros(len(self.coins))
        pos = self._positions()
        return np.array([pos[c] * prices[i] / eq for i, c in enumerate(self.coins)])

    def sync(self, target: np.ndarray, prices: np.ndarray, ts=None) -> int:
        eq = self.equity(prices)
        current = self.weights(prices)
        n = 0

        for i, coin in enumerate(self.coins):
            delta_value = (target[i] - current[i]) * eq
            if abs(delta_value) < 10:
                continue
            amount = abs(delta_value) / prices[i]
            side = "buy" if delta_value > 0 else "sell"
            sym = self.symbols[i]
            try:
                amount = float(self.ex.amount_to_precision(sym, amount))
                if amount <= 0:
                    continue
                order = self.ex.create_order(sym, "market", side, amount)
                fill = order.get("average") or order.get("price") or prices[i]
                drift = (float(fill) / prices[i] - 1) * 100 if fill else 0.0
                print(
                    f"  [bybit demo] {side} {amount} {coin} at {fill} "
                    f"(deviation from mainnet price {drift:+.2f}%)"
                )
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [bybit demo] {side} order for {coin} failed: {type(e).__name__}: {e}")
        return n
