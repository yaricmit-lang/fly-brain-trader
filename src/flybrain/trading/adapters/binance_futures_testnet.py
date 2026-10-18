"""
Binance Futures Testnet demo account.

Why the futures testnet and not the spot one: the spot testnet has no
trading site, only key issuance. The futures testnet has a full web UI, so
the fly's trades are visible next to the usual price chart.

Leverage is hard-set to 1x and there are no shorts, so a futures position
behaves like spot and metrics stay comparable with the paper account.

One subtlety: ccxt's sandbox mode for binanceusdm has historically used
the wrong endpoints, so the base URL is set explicitly.
"""

from __future__ import annotations

import numpy as np

from ...config import Config, env
from .base import Adapter


class BinanceFuturesTestnetAdapter(Adapter):
    name = "binance_futures_testnet"

    def __init__(self, cfg: Config) -> None:
        import ccxt  # noqa: PLC0415

        self.cfg = cfg
        self.coins = [s.split("/")[0] for s in cfg.market.symbols]
        self.symbols = [f"{c}/USDT:USDT" for c in self.coins]

        key, secret = env("BINANCE_TESTNET_KEY"), env("BINANCE_TESTNET_SECRET")
        if not key or not secret:
            raise RuntimeError(
                "Testnet API keys are missing. Register at "
                "https://testnet.binancefuture.com, create an API key "
                "and put it in .env as BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET"
            )

        self.ex = ccxt.binanceusdm({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.ex.set_sandbox_mode(True)
        # safeguard: pin the testnet endpoint explicitly
        base = cfg.trading.executor.testnet_base_url
        for k in ("fapiPublic", "fapiPrivate", "fapiPublicV2", "fapiPrivateV2", "fapiData"):
            if k in self.ex.urls["api"]:
                self.ex.urls["api"][k] = f"{base}/fapi/v1" if k.endswith(("Public", "Private")) else self.ex.urls["api"][k]

        self.ex.load_markets()
        self._set_leverage()

    # ------------------------------------------------------------------
    def _set_leverage(self) -> None:
        """Set 1x leverage on every instrument so it behaves like spot."""
        for sym in self.symbols:
            try:
                self.ex.set_leverage(int(self.cfg.trading.leverage), sym)
            except Exception as e:  # noqa: BLE001
                print(f"  could not set leverage for {sym}: {type(e).__name__}")

    def _balance_usdt(self) -> float:
        bal = self.ex.fetch_balance()
        return float(bal["total"].get("USDT", 0.0))

    def _positions(self) -> dict[str, float]:
        """Signed position size in coins for each instrument."""
        out = {c: 0.0 for c in self.coins}
        for p in self.ex.fetch_positions(self.symbols):
            sym = p.get("symbol", "")
            coin = sym.split("/")[0]
            if coin in out:
                amt = float(p.get("contracts") or 0.0)
                side = (p.get("side") or "long").lower()
                out[coin] = amt if side == "long" else -amt
        return out

    # ------------------------------------------------------------------
    def equity(self, prices: np.ndarray) -> float:
        return self._balance_usdt()

    def weights(self, prices: np.ndarray) -> np.ndarray:
        eq = self.equity(prices)
        if eq <= 0:
            return np.zeros(len(self.coins))
        pos = self._positions()
        return np.array([pos[c] * prices[i] / eq for i, c in enumerate(self.coins)])

    def sync(self, target: np.ndarray, prices: np.ndarray, ts=None) -> int:
        """Bring each position to its target weight with market orders."""
        eq = self.equity(prices)
        current = self.weights(prices)
        n = 0

        for i, coin in enumerate(self.coins):
            delta_value = (target[i] - current[i]) * eq
            if abs(delta_value) < 10:          # below the exchange minimum order size
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
                    f"  [testnet] {side} {amount} {coin} at {fill} "
                    f"(deviation from mainnet price {drift:+.2f}%)"
                )
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [testnet] {side} order for {coin} failed: {type(e).__name__}: {e}")
        return n
