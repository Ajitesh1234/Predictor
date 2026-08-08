import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

PAPER_BASE_URL_FRAGMENT = "paper-api.alpaca.markets"


@dataclass(frozen=True)
class AlpacaAccountSnapshot:
    account_id: str
    equity: float
    buying_power: float
    trading_blocked: bool
    account_blocked: bool


def require_paper_alpaca_config() -> dict[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    base_url = os.environ.get("APCA_API_BASE_URL")

    missing = [
        name for name, value in {
            "ALPACA_API_KEY": api_key,
            "ALPACA_SECRET_KEY": secret_key,
            "APCA_API_BASE_URL": base_url,
        }.items()
        if not value
    ]
    if missing:
        message = f"ALPACA PAPER SAFETY ERROR: missing env vars: {', '.join(missing)}"
        logger.critical(message)
        raise SystemExit(message)

    normalized_base_url = base_url.rstrip("/")
    if PAPER_BASE_URL_FRAGMENT not in normalized_base_url:
        message = (
            "ALPACA PAPER SAFETY ERROR: APCA_API_BASE_URL must contain "
            f"{PAPER_BASE_URL_FRAGMENT!r}; got {normalized_base_url!r}"
        )
        logger.critical(message)
        raise SystemExit(message)

    return {
        "api_key": api_key,
        "secret_key": secret_key,
        "base_url": normalized_base_url,
    }


class AlpacaPaperExecutionEngine:
    def __init__(self, total_capital: float | None = None, max_position_pct: float = 0.02):
        config = require_paper_alpaca_config()
        self.api_key = config["api_key"]
        self.secret_key = config["secret_key"]
        self.base_url = config["base_url"]
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.account = self.get_account()
        self._print_account_mode()

    def _print_account_mode(self):
        print("=" * 72)
        print("ALPACA CONNECTION MODE: PAPER")
        print(f"APCA_API_BASE_URL: {self.base_url}")
        print(f"Account ID: {self.account.account_id}")
        print(f"Paper Equity: ${self.account.equity:,.2f}")
        print("=" * 72)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict | list:
        url = f"{self.base_url}{path}"
        body = None
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read()
                if not response_body:
                    return {}
                return json.loads(response_body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            message = f"Alpaca paper API {method} {path} failed: HTTP {exc.code} {error_body}"
            logger.error(message)
            raise RuntimeError(message) from exc
        except Exception:
            logger.exception("Alpaca paper API %s %s failed", method, path)
            raise

    def get_account(self) -> AlpacaAccountSnapshot:
        payload = self._request("GET", "/v2/account")
        return AlpacaAccountSnapshot(
            account_id=str(payload["id"]),
            equity=float(payload["equity"]),
            buying_power=float(payload["buying_power"]),
            trading_blocked=bool(payload.get("trading_blocked", False)),
            account_blocked=bool(payload.get("account_blocked", False)),
        )

    def get_positions(self) -> dict[str, float]:
        positions = self._request("GET", "/v2/positions")
        return {str(position["symbol"]): float(position["qty"]) for position in positions}

    def submit_market_order(self, symbol: str, quantity: int, side: str) -> dict:
        if quantity <= 0:
            raise ValueError(f"Order quantity must be positive for {symbol}: {quantity}")
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported Alpaca order side for {symbol}: {side}")

        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        order = self._request("POST", "/v2/orders", payload)
        print(f"[PAPER ORDER] {side.upper()} {quantity} {symbol} -> id={order.get('id')}")
        return order

    def rebalance_portfolio(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        dry_run: bool = False,
    ) -> list[dict]:
        account = self.get_account()
        if account.account_blocked or account.trading_blocked:
            message = "Alpaca paper account is blocked or trading-blocked; refusing orders."
            logger.error(message)
            raise RuntimeError(message)

        positions = self.get_positions()
        max_position_value = account.equity * self.max_position_pct
        orders = []

        print("[+] Initiating Alpaca PAPER portfolio rebalance...")
        print(f"[+] Equity=${account.equity:,.2f}; hard per-position cap=${max_position_value:,.2f}")

        for ticker, target_weight in target_weights.items():
            current_price = current_prices[ticker]
            if current_price <= 0:
                raise RuntimeError(f"Invalid current price for {ticker}: {current_price}")

            capped_target_value = max(
                -max_position_value,
                min(account.equity * target_weight, max_position_value),
            )
            target_shares = int(capped_target_value / current_price)
            current_shares = int(positions.get(ticker, 0.0))
            shares_delta = target_shares - current_shares

            event = {
                "symbol": ticker,
                "target_weight": target_weight,
                "capped_target_value": capped_target_value,
                "current_price": current_price,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "shares_delta": shares_delta,
                "dry_run": dry_run,
            }

            if shares_delta == 0:
                print(f"[HOLD] {ticker}: current={current_shares}, target={target_shares}")
                orders.append({**event, "action": "hold"})
                continue

            side = "buy" if shares_delta > 0 else "sell"
            quantity = abs(shares_delta)
            if dry_run:
                print(f"[DRY PAPER ORDER] {side.upper()} {quantity} {ticker} @ ~${current_price:.2f}")
                orders.append({**event, "action": side, "order": None})
                continue

            order = self.submit_market_order(ticker, quantity, side)
            orders.append({**event, "action": side, "order": order})

        return orders
