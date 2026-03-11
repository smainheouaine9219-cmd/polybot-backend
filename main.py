"""
Polymarket Trading Bot — Full Backend
Supports: manual trades, auto-trading strategies, live P&L tracking
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx
import json
import os
import time
import hmac
import hashlib
import asyncio
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Polymarket Bot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down to your Lovable URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG ────────────────────────────────────────────────────────────────
CLOB_HOST      = "https://clob.polymarket.com"
GAMMA_HOST     = "https://gamma-api.polymarket.com"
POLY_APP       = "https://polymarket.com/event"

API_KEY        = os.getenv("POLYMARKET_API_KEY", "")
API_SECRET     = os.getenv("POLYMARKET_SECRET", "")
API_PASSPHRASE = os.getenv("POLYMARKET_PASSPHRASE", "")
WALLET         = os.getenv("WALLET_ADDRESS", "")

IS_CONFIGURED  = bool(API_KEY and API_SECRET and WALLET)


# ─── IN-MEMORY STATE ───────────────────────────────────────────────────────
# In production, replace these with a real DB (Postgres, Supabase, etc.)

trade_log: List[Dict]    = []   # All placed trades
positions: Dict[str, Dict] = {} # market_id → position data
pnl_history: List[Dict]  = []   # Timestamped P&L snapshots
bot_running: bool        = False
auto_task               = None


# ─── MODELS ────────────────────────────────────────────────────────────────

class TradeRequest(BaseModel):
    market_id:    str
    side:         str            # "YES" or "NO"
    amount_usdc:  float
    price:        Optional[float] = None   # None = market order
    slippage:     float = 0.02
    note:         Optional[str] = None    # your label

class SearchRequest(BaseModel):
    keyword: str
    limit:   int = 12

class Strategy(BaseModel):
    name:          str
    keywords:      List[str] = []          # markets to watch
    side:          str = "YES"             # default side
    bet_usdc:      float = 5.0             # per-trade size
    min_price:     float = 0.10            # only trade if price > this
    max_price:     float = 0.90            # only trade if price < this
    min_volume:    float = 1000.0          # skip low-volume markets
    cooldown_mins: int   = 60              # don't re-enter same market within N mins

class BotSettings(BaseModel):
    enabled:        bool = False
    paper_mode:     bool = True            # always True unless API configured
    max_total_usdc: float = 100.0          # hard cap on total exposure
    strategies:     List[Strategy] = []

class CancelRequest(BaseModel):
    order_id: str


# ─── LIVE STATE ────────────────────────────────────────────────────────────

bot_settings = BotSettings(strategies=[
    Strategy(
        name="Default Strategy",
        keywords=[],
        side="YES",
        bet_usdc=5.0,
        min_price=0.10,
        max_price=0.60,
    )
])

last_traded: Dict[str, datetime] = {}   # market_id → last trade time


# ─── HELPERS ───────────────────────────────────────────────────────────────

def auth_headers(method: str, path: str, body: str = "") -> Dict:
    ts  = str(int(time.time()))
    msg = ts + method.upper() + path + body
    sig = hmac.new(API_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return {
        "POLY-ADDRESS":    WALLET,
        "POLY-API-KEY":    API_KEY,
        "POLY-SIGNATURE":  sig,
        "POLY-TIMESTAMP":  ts,
        "POLY-PASSPHRASE": API_PASSPHRASE,
        "Content-Type":    "application/json",
    }

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def snapshot_pnl():
    total = sum(p.get("unrealized_pnl", 0) for p in positions.values())
    realized = sum(
        t.get("realized_pnl", 0) for t in trade_log if t.get("status") == "FILLED"
    )
    pnl_history.append({
        "ts": now_iso(),
        "unrealized": round(total, 4),
        "realized":   round(realized, 4),
        "total":      round(total + realized, 4),
    })
    # Keep last 288 points (24h at 5-min intervals)
    if len(pnl_history) > 288:
        pnl_history.pop(0)


def make_paper_trade(req: TradeRequest) -> Dict:
    price = req.price or (0.55 if req.side == "YES" else 0.45)
    shares = round(req.amount_usdc / price, 2)
    trade = {
        "id":          f"paper_{uuid.uuid4().hex[:8]}",
        "market_id":   req.market_id,
        "side":        req.side.upper(),
        "amount_usdc": req.amount_usdc,
        "price":       price,
        "shares":      shares,
        "status":      "PAPER_FILLED",
        "mode":        "paper",
        "note":        req.note or "",
        "ts":          now_iso(),
    }
    # Update virtual position
    pos = positions.get(req.market_id, {
        "market_id": req.market_id,
        "side": req.side.upper(),
        "shares": 0,
        "avg_price": 0,
        "cost_basis": 0,
        "current_price": price,
        "unrealized_pnl": 0,
        "realized_pnl": 0,
    })
    new_cost   = pos["cost_basis"] + req.amount_usdc
    new_shares = pos["shares"] + shares
    pos["shares"]      = round(new_shares, 4)
    pos["avg_price"]   = round(new_cost / new_shares, 4) if new_shares else 0
    pos["cost_basis"]  = round(new_cost, 4)
    pos["side"]        = req.side.upper()
    positions[req.market_id] = pos
    return trade


# ─── AUTO-TRADING ENGINE ───────────────────────────────────────────────────

async def auto_trade_loop():
    """Background task: poll markets & fire trades based on strategies."""
    global bot_running
    print("🤖 Auto-trade loop started")
    while bot_running:
        try:
            for strategy in bot_settings.strategies:
                if not strategy.keywords:
                    continue
                for kw in strategy.keywords:
                    await run_strategy_for_keyword(strategy, kw)
        except Exception as e:
            print(f"Auto-trade error: {e}")
        # Poll every 5 minutes
        await asyncio.sleep(300)
    print("🤖 Auto-trade loop stopped")


async def run_strategy_for_keyword(strategy: Strategy, keyword: str):
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{GAMMA_HOST}/markets",
                params={"keyword": keyword, "limit": 5, "active": True, "closed": False},
            )
            if resp.status_code != 200:
                return
            markets = resp.json()
            if isinstance(markets, dict):
                markets = markets.get("markets", [])
        except Exception:
            return

    for m in markets:
        mid = m.get("conditionId") or m.get("id", "")
        if not mid:
            continue

        # Cooldown check
        last = last_traded.get(mid)
        if last and (datetime.utcnow() - last) < timedelta(minutes=strategy.cooldown_mins):
            continue

        # Price filters
        prices = m.get("outcomePrices", [])
        try:
            yes_price = float(prices[0]) if prices else 0.5
        except (ValueError, TypeError):
            yes_price = 0.5
        no_price = 1 - yes_price

        target_price = yes_price if strategy.side == "YES" else no_price
        if not (strategy.min_price <= target_price <= strategy.max_price):
            continue

        # Volume filter
        vol = float(m.get("volume") or 0)
        if vol < strategy.min_volume:
            continue

        # Total exposure cap
        total_exposure = sum(p.get("cost_basis", 0) for p in positions.values())
        if total_exposure + strategy.bet_usdc > bot_settings.max_total_usdc:
            print(f"⛔ Exposure cap reached ({total_exposure:.2f} USDC)")
            continue

        # Place the trade
        req = TradeRequest(
            market_id=mid,
            side=strategy.side,
            amount_usdc=strategy.bet_usdc,
            price=target_price,
            note=f"Auto: {strategy.name} / {keyword}",
        )
        trade = make_paper_trade(req) if (bot_settings.paper_mode or not IS_CONFIGURED) else await place_live_trade(req)
        trade_log.append(trade)
        last_traded[mid] = datetime.utcnow()
        print(f"✅ Auto trade: {strategy.side} ${strategy.bet_usdc} on {m.get('question','?')[:60]}")
        snapshot_pnl()


async def place_live_trade(req: TradeRequest) -> Dict:
    order = {
        "orderType": "LIMIT" if req.price else "MARKET",
        "tokenID":   req.market_id,
        "side":      "BUY",
        "size":      str(req.amount_usdc),
        "price":     str(req.price) if req.price else None,
        "funder":    WALLET,
        "maker":     WALLET,
        "signer":    WALLET,
        "taker":     "0x0000000000000000000000000000000000000000",
        "outcome":   req.side.upper(),
    }
    body = json.dumps(order)
    headers = auth_headers("POST", "/order", body)
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{CLOB_HOST}/order", content=body, headers=headers)
        if resp.status_code not in (200, 201):
            raise HTTPException(resp.status_code, f"Order failed: {resp.text}")
        data = resp.json()
    return {**data, "side": req.side, "amount_usdc": req.amount_usdc, "ts": now_iso(), "mode": "live", "note": req.note or ""}


# ─── ROUTES ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "online",
        "configured":    IS_CONFIGURED,
        "paper_mode":    not IS_CONFIGURED or bot_settings.paper_mode,
        "bot_running":   bot_running,
        "wallet":        (WALLET[:6] + "…" + WALLET[-4:]) if WALLET else None,
        "ts":            now_iso(),
    }


# ── MARKETS ────────────────────────────────────────────────────────────────

@app.post("/markets/search")
async def search_markets(req: SearchRequest):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{GAMMA_HOST}/markets",
            params={"keyword": req.keyword, "limit": req.limit, "active": True, "closed": False},
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"Gamma API: {resp.text}")
    raw = resp.json()
    markets = raw if isinstance(raw, list) else raw.get("markets", [])
    return {
        "markets": [
            {
                "id":        m.get("conditionId") or m.get("id"),
                "question":  m.get("question"),
                "yes_price": m.get("outcomePrices", ["0.5"])[0],
                "no_price":  m.get("outcomePrices", ["0.5", "0.5"])[1] if len(m.get("outcomePrices", [])) > 1 else "0.5",
                "volume":    m.get("volume", 0),
                "end_date":  m.get("endDate"),
                "category":  m.get("category"),
                "slug":      m.get("slug"),
                "url":       f"{POLY_APP}/{m.get('slug', '')}",
            }
            for m in markets if m.get("conditionId") or m.get("id")
        ]
    }


@app.get("/markets/trending")
async def trending_markets():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{GAMMA_HOST}/markets",
            params={"limit": 8, "active": True, "closed": False, "order": "volume", "ascending": False},
        )
    if resp.status_code != 200:
        raise HTTPException(502, resp.text)
    raw = resp.json()
    markets = raw if isinstance(raw, list) else raw.get("markets", [])
    return {
        "markets": [
            {
                "id":        m.get("conditionId") or m.get("id"),
                "question":  m.get("question"),
                "yes_price": m.get("outcomePrices", ["0.5"])[0],
                "no_price":  (m.get("outcomePrices") or ["0.5","0.5"])[1] if len(m.get("outcomePrices") or []) > 1 else "0.5",
                "volume":    m.get("volume", 0),
                "end_date":  m.get("endDate"),
                "category":  m.get("category"),
                "url":       f"{POLY_APP}/{m.get('slug', '')}",
            }
            for m in markets
        ]
    }


# ── ACCOUNT ────────────────────────────────────────────────────────────────

@app.get("/account/balance")
async def get_balance():
    if not IS_CONFIGURED:
        return {"balance": 1000.0, "mode": "paper", "note": "Simulated balance — add API keys for real balance"}
    headers = auth_headers("GET", "/balance")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{CLOB_HOST}/balance", headers=headers)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)
    return {**resp.json(), "mode": "live"}


@app.get("/account/positions")
def get_positions():
    pos_list = []
    for mid, pos in positions.items():
        # Refresh unrealized P&L from current price
        cp = pos.get("current_price", pos.get("avg_price", 0.5))
        shares = pos.get("shares", 0)
        cost   = pos.get("cost_basis", 0)
        current_val = shares * cp
        pos["unrealized_pnl"] = round(current_val - cost, 4)
        pos_list.append(pos)
    return {"positions": pos_list, "count": len(pos_list)}


@app.get("/account/orders")
async def get_orders():
    if not IS_CONFIGURED:
        return {"orders": [], "note": "Paper mode"}
    headers = auth_headers("GET", "/orders")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{CLOB_HOST}/orders", headers=headers)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)
    return resp.json()


# ── TRADING ────────────────────────────────────────────────────────────────

@app.post("/trade")
async def place_trade(req: TradeRequest):
    if req.amount_usdc <= 0:
        raise HTTPException(400, "Amount must be > 0")
    if req.side.upper() not in ("YES", "NO"):
        raise HTTPException(400, "Side must be YES or NO")

    # Paper or live
    if not IS_CONFIGURED or bot_settings.paper_mode:
        trade = make_paper_trade(req)
    else:
        trade = await place_live_trade(req)

    trade_log.append(trade)
    snapshot_pnl()
    return {"success": True, "trade": trade}


@app.delete("/order/{order_id}")
async def cancel_order(order_id: str):
    if not IS_CONFIGURED:
        return {"success": True, "note": "Paper mode — nothing to cancel"}
    headers = auth_headers("DELETE", f"/order/{order_id}")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(f"{CLOB_HOST}/order/{order_id}", headers=headers)
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)
    return {"success": True, "order_id": order_id}


# ── BOT CONTROL ────────────────────────────────────────────────────────────

@app.get("/bot/settings")
def get_bot_settings():
    return bot_settings

@app.put("/bot/settings")
def update_bot_settings(settings: BotSettings):
    global bot_settings
    bot_settings = settings
    return {"success": True, "settings": bot_settings}

@app.post("/bot/start")
async def start_bot(background_tasks: BackgroundTasks):
    global bot_running, auto_task
    if bot_running:
        return {"status": "already_running"}
    bot_running = True
    background_tasks.add_task(auto_trade_loop)
    return {"status": "started", "paper_mode": bot_settings.paper_mode or not IS_CONFIGURED}

@app.post("/bot/stop")
def stop_bot():
    global bot_running
    bot_running = False
    return {"status": "stopped"}

@app.get("/bot/status")
def bot_status():
    return {
        "running":       bot_running,
        "paper_mode":    bot_settings.paper_mode or not IS_CONFIGURED,
        "strategies":    len(bot_settings.strategies),
        "trades_placed": len(trade_log),
        "open_positions": len(positions),
    }


# ── P&L & ANALYTICS ───────────────────────────────────────────────────────

@app.get("/pnl/history")
def get_pnl_history(hours: int = 24):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    filtered = [p for p in pnl_history if datetime.fromisoformat(p["ts"].replace("Z","")) > cutoff]
    return {"history": filtered, "points": len(filtered)}

@app.get("/pnl/summary")
def get_pnl_summary():
    unrealized = sum(p.get("unrealized_pnl", 0) for p in positions.values())
    realized   = sum(t.get("realized_pnl", 0) for t in trade_log if t.get("realized_pnl"))
    total_invested = sum(t.get("amount_usdc", 0) for t in trade_log)
    return {
        "unrealized_pnl":  round(unrealized, 4),
        "realized_pnl":    round(realized, 4),
        "total_pnl":       round(unrealized + realized, 4),
        "total_invested":  round(total_invested, 4),
        "open_positions":  len(positions),
        "total_trades":    len(trade_log),
        "win_rate":        _calc_win_rate(),
    }

def _calc_win_rate() -> float:
    closed = [t for t in trade_log if t.get("realized_pnl") is not None]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if t.get("realized_pnl", 0) > 0)
    return round(wins / len(closed) * 100, 1)

@app.get("/trades")
def get_trades(limit: int = 50):
    return {"trades": list(reversed(trade_log))[:limit], "total": len(trade_log)}


# ── PRICE UPDATE (called by frontend to refresh P&L) ──────────────────────

@app.post("/positions/{market_id}/update_price")
def update_position_price(market_id: str, price: float):
    if market_id in positions:
        positions[market_id]["current_price"] = price
        shares = positions[market_id].get("shares", 0)
        cost   = positions[market_id].get("cost_basis", 0)
        positions[market_id]["unrealized_pnl"] = round(shares * price - cost, 4)
    return {"success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
