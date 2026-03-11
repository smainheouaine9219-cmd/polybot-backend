"""
Polymarket Trading Bot — Safest Bets Engine
Scans all markets, scores them by safety, trades the best ones multiple times/day
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
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

app = FastAPI(title="Polymarket SafeBot API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG ────────────────────────────────────────────────────────────────
CLOB_HOST   = "https://clob.polymarket.com"
GAMMA_HOST  = "https://gamma-api.polymarket.com"

API_KEY        = os.getenv("POLYMARKET_API_KEY", "")
API_SECRET     = os.getenv("POLYMARKET_SECRET", "")
API_PASSPHRASE = os.getenv("POLYMARKET_PASSPHRASE", "")
WALLET         = os.getenv("WALLET_ADDRESS", "")
IS_CONFIGURED  = bool(API_KEY and API_SECRET and WALLET)

# ─── STATE ─────────────────────────────────────────────────────────────────
trade_log:    List[Dict] = []
positions:    Dict[str, Dict] = {}
pnl_history:  List[Dict] = []
bot_running:  bool = False
last_scan:    Optional[str] = None
todays_bets:  List[Dict] = []   # cached best bets for today
last_traded:  Dict[str, datetime] = {}

# ─── SAFETY CONFIG (editable via API) ──────────────────────────────────────
class SafetyConfig(BaseModel):
    # Price range — only bet on near-certain outcomes
    min_price: float = 0.80        # Only trade if YES price > 80% (very likely)
    max_price: float = 0.97        # Avoid 99%+ (no profit left)
    # Volume — only liquid markets
    min_volume: float = 50000.0    # At least $50k volume
    # Liquidity
    min_liquidity: float = 10000.0 # At least $10k liquidity
    # Days until close — avoid markets closing too soon or too far
    min_days_left: int = 1
    max_days_left: int = 30
    # Trade sizing
    bet_usdc: float = 5.0          # Small bets = safer
    max_daily_usdc: float = 50.0   # Max spend per day
    max_positions: int = 10        # Max open positions
    # Scan frequency (minutes)
    scan_interval_mins: int = 60   # Scan every hour
    # Cooldown per market
    cooldown_hours: int = 24       # Don't re-enter same market for 24h
    # Categories to focus on (empty = all)
    categories: List[str] = ["politics", "crypto", "sports", "finance"]
    paper_mode: bool = True

safety_config = SafetyConfig()

# ─── AUTH ──────────────────────────────────────────────────────────────────
def auth_headers(method: str, path: str, body: str = "") -> Dict:
    ts  = str(int(time.time()))
    msg = ts + method.upper() + path + body
    sig = hmac.new(API_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return {
        "POLY-ADDRESS": WALLET, "POLY-API-KEY": API_KEY,
        "POLY-SIGNATURE": sig, "POLY-TIMESTAMP": ts,
        "POLY-PASSPHRASE": API_PASSPHRASE, "Content-Type": "application/json",
    }

def now_iso(): return datetime.utcnow().isoformat() + "Z"

# ─── SAFETY SCORER ─────────────────────────────────────────────────────────
def score_market(m: Dict, cfg: SafetyConfig) -> Dict:
    """
    Score a market's safety from 0-100.
    Higher = safer bet.
    
    Safety factors:
    1. Price certainty — closer to 1.0 = more certain outcome
    2. High volume — more traders = more accurate price
    3. High liquidity — easy to exit
    4. Time remaining — not expiring immediately
    5. Category trust — known reliable categories
    """
    try:
        prices = m.get("outcomePrices") or ["0.5", "0.5"]
        yes_p  = float(prices[0]) if prices else 0.5
        no_p   = 1 - yes_p

        # Pick the more likely side
        best_p = max(yes_p, no_p)
        side   = "YES" if yes_p >= no_p else "NO"

        volume    = float(m.get("volume") or 0)
        liquidity = float(m.get("liquidityNum") or m.get("liquidity") or 0)

        # Days until close
        end_date = m.get("endDate") or m.get("endDateIso")
        days_left = 999
        if end_date:
            try:
                end = datetime.fromisoformat(end_date.replace("Z", ""))
                days_left = max(0, (end - datetime.utcnow()).days)
            except Exception:
                days_left = 999

        # ── FILTER: hard disqualifiers ──
        if best_p < cfg.min_price:      return {"score": 0, "reason": f"Price {best_p:.0%} below min {cfg.min_price:.0%}"}
        if best_p > cfg.max_price:      return {"score": 0, "reason": f"Price {best_p:.0%} too certain, no profit"}
        if volume < cfg.min_volume:     return {"score": 0, "reason": f"Volume ${volume:,.0f} too low"}
        if liquidity < cfg.min_liquidity: return {"score": 0, "reason": f"Liquidity ${liquidity:,.0f} too low"}
        if days_left < cfg.min_days_left: return {"score": 0, "reason": "Expires too soon"}
        if days_left > cfg.max_days_left: return {"score": 0, "reason": f"{days_left} days too far out"}

        # ── SCORE: weighted factors ──
        # 1. Price certainty (40 pts) — higher = better
        price_score = ((best_p - cfg.min_price) / (cfg.max_price - cfg.min_price)) * 40

        # 2. Volume score (30 pts) — log scale
        import math
        vol_score = min(30, (math.log10(max(volume, 1)) - math.log10(cfg.min_volume)) * 10)

        # 3. Liquidity score (20 pts)
        liq_score = min(20, (math.log10(max(liquidity, 1)) - math.log10(cfg.min_liquidity)) * 7)

        # 4. Time score (10 pts) — sweet spot 3-14 days
        if 3 <= days_left <= 14:
            time_score = 10
        elif days_left < 3:
            time_score = days_left * 3
        else:
            time_score = max(0, 10 - (days_left - 14) * 0.3)

        total = round(price_score + vol_score + liq_score + time_score, 1)

        return {
            "score":      total,
            "side":       side,
            "price":      best_p,
            "yes_price":  yes_p,
            "no_price":   no_p,
            "volume":     volume,
            "liquidity":  liquidity,
            "days_left":  days_left,
            "reason":     f"Score {total:.0f}/100 — {side} @ {best_p:.0%}, {days_left}d left",
        }
    except Exception as e:
        return {"score": 0, "reason": str(e)}


# ─── MARKET SCANNER ────────────────────────────────────────────────────────
async def scan_safest_bets() -> List[Dict]:
    """Fetch all active markets, score them, return top 10 safest."""
    global last_scan, todays_bets

    all_markets = []
    async with httpx.AsyncClient(timeout=15) as client:
        # Fetch from multiple categories for breadth
        for offset in [0, 100, 200, 300]:
            try:
                resp = await client.get(
                    f"{GAMMA_HOST}/markets",
                    params={
                        "limit": 100, "offset": offset,
                        "active": True, "closed": False,
                        "order": "volume", "ascending": False,
                    },
                )
                if resp.status_code == 200:
                    raw = resp.json()
                    batch = raw if isinstance(raw, list) else raw.get("markets", [])
                    all_markets.extend(batch)
            except Exception:
                pass

    # Score every market
    scored = []
    for m in all_markets:
        mid = m.get("conditionId") or m.get("id", "")
        if not mid:
            continue

        # Skip if recently traded
        last = last_traded.get(mid)
        if last and (datetime.utcnow() - last) < timedelta(hours=safety_config.cooldown_hours):
            continue

        result = score_market(m, safety_config)
        if result["score"] > 0:
            scored.append({
                "id":         mid,
                "question":   m.get("question", ""),
                "score":      result["score"],
                "side":       result["side"],
                "price":      result["price"],
                "yes_price":  result["yes_price"],
                "no_price":   result["no_price"],
                "volume":     result["volume"],
                "liquidity":  result["liquidity"],
                "days_left":  result["days_left"],
                "reason":     result["reason"],
                "category":   m.get("category", ""),
                "end_date":   m.get("endDate", ""),
                "url":        f"https://polymarket.com/event/{m.get('slug','')}",
            })

    # Sort by score, take top 10
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:10]

    last_scan = now_iso()
    todays_bets = top
    print(f"✅ Scanned {len(all_markets)} markets → {len(scored)} qualified → top {len(top)} selected")
    return top


# ─── AUTO TRADE LOOP ───────────────────────────────────────────────────────
async def safe_bot_loop():
    global bot_running
    print("🤖 SafeBot started")

    while bot_running:
        try:
            # Check daily spend cap
            today = datetime.utcnow().date().isoformat()
            daily_spent = sum(
                t.get("amount_usdc", 0) for t in trade_log
                if t.get("ts", "")[:10] == today
            )

            if daily_spent >= safety_config.max_daily_usdc:
                print(f"⛔ Daily cap reached: ${daily_spent:.2f} / ${safety_config.max_daily_usdc}")
                await asyncio.sleep(60 * 10)
                continue

            if len(positions) >= safety_config.max_positions:
                print(f"⛔ Max positions reached: {len(positions)}")
                await asyncio.sleep(60 * 10)
                continue

            # Scan for best bets
            bets = await scan_safest_bets()

            for bet in bets:
                if not bot_running:
                    break

                # Re-check daily cap per trade
                daily_spent = sum(
                    t.get("amount_usdc", 0) for t in trade_log
                    if t.get("ts", "")[:10] == today
                )
                if daily_spent + safety_config.bet_usdc > safety_config.max_daily_usdc:
                    print(f"⛔ Would exceed daily cap, stopping trades")
                    break

                if len(positions) >= safety_config.max_positions:
                    break

                # Skip if already have this position
                if bet["id"] in positions:
                    continue

                # Place the trade
                trade = await execute_trade(
                    market_id=bet["id"],
                    side=bet["side"],
                    amount_usdc=safety_config.bet_usdc,
                    price=bet["price"],
                    note=f"SafeBot Score {bet['score']:.0f}/100 — {bet['reason']}"
                )
                trade_log.append(trade)
                last_traded[bet["id"]] = datetime.utcnow()
                snapshot_pnl()

                print(f"⚡ Traded: {bet['side']} ${safety_config.bet_usdc} on '{bet['question'][:50]}' (score {bet['score']:.0f})")

                # Small delay between trades
                await asyncio.sleep(5)

        except Exception as e:
            print(f"SafeBot error: {e}")

        # Wait for next scan
        wait_secs = safety_config.scan_interval_mins * 60
        print(f"⏳ Next scan in {safety_config.scan_interval_mins} minutes")
        await asyncio.sleep(wait_secs)

    print("🤖 SafeBot stopped")


async def execute_trade(market_id: str, side: str, amount_usdc: float, price: float, note: str) -> Dict:
    """Execute a trade — paper or live."""
    if not IS_CONFIGURED or safety_config.paper_mode:
        shares = round(amount_usdc / price, 2) if price else 0
        trade = {
            "id": f"safe_{uuid.uuid4().hex[:8]}",
            "market_id": market_id, "side": side,
            "amount_usdc": amount_usdc, "price": price,
            "shares": shares, "status": "PAPER_FILLED",
            "mode": "paper", "note": note, "ts": now_iso(),
        }
        # Update position
        pos = positions.get(market_id, {
            "market_id": market_id, "side": side, "shares": 0,
            "avg_price": 0, "cost_basis": 0, "current_price": price,
            "unrealized_pnl": 0,
        })
        pos["shares"] = round(pos["shares"] + shares, 4)
        pos["cost_basis"] = round(pos["cost_basis"] + amount_usdc, 4)
        pos["avg_price"] = round(pos["cost_basis"] / pos["shares"], 4) if pos["shares"] else 0
        positions[market_id] = pos
        return trade
    else:
        # Live trade
        order = {
            "orderType": "LIMIT", "tokenID": market_id,
            "side": "BUY", "size": str(amount_usdc),
            "price": str(price), "funder": WALLET,
            "maker": WALLET, "signer": WALLET,
            "taker": "0x0000000000000000000000000000000000000000",
            "outcome": side.upper(),
        }
        body = json.dumps(order)
        headers = auth_headers("POST", "/order", body)
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{CLOB_HOST}/order", content=body, headers=headers)
            if resp.status_code not in (200, 201):
                raise Exception(f"Order failed: {resp.text}")
            data = resp.json()
        return {**data, "side": side, "amount_usdc": amount_usdc, "ts": now_iso(), "mode": "live", "note": note}


def snapshot_pnl():
    total = sum(p.get("unrealized_pnl", 0) for p in positions.values())
    realized = sum(t.get("realized_pnl", 0) for t in trade_log if t.get("realized_pnl"))
    pnl_history.append({
        "ts": now_iso(), "unrealized": round(total, 4),
        "realized": round(realized, 4), "total": round(total + realized, 4),
    })
    if len(pnl_history) > 288:
        pnl_history.pop(0)


# ─── ROUTES ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    today = datetime.utcnow().date().isoformat()
    daily_spent = sum(t.get("amount_usdc", 0) for t in trade_log if t.get("ts", "")[:10] == today)
    return {
        "status": "online", "configured": IS_CONFIGURED,
        "paper_mode": safety_config.paper_mode or not IS_CONFIGURED,
        "bot_running": bot_running,
        "wallet": (WALLET[:6] + "…" + WALLET[-4:]) if WALLET else None,
        "daily_spent": round(daily_spent, 2),
        "daily_cap": safety_config.max_daily_usdc,
        "open_positions": len(positions),
        "total_trades": len(trade_log),
        "last_scan": last_scan,
        "ts": now_iso(),
    }


# ── BOT CONTROL ────────────────────────────────────────────────────────────

@app.post("/bot/start")
async def start_bot(background_tasks: BackgroundTasks):
    global bot_running
    if bot_running:
        return {"status": "already_running"}
    bot_running = True
    background_tasks.add_task(safe_bot_loop)
    return {"status": "started", "paper_mode": safety_config.paper_mode or not IS_CONFIGURED}

@app.post("/bot/stop")
def stop_bot():
    global bot_running
    bot_running = False
    return {"status": "stopped"}

@app.get("/bot/status")
def bot_status():
    today = datetime.utcnow().date().isoformat()
    daily_spent = sum(t.get("amount_usdc", 0) for t in trade_log if t.get("ts", "")[:10] == today)
    return {
        "running": bot_running,
        "paper_mode": safety_config.paper_mode or not IS_CONFIGURED,
        "daily_spent": round(daily_spent, 2),
        "daily_cap": safety_config.max_daily_usdc,
        "open_positions": len(positions),
        "max_positions": safety_config.max_positions,
        "last_scan": last_scan,
        "todays_bets": len(todays_bets),
        "scan_interval_mins": safety_config.scan_interval_mins,
    }

@app.get("/bot/settings")
def get_settings():
    return safety_config

@app.put("/bot/settings")
def update_settings(cfg: SafetyConfig):
    global safety_config
    safety_config = cfg
    return {"success": True, "settings": safety_config}


# ── SAFE BETS ──────────────────────────────────────────────────────────────

@app.get("/safebets")
async def get_safe_bets(refresh: bool = False):
    """Get today's safest bets. Pass refresh=true to rescan."""
    global todays_bets
    if refresh or not todays_bets:
        todays_bets = await scan_safest_bets()
    return {
        "bets": todays_bets,
        "count": len(todays_bets),
        "last_scan": last_scan,
        "config": {
            "min_price": safety_config.min_price,
            "min_volume": safety_config.min_volume,
        }
    }

@app.post("/safebets/scan")
async def force_scan():
    """Force a fresh scan right now."""
    bets = await scan_safest_bets()
    return {"bets": bets, "count": len(bets), "last_scan": last_scan}


# ── MARKETS ────────────────────────────────────────────────────────────────

@app.post("/markets/search")
async def search_markets(body: dict):
    keyword = body.get("keyword", "")
    limit   = body.get("limit", 12)
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{GAMMA_HOST}/markets",
            params={"keyword": keyword, "limit": limit, "active": True, "closed": False},
        )
    if resp.status_code != 200:
        return {"markets": []}
    raw = resp.json()
    markets = raw if isinstance(raw, list) else raw.get("markets", [])
    return {
        "markets": [{
            "id":        m.get("conditionId") or m.get("id"),
            "question":  m.get("question"),
            "yes_price": (m.get("outcomePrices") or ["0.5"])[0],
            "no_price":  (m.get("outcomePrices") or ["0.5","0.5"])[1] if len(m.get("outcomePrices") or []) > 1 else "0.5",
            "volume":    m.get("volume", 0),
            "end_date":  m.get("endDate"),
            "category":  m.get("category"),
            "url":       f"https://polymarket.com/event/{m.get('slug','')}",
        } for m in markets if m.get("conditionId") or m.get("id")]
    }

@app.get("/markets/trending")
async def trending():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{GAMMA_HOST}/markets",
            params={"limit": 8, "active": True, "closed": False, "order": "volume", "ascending": False},
        )
    if resp.status_code != 200:
        return {"markets": []}
    raw = resp.json()
    markets = raw if isinstance(raw, list) else raw.get("markets", [])
    return {"markets": [{
        "id":        m.get("conditionId") or m.get("id"),
        "question":  m.get("question"),
        "yes_price": (m.get("outcomePrices") or ["0.5"])[0],
        "no_price":  (m.get("outcomePrices") or ["0.5","0.5"])[1] if len(m.get("outcomePrices") or []) > 1 else "0.5",
        "volume":    m.get("volume", 0),
        "end_date":  m.get("endDate"),
        "category":  m.get("category"),
    } for m in markets]}


# ── ACCOUNT ────────────────────────────────────────────────────────────────

@app.get("/account/balance")
async def get_balance():
    if not IS_CONFIGURED:
        return {"balance": 1000.0, "mode": "paper"}
    headers = auth_headers("GET", "/balance")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{CLOB_HOST}/balance", headers=headers)
    return {**resp.json(), "mode": "live"} if resp.status_code == 200 else {"balance": 0, "error": resp.text}

@app.get("/account/positions")
def get_positions():
    for mid, pos in positions.items():
        cp = pos.get("current_price", pos.get("avg_price", 0.5))
        pos["unrealized_pnl"] = round(pos.get("shares", 0) * cp - pos.get("cost_basis", 0), 4)
    return {"positions": list(positions.values()), "count": len(positions)}


# ── TRADING ────────────────────────────────────────────────────────────────

@app.post("/trade")
async def manual_trade(body: dict):
    trade = await execute_trade(
        market_id=body["market_id"], side=body["side"],
        amount_usdc=float(body["amount_usdc"]),
        price=float(body.get("price") or 0.5),
        note=body.get("note") or "Manual trade",
    )
    trade_log.append(trade)
    snapshot_pnl()
    return {"success": True, "trade": trade}


# ── P&L ────────────────────────────────────────────────────────────────────

@app.get("/pnl/summary")
def pnl_summary():
    unrealized = sum(p.get("unrealized_pnl", 0) for p in positions.values())
    realized   = sum(t.get("realized_pnl", 0) for t in trade_log if t.get("realized_pnl"))
    invested   = sum(t.get("amount_usdc", 0) for t in trade_log)
    today      = datetime.utcnow().date().isoformat()
    daily_spent = sum(t.get("amount_usdc", 0) for t in trade_log if t.get("ts", "")[:10] == today)
    return {
        "unrealized_pnl": round(unrealized, 4),
        "realized_pnl":   round(realized, 4),
        "total_pnl":      round(unrealized + realized, 4),
        "total_invested": round(invested, 4),
        "daily_spent":    round(daily_spent, 4),
        "daily_cap":      safety_config.max_daily_usdc,
        "open_positions": len(positions),
        "total_trades":   len(trade_log),
        "win_rate":       0.0,
    }

@app.get("/pnl/history")
def pnl_history_route():
    return {"history": pnl_history, "points": len(pnl_history)}

@app.get("/trades")
def get_trades(limit: int = 50):
    return {"trades": list(reversed(trade_log))[:limit], "total": len(trade_log)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
