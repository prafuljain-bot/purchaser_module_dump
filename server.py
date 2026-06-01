#!/usr/bin/env python3
"""
Procurement Analytics API Server
Reads data from GitHub CSV, computes metrics, serves frontend + Claude Q&A
"""
import os, io, time, csv, json, statistics
from collections import defaultdict, Counter
from datetime import datetime
from functools import lru_cache

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Config ────────────────────────────────────────────────────────────────────
DATA_URL    = os.environ.get("DATA_URL",
    "https://raw.githubusercontent.com/prafuljain-bot/purchaser_module_dump/main/data.csv")
GROQ_KEY      = os.environ.get("GROQ_API_KEY", "")
CACHE_TTL   = int(os.environ.get("CACHE_TTL", "300"))   # seconds
DEFAULT_MONTHS = ['2026-03','2026-04','2026-05']
CITIES      = ['Jaipur','Chandigarh']
BRAND_PAL   = ['#4ade80','#60a5fa','#f87171','#f59e0b','#a78bfa',
               '#fb923c','#22d3ee','#818cf8','#34d399','#fbbf24','#64748b']

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache = {"data": None, "ts": 0}

def fetch_rows():
    now = time.time()
    if _cache["data"] is not None and now - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    rows = list(reader)  # fetch ALL rows; date filtering in compute_payload
    _cache["data"] = rows
    _cache["ts"]   = now
    return rows

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_bilaspur(r):
    return "Bilaspur" in [c.strip() for c in r.get("total_cities","").split(",") if c.strip()]

def is_ci_near(r):
    return r.get("block_type") == "CROSS_INVENTORY" and r.get("transfer_order_city") == "Bilaspur"

def has_ci_far(r):
    return any(c != "Bilaspur" for c in [c.strip() for c in r.get("total_cities","").split(",") if c.strip()])

def block_label(r):
    bt = r.get("block_type",""); tc = r.get("transfer_order_city","")
    if bt == "PREFERRED_VENDOR":      return "PV"
    if bt == "CROSS_INVENTORY":       return "CI Near" if tc == "Bilaspur" else "CI Far"
    if bt == "OPEN_MARKET":           return "LOM"
    if bt == "PAN_INDIA_OPEN_MARKET": return "PIOM Near" if tc == "Bilaspur" else "PIOM Far"
    return "Other"

def parse_dt(s):
    if not s: return None
    try: return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")
    except: return None

def med(lst): return round(statistics.median(lst), 2) if lst else None

def ci_hrs(r):
    t1 = parse_dt(r.get("cross_inventory_started_at"))
    t2 = parse_dt(r.get("hard_confirmed_at"))
    if t1 and t2 and (t2-t1).total_seconds() >= 0:
        return (t2-t1).total_seconds() / 3600
    return None

def detect_groups(rows, bucket=15, threshold=8, min_gap=2):
    buckets = Counter()
    for r in rows:
        bt = r.get("batch_created_at","")
        if len(bt) < 16: continue
        h, m = int(bt[11:13]), int(bt[14:16])
        buckets[(h*60+m)//bucket*bucket] += 1
    def fmt(m): return f"{m//60:02d}:{m%60:02d}"
    clusters, current, gap = [], None, 0
    for mins in range(6*60, 21*60, bucket):
        cnt = buckets.get(mins, 0)
        if cnt >= threshold:
            gap = 0
            if current is None: current = {"start": mins, "end": mins+bucket}
            else: current["end"] = mins+bucket
        else:
            gap += 1
            if current and gap >= min_gap:
                clusters.append(current); current = None
    if current: clusters.append(current)
    return [(f"{fmt(c['start'])}–{fmt(c['end'])}", c["start"], c["end"]) for c in clusters]

def assign_group(bt, groups):
    if not bt or len(bt) < 16: return groups[-1][0] if groups else "Late/Other"
    h, m = int(bt[11:13]), int(bt[14:16])
    mins = h*60+m
    for label,start,end in groups:
        if start <= mins < end: return label
    return "Late/Other"

# ── Share computation ─────────────────────────────────────────────────────────
BLOCKS = ["PV","CI Near","CI Far","LOM","PIOM Near","PIOM Far"]

def block_match(r, b):
    bt = r.get("block_type",""); tc = r.get("transfer_order_city","")
    cities = [c.strip() for c in r.get("total_cities","").split(",") if c.strip()]
    if b == "PV":        return bt == "PREFERRED_VENDOR"
    if b == "CI Near":   return bt == "CROSS_INVENTORY" and tc == "Bilaspur"
    if b == "CI Far":    return bt == "CROSS_INVENTORY" and tc != "Bilaspur"
    if b == "LOM":       return bt == "OPEN_MARKET"
    if b == "PIOM Near": return bt == "PAN_INDIA_OPEN_MARKET" and tc == "Bilaspur"
    if b == "PIOM Far":  return bt == "PAN_INDIA_OPEN_MARKET" and tc != "Bilaspur"
    return False

def avail_for(r, b):
    cities = [c.strip() for c in r.get("total_cities","").split(",") if c.strip()]
    if b == "CI Near":   return "Bilaspur" in cities
    if b == "CI Far":    return any(c != "Bilaspur" for c in cities)
    if b == "PV":        return bool(r.get("preferred_vendor_started_at"))
    if b == "LOM":       return bool(r.get("open_market_started_at"))
    if b in ("PIOM Near","PIOM Far"): return bool(r.get("pan_india_started_at"))
    return False

def trig_for(r, b):
    cities = [c.strip() for c in r.get("total_cities","").split(",") if c.strip()]
    if b == "CI Near":   return "Bilaspur" in cities
    if b == "CI Far":    return any(c != "Bilaspur" for c in cities)
    if b == "PV":        return bool(r.get("preferred_vendor_started_at"))
    if b == "LOM":       return bool(r.get("open_market_started_at"))
    if b in ("PIOM Near","PIOM Far"): return bool(r.get("pan_india_started_at"))
    return False

def compute_block_group(grp_rows, overall_total):
    total = len(grp_rows)
    result = {"total": total, "overall_total": overall_total}
    for b in BLOCKS:
        conf  = sum(1 for r in grp_rows if block_match(r, b))
        avail = sum(1 for r in grp_rows if avail_for(r, b))
        trig  = sum(1 for r in grp_rows if trig_for(r, b))
        pct       = round(conf/overall_total*100, 1) if overall_total else 0
        trig_opct = round(trig/overall_total*100, 1) if overall_total else 0
        result[b] = {"pct": pct, "overall_pct": pct, "trig_opct": trig_opct,
                     "confirmed": conf, "available": avail, "triggered": trig}
    return result

def compute_share(rows, groups, months=None):
    group_labels = [g[0] for g in groups]
    overall, by_batch = {}, {}
    for m in (months or DEFAULT_MONTHS):
        mo = [r for r in rows if r.get("part_received_at","")[:7] == m]
        total = len(mo)
        overall[m] = compute_block_group(mo, total)
        by_batch[m] = {}
        for g in group_labels:
            grp = [r for r in mo if assign_group(r.get("batch_created_at",""), groups) == g]
            by_batch[m][g] = compute_block_group(grp, total)
    return {"overall": overall, "by_batch": by_batch}

# ── Brand computation ─────────────────────────────────────────────────────────
def compute_brand(rows, groups, months=None):
    group_labels = [g[0] for g in groups]
    bc = Counter(r.get("requested_part_brand","") for r in rows if r.get("requested_part_brand"))
    top10  = [b for b,_ in bc.most_common(10)]
    brands = top10 + ["Other"]
    colors = {b: BRAND_PAL[i] for i,b in enumerate(brands)}
    def brand_of(r):
        b = r.get("requested_part_brand","")
        return b if b in top10 else "Other"
    def block_for_brand(br_rows, overall_total):
        total = len(br_rows)
        result = {"total": total}
        for bl in BLOCKS:
            conf  = sum(1 for r in br_rows if block_match(r, bl))
            avail = sum(1 for r in br_rows if avail_for(r, bl))
            trig  = sum(1 for r in br_rows if trig_for(r, bl))
            result[bl] = {
                "pct":         round(conf/total*100, 1)         if total         else 0,
                "overall_pct": round(conf/overall_total*100, 1) if overall_total else 0,
                "trig_opct":   round(trig/overall_total*100, 1) if overall_total else 0,
                "confirmed": conf, "available": avail, "triggered": trig
            }
        return result
    by_brand = {}
    for brand in brands:
        by_brand[brand] = {}
        for m in (months or DEFAULT_MONTHS):
            mo = [r for r in rows if r.get("part_received_at","")[:7] == m]
            br = [r for r in mo if brand_of(r) == brand]
            by_brand[brand][m] = block_for_brand(br, len(mo))
    return {"top10": top10, "brands": brands, "colors": colors, "blocks": BLOCKS, "by_brand": by_brand}

# ── PIOM cascade ──────────────────────────────────────────────────────────────
def compute_piom_cascade(rows, groups, months=None):
    bc = Counter(r.get("requested_part_brand","") for r in rows if r.get("requested_part_brand"))
    top10  = [b for b,_ in bc.most_common(10)]
    brands = top10 + ["Other"]
    def brand_of(r):
        b = r.get("requested_part_brand","")
        return b if b in top10 else "Other"
    def cascade_cell(piom_rows, mo_total):
        n = len(piom_rows)
        if n == 0: return {"piom": 0}
        def cnt(fn): return sum(1 for r in piom_rows if fn(r))
        def pct(c):  return round(c/n*100, 1) if n else 0
        def opct(c): return round(c/mo_total*100, 1) if mo_total else 0
        cn = cnt(is_bilaspur)
        cf = cnt(has_ci_far)
        return {"piom": n, "piom_overall_pct": opct(n),
                "ci_near_pct": pct(cn), "ci_near_overall_pct": opct(cn),
                "ci_far_pct":  pct(cf), "ci_far_overall_pct":  opct(cf)}
    result = {"overall": {}, "by_brand": {}, "brands": brands}
    for m in (months or DEFAULT_MONTHS):
        mo_total = sum(1 for r in rows if r.get("part_received_at","")[:7] == m)
        mo_piom  = [r for r in rows if r.get("part_received_at","")[:7] == m
                    and r.get("block_type") == "PAN_INDIA_OPEN_MARKET"]
        result["overall"][m] = cascade_cell(mo_piom, mo_total)
        for brand in brands:
            if brand not in result["by_brand"]: result["by_brand"][brand] = {}
            result["by_brand"][brand][m] = cascade_cell(
                [r for r in mo_piom if brand_of(r) == brand], mo_total)
    return result

# ── Further RCA ───────────────────────────────────────────────────────────────
def compute_further_rca(rows, groups, months=None):
    def block_of(r):
        bt = r.get("block_type",""); tc = r.get("transfer_order_city","")
        if bt == "PREFERRED_VENDOR": return "PV"
        if bt == "CROSS_INVENTORY":  return "CI Near" if tc == "Bilaspur" else "CI Far"
        if bt == "OPEN_MARKET":      return "LOM"
        if bt == "PAN_INDIA_OPEN_MARKET": return "PIOM"
        return "Other"
    def cities_of(r):
        return [c.strip() for c in r.get("total_cities","").split(",") if c.strip()]
    overall_ct = {}
    for m in (months or DEFAULT_MONTHS):
        mo_conf = [r for r in rows if r.get("part_received_at","")[:7] == m and is_ci_near(r)]
        times = [h for r in mo_conf for h in [ci_hrs(r)] if h is not None]
        overall_ct[m] = med(times)
    def spill_cell(rej_rows, m, spill_blocks, mo_total):
        n = len(rej_rows)
        if n == 0: return {"rejected": 0}
        wt = [h for r in rej_rows for h in [ci_hrs(r)] if h is not None]
        mct = overall_ct[m]; mwt = med(wt)
        verdict = None
        if mwt is not None and mct is not None:
            verdict = "less_time" if mwt < mct else "ops_slow"
        bc = Counter(block_of(r) for r in rej_rows)
        return {
            "rejected": n,
            "rejected_opct": round(n/mo_total*100, 1) if mo_total else 0,
            "confirm_time": mct, "window": mwt, "verdict": verdict,
            "spilled_to": {b: {"count": bc.get(b,0),
                               "pct": round(bc.get(b,0)/n*100,1) if n else 0,
                               "overall_pct": round(bc.get(b,0)/mo_total*100,1) if mo_total else 0}
                           for b in spill_blocks}
        }
    result = {"ci_near": {}, "ci_far": {}}
    for m in (months or DEFAULT_MONTHS):
        mo = [r for r in rows if r.get("part_received_at","")[:7] == m]
        mo_total = len(mo)
        ci_near_rej = [r for r in mo if is_bilaspur(r) and not is_ci_near(r)]
        result["ci_near"][m] = spill_cell(ci_near_rej, m, ["PV","CI Far","LOM","PIOM"], mo_total)
        ci_far_rej  = [r for r in mo
                       if any(c != "Bilaspur" for c in cities_of(r))
                       and not (r.get("block_type") == "CROSS_INVENTORY" and r.get("transfer_order_city") != "Bilaspur")]
        result["ci_far"][m]  = spill_cell(ci_far_rej, m, ["PV","CI Near","LOM","PIOM"], mo_total)
    return result

# ── Summary stats for Claude context ─────────────────────────────────────────
def build_context_summary(payload, months=None):
    lines = ["=== PROCUREMENT ANALYTICS CONTEXT ===\n"]
    lines.append("Cities: Jaipur, Chandigarh | Period: Mar–May 2026 | JIT Non-Adhoc\n")
    lines.append("Cascade: CI Near → PV → LOM → CI Far (May+) → PIOM Near → PIOM Far\n")
    lines.append("CI Near = own inventory at Bilaspur. CI Far = own inventory other cities.\n")
    lines.append("PIOM = fresh purchase from external vendor.\n")
    lines.append("PV = Preferred Vendor mapped by brand/category.\n\n")
    for city in CITIES:
        lines.append(f"--- {city} ---")
        grps  = payload["groups"].get(city, [])
        share = payload[city]["share"]
        for m in (months or DEFAULT_MONTHS):
            ov = share["overall"].get(m, {})
            total = ov.get("total", 0)
            if not total: continue
            mn = {"2026-01":"Jan","2026-02":"Feb","2026-03":"Mar","2026-04":"Apr","2026-05":"May","2026-06":"Jun","2026-07":"Jul","2026-08":"Aug","2026-09":"Sep","2026-10":"Oct","2026-11":"Nov","2026-12":"Dec"}.get(m, m)
            lines.append(f"\n{mn} ({total} parts):")
            for b in BLOCKS:
                bd = ov.get(b, {})
                if bd.get("confirmed", 0) > 0:
                    lines.append(f"  {b}: {bd['pct']}% conf ({bd['confirmed']}) avail={bd['available']} trig={bd['triggered']}")
        lines.append(f"\nBatch groups ({city}): {grps}")
        brands = payload[city]["brand"]
        lines.append(f"\nTop brands: {brands['top10']}")
        lines.append("")
    return "\n".join(lines)

# ── Compute everything ────────────────────────────────────────────────────────
def compute_payload(start: str = None, end: str = None):
    all_rows = fetch_rows()

    # Filter by date range if provided
    def in_range(r):
        d = r.get("part_received_at", "")[:10]
        if not d: return False
        if start and d < start: return False
        if end   and d > end:   return False
        return True

    rows = [r for r in all_rows if in_range(r)] if (start or end) else all_rows

    # Detect months dynamically from filtered rows
    months = sorted(set(r["part_received_at"][:7]
                        for r in rows if r.get("part_received_at")))
    if not months:
        months = DEFAULT_MONTHS

    payload = {"groups": {}, "months": months, "data_ts": time.time(),
               "date_range": {"start": start or "", "end": end or ""}}
    for city in CITIES:
        cr   = [r for r in rows if r.get("entity_city") == city]
        grps = detect_groups(cr)
        payload[city] = {
            "share":   compute_share(cr, grps, months),
            "brand":   compute_brand(cr, grps, months),
            "piom":    compute_piom_cascade(cr, grps, months),
            "further": compute_further_rca(cr, grps, months),
        }
        payload["groups"][city] = [g[0] for g in grps]
    payload["context"] = build_context_summary(payload, months)
    return payload

_payload_cache = {"data": None, "ts": 0}

def get_payload(start=None, end=None):
    # If date range specified, always recompute (no cache)
    if start or end:
        return compute_payload(start, end)
    now = time.time()
    if _payload_cache["data"] is None or now - _payload_cache["ts"] > CACHE_TTL:
        _payload_cache["data"] = compute_payload()
        _payload_cache["ts"]   = now
    return _payload_cache["data"]

# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/api/data")
def api_data(start: Optional[str] = Query(None), end: Optional[str] = Query(None)):
    return get_payload(start, end)

@app.get("/api/download")
def api_download(
    city:        Optional[str] = Query("Jaipur"),
    start:       Optional[str] = Query(None),
    end:         Optional[str] = Query(None),
    block:       Optional[str] = Query(None),
    batch_group: Optional[str] = Query(None),
):
    from fastapi.responses import StreamingResponse
    rows = fetch_rows()
    f = [r for r in rows if r.get("entity_city","") == city]
    if start: f = [r for r in f if r.get("part_received_at","")[:10] >= start]
    if end:   f = [r for r in f if r.get("part_received_at","")[:10] <= end]
    if block:
        f = [r for r in f if block_label(r) == block]
    if batch_group and batch_group != "overall":
        grps = detect_groups(f)
        f = [r for r in f if assign_group(r.get("batch_created_at",""), grps) == batch_group]
    out = io.StringIO()
    if f:
        writer = csv.DictWriter(out, fieldnames=list(f[0].keys()))
        writer.writeheader()
        writer.writerows(f)
    parts = [city, start or "all", end or "all"]
    if block: parts.append(block.replace(" ","_"))
    fname = "_".join(parts) + ".csv"
    return StreamingResponse(
        io.BytesIO(out.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )

@app.get("/api/refresh")
def api_refresh():
    _cache["data"] = None
    _payload_cache["data"] = None
    return {"status": "refreshed"}

class ChatRequest(BaseModel):
    messages: list
    city: str = "Jaipur"

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set")
    payload = get_payload()
    context = payload.get("context", "")
    system  = f"""You are an expert procurement analytics assistant for a spare parts procurement system.
Answer questions clearly and concisely. Use numbers from the context when available.
Be specific — cite actual percentages and counts.

{context}

Current city filter: {req.city}

Key rules:
- CI Near = Bilaspur own inventory (fast route). CI Far = other cities own inventory (added May 2026).
- PIOM = purchased fresh from external vendor in another city.
- PV = Preferred Vendor, mapped by brand or brand+category.
- Fill rate = confirmed / available. Triggered = part had this option available.
- Batch timing matters: earlier batches (8-9AM) had better CI Near rates because PV vendors inactive before 10AM.
- April had low PIOM because Mahindra volume dropped and Ford vendor mapping improved.
- May PIOM grew because LOM fill rate collapsed (82%→55%) and volume surged (Hyundai/Honda/Tata).

DOWNLOAD INSTRUCTIONS:
When the user asks to download, export, or get raw data, respond with your explanation AND include
a download link in this EXACT format on its own line (replace values appropriately):
[Download CSV](/api/download?city=CITY&start=START&end=END)
Examples:
- "download March data" → [Download CSV](/api/download?city={req.city}&start=2026-03-01&end=2026-03-31)
- "download CI Near data" → [Download CSV](/api/download?city={req.city}&start=2026-03-01&end=2026-05-31&block=CI+Near)
- "download all Chandigarh data" → [Download CSV](/api/download?city=Chandigarh&start=2026-03-01&end=2026-05-31)
Use the current city ({req.city}) unless user specifies another city.
"""
    client = Groq(api_key=GROQ_KEY)
    resp   = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1024,
        messages=[{"role": "system", "content": system}] + req.messages
    )
    return {"response": resp.choices[0].message.content}

# ── Serve static HTML ─────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
