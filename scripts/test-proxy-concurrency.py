#!/usr/bin/env python3
"""Test max parallel proxy calls to find optimal concurrency for eval speedup."""

import asyncio
import time
import os
import sys
import httpx

# Load .env with override to pick up proxy key
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# Clear auth token to avoid Bearer bug
os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

PROXY_URL = "http://localhost:3210/proxy"
MODEL = "claude-sonnet-4-20250514"
API_KEY = os.getenv("ANTHROPIC_API_KEY") or "proxy-key"
PROMPT = "Trả lời ngắn gọn: 1+1=?"
MAX_CONCURRENCY = 20


async def single_call(client: httpx.AsyncClient, idx: int) -> tuple[int, float, str]:
    """Make one streaming proxy call via raw HTTP (matching how LLMProvider works)."""
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{PROXY_URL}/v1/messages",
            headers={
                "x-api-key": API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": MODEL,
                "max_tokens": 50,
                "stream": True,
                "messages": [{"role": "user", "content": PROMPT}],
            },
            timeout=60.0,
        )
        # Collect SSE stream
        text_parts = []
        async for line in resp.aiter_lines():
            line = line.strip()
            if line.startswith("data:"):
                import json
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                    if data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_parts.append(delta.get("text", ""))
                except json.JSONDecodeError:
                    continue

        latency = time.monotonic() - start
        text = "".join(text_parts)[:40] or "(empty)"
        return idx, latency, f"OK: {text}"
    except Exception as e:
        latency = time.monotonic() - start
        return idx, latency, f"ERR: {type(e).__name__}: {e}"


async def test_concurrency(n: int) -> tuple[float, int, int, list[float]]:
    """Fire n parallel calls, return (total_time, ok_count, err_count, latencies)."""
    async with httpx.AsyncClient() as client:
        start = time.monotonic()
        results = await asyncio.gather(*[single_call(client, i) for i in range(n)])
        total = time.monotonic() - start

    ok = sum(1 for _, _, s in results if s.startswith("OK"))
    err = sum(1 for _, _, s in results if s.startswith("ERR"))
    latencies = [lat for _, lat, _ in results]

    for idx, lat, status in results:
        if status.startswith("ERR"):
            print(f"  [#{idx}] {lat:.1f}s — {status}")

    return total, ok, err, latencies


async def main():
    print(f"Proxy: {PROXY_URL}")
    print(f"Model: {MODEL}")
    print(f"API Key: {API_KEY[:10]}...")
    print(f"Testing concurrency 1 → {MAX_CONCURRENCY}\n")
    print(f"{'N':>3} | {'Total(s)':>8} | {'Avg(s)':>7} | {'Max(s)':>7} | {'OK':>3} | {'ERR':>3} | {'Throughput':>10}")
    print("-" * 65)

    # Warm up
    print("Warming up...")
    await test_concurrency(1)
    print("Warm up done.\n")

    results_summary = []
    for n in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        if n > MAX_CONCURRENCY:
            break
        total, ok, err, latencies = await test_concurrency(n)
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        throughput = ok / total if total > 0 else 0
        results_summary.append((n, total, throughput, err))
        print(f"{n:>3} | {total:>8.1f} | {avg_lat:>7.1f} | {max_lat:>7.1f} | {ok:>3} | {err:>3} | {throughput:>8.1f} q/s")

        if err > n * 0.5:
            print(f"\n⚠ >50% errors at N={n}, stopping.")
            break

    # Find sweet spot
    print("\n--- Summary ---")
    best_n, best_tp = 1, 0
    for n, total, tp, err in results_summary:
        if err == 0 and tp > best_tp:
            best_n, best_tp = n, tp
    print(f"Best concurrency (0 errors): N={best_n} ({best_tp:.1f} q/s)")

    best_n2, best_tp2 = 1, 0
    for n, total, tp, err in results_summary:
        if tp > best_tp2:
            best_n2, best_tp2 = n, tp
    if best_n2 != best_n:
        print(f"Best concurrency (with errors): N={best_n2} ({best_tp2:.1f} q/s)")


if __name__ == "__main__":
    asyncio.run(main())
