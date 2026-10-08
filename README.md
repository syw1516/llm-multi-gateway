# LLM Multi-Model Gateway

Production-grade LLM API gateway: multi-provider routing, multi-key rotation with circuit breaker,
streaming, OpenAI-compatible + Anthropic-compatible APIs, per-key rate limiting, and observability.
Built and running in production; used to aggregate heterogeneous model backends
(NVIDIA NIM, OpenAI-compatible endpoints, third-party aggregators) behind a single OpenAI-compatible URL.

## Capabilities

- **Multi-provider routing** — one OpenAI-compatible endpoint fans out to NVIDIA NIM, OpenAI-compatible
  upstreams, and aggregator APIs; model names resolved per upstream.
- **4-key rotation + circuit breaker** — per-provider key pools with round-robin rotation; a 429 on one key
  triggers exponential backoff (1s → 2s → … up to 8s) while other keys keep serving.
- **Sliding-window rate limiter** — per-key request windows with per-key backoff state.
- **Streaming (SSE) pass-through** — chunked `/v1/chat/completions` streams relayed end-to-end with
  tool-call and reasoning deltas intact.
- **OpenAI ↔ Anthropic protocol translation** — bidirectional conversion layer so Claude-Code-style
  clients (`/v1/messages`, `tool_use`/`tool_result`, `stop_reason`) can run on top of OpenAI upstreams
  and vice versa (`claude_compatibility.py` + `e2e_tool_loop_test.py` verifies a full 2-round tool loop).
- **Observability** — structured logging, per-request timing, per-key hit labels (e.g. `Key1(nvapi-ab12cd34)`).

## Layout

| File | What it shows |
|---|---|
| `gateway_core.py` | Core FastAPI/uvicorn gateway: key rotation, rate limiter, streaming relay, provider routing, observability (~1700 lines, battle-tested) |
| `claude_compatibility.py` | OpenAI ↔ Anthropic message/tool-call translation layer (249 lines) |
| `e2e_tool_loop_test.py` | E2E test: real upstream, 2-round agentic tool loop via the Anthropic surface, asserts `tool_use` → `tool_result` → `end_turn` |
| `model_benchmark_results.json` | Real benchmark matrix across a large model catalog (champion + per-model success/status) — output of capability-testing the gateway against many upstream models |

## Quick start

```bash
# 1. Create .env (never commit real keys):
#    NVIDIA_API_KEY_1=...  NVIDIA_API_KEY_2=...  (up to 4 keys per provider)
# 2. Run:
python3 -m uvicorn gateway_core:app --host 0.0.0.0 --port 8085
# 3. Use as any OpenAI-compatible base URL:
curl http://localhost:8085/v1/chat/completions -H "Authorization: Bearer $ANY_KEY" \
  -d '{"model":"<upstream-model>","messages":[{"role":"user","content":"hi"}]}'
```

## Use case

Aggregating heterogeneous model backends (NVIDIA NIM + OpenAI-compatible + third-party) behind a single
OpenAI-compatible URL for enterprise/internal tooling, agent frameworks, and cost/rate-limit isolation.
The gateway has been running multi-month in production for personal & team workloads; ports 8082/8085/8086
instances coexist with independent key pools.

## Notes on publishing

All secrets are environment-sourced (`.env`, never committed). The `Bearer {key}` strings in the source are
template variables only. No client API keys, no customer data, no PII.
