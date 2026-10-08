#!/usr/bin/env python3
import asyncio
import os
import json
import uuid
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn
import httpx
import logging
import time
from collections import defaultdict

# 滑动窗口限流器
class RateLimiter:
    def __init__(self, max_requests=20, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.backoff = defaultdict(float)  # per-key cooldown after 429

    def allow_request(self, key):
        now = time.time()
        self.requests[key] = [t for t in self.requests[key] if now - t < self.window_seconds]
        if len(self.requests[key]) >= self.max_requests:
            return False
        cooldown = self.backoff.get(key, 0)
        if cooldown > 0 and now - self.requests[key][-1] < cooldown:
            return False
        return True

    def record_request(self, key):
        self.requests[key].append(time.time())

    def mark_rate_limited(self, key):
        self.backoff[key] = min((self.backoff.get(key) or 1) * 2, 8)

    def clear_backoff(self, key):
        self.backoff[key] = 0

    def get_wait_time(self, key):
        if not self.requests.get(key):
            return 0
        last = self.requests[key][-1]
        if time.time() - last >= self.window_seconds:
            return 0
        return int(self.window_seconds - (time.time() - last)) + 1

rate_limiter = RateLimiter()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("\'\"")
load_env()

# ============================================
# 4 Key 轮询
# ============================================
def get_nvidia_keys():
    keys = []
    for i in range(1, 5):
        key = os.getenv(f"NVIDIA_API_KEY_{i}")
        if key:
            keys.append(key)
    if not keys:
        main_key = os.getenv("NVIDIA_API_KEY")
        if main_key:
            keys.append(main_key)
    return keys

NVIDIA_API_KEYS = get_nvidia_keys()
if not NVIDIA_API_KEYS:
    raise RuntimeError("缺少 NVIDIA_API_KEY")

_key_index = 0

def get_next_key():
    global _key_index
    key = NVIDIA_API_KEYS[_key_index]
    _key_index = (_key_index + 1) % len(NVIDIA_API_KEYS)
    return key

def get_key_label(key: str) -> str:
    try:
        idx = NVIDIA_API_KEYS.index(key) + 1
        if key.startswith("nvapi-"):
            suffix = key.replace("nvapi-", "")[:8]
            return f"Key{idx}({suffix})"
        return f"Key{idx}({key[:8]})"
    except ValueError:
        return key[:12]

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# AtomGit/DeepSeek 端点（atomcode 配置为 llm-api.atomgit.com，但该端点 DeepSeek 返回 403；
# api-ai.gitcode.com 是同一账号下的通用 LLM 接口，支持全部模型）
ATOMGIT_BASE_URL = os.getenv("ATOMGIT_BASE_URL", "https://api-ai.gitcode.com/v1")
_ATOMGIT_TOKENS_RAW = os.getenv("ATOMGIT_TOKENS") or os.getenv("ATOMGIT_API_KEYS") or os.getenv("ATOMGIT_API_KEY") or ""
ATOMGIT_TOKENS = [k.strip() for k in _ATOMGIT_TOKENS_RAW.split(",") if k.strip()]
_ATOMGIT_TOKEN_IDX = 0
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8082"))

# trust_env=False：环境变量里的 ALL_PROXY=socks:// 是 httpx 不支持的协议，
# 不屏蔽会在启动时直接 ValueError（与 proxy_codex.py 同款根治）
async_client = httpx.AsyncClient(
    timeout=httpx.Timeout(300.0, connect=15.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    http2=False,
    trust_env=False
)

# ============================================
# 模型映射
# ============================================
MODEL_MAP = {
    "nvidia-coder": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "gpt-4": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "gpt-4-turbo": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "gpt-4o": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "gpt-3.5-turbo": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia-test": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-sonnet-4-5": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-sonnet-4": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-opus-4-5": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-opus-4": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-haiku-4-5": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-haiku-4": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-sonnet-4-8": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-sonnet-5": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-sonnet-5-1": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-opus-5": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-haiku-5": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-3-7-sonnet": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-3-7-opus": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-opus-4-8": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-haiku-4-8": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-3-7-sonnet": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-3-7-opus": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-3-5-sonnet": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-3-5-opus": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "gpt-5.6-luna": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "claude-3-5-haiku": "nvidia/nemotron-3.5-lightning-30b-a3b",
}
# Claude Code 新 auto 模式分类器用 claude-classifier-5；上游没有这些名字，
# 兜底落到 CHAMPION_MODEL 而不是原样透传（透传 → 上游 404，auto 模式分类器挂死）
CHAMPION_MODEL = None
_rank_file = Path(__file__).parent / "agent_rank.json"
if _rank_file.exists():
    try:
        _rank = json.loads(_rank_file.read_text())
        CHAMPION_MODEL = _rank.get("champion") or None
    except Exception:
        pass
if not CHAMPION_MODEL:
    CHAMPION_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

def map_model_upstream(original_model):
    return MODEL_MAP.get(original_model) or CHAMPION_MODEL

# ============================================
# 多 Provider 路由（NVIDIA + AtomGit/DeepSeek）
# ============================================
DEEPSEEK_KEYWORDS = ("deepseek", "deep_seek", "glm", "qwen")
AGENT_MODEL_KEYWORDS = ("gpt", "coder", "code", "instruct", "reasoning", "nemotron", "kimi", "large", "claude")

def get_model_upstream_config(original_model: str):
    """返回 (base_url, api_key, provider_name, mapped_model)。
    model 命中 deepseek 关键词 → AtomGit；否则走 NVIDIA 映射。"""
    # 直接命中 DeepSeek → 走 AtomGit（atomcode 同款端点 + OAuth token）
    if any(kw in original_model.lower() for kw in DEEPSEEK_KEYWORDS):
        return ATOMGIT_BASE_URL, ATOMGIT_TOKENS, "atomgit", original_model
    # 模型映射（NVIDIA）
    mapped = MODEL_MAP.get(original_model) or CHAMPION_MODEL
    return NVIDIA_BASE_URL, NVIDIA_API_KEYS[0] if NVIDIA_API_KEYS else "", "nvidia", mapped

def get_provider_attempt_count(provider_name: str) -> int:
    if provider_name == "nvidia":
        return len(NVIDIA_API_KEYS)
    return 1

def get_provider_key(provider_name: str, keys, idx: int) -> str:
    if provider_name == "nvidia" and keys:
        global _nvidia_key_index
        key = keys[_nvidia_key_index % len(keys)]
        _nvidia_key_index = (_nvidia_key_index + 1) % len(keys)
        return key
    if provider_name == "atomgit" and keys:
        global _atomgit_key_index
        key = keys[_atomgit_key_index % len(keys)]
        _atomgit_key_index = (_atomgit_key_index + 1) % len(keys)
        return key
    return keys[idx % len(keys)] if keys else ""

_nvidia_key_index = 0
_atomgit_key_index = 0

async def call_upstream(base_url, api_key, body, is_stream, provider_name):
    """统一上游调用，处理 429/503 重试，返回 (resp, error_msg)"""
    max_attempts = 1 if provider_name != "nvidia" else len(NVIDIA_API_KEYS)
    for attempt in range(max_attempts):
        if provider_name == "nvidia":
            key = get_provider_key(provider_name, NVIDIA_API_KEYS, attempt)
        else:
            key = api_key
        key_label = f"{provider_name}_{attempt+1}" if provider_name != "nvidia" else get_key_label(key)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            req = async_client.build_request("POST", f"{base_url}/chat/completions", json=body, headers=headers)
            resp = await async_client.send(req, stream=is_stream)
            if resp.status_code == 429:
                logger.warning(f"⚠️ {key_label} 429 Rate Limit")
                await resp.aclose()
                wait = 3 if provider_name != "nvidia" else rate_limiter.get_wait_time(key)
                if wait > 0:
                    await asyncio.sleep(wait)
                continue
            if resp.status_code == 503:
                logger.warning(f"⚠️ {key_label} 503 Service Unavailable")
                await resp.aclose()
                await asyncio.sleep(3)
                continue
            return resp, None
        except Exception as e:
            logger.error(f"💥 {key_label} 异常: {str(e)}")
            continue
    return None, "all upstream keys exhausted"


# 本地分类器固定用 sonnet 档模型名（不吃 CLAUDE_CODE_AUTO_MODE_MODEL），
# 所以 "classifier" in model 永远 False——必须靠 system prompt 指纹识别。
# CLI 把完整 transcript（几万~8 万+ token）喂给分类器，任何上游在 60s 单次
# 超时内都顶不住（prefill 占满时间）→ 时好时坏超时。正解：指纹命中后
# 截断 transcript——留 开头 HEAD 条（用户任务/意图）+ 结尾 TAIL 条（最近
# 上下文，含被评估的 action），中间大块（工具调用历史）丢弃。
# 只留尾巴会把开头的用户意图丢掉——"Out-of-Place Publication" 等
# SOFT BLOCK 规则靠 transcript 里的用户意图解除，丢了就误拦。
CLASSIFIER_FINGERPRINT = "security monitor for autonomous AI coding agents"
CLASSIFIER_TRANSCRIPT_HEAD = 6      # 保留开头 entry 数（任务指令/用户意图）
CLASSIFIER_TRANSCRIPT_TAIL = 12     # 保留结尾 entry 数（含被评估的 action）
CLASSIFIER_ENTRY_TEXT_CAP = 20000   # 非 action entry 的单个 text 块上限（防单条大工具输出重新撑爆输入）
CLASSIFIER_HEAD_TEXT_CAP = 2000     # 开头 entry 的 text 上限（意图够用即可，防超长系统注入撑爆预算）
CLASSIFIER_TOTAL_BUDGET = 25000     # 截断后 transcript 总字符预算（≈6k token，确保 45s 内返回）
CLASSIFIER_TIMEOUT = 45             # 分类器请求硬超时（秒），超出则返回默认 allow，避免 CLI 60s 超时报 unavailable

def _classifier_default_response(original_model):
    """分类器超时/异常兜底：返回 allow 响应，避免 CLI 报 'unavailable'。"""
    return Response(
        json.dumps({
            "type": "message",
            "id": str(uuid.uuid4()),
            "model": original_model,
            "role": "assistant",
            "content": [{"type": "text", "text": "allowed"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
        }, ensure_ascii=False),
        media_type="application/json"
    )

def _msg_text_len(msg):
    c = msg.get("content")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        n = 0
        for b in c:
            if isinstance(b, dict):
                n += len(b.get("text", "") or "") + len(b.get("arguments", "") or "")
        return n
    return 0

def is_classifier_request(body):
    """system prompt 指纹识别（8081 同款逻辑）。system 可能是顶层字符串、
    block 列表，或 messages 里的 role=system；拼接后再匹配，避免
    cache_control 拆块导致指纹跨 block 边界漏判。"""
    parts = []
    sys_field = body.get("system")
    if isinstance(sys_field, str):
        parts.append(sys_field)
    elif isinstance(sys_field, list):
        for blk in sys_field:
            if isinstance(blk, dict):
                parts.append(str(blk.get("text", "")))
    for msg in body.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict):
                        parts.append(str(blk.get("text", "")))
    return CLASSIFIER_FINGERPRINT in "".join(parts)

def _cap_entry_text(entry):
    """截断 transcript entry（segmented 的消息 或 single-msg 的 block）里的超长文本。"""
    if not isinstance(entry, dict):
        return
    target = entry.get("content")
    if target is None:
        if isinstance(entry.get("text"), str) and len(entry["text"]) > CLASSIFIER_ENTRY_TEXT_CAP:
            entry["text"] = entry["text"][:CLASSIFIER_ENTRY_TEXT_CAP] + "\n...[truncated]..."
        return
    if isinstance(target, str):
        if len(target) > CLASSIFIER_ENTRY_TEXT_CAP:
            entry["content"] = target[:CLASSIFIER_ENTRY_TEXT_CAP] + "\n...[truncated]..."
    elif isinstance(target, list):
        for blk in target:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str) and len(blk["text"]) > CLASSIFIER_ENTRY_TEXT_CAP:
                blk["text"] = blk["text"][:CLASSIFIER_ENTRY_TEXT_CAP] + "\n...[truncated]..."

def _entry_chars(e):
    """entry 的字符量（block 或消息，content 可为 str 或 block 列表）。"""
    if not isinstance(e, dict):
        return 0
    if isinstance(e.get("text"), str):
        return len(e["text"])
    c = e.get("content")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        n = 0
        for b in c:
            if isinstance(b, dict):
                n += len(b.get("text", "") or "") + len(b.get("arguments", "") or "")
    return n

def _cap_entry(e, cap):
    """把 entry 里的 text 块截头到 cap（block 或消息都支持）。"""
    if not isinstance(e, dict):
        return
    if isinstance(e.get("text"), str) and len(e["text"]) > cap:
        e["text"] = e["text"][:cap] + "\n...[truncated]..."
        return
    c = e.get("content")
    if isinstance(c, str):
        if len(c) > cap:
            e["content"] = c[:cap] + "\n...[truncated]..."
    elif isinstance(c, list):
        for blk in c:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str) and len(blk["text"]) > cap:
                blk["text"] = blk["text"][:cap] + "\n...[truncated]..."

def truncate_classifier_transcript(body):
    """截断分类器 transcript：留 开头 HEAD 条 + 结尾 TAIL 条，丢中间。

    CLI 有两种结构（zCo 构造）：
      - segmented（CLAUDE_CODE_AUTO_MODE_SEGMENTED_TRANSCRIPT=true，本机）：
        每个 entry 是独立 user 消息，<transcript>/</transcript> 标记各自单独成消息
      - 非 segmented：所有 entry 在同一条 user 消息里，标记是 content 里的 block
    保留规则：
      - 开头 HEAD 条（用户任务/意图）：text 截头到 HEAD_CAP。
        SOFT BLOCK 规则（如 Out-of-Place Publication）靠 transcript 里看到的
        用户意图解除——只留尾巴会把开头的任务指令丢掉 → 误拦 git push。
      - 结尾 TAIL 条（最近上下文）：text 截头到 CAP；最后一条是被评估的
        action，完整保留不截。
      - 总字符超 BUDGET 时从开头往末尾砍（tail 永不砍）。
    <transcript> 前的消息与 </transcript> 之后的内容（Forwarded user turns 等）
    原样保留。"""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    open_i = close_i = -1
    for i, m in enumerate(msgs):
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for blk in c:
            if not isinstance(blk, dict):
                continue
            t = str(blk.get("text", "")).strip()
            if t == "<transcript>" and open_i < 0:
                open_i = i
            elif t == "</transcript>":
                close_i = i
    if open_i < 0 or close_i < open_i:
        return False  # 非预期结构（CLI 改版？），不动
    before_chars = sum(_msg_text_len(m) for m in msgs)
    if close_i > open_i:
        # segmented：entry 是两个标记消息之间的消息
        entries = msgs[open_i + 1:close_i]
        reassemble = lambda kept: (msgs[:open_i + 1] + kept + msgs[close_i:])
        kind = "segmented"
    else:
        # 非 segmented：entry 是同一条消息里两个标记 block 之间的 block
        content = msgs[open_i]["content"]
        ob = cb = -1
        for j, blk in enumerate(content):
            if not isinstance(blk, dict):
                continue
            t = str(blk.get("text", "")).strip()
            if t == "<transcript>" and ob < 0:
                ob = j
            elif t == "</transcript>":
                cb = j
        if cb <= ob:
            return False
        entries = content[ob + 1:cb]
        reassemble = lambda kept: (content[:ob + 1] + kept + content[cb:])
        kind = "single-msg"
    n = len(entries)
    if n <= CLASSIFIER_TRANSCRIPT_HEAD + CLASSIFIER_TRANSCRIPT_TAIL:
        return False
    head, tail = entries[:CLASSIFIER_TRANSCRIPT_HEAD], entries[-CLASSIFIER_TRANSCRIPT_TAIL:]
    for e in head:
        _cap_entry(e, CLASSIFIER_HEAD_TEXT_CAP)
    # 预算分配：head 占用后剩余均分给 tail 的每条（action 条不截，永不砍 tail）；
    # 不能先按大 CAP 截尾再回头砍 head——尾部大工具输出会把 head 全挤掉，
    # 意图丢失 → SOFT BLOCK 误拦（如 git push 被判 Out-of-Place Publication）
    head_chars = sum(_entry_chars(e) for e in head)
    tail_cap = min(CLASSIFIER_ENTRY_TEXT_CAP,
                   max((CLASSIFIER_TOTAL_BUDGET - head_chars) // max(len(tail), 1), 500))
    for e in tail[:-1]:
        _cap_entry(e, tail_cap)
    kept = head + tail
    # 极端情况（action 本身超预算）：仍从 head 砍（tail 含 action，永不砍）
    while sum(_entry_chars(e) for e in kept) > CLASSIFIER_TOTAL_BUDGET and len(head) > 0:
        head.pop(0)
        kept = head + tail
    if kind == "segmented":
        body["messages"] = reassemble(kept)
    else:
        msgs[open_i]["content"] = reassemble(kept)
    after_chars = sum(_msg_text_len(m) for m in body["messages"])
    logger.info(f"[classifier] transcript 截断 ({kind}): {n}→{len(kept)} 条 entry (头{len(head)}+尾{len(tail)}), "
                f"{before_chars}→{after_chars} chars")
    return True

def apply_classifier_handling(body):
    """分类器指纹命中 → 立即截断 body["messages"]。
    必须在端点把 body 转成 openai 消息（扁平化/转换）之前调用，
    否则截断不生效。重复调用幂等（已截断的会直接 return）。"""
    if is_classifier_request(body):
        truncate_classifier_transcript(body)
        return True
    return False

def resolve_classifier(body, original_model):
    """指纹命中：截尾 transcript + 返回 True（调用方需设 reasoning_effort=none）。"""
    if is_classifier_request(body):
        truncate_classifier_transcript(body)
        logger.info(f"[classifier] 指纹命中：model={original_model}，截尾+关推理（防 CLI 60s 超时）")
        return True
    return False

app = FastAPI(title="NVIDIA Proxy")

# ============================================
# 工具函数
# ============================================
def extract_content_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text += item.get("text", "")
                elif "content" in item:
                    text += extract_content_text(item.get("content", ""))
            elif isinstance(item, str):
                text += item
        return text
    elif isinstance(content, dict):
        return content.get("text", "") or content.get("content", "")
    return str(content) if content else ""

def normalize_role(role):
    if role == "developer":
        return "system"
    return role

# ============================================
# Anthropic 协议转换层（从 8081 proxy_codex.py 移植，上游为 NVIDIA OpenAI 格式）
# 旧 /v1/messages 直接透传 OpenAI SSE、非流式响应缺 stop_reason、不转发 tools
# → Claude Code 解析不了流/工具调用全失败（claude cli 切 8082 不能用的根因）
# ============================================
IDLE_TIMEOUT = 120           # 流式: 单行之间最长等待秒数（累计空闲时长）
STREAM_TOTAL_TIMEOUT = 600   # 流式: 整个流式过程最长总时长
POLL_INTERVAL = 5            # 等下一行数据时，每次最多等这么久就回来检查一次 ping/超时
KEEPALIVE_INTERVAL = 10      # 流式等待期间无数据时，向下游发 keep-alive 信号的最小间隔（秒）

def _sse_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

class UpstreamStreamTimeout(Exception):
    """上游流空闲超时或总时长超限（流式读取被 poll_aiter 掐断时抛出）"""
    pass

async def poll_aiter(aiter, keepalive=None):
    """短轮询迭代器：给 aiter 加上空闲/总时长保护，防止上游"开了头突然哑火"卡死。
    复用 8081 已验证的机制（POLL_INTERVAL 短轮询 + 空闲/总超时），
    等待期间可经哨兵向下游产出 keepalive 项（SSE ping，客户端会忽略）。"""
    start_time = time.monotonic()
    item_count = 0
    idle_since_data = 0.0
    last_alive_time = start_time
    pending = None
    try:
        while True:
            if time.monotonic() - start_time > STREAM_TOTAL_TIMEOUT:
                raise UpstreamStreamTimeout(
                    f"upstream stream exceeded total timeout of {STREAM_TOTAL_TIMEOUT}s ({item_count} items)")
            if idle_since_data > IDLE_TIMEOUT:
                raise UpstreamStreamTimeout(
                    f"no data from upstream for {IDLE_TIMEOUT}s (stalled stream, {item_count} items so far)")
            if pending is None:
                pending = asyncio.ensure_future(aiter.__anext__())
            try:
                item = await asyncio.wait_for(asyncio.shield(pending), timeout=POLL_INTERVAL)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                idle_since_data += POLL_INTERVAL
                if keepalive is not None and time.monotonic() - last_alive_time >= KEEPALIVE_INTERVAL:
                    last_alive_time = time.monotonic()
                    yield keepalive
                continue
            pending = None
            idle_since_data = 0.0
            item_count += 1
            yield item
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except BaseException:
                pass

def _json_complete(s: str) -> bool:
    """判断字符串是否为完整合法 JSON（用于检测工具调用参数是否被截断）。空串视为零参数 {}。"""
    if not s or not s.strip():
        return True
    try:
        json.loads(s)
        return True
    except Exception:
        return False

def normalize_tools(tools):
    """自动转换各种 tools 格式为 OpenAI 标准格式，并过滤掉无效/占位工具（name 为空）"""
    if not tools:
        return None
    converted = []
    skipped = 0
    for t in tools:
        if isinstance(t, dict):
            if "function" in t and "type" in t:
                name = t.get("function", {}).get("name")
                if not name:
                    skipped += 1
                    continue
                converted.append(t)
            else:
                name = t.get("name") or t.get("function", {}).get("name")
                if not name:
                    skipped += 1
                    continue
                desc = t.get("description") or t.get("function", {}).get("description", "")
                # Claude 格式工具把参数 schema 放在 input_schema（OpenAI 格式才是 parameters）
                params = (t.get("parameters")
                          or t.get("input_schema")
                          or t.get("function", {}).get("parameters")
                          or {})
                converted.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": params
                    }
                })
    if skipped:
        logger.info(f"[Tools] 已跳过 {skipped} 个无效工具定义（name 为空）")
    return converted

def convert_claude_messages_to_openai(body):
    """Claude (Anthropic) 格式 → OpenAI chat messages，保留工具循环上下文：
      - assistant 消息里的 tool_use block → {role: assistant, tool_calls: [...]}
      - user 消息里的 tool_result block → 逐条展开为 {role: tool, tool_call_id} 消息
      - 多个 text block 拼接；thinking 等无关 block 丢弃
    """
    messages = []
    system_text = ""
    system = body.get("system")
    if isinstance(system, str):
        system_text = system
    elif isinstance(system, list):
        system_text = "\n".join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")
    elif isinstance(system, dict):
        system_text = system.get("text", "")

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("role"))
        content = msg.get("content", "")

        # 非 list content：按普通文本处理（str 直接用；dict/None 交给 extract）
        if not isinstance(content, list):
            text = extract_content_text(content)
            if role == "system":
                if text:
                    system_text = (system_text + "\n" + text).strip()
            elif role in ("user", "assistant") and text:
                messages.append({"role": role, "content": text})
            continue

        text_parts = []
        tool_calls = []
        tool_results = []
        for block in content:
            if not isinstance(block, dict):
                if block:
                    text_parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                if block.get("text"):
                    text_parts.append(block["text"])
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}) or {}, ensure_ascii=False)
                    }
                })
            elif btype == "tool_result":
                tc_content = block.get("content", "")
                if isinstance(tc_content, list):
                    tc_content = "\n".join(
                        b.get("text", "") for b in tc_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(tc_content, dict):
                    tc_content = json.dumps(tc_content, ensure_ascii=False)
                else:
                    tc_content = str(tc_content or "")
                if block.get("is_error"):
                    tc_content = f"[tool error] {tc_content}"
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": tc_content
                })
            # 其余 block（thinking / redacted_thinking 等）丢弃

        if role == "system":
            if text_parts:
                system_text = (system_text + "\n" + "\n".join(text_parts)).strip()
        elif role == "assistant":
            if tool_calls:
                out = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
                out["tool_calls"] = tool_calls
                messages.append(out)
            elif text_parts:
                messages.append({"role": "assistant", "content": "\n".join(text_parts)})
        else:
            # user（及兜底）：tool_result 先展开（tool 消息需紧随上一条 assistant 的
            # tool_calls，中间插 user 文本会破坏关联），再放普通文本
            messages.extend(tool_results)
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})

    if system_text:
        messages.insert(0, {"role": "system", "content": system_text})
    return messages

def parse_tool_arguments(args):
    """tool_call arguments → dict（非流式响应转 Claude tool_use.input 用）。
    空/缺失 → {}；dict 直接返回；截断/非法 JSON → 记 warning 并返回 {}，
    不抛异常（参数残缺 ≠ API key 额度失败，不能触发换 key 重试）。"""
    if not args:
        return {}
    if isinstance(args, dict):
        return args
    try:
        v = json.loads(args)
        return v if isinstance(v, dict) else {"value": v}
    except Exception:
        logger.warning(f"tool arguments 截断/非法，降级为空对象: {str(args)[:120]!r}")
        return {}

async def openai_stream_to_anthropic(nvidia_resp, model: str) -> "async generator":
    """把上游(OpenAI 格式) SSE 流实时转成 Claude(Anthropic) 事件流。
    处理: content → content_block_delta; tool_calls 增量 → input_json_delta;
    choices 为空的 usage/心跳 chunk; reasoning_content(丢弃); [DONE]; 空闲/总超时。
    出错(上游错误帧/超时/截断)只发 error 事件就结束，不发 content_block_stop/
    message_delta/message_stop，让客户端识别失败并自行重试。"""
    msg_id = f"msg_{uuid.uuid4().hex[:16]}"
    text_index = 0
    next_index = 1
    active_tools = {}       # 上游 tool index -> {"id","name"}
    tool_anthropic_index = {}  # 上游 tool index -> anthropic block index
    error_message = None
    finish_reason_seen = None
    line_count = 0

    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    text_started = False
    stop_reason = "end_turn"

    # poll_aiter（shield + 分片 wait_for）：裸 wait_for 超时会 cancel 掉
    # 正在进行的 __anext__()，可能破坏 httpx 底层流状态（"说完就断"病根）；
    # poll_aiter 只分片等待、不 cancel 读取。空闲期间经哨兵每 10s 发 ping。
    PING_SENTINEL = "\x00__PING__\x00"

    try:
        async for line in poll_aiter(nvidia_resp.aiter_lines(), keepalive=PING_SENTINEL):
            if line == PING_SENTINEL:
                yield _sse_event("ping", {"type": "ping"})
                continue

            line_count += 1
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue

            # 暴露上游 SSE 错误帧：流中途推 error 时立即记录并中断，后续只发 error 事件
            upstream_err = chunk.get("error")
            if upstream_err is not None or chunk.get("type") == "error":
                if isinstance(upstream_err, dict):
                    err_detail = upstream_err.get("message") or json.dumps(upstream_err, ensure_ascii=False)
                elif isinstance(upstream_err, str):
                    err_detail = upstream_err
                else:
                    err_detail = json.dumps(chunk, ensure_ascii=False)[:500]
                error_message = f"upstream SSE error frame: {err_detail}"
                logger.error(f"=== 上游流内推送错误帧({line_count} 行): {error_message} ===")
                break

            choices = chunk.get("choices") or []
            if not choices:
                # usage / 心跳 chunk，无 choices，跳过
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            # 普通文本
            if delta.get("content"):
                if not text_started:
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start", "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    text_started = True
                yield _sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": text_index,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                })

            # 工具调用（arguments 为增量片段，逐块转发为 input_json_delta）
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                if idx not in active_tools:
                    tool_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
                    func = tc.get("function") or {}
                    active_tools[idx] = {"id": tool_id, "name": func.get("name", ""), "args": ""}
                    tool_anthropic_index[idx] = next_index
                    next_index += 1
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start", "index": tool_anthropic_index[idx],
                        "content_block": {"type": "tool_use", "id": tool_id, "name": active_tools[idx]["name"], "input": {}},
                    })
                args = (tc.get("function") or {}).get("arguments")
                if args:
                    active_tools[idx]["args"] += args
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta", "index": tool_anthropic_index[idx],
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    })

            # finish_reason 必须在 delta 处理之后判断：上游常把最后一个
            # tool_calls delta 和 finish_reason 放在同一个 chunk
            if choice.get("finish_reason"):
                finish_reason_seen = choice["finish_reason"]
                # 截断(length)必须报 max_tokens
                fr = choice["finish_reason"]
                if fr in ("length", "max_tokens"):
                    stop_reason = "max_tokens"
                elif active_tools:
                    stop_reason = "tool_use"
                else:
                    stop_reason = "end_turn"
    except UpstreamStreamTimeout as e:
        logger.error(f"=== 上游流超时/哑火被掐断: {e} ===")
        error_message = str(e)
    except Exception as e:
        logger.error(f"流式读取错误: {e!r}")
        error_message = f"connection to upstream failed or timed out: {e}"
    finally:
        try:
            await nvidia_resp.aclose()
        except Exception:
            pass

    # 检测"静默截断"：上游中途断流（有 tool_use 但无 finish_reason）或空流（无任何内容）。
    # 不把残缺的 tool_use 伪装成成功完成，发 error 让 Claude Code 触发自己的重试。
    if not error_message:
        if active_tools:
            accepted_tool_finish = {"tool_calls", "tool_use", "function_call", "stop", "end_turn", "length"}
            args_all_valid = all(_json_complete(t.get("args") or "") for t in active_tools.values())
            if finish_reason_seen is None and args_all_valid:
                logger.warning(f"=== 流结束带工具调用但无 finish_reason，参数 JSON 完整，按正常完成处理 ===")
                stop_reason = "tool_use"  # 必须补上，否则客户端不会去执行工具
            elif finish_reason_seen not in accepted_tool_finish:
                error_message = (f"upstream closed stream mid tool-call without valid finish_reason "
                                 f"({line_count} lines, finish_reason={finish_reason_seen}, args_complete={args_all_valid})")
        elif not text_started and not active_tools:
            error_message = f"upstream closed stream with no content ({line_count} lines)"
        if error_message:
            logger.error(f"=== abnormal stream end: {error_message} ===")

    if error_message:
        logger.error(f"=== 流式错误, 发送 error 事件: {error_message} ===")
        yield _sse_event("error", {"type": "error", "error": {"type": "api_error", "message": error_message}})
        return

    if text_started:
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": text_index})
    for anthropic_idx in tool_anthropic_index.values():
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": anthropic_idx})

    logger.info(f"=== 流完成: {line_count} 行, finish_reason={finish_reason_seen}, "
                f"stop_reason={stop_reason}, text={text_started}, tools={len(active_tools)} ===")

    yield _sse_event("message_delta", {
        "type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 0},
    })
    yield _sse_event("message_stop", {"type": "message_stop"})

# ============================================
# Responses API 流式转换器
# ============================================
async def convert_nvidia_stream_to_responses(stream_generator, model_name: str):
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    item_id = f"item_{uuid.uuid4().hex[:8]}"
    full_text = ""
    
    yield f'data: {json.dumps({"type": "response.created", "response": {"id": response_id, "status": "in_progress"}})}\n\n'
    
    yield f'data: {json.dumps({
        "type": "response.output_item.added",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "content": []
        }
    })}\n\n'
    
    try:
        async for line in stream_generator:
            if not line.startswith("data: "):
                continue
                
            data_str = line[6:]
            if data_str == "[DONE]":
                continue
                
            try:
                chunk = json.loads(data_str)
                
                if chunk.get("choices"):
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    
                    content = delta.get("content", "")
                    if not content:
                        content = delta.get("reasoning", "")
                    
                    if content:
                        full_text += content
                        yield f'data: {json.dumps({
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": content
                        })}\n\n'
                        
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        logger.error(f"流式转换错误: {e}")
        yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'
    
    yield f'data: {json.dumps({
        "type": "response.output_text.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": full_text
    })}\n\n'
    
    yield f'data: {json.dumps({
        "type": "response.output_item.done",
        "response_id": response_id,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "role": "assistant"
        }
    })}\n\n'
    
    yield f'data: {json.dumps({
        "type": "response.completed",
        "response": {
            "id": response_id,
            "status": "completed"
        }
    })}\n\n'
    
    yield "data: [DONE]\n\n"

# ============================================
# /v1/responses 端点
# ============================================
@app.api_route("/v1/responses", methods=["POST"])
async def codex_responses(request: Request):
    body = await request.json()
    is_stream = body.get("stream", False)
    
    logger.info(f"[Responses] stream={is_stream}")
    
    messages = []
    
    if body.get("instructions"):
        instructions = body.get("instructions")
        if isinstance(instructions, str):
            messages.append({"role": "system", "content": instructions})
        elif isinstance(instructions, dict):
            content = extract_content_text(instructions.get("content", ""))
            if content:
                messages.append({"role": "system", "content": content})
    
    if "input" in body:
        input_data = body["input"]
        if isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, dict):
                    role = normalize_role(item.get("role", "user"))
                    content = item.get("content", "")
                    content = extract_content_text(content)
                    if role and content:
                        messages.append({"role": role, "content": content})
                elif isinstance(item, str):
                    messages.append({"role": "user", "content": item})
    
    if not messages and "messages" in body:
        for msg in body["messages"]:
            if isinstance(msg, dict):
                role = normalize_role(msg.get("role", "user"))
                content = msg.get("content", "")
                content = extract_content_text(content)
                if role and content:
                    messages.append({"role": role, "content": content})
    
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    
    original_model = body.get("model", "nvidia-coder")
    base_url, api_key, provider_name, mapped_model = get_model_upstream_config(original_model)

    logger.info(f"[Responses/{provider_name}] {original_model} -> {mapped_model}")
    logger.info(f"[Responses] messages: {json.dumps(messages, ensure_ascii=False)[:200]}")

    nvidia_body = {
        "model": mapped_model,
        "messages": messages,
        "stream": is_stream,
        "max_tokens": body.get("max_tokens", body.get("max_output_tokens", 4096)),
        "temperature": body.get("temperature", 1.0),
    }

    if "top_p" in body:
        nvidia_body["top_p"] = body["top_p"]

    upstream_keys = NVIDIA_API_KEYS if provider_name == "nvidia" else (api_key if isinstance(api_key, list) else [api_key] if api_key else [])
    max_attempts = len(upstream_keys) if upstream_keys else 1

    for attempt in range(max_attempts):
        key = get_provider_key(provider_name, upstream_keys, attempt)
        key_label = get_key_label(key) if provider_name == "nvidia" else provider_name

        if not rate_limiter.allow_request(key):
            logger.info(f"⏳ {key_label} 本地限流，跳过")
            continue

        if attempt > 0:
            logger.info(f"🔄 重试 {attempt + 1}/{max_attempts}: {key_label}")
        else:
            logger.info(f"🔑 使用 {key_label}")

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        try:
            req = async_client.build_request(
                "POST",
                f"{base_url}/chat/completions",
                json=nvidia_body,
                headers=headers
            )

            resp = await async_client.send(req, stream=is_stream)

            if resp.status_code == 429:
                logger.warning(f"⚠️ {key_label} 429 Rate Limit")
                if provider_name == "nvidia":
                    rate_limiter.mark_rate_limited(key)
                await resp.aclose()
                wait = rate_limiter.get_wait_time(key) if provider_name == "nvidia" else 3
                if wait > 0:
                    logger.info(f"⏳ 等待 {wait}s 后重试 {key_label}")
                    await asyncio.sleep(wait)
                continue

            if resp.status_code != 200:
                error_text = await resp.aread()
                error_msg = error_text.decode('utf-8', errors='ignore')
                logger.error(f"{provider_name}错误: {resp.status_code} - {error_msg}")
                await resp.aclose()
                return Response(
                    json.dumps({"error": {"message": error_msg}}),
                    status_code=resp.status_code,
                    media_type="application/json"
                )

            if provider_name == "nvidia":
                rate_limiter.clear_backoff(key)
                rate_limiter.record_request(key)

            if is_stream:
                async def stream_gen():
                    try:
                        async for line in resp.aiter_lines():
                            if line:
                                yield line
                    except Exception as e:
                        logger.error(f"流式读取错误: {e}")
                        yield f'data: {{"error": "{str(e)}"}}\n\n'
                    finally:
                        await resp.aclose()

                return StreamingResponse(
                    convert_nvidia_stream_to_responses(
                        stream_gen(),
                        original_model
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                data = await resp.aread()
                data = json.loads(data)
                await resp.aclose()

                content = ""
                if data.get("choices"):
                    for choice in data["choices"]:
                        message = choice.get("message", {})
                        content = message.get("content") or message.get("reasoning") or message.get("reasoning_content") or ""

                return Response(
                    json.dumps({
                        "id": data.get("id", f"resp_{uuid.uuid4().hex[:12]}"),
                        "object": "response",
                        "model": original_model,
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": content}
                                ]
                            }
                        ],
                        "usage": data.get("usage", {})
                    }),
                    media_type="application/json"
                )
                
        except Exception as e:
            logger.error(f"💥 {key_label} 异常: {str(e)}")
            continue
    
    return Response(
        json.dumps({"error": {"message": "All API keys rate limited"}}),
        status_code=429,
        media_type="application/json"
    )

# ============================================
# /v1/chat/completions
# ============================================
@app.api_route("/v1/chat/completions", methods=["POST"])
async def codex_chat_completions(request: Request):
    body = await request.json()
    
    original_model = body.get("model", "nvidia-coder")
    base_url, api_key, provider_name, mapped_model = get_model_upstream_config(original_model)
    is_stream = body.get("stream", False)

    messages = []
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        msg_copy = dict(msg)
        msg_copy["role"] = normalize_role(msg_copy.get("role", "user"))
        messages.append(msg_copy)

    nvidia_body = {
        "model": mapped_model,
        "messages": messages,
        "stream": is_stream,
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 1.0),
    }

    passthrough_fields = [
        "top_p", "tools", "tool_choice", "parallel_tool_calls",
        "response_format", "stop", "seed", "presence_penalty",
        "frequency_penalty", "logprobs", "top_logprobs", "user", "n",
    ]
    for field in passthrough_fields:
        if field in body:
            nvidia_body[field] = body[field]

    logger.info(f"[{provider_name}/chat] {original_model} -> {mapped_model}")

    upstream_keys = NVIDIA_API_KEYS if provider_name == "nvidia" else (api_key if isinstance(api_key, list) else [api_key] if api_key else [])
    max_attempts = len(upstream_keys) if upstream_keys else 1

    for attempt in range(max_attempts):
        key = get_provider_key(provider_name, upstream_keys, attempt)
        key_label = get_key_label(key) if provider_name == "nvidia" else provider_name
        if attempt > 0:
            logger.info(f"🔄 [{provider_name}/chat] 重试 {attempt + 1}/{max_attempts}: {key_label}")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            req = async_client.build_request("POST", f"{base_url}/chat/completions", json=nvidia_body, headers=headers)
            resp = await async_client.send(req, stream=is_stream)
            if resp.status_code == 429:
                logger.warning(f"⚠️ {key_label} 429 Rate Limit")
                if provider_name == "nvidia":
                    rate_limiter.mark_rate_limited(key)
                await resp.aclose()
                wait = rate_limiter.get_wait_time(key) if provider_name == "nvidia" else 3
                if wait > 0:
                    await asyncio.sleep(wait)
                continue
            if resp.status_code == 503:
                logger.warning(f"⚠️ {key_label} 503 Service Unavailable")
                await resp.aclose()
                await asyncio.sleep(3)
                continue
            if resp.status_code != 200:
                error_msg = (await resp.aread()).decode('utf-8', errors='ignore')
                logger.error(f"{provider_name}错误: {resp.status_code} - {error_msg}")
                await resp.aclose()
                return Response(json.dumps({"error": {"message": error_msg}}), status_code=resp.status_code, media_type="application/json")
            if provider_name == "nvidia":
                rate_limiter.clear_backoff(key)
                rate_limiter.record_request(key)
            if is_stream:
                if provider_name == "atomgit":
                    # gitcode SSE 流在 finish_reason 帧之后还会跟一帧
                    # choices=[{delta:{}}] + usage（无 finish_reason），
                    # pi-ai 严格解析器会报 "Stream ended without finish_reason"。
                    # 帧级修补：跳过无 finish_reason 且无有效 delta 的尾帧。
                    async def atomgit_stream_gen():
                        buf = b""
                        async for raw_chunk in resp.aiter_bytes():
                            buf += raw_chunk
                            # 按 SSE 事件（\n\n 分隔）切分，保留不完整的尾部
                            parts = buf.split(b"\n\n")
                            buf = parts[-1]
                            for part in parts[:-1]:
                                if part.startswith(b"data:"):
                                    payload = part[len(b"data:"):].strip()
                                    if payload == b"[DONE]":
                                        yield part + b"\n\n"
                                        continue
                                    try:
                                        chunk_obj = json.loads(payload)
                                        choices = chunk_obj.get("choices") or []
                                        has_finish = any(
                                            c.get("finish_reason") for c in choices
                                        )
                                        has_content = any(
                                            c.get("delta") for c in choices
                                        )
                                        if not has_finish and not has_content:
                                            # usage 尾帧 / 心跳帧：丢弃
                                            continue
                                    except Exception:
                                        pass  # 无法解析就原样透传
                                yield part + b"\n\n"
                        # 冲刷尾部（含最后的 [DONE] 事件）
                        if buf:
                            yield buf
                        await resp.aclose()
                    return StreamingResponse(atomgit_stream_gen(), media_type="text/event-stream")
                async def chat_stream_gen():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    except Exception as e:
                        logger.error(f"流式读取错误: {e}")
                    finally:
                        await resp.aclose()
                return StreamingResponse(chat_stream_gen(), media_type="text/event-stream")
            data = await resp.aread()
            data = json.loads(data)
            await resp.aclose()

            # ========================================================
            # NVIDIA -> OpenAI tool_calls compatibility
            # NVIDIA Nemotron 当前会把工具调用放在 message.content
            # 中，格式类似：
            # [[{"name":"test_tool","parameters":{"command":"HELLO"}}]]
            #
            # OpenHands/LiteLLM 需要标准 OpenAI:
            # message.tool_calls[]
            # ========================================================
            if data.get("choices"):
                for choice in data["choices"]:
                    message = choice.get("message", {})

                    raw_content = message.get("content")

                    if isinstance(raw_content, str):
                        try:
                            # NVIDIA 有时返回近似 JSON：
                            # [[{"name":"test_tool","parameters":{"command":"HELLO"}}]
                            # 外层可能少一个 ]，因此先正常解析，
                            # 失败后再从 content 中提取工具对象。
                            parsed = None

                            try:
                                parsed = json.loads(raw_content.strip())
                            except json.JSONDecodeError:
                                decoder = json.JSONDecoder()
                                pos = raw_content.find("{")

                                if pos >= 0:
                                    candidate, _ = decoder.raw_decode(
                                        raw_content[pos:]
                                    )

                                    if (
                                        isinstance(candidate, dict)
                                        and candidate.get("name")
                                        and "parameters" in candidate
                                    ):
                                        parsed = [candidate]

                            # NVIDIA 工具调用兼容：
                            # [{"name":"test_tool","parameters":{...}}]
                            # [[{"name":"test_tool","parameters":{...}}]]
                            tool_items = []

                            if isinstance(parsed, list):
                                for item in parsed:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("name")
                                        and "parameters" in item
                                    ):
                                        tool_items.append(item)

                                    elif isinstance(item, list):
                                        for sub_item in item:
                                            if (
                                                isinstance(sub_item, dict)
                                                and sub_item.get("name")
                                                and "parameters" in sub_item
                                            ):
                                                tool_items.append(sub_item)

                            if tool_items:
                                converted_tools = []

                                for item in tool_items:
                                    converted_tools.append({
                                        "id": f"call_{uuid.uuid4().hex[:24]}",
                                        "type": "function",
                                        "function": {
                                            "name": item["name"],
                                            "arguments": json.dumps(
                                                item.get("parameters", {}),
                                                ensure_ascii=False,
                                                separators=(",", ":")
                                            )
                                        }
                                    })

                                message["tool_calls"] = converted_tools
                                message["content"] = None

                                logger.info(
                                    f"🔧 [tool-call] NVIDIA特殊格式 -> OpenAI tool_calls: "
                                    f"{[x['function']['name'] for x in converted_tools]}"
                                )

                        except (json.JSONDecodeError, TypeError, ValueError) as e:
                            logger.warning(
                                f"⚠️ [tool-call] NVIDIA content JSON解析失败: {e}"
                            )

                    # 原有 reasoning fallback
                    if (
                        not message.get("content")
                        and not message.get("tool_calls")
                        and message.get("reasoning")
                    ):
                        message["content"] = message["reasoning"]

            return Response(json.dumps(data), media_type="application/json")

        except Exception as e:
            logger.error(f"💥 {key_label} 异常: {str(e)}")
            continue

    return Response(
        json.dumps({"error": {"message": "All API keys rate limited"}}),
        status_code=429,
        media_type="application/json"
    )

# ============================================
# /v1/models
# ============================================
# 上游真实模型列表缓存（/v1/models 动态获取，失败时用上次成功结果兜底）
_upstream_models_cache = []  # [{"id":..., "owned_by":...}, ...]

# coding-plan 专属模型：chat 可用但上游 /v1/models 不返回，需手动补充展示
EXTRA_MODELS = ["glm5.3-flash", "qwen3.8-27b"]

async def _fetch_upstream_models(base_url: str, api_key: str, owned_by: str):
    """从上游 /v1/models 拉取真实模型 ID。"""
    try:
        resp = await async_client.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", []) if m.get("id")]
        logger.info(f"拉取上游模型列表成功 ({owned_by}, token={api_key[:6]}***): {len(ids)} 个")
        return [{"id": i, "owned_by": owned_by} for i in ids]
    except Exception as e:
        logger.warning(f"拉取上游模型列表失败 ({owned_by}): {e}")
        return []

@app.get("/v1/models")
async def proxy_models():
    global _upstream_models_cache
    atomgit_key = ATOMGIT_TOKENS[0] if ATOMGIT_TOKENS else ""
    # 只展示 AtomGit 上游真实模型名（NVIDIA 侧不对外展示）
    fetched = await _fetch_upstream_models(ATOMGIT_BASE_URL, atomgit_key, "atomgit")
    # coding-plan 专属模型（glm/qwen3.8 等）不在上游 /v1/models 列表里，但 chat 可用，
    # 实测可用后合并进来，避免从列表里消失
    fetched_ids = {m["id"] for m in fetched}
    for extra in EXTRA_MODELS:
        if extra not in fetched_ids:
            fetched.append({"id": extra, "owned_by": "atomgit"})
    if fetched:
        _upstream_models_cache = fetched
    models = []
    for m in _upstream_models_cache:
        models.append({
            "id": m["id"],
            "object": "model",
            "owned_by": m["owned_by"],
            "created": 1700000000,
            "context_window": 128000,
            "max_context_tokens": 128000,
            "max_output_tokens": 16384,
            "capabilities": {
                "chat_completions": True,
                "responses": True,
                "streaming": True,
                "reasoning": True
            }
        })
    return {"object": "list", "data": models}

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    # 直通模式：任何 model_id 都视为合法（路由按关键词分流，未命中走 NVIDIA 兜底）
    owned_by = "openai"
    for m in _upstream_models_cache:
        if m["id"] == model_id:
            owned_by = m["owned_by"]
            break
    return {
        "id": model_id,
        "object": "model",
        "owned_by": owned_by,
        "created": 1700000000,
        "context_window": 128000,
        "max_context_tokens": 128000,
        "max_output_tokens": 16384,
        "capabilities": {
            "chat_completions": True,
            "responses": True,
            "streaming": True,
            "reasoning": True
        }
    }

# ============================================
# Claude 兼容
# ============================================
@app.api_route("/v1/v1/messages", methods=["POST"])
@app.api_route("/v1/v1/messages/", methods=["POST"])
async def claude_messages(request: Request):
    body = await request.json()
    # 分类器：先截断 body["messages"] 再做下面的转换，否则截断不生效
    apply_classifier_handling(body)

    original_model = body.get("model", "nvidia-coder")
    base_url, api_key, provider_name, mapped_model = get_model_upstream_config(original_model)
    is_stream = body.get("stream", False)

    nvidia_body = {
        "model": mapped_model,
        "messages": convert_claude_messages_to_openai(body),
        "stream": is_stream,
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 1.0),
    }

    if resolve_classifier(body, original_model):
        nvidia_body["reasoning_effort"] = "none"

    REASONING_MODELS = [
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ]
    if mapped_model in REASONING_MODELS:
        nvidia_body["reasoning_effort"] = "none"
        logger.info(f"🧠 [推理修复] {mapped_model} 已关闭推理以启用工具调用")

    if body.get("tools"):
        _tools = normalize_tools(body.get("tools"))
        if _tools:
            nvidia_body["tools"] = _tools
        logger.info(f"[Claude] 转发 {len(_tools or [])} 个工具定义（已标准化）")

    logger.info(f"[{provider_name}] /v1/v1/messages {original_model} -> {mapped_model}")

    upstream_keys = NVIDIA_API_KEYS if provider_name == "nvidia" else (api_key if isinstance(api_key, list) else [api_key] if api_key else [])
    max_attempts = len(upstream_keys) if upstream_keys else 1

    for attempt in range(max_attempts):
        key = get_provider_key(provider_name, upstream_keys, attempt)
        key_label = get_key_label(key) if provider_name == "nvidia" else provider_name
        if attempt > 0:
            logger.info(f"🔄 [{provider_name}] 重试 {attempt + 1}/{max_attempts}: {key_label}")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            req = async_client.build_request("POST", f"{base_url}/chat/completions", json=nvidia_body, headers=headers)
            resp = await async_client.send(req, stream=is_stream)
            if resp.status_code == 429:
                logger.warning(f"⚠️ {key_label} 429 Rate Limit")
                if provider_name == "nvidia":
                    rate_limiter.mark_rate_limited(key)
                await resp.aclose()
                wait = rate_limiter.get_wait_time(key) if provider_name == "nvidia" else 3
                if wait > 0:
                    await asyncio.sleep(wait)
                continue
            if resp.status_code == 503:
                logger.warning(f"⚠️ {key_label} 503 Service Unavailable")
                await resp.aclose()
                await asyncio.sleep(3)
                continue
            if resp.status_code != 200:
                error_msg = (await resp.aread()).decode('utf-8', errors='ignore')
                logger.error(f"{provider_name}错误: {resp.status_code} - {error_msg}")
                await resp.aclose()
                return Response(json.dumps({"error": {"message": error_msg}}), status_code=resp.status_code, media_type="application/json")
            if provider_name == "nvidia":
                rate_limiter.clear_backoff(key)
                rate_limiter.record_request(key)
            if is_stream:
                return StreamingResponse(openai_stream_to_anthropic(resp, original_model), media_type="text/event-stream")
            data = await resp.aread()
            data = json.loads(data)
            await resp.aclose()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") or message.get("reasoning") or message.get("reasoning_content") or ""
            tool_calls = message.get("tool_calls", [])
            finish_reason = choice.get("finish_reason") or "stop"
            stop_reason = "tool_use" if tool_calls else ("max_tokens" if finish_reason in ("length", "max_tokens") else "end_turn")
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for tc in tool_calls:
                content_blocks.append({"type": "tool_use", "id": tc.get("id"), "name": tc.get("function", {}).get("name"), "input": parse_tool_arguments(tc.get("function", {}).get("arguments"))})
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            return Response(json.dumps({"type": "message", "id": data.get("id", "msg_001"), "model": original_model, "role": "assistant", "content": content_blocks, "stop_reason": stop_reason, "stop_sequence": None, "usage": data.get("usage", {})}), media_type="application/json")
        except Exception as e:
            logger.error(f"💥 {key_label} 异常: {str(e)}")
            continue
    return Response(json.dumps({"error": {"message": "All API keys rate limited"}}), status_code=429, media_type="application/json")

@app.api_route("/v1/messages", methods=["POST"])
async def claude_code_messages(request: Request):
    body = await request.json()
    # 分类器：先截断 body["messages"] 再做下面的转换，否则截断不生效
    apply_classifier_handling(body)

    original_model = body.get("model", "nvidia-coder")
    base_url, api_key, provider_name, mapped_model = get_model_upstream_config(original_model)
    is_stream = body.get("stream", False)

    nvidia_body = {
        "model": mapped_model,
        "messages": convert_claude_messages_to_openai(body),
        "stream": is_stream,
        "max_tokens": body.get("max_tokens", 4096),
        "temperature": body.get("temperature", 1.0),
    }

    if resolve_classifier(body, original_model):
        nvidia_body["reasoning_effort"] = "none"

    REASONING_MODELS = [
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ]
    if mapped_model in REASONING_MODELS:
        nvidia_body["reasoning_effort"] = "none"
        logger.info(f"🧠 [推理修复] {mapped_model} 已关闭推理以启用工具调用")

    if body.get("tools"):
        _tools = normalize_tools(body.get("tools"))
        if _tools:
            nvidia_body["tools"] = _tools
        logger.info(f"[Claude Code] 转发 {len(_tools or [])} 个工具定义（已标准化）")

    logger.info(f"[{provider_name}] /v1/messages {original_model} -> {mapped_model}")

    upstream_keys = NVIDIA_API_KEYS if provider_name == "nvidia" else (api_key if isinstance(api_key, list) else [api_key] if api_key else [])
    max_attempts = len(upstream_keys) if upstream_keys else 1

    for attempt in range(max_attempts):
        key = get_provider_key(provider_name, upstream_keys, attempt)
        key_label = get_key_label(key) if provider_name == "nvidia" else provider_name
        if attempt > 0:
            logger.info(f"🔄 [{provider_name}] 重试 {attempt + 1}/{max_attempts}: {key_label}")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            req = async_client.build_request("POST", f"{base_url}/chat/completions", json=nvidia_body, headers=headers)
            resp = await async_client.send(req, stream=is_stream)
            if resp.status_code == 429:
                logger.warning(f"⚠️ {key_label} 429 Rate Limit")
                if provider_name == "nvidia":
                    rate_limiter.mark_rate_limited(key)
                await resp.aclose()
                wait = rate_limiter.get_wait_time(key) if provider_name == "nvidia" else 3
                if wait > 0:
                    await asyncio.sleep(wait)
                continue
            if resp.status_code == 503:
                logger.warning(f"⚠️ {key_label} 503 Service Unavailable")
                await resp.aclose()
                await asyncio.sleep(3)
                continue
            if resp.status_code != 200:
                error_msg = (await resp.aread()).decode('utf-8', errors='ignore')
                logger.error(f"{provider_name}错误: {resp.status_code} - {error_msg}")
                await resp.aclose()
                return Response(json.dumps({"error": {"message": error_msg}}), status_code=resp.status_code, media_type="application/json")
            if provider_name == "nvidia":
                rate_limiter.clear_backoff(key)
                rate_limiter.record_request(key)
            if is_stream:
                return StreamingResponse(openai_stream_to_anthropic(resp, original_model), media_type="text/event-stream")
            data = await resp.aread()
            data = json.loads(data)
            await resp.aclose()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") or message.get("reasoning") or message.get("reasoning_content") or ""
            tool_calls = message.get("tool_calls", [])
            finish_reason = choice.get("finish_reason") or "stop"
            stop_reason = "tool_use" if tool_calls else ("max_tokens" if finish_reason in ("length", "max_tokens") else "end_turn")
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for tc in tool_calls:
                content_blocks.append({"type": "tool_use", "id": tc.get("id"), "name": tc.get("function", {}).get("name"), "input": parse_tool_arguments(tc.get("function", {}).get("arguments"))})
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            return Response(json.dumps({"type": "message", "id": data.get("id", "msg_001"), "model": original_model, "role": "assistant", "content": content_blocks, "stop_reason": stop_reason, "stop_sequence": None, "usage": data.get("usage", {})}), media_type="application/json")
        except Exception as e:
            logger.error(f"💥 {key_label} 异常: {str(e)}")
            continue
    return Response(json.dumps({"error": {"message": "All API keys rate limited"}}), status_code=429, media_type="application/json")

# ============================================
# 健康检查
# ============================================
@app.get("/v1/")
async def v1_root():
    return {"status": "ok", "message": "NVIDIA/AtomGit Proxy v3.1 + 多Provider"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "nvidia_keys": len(NVIDIA_API_KEYS),
        "atomgit_configured": bool(ATOMGIT_TOKENS),
        "providers": ["nvidia", "atomgit"] if ATOMGIT_TOKENS else ["nvidia"],
        "atomgit_tokens": len(ATOMGIT_TOKENS),
    }

@app.get("/")
async def root():
    return {
        "name": "NVIDIA/AtomGit Proxy",
        "version": "3.1 + 多Provider",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "responses": "/v1/responses",
            "claude_v1": "/v1/v1/messages",
            "claude_v2": "/v1/messages",
            "models": "/v1/models"
        },
        "api_keys_count": len(NVIDIA_API_KEYS),
        "atomgit_configured": bool(ATOMGIT_TOKENS)
    }

@app.on_event("shutdown")
async def shutdown():
    await async_client.aclose()

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🚀 NVIDIA/AtomGit Proxy v3.1 + 多Provider")
    logger.info("=" * 50)
    logger.info(f"🔑 NVIDIA Key: {len(NVIDIA_API_KEYS)} 个")
    for i, key in enumerate(NVIDIA_API_KEYS):
        logger.info(f"   Key{i+1}: {key[:12]}...")
    logger.info(f"🔑 AtomGit: {len(ATOMGIT_TOKENS)} 个 token（api-ai.gitcode.com）")
    logger.info(f"📍 http://{HOST}:{PORT}")
    logger.info("✅ Provider: NVIDIA (4Key轮询) + AtomGit (deepseek路由)")
    logger.info("✅ 流式: build_request + send(stream=True) 生命周期正确")
    logger.info("=" * 50)
    uvicorn.run(app, host=HOST, port=PORT)
