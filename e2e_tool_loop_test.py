"""E2E：真实 Agnes 上游，模拟 Claude Code 两轮工具循环。
第 1 轮：user 让模型调 read_file → 期望 stop_reason=tool_use
第 2 轮：把 tool_use 和 tool_result 按 Anthropic 格式回传 → 期望模型基于工具结果正常收尾（end_turn，文本提到结果内容）
验证点：
  1. 第 1 轮 stop_reason 正确（tool_use 而非 end_turn）
  2. 第 2 轮历史转换后模型能看到工具结果（回复内容应引用文件内容）—— 旧实现在这里上下文断裂
  3. 两轮 SSE 事件序列完整
"""
import json
import httpx

BASE = "http://127.0.0.1:8081/v1/messages?beta=true"
MODEL = "claude-sonnet-5"
HEADERS = {"anthropic-version": "2023-06-01"}

tools = [
    {"name": "read_file", "description": "Read a file from disk",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]


def run_round(payload, label):
    print(f"\n== {label} ==")
    events = []
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0), trust_env=False) as c:
        with c.stream("POST", BASE, json=payload, headers=HEADERS) as resp:
            assert resp.status_code == 200, f"HTTP {resp.status_code}"
            buf = b""
            for chunk in resp.iter_bytes():
                buf += chunk
                while b"\n\n" in buf:
                    block, buf = buf.split(b"\n\n", 1)
                    if not block.strip():
                        continue
                    elines = block.decode("utf-8", errors="replace").split("\n")
                    ev = [l[7:] for l in elines if l.startswith("event: ")]
                    data = [l[6:] for l in elines if l.startswith("data: ")]
                    name = ev[0] if ev else "?"
                    try:
                        d = json.loads(data[0]) if data else {}
                    except Exception:
                        d = {}
                    events.append((name, d))
    names = [e[0] for e in events]
    # 提取 stop_reason、text、tool_use
    stop_reason = None
    for n, d in events:
        if n == "message_delta":
            stop_reason = d.get("delta", {}).get("stop_reason")
    text = ""
    tool_uses = []
    for n, d in events:
        if n == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta":
            text += d["delta"]["text"]
        if n == "content_block_start" and d.get("content_block", {}).get("type") == "tool_use":
            tool_uses.append({"id": d["content_block"].get("id"), "name": d["content_block"].get("name")})
        if n == "content_block_delta" and d.get("delta", {}).get("type") == "input_json_delta":
            tool_uses[-1].setdefault("args", "")
            tool_uses[-1]["args"] += d["delta"]["partial_json"]
    print(f"   事件序列: {' → '.join(dict.fromkeys(names))}")
    print(f"   stop_reason={stop_reason}")
    print(f"   text: {text[:200]!r}")
    print(f"   tool_uses: {json.dumps(tool_uses, ensure_ascii=False)[:300]}")
    assert names[0] == "message_start" and names[-1] == "message_stop", f"事件序列不完整: {names[:3]}...{names[-3:]}"
    assert not any(n == "error" for n in names), "出现 error 事件"
    return stop_reason, text, tool_uses


def main():
    # 第 1 轮
    round1 = {
        "model": MODEL, "max_tokens": 2048, "stream": True,
        "system": "You are a coding agent. Use the read_file tool to read files, then answer.",
        "tools": tools,
        "messages": [
            {"role": "user", "content": "Read the file /tmp/e2e_target.txt and tell me exactly what it contains."}
        ],
    }
    # 造一个真实文件让 read_file 有东西可读（工具实际不会执行，结果由我们伪造回传）
    with open("/tmp/e2e_target.txt", "w") as f:
        f.write("SECRET_ANSWER_42\n")

    sr1, text1, tools1 = run_round(round1, "第1轮：要求读文件")
    assert sr1 == "tool_use", f"BUG: 第1轮 stop_reason={sr1}, 期望 tool_use"
    assert tools1 and tools1[0]["name"] == "read_file", f"BUG: 第1轮未调用 read_file: {tools1}"
    args1 = json.loads(tools1[0].get("args") or "{}")
    print(f"   [OK] stop_reason=tool_use, 工具参数={args1}")

    # 第 2 轮：回传 tool_use + tool_result（Anthropic 格式，正是旧实现会丢上下文的地方）
    round2 = {
        "model": MODEL, "max_tokens": 2048, "stream": True,
        "system": "You are a coding agent. Use the read_file tool to read files, then answer.",
        "tools": tools,
        "messages": [
            {"role": "user", "content": "Read the file /tmp/e2e_target.txt and tell me exactly what it contains."},
            {"role": "assistant", "content": [
                {"type": "text", "text": text1 or "Reading the file."},
                {"type": "tool_use", "id": tools1[0]["id"], "name": "read_file", "input": args1},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tools1[0]["id"], "content": "SECRET_ANSWER_42"},
            ]},
        ],
    }
    sr2, text2, tools2 = run_round(round2, "第2轮：回传工具结果")
    assert sr2 == "end_turn", f"BUG: 第2轮 stop_reason={sr2}, 期望 end_turn"
    assert "SECRET_ANSWER_42" in text2 or "42" in text2, \
        f"BUG: 模型没看到工具结果（上下文断裂！）: {text2[:200]!r}"
    print(f"   [OK] stop_reason=end_turn, 模型正确引用了文件内容")

    print("\nE2E ALL PASS：两轮工具循环上下文完整，stop_reason 全对")


if __name__ == "__main__":
    main()
