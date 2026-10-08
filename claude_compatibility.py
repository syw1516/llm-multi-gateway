# Claude Messages API 兼容层（OpenAI 上游 <-> Anthropic 协议双向转换）
import json
import logging
import uuid

logger = logging.getLogger(__name__)


def build_claude_to_openai_messages(body):
    """Anthropic messages + 顶层 system -> OpenAI messages（system 作为独立消息，不合并进首条）"""
    messages = []
    system_msg = ""

    top_system = body.get("system")
    if top_system:
        if isinstance(top_system, str):
            system_msg += top_system
        elif isinstance(top_system, list):
            for block in top_system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_msg += block.get("text", "") + "\n"
                elif isinstance(block, str):
                    system_msg += block + "\n"

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_msg += content + "\n"
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        system_msg += b.get("text", "") + "\n"
            continue

        content = msg.get("content")
        text = None
        oai_tool_calls = []
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "text":
                    text_parts.append(b.get("text", ""))
                elif btype == "tool_use":
                    oai_tool_calls.append({
                        "id": b.get("id", "toolu_0"),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                        },
                    })
                elif btype == "tool_result":
                    cid = b.get("tool_use_id", "")
                    rc = b.get("content", "")
                    if isinstance(rc, list):
                        rc = "".join(
                            x.get("text", "") if isinstance(x, dict) else str(x)
                            for x in rc
                        )
                    elif not isinstance(rc, str):
                        rc = json.dumps(rc, ensure_ascii=False)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": str(rc),
                    })
            text = "".join(text_parts) if text_parts else None

        if role == "assistant":
            oai_msg = {"role": "assistant", "content": text if text is not None else ""}
            if oai_tool_calls:
                oai_msg["tool_calls"] = oai_tool_calls
            messages.append(oai_msg)
        else:
            messages.append({"role": "user", "content": text or ""})

    if system_msg:
        messages.insert(0, {"role": "system", "content": system_msg.strip()})
    return messages


def build_claude_response(openai_data: dict, original_model: str) -> dict:
    """OpenAI chat.completions 非流式响应 -> Anthropic message 响应"""
    message = {}
    for choice in openai_data.get("choices", []):
        message = choice.get("message") or {}
        break

    content_blocks = []
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in (message.get("tool_calls") or []):
        fn = tc.get("function", {}) or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            inp = fn.get("arguments") or {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", "toolu_0"),
            "name": fn.get("name", ""),
            "input": inp,
        })
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    has_tool = any(b["type"] == "tool_use" for b in content_blocks)
    usage_src = openai_data.get("usage") or {}
    return {
        "type": "message",
        "id": openai_data.get("id", "msg_001"),
        "model": original_model,
        "role": "assistant",
        "content": content_blocks,
        "stop_reason": "tool_use" if has_tool else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage_src.get("prompt_tokens", 0),
            "output_tokens": usage_src.get("completion_tokens", 0),
        },
    }


async def convert_openai_stream_to_claude(stream_generator, original_model: str, input_tokens: int = 0):
    """OpenAI SSE 流 -> Anthropic 流事件。

    事件序列：message_start / content_block_start / content_block_delta* /
    content_block_stop / message_delta(stop_reason) / message_stop。
    文本与 tool_calls 增量都支持。
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def data_line(payload: str) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # message_start
    yield data_line({
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": original_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })

    block_index = -1
    active_text_block = -1
    tool_blocks = {}   # oai tc index -> (block_index, args)
    output_tokens = 0
    stop_reason = "end_turn"

    def next_block() -> int:
        nonlocal block_index
        block_index += 1
        return block_index

    try:
        async for raw in stream_generator:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            line = (raw or "").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                continue
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

            for choice in (chunk.get("choices") or []):
                delta = choice.get("delta") or {}

                text = delta.get("content") or delta.get("reasoning") or ""
                if text:
                    if active_text_block == -1:
                        active_text_block = next_block()
                        yield data_line({
                            "type": "content_block_start",
                            "index": active_text_block,
                            "content_block": {"type": "text", "text": ""},
                        })
                    yield data_line({
                        "type": "content_block_delta",
                        "index": active_text_block,
                        "delta": {"type": "text_delta", "text": text},
                    })

                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    if idx not in tool_blocks:
                        bi = next_block()
                        tool_blocks[idx] = [bi, ""]
                        tname = (tc.get("function", {}) or {}).get("name", "")
                        yield data_line({
                            "type": "content_block_start",
                            "index": bi,
                            "content_block": {"type": "tool_use", "id": tc.get("id", "toolu_0"), "name": tname, "input": {}},
                        })
                    args_piece = (tc.get("function", {}) or {}).get("arguments", "")
                    if args_piece:
                        tool_blocks[idx][1] += args_piece
                        yield data_line({
                            "type": "content_block_delta",
                            "index": tool_blocks[idx][0],
                            "delta": {"type": "input_json_delta", "partial_json": args_piece},
                        })

                if choice.get("finish_reason") == "tool_calls":
                    stop_reason = "tool_use"
    except Exception as e:
        logger.error(f"Claude 流式转换错误: {e}")
        yield data_line({
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        })
        return

    # 收尾：依次结束各 block
    if active_text_block != -1:
        yield data_line({"type": "content_block_stop", "index": active_text_block})
    for bi, _args in tool_blocks.values():
        yield data_line({"type": "content_block_stop", "index": bi})

    yield data_line({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield data_line({"type": "message_stop"})
