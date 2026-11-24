"""
gpqa_openai_stream_client.py — one streamed /v1/chat/completions call with per-request timing.

Streams so TTFT is real (time to the first reasoning OR content token), and asks for
stream_options.include_usage so completion_tokens comes from the server, not a chunk count.
Decode tok/s = (completion_tokens - 1) / (t_last_token - t_first_token): the inter-token rate
after the first token, excluding prefill. Works against Pantheon and vLLM (reasoning arrives
in delta.reasoning_content or delta.reasoning on both when a reasoning parser is on).

Status: works (stdlib only).
Issues: None
Todo: None
"""
import http.client
import json
import time
import urllib.parse


class ServeRequestError(Exception):
    """The request failed at the transport/HTTP level; the item is retried on resume."""


def open_connection(base_url, idle_timeout_s):
    parsed = urllib.parse.urlparse(base_url)
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return connection_class(parsed.hostname, parsed.port, timeout=idle_timeout_s), parsed.path.rstrip("/")


def iter_sse_data(response):
    """Yield each `data:` payload of a server-sent-event stream (stops at [DONE])."""
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        yield payload


def stream_chat_completion(base_url, request_body, api_key=None, idle_timeout_s=900):
    """POST a streamed chat completion; return content, reasoning, usage and timings."""
    connection, base_path = open_connection(base_url, idle_timeout_s)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = dict(request_body, stream=True, stream_options={"include_usage": True})
    content_parts, reasoning_parts = [], []
    usage, finish_reason, t_first, t_last, token_chunks = None, None, None, None, 0
    t_start = time.monotonic()
    try:
        connection.request("POST", base_path + "/v1/chat/completions", json.dumps(body), headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ServeRequestError(f"HTTP {response.status}: {response.read(500)!r}")
        for payload in iter_sse_data(response):
            chunk = json.loads(payload)
            if chunk.get("error"):
                raise ServeRequestError(f"stream error: {chunk['error']}")
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                content = delta.get("content") or ""
                if reasoning or content:
                    now = time.monotonic()
                    t_first = t_first if t_first is not None else now
                    t_last = now
                    token_chunks += 1
                    reasoning_parts.append(reasoning)
                    content_parts.append(content)
                finish_reason = choice.get("finish_reason") or finish_reason
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
        raise ServeRequestError(f"{type(e).__name__}: {e}") from e
    finally:
        connection.close()
    t_end = time.monotonic()
    return summarize_stream(content_parts, reasoning_parts, usage, finish_reason,
                            t_start, t_first, t_last, t_end, token_chunks)


def summarize_stream(content_parts, reasoning_parts, usage, finish_reason,
                     t_start, t_first, t_last, t_end, token_chunks):
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens") or token_chunks
    details = usage.get("completion_tokens_details") or {}
    decode_window = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "completion_tokens_from_usage": bool(usage.get("completion_tokens")),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "stream_token_chunks": token_chunks,
        "ttft_s": (t_first - t_start) if t_first is not None else None,
        "total_s": t_end - t_start,
        "decode_tok_s": ((completion_tokens - 1) / decode_window) if decode_window > 0 else None,
    }
