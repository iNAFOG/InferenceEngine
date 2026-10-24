import json

import httpx

# the server sends one json object per line (ndjson):
#   {"type": "token", "text": " Machine"}
#   ...
#   {"type": "done", "generated_tokens": 32, "latency_ms": ...}
with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/generate/stream",
    json={
        "prompt": "Explain machine learning",
        "max_new_tokens": 32,
        "do_sample": False,
        "top_k": 0,
    },
    timeout=60,
) as response:
    for line in response.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        else:
            print(f"\n\n[{event['generated_tokens']} tokens, {event['tokens_per_sec']} tok/s, "
                  f"ttft {event['ttft_ms']} ms]")
