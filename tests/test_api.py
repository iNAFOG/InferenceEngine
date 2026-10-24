import json

import pytest
from fastapi.testclient import TestClient

GREEDY = {"max_new_tokens": 8, "do_sample": False, "top_k": 0}


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("INFENG_MODEL", "sshleifer/tiny-gpt2")
    mp.setenv("INFENG_DEVICE", "cpu")
    from infeng.api import app

    with TestClient(app) as c:  # "with" runs the lifespan, which loads the model
        yield c
    mp.undo()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


def test_generate(client):
    r = client.post("/generate", json={"prompt": "Hello", **GREEDY})
    assert r.status_code == 200
    assert r.json()["generated_tokens"] >= 1


def test_generate_batch(client):
    r = client.post("/generate/batch", json={"prompts": ["Hi", "Hello there"], **GREEDY})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_stream_is_ndjson(client):
    with client.stream("POST", "/generate/stream", json={"prompt": "Hello", **GREEDY}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert r.status_code == 200
    assert events[-1]["type"] == "done"
    assert all(e["type"] == "token" for e in events[:-1])


def test_schema_rejects_bad_request(client):
    # pydantic catches this before the engine is touched -> 422
    assert client.post("/generate", json={"prompt": "", **GREEDY}).status_code == 422


def test_engine_rejects_inconsistent_params(client):
    # each field is valid alone, but greedy + top_k=50 together is not -> engine says 400
    r = client.post("/generate", json={"prompt": "Hello", "do_sample": False, "top_k": 50})
    assert r.status_code == 400
