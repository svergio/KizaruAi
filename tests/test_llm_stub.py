"""LLM-сервер (llama-swap или аналог, OpenAI-совместимый API). В тестах только заглушка fake_llm_server.
Клиента в core/llm ещё нет, поэтому тест клиента пропускается."""
import http.client
import json
from urllib.parse import urlparse

import pytest

import core.llm


def _post(base_url, path, payload):
    url = urlparse(base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    try:
        conn.request("POST", url.path + path, body=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def test_fake_llm_server_speaks_openai_format(fake_llm_server):
    status, data = _post(
        fake_llm_server.base_url,
        "/chat/completions",
        {"model": "qwen", "messages": [{"role": "user", "content": "Оцени рифму"}]},
    )
    assert status == 200
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == fake_llm_server.reply
    assert fake_llm_server.requests[0]["body"]["messages"][0]["content"] == "Оцени рифму"
    assert fake_llm_server.base_url.startswith("http://127.0.0.1:")


def _client_entrypoint():
    names = [n for n in dir(core.llm) if not n.startswith("_")]
    return [n for n in names if callable(getattr(core.llm, n))]


@pytest.mark.skipif(not _client_entrypoint(), reason="клиент llm-server в core/llm ещё не написан")
def test_llm_client_against_fake_server(fake_llm_server):  # pragma: no cover - включится вместе с клиентом
    pytest.fail("клиент появился: напиши здесь проверку против fake_llm_server.base_url")
