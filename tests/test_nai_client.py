import random

from core.modules import nai_client as nai


def run_parser(chunks):
    parser = nai.ThinkTagParser()
    reasoning, answer = "", ""
    events = []
    for c in chunks:
        events += parser.feed(c)
    events += parser.flush()
    for kind, text in events:
        if kind == "reasoning":
            reasoning += text
        elif kind == "content":
            answer += text
        elif kind == "reclassify":
            reasoning, answer = reasoning + answer, ""
    return reasoning, answer


def random_split(text, seed):
    rnd = random.Random(seed)
    chunks, i = [], 0
    while i < len(text):
        n = rnd.randint(1, 6)
        chunks.append(text[i:i + n])
        i += n
    return chunks


def test_parser_plain_text():
    assert run_parser(["Olá ", "mundo"]) == ("", "Olá mundo")


def test_parser_think_tags_any_split():
    text = "<think>Pensando bem</think>\n\nResposta final."
    for seed in range(50):
        r, a = run_parser(random_split(text, seed))
        assert r == "Pensando bem", seed
        assert a == "Resposta final.", seed


def test_parser_orphan_close_tag_deepseek_r1():
    text = "Raciocínio sem abrir\n</think>\n\nResposta."
    for seed in range(50):
        r, a = run_parser(random_split(text, seed))
        assert r.strip() == "Raciocínio sem abrir", seed
        assert a == "Resposta.", seed


def test_parser_keeps_lone_angle_bracket():
    assert run_parser(["a < b e c <", "d"]) == ("", "a < b e c <d")


def test_split_reasoning_non_stream():
    assert nai.split_reasoning("<think>x</think>y") == ("x", "y")
    assert nai.split_reasoning("só resposta") == ("", "só resposta")


def test_models_url():
    assert nai.models_url("https://h/enterpriseai/gateway/v1/chat/completions") == "https://h/enterpriseai/gateway/v1/models"
    assert nai.models_url("https://h/v1/chat/completions/") == "https://h/v1/models"
    assert nai.models_url("https://h/v1") == "https://h/v1/models"


def test_clean_history_filters_and_limits():
    hist = [
        {"role": "assistant", "content": "começa com assistente"},
        {"role": "user", "content": "u1", "extra": 1},
        {"role": "assistant", "content": "a1", "metrics": {}, "reasoning": "r"},
        {"role": "system", "content": "ignorar"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "u2"},
    ]
    assert nai.clean_history(hist, limit=20) == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    # Corte nas últimas 2 deixaria "a1" primeiro; deve começar no usuário
    assert nai.clean_history(hist, limit=2) == [{"role": "user", "content": "u2"}]
    assert nai.clean_history(hist, limit=0) == []


def test_build_payload_with_document_truncation(monkeypatch):
    monkeypatch.setattr(nai, "DOC_MAX_CHARS", 10)
    payload, truncated = nai.build_payload(
        "pergunta", [], "m", system_prompt="SYS", temperature=0.5, max_tokens=99,
        document_text="0123456789ABCDEF", document_name="doc.txt",
    )
    assert truncated is True
    system = payload["messages"][0]
    assert system["role"] == "system"
    assert "SYS" in system["content"] and "0123456789" in system["content"]
    assert "ABCDEF" not in system["content"]
    assert payload["temperature"] == 0.5 and payload["max_tokens"] == 99
    assert payload["messages"][-1] == {"role": "user", "content": "pergunta"}


def test_build_payload_empty_system_prompt_is_omitted():
    payload, _ = nai.build_payload("oi", [], "m", system_prompt="")
    assert payload["messages"] == [{"role": "user", "content": "oi"}]


def test_metrics_with_and_without_usage():
    m = nai.build_metrics(0.0, 0.5, 2.5, {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}, 90)
    assert m["ttft"] == 0.5 and m["tps"] == 50.0 and m["estimated"] is False and m["total_tokens"] == 110
    m = nai.build_metrics(0.0, 0.5, 2.5, None, 40)
    assert m["completion_tokens"] == 40 and m["estimated"] is True and m["tps"] == 20.0
    m = nai.build_metrics(0.0, None, 2.0, {"completion_tokens": 10}, 0)
    assert m["ttft"] is None and m["tps"] == 5.0
