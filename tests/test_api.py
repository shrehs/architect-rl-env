from fastapi.testclient import TestClient

from api.server import app


client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_reset_and_step_contract_shape() -> None:
    reset_resp = client.post("/reset", json={"task_id": "easy"})
    assert reset_resp.status_code == 200

    body = reset_resp.json()
    assert "last_assistant_message" in body
    assert "constraints_collected" in body
    assert "missing_constraints" in body

    step_resp = client.post(
        "/step",
        json={"action": {"user_reply": "use case is forecasting and latency is 30ms"}},
    )
    assert step_resp.status_code == 200

    payload = step_resp.json()
    assert set(payload.keys()) == {"observation", "reward", "done", "info"}
    assert isinstance(payload["reward"], float)
    assert isinstance(payload["done"], bool)
    assert isinstance(payload["info"], dict)


def test_state_endpoint() -> None:
    client.post("/reset", json={"task_id": "easy"})
    resp = client.get("/state")

    assert resp.status_code == 200
    state = resp.json()
    assert "messages" in state
    assert "constraints" in state
    assert "step_count" in state


def test_step_after_done_returns_conflict() -> None:
    client.post("/reset", json={"task_id": "easy"})
    for _ in range(8):
        client.post("/step", json={"action": {"user_reply": "x"}})

    resp = client.post("/step", json={"action": {"user_reply": "one more"}})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Episode already finished. Call reset()."


def test_get_reset_compatibility_route() -> None:
    resp = client.get("/reset")
    assert resp.status_code == 200
    payload = resp.json()
    assert "last_assistant_message" in payload
    assert "constraints_collected" in payload


def test_step_accepts_flat_user_reply_payload() -> None:
    client.get("/reset")
    resp = client.post("/step", json={"user_reply": "test"})
    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload.keys()) == {"observation", "reward", "done", "info"}
