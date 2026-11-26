import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.pipeline import Providers
from app.providers.demo import DemoPrices, DemoTranscriber, DemoVision
from app.store import Store


def png(w=600, h=400) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (w, h), (200, 200, 200)).save(b, "PNG")
    return b.getvalue()


@pytest.fixture
def client():
    cfg = Settings(free_monthly_limit=3, db_path=":memory:")
    app = create_app(cfg, Providers(DemoVision(), DemoTranscriber(), DemoPrices(), demo=True),
                     Store(":memory:", 3))
    return TestClient(app)


def key_for(c, email="a@shop.com"):
    r = c.post("/api/v1/signup", json={"email": email})
    assert r.status_code == 201
    return {"X-API-Key": r.json()["api_key"]}


def analyze(c, h, **kw):
    files = {"image": ("p.png", png(), "image/png")}
    return c.post("/api/v1/parts/analyze", files=files, data=kw, headers=h)


def test_health_and_index(client):
    assert client.get("/health").json()["demo"] is True
    assert "PartAi" in client.get("/").text
    assert client.get("/manifest.json").status_code == 200
    assert client.get("/sw.js").status_code == 200


def test_auth_required(client):
    assert client.get("/api/v1/me").status_code == 401
    assert client.get("/api/v1/me", headers={"X-API-Key": "pk_bad"}).status_code == 401
    assert analyze(client, {}).status_code == 401


def test_signup_validation_and_duplicates(client):
    assert client.post("/api/v1/signup", json={"email": "nope"}).status_code == 422
    key_for(client)
    assert client.post("/api/v1/signup", json={"email": "A@shop.com"}).status_code == 409


def test_analyze_full_flow_and_demo_flag(client):
    h = key_for(client)
    r = analyze(client, h, note="tested working")
    assert r.status_code == 200
    j = r.json()
    assert j["demo"] is True and j["part_number"] == "00145787"
    assert j["estimated_price_usd"]["low"] <= j["estimated_price_usd"]["high"]
    assert client.get("/api/v1/me", headers=h).json()["used_this_month"] == 1
    assert len(client.get("/api/v1/parts", headers=h).json()) == 1


def test_free_quota_enforced(client):
    h = key_for(client)
    for _ in range(3):
        assert analyze(client, h).status_code == 200
    r = analyze(client, h)
    assert r.status_code == 402 and "limit" in r.json()["detail"].lower()


def test_quotas_are_per_tenant(client):
    h1, h2 = key_for(client, "a@x.com"), key_for(client, "b@x.com")
    for _ in range(3):
        analyze(client, h1)
    assert analyze(client, h1).status_code == 402
    assert analyze(client, h2).status_code == 200
    assert client.get("/api/v1/parts", headers=h2).json()[0]["part_number"]
    assert len(client.get("/api/v1/parts", headers=h2).json()) == 1


def test_bad_uploads_do_not_consume_quota(client):
    h = key_for(client)
    r = client.post("/api/v1/parts/analyze", files={"image": ("x.txt", b"hello", "text/plain")}, headers=h)
    assert r.status_code == 415
    tiny = client.post("/api/v1/parts/analyze", files={"image": ("t.png", png(50, 50), "image/png")}, headers=h)
    assert tiny.status_code == 415
    assert client.get("/api/v1/me", headers=h).json()["used_this_month"] == 0


def test_vision_outage_is_502_and_free(client):
    class Boom:
        async def analyze(self, *a):
            from app.providers.base import ProviderError
            raise ProviderError("down")
    c = TestClient(create_app(Settings(db_path=":memory:"),
                              Providers(Boom(), None, DemoPrices()), Store(":memory:", 10)))
    h = key_for(c)
    assert analyze(c, h).status_code == 502
    assert c.get("/api/v1/me", headers=h).json()["used_this_month"] == 0


def test_pro_plan_unlimited(client):
    store = Store(":memory:", 1)
    c = TestClient(create_app(Settings(db_path=":memory:"),
                              Providers(DemoVision(), None, DemoPrices(), demo=True), store))
    h = key_for(c)
    analyze(c, h)
    assert analyze(c, h).status_code == 402
    assert store.set_plan("a@shop.com", "pro")
    assert analyze(c, h).status_code == 200
    assert c.get("/api/v1/me", headers=h).json()["monthly_limit"] is None


def test_exports(client):
    h = key_for(client)
    analyze(client, h)
    inv = client.get("/api/v1/parts/export?fmt=inventory_csv", headers=h)
    assert inv.status_code == 200 and "00145787" in inv.text and inv.headers["content-type"].startswith("text/csv")
    assert "Add" in client.get("/api/v1/parts/export?fmt=ebay_csv", headers=h).text
    ej = client.get("/api/v1/parts/export?fmt=ebay_json", headers=h).json()
    assert ej[0]["aspects"]["MPN"] == ["00145787"] and ej[0]["conditionId"] == 3000
    assert client.get("/api/v1/parts/export?fmt=xml", headers=h).status_code == 422


def test_live_mode_requires_keys():
    from app.main import build_providers
    with pytest.raises(RuntimeError):
        build_providers(Settings(mode="live"))
