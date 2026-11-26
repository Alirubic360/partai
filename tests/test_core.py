import asyncio
import json
import time

import httpx
import pytest

from app.config import Settings
from app.export import to_ebay_csv, to_inventory_csv
from app.listing import build_description, build_title, shipping_weight
from app.notes import parse_note
from app.pipeline import AnalysisError, Providers, analyze_part
from app.pricing import compute_pricing
from app.providers.base import ProviderError
from app.providers.demo import DemoPrices, DemoTranscriber, DemoVision
from app.providers.openai_client import OpenAIVision, OpenAIWhisper
from app.providers.serpapi_client import SerpApiEbay, parse_comps
from app.schemas import Comp, NoteFacts, VisionResult

CFG = Settings()
IMG = b"jpeg-bytes"


def run(coro):
    return asyncio.run(coro)


# ---------- notes ----------
@pytest.mark.parametrize("text,status", [
    ("tested working, pulled from a Bosch washer", "working"),
    ("drain pump works fine", "working"),
    ("not tested, as is", "untested"),
    ("untested", "untested"),
    ("not working, burnt connector", "for_parts"),
    ("for parts only", "for_parts"),
    ("", "untested"),
    ("pulled from a Bosch series 8", "untested"),   # no claim of working -> never assume
])
def test_note_status(text, status):
    assert parse_note(text).functional_status == status


def test_note_models():
    assert "WAE24060UC" in parse_note("from WAE24060UC washer, tested working").model_numbers
    assert parse_note("series 8").model_numbers == []


# ---------- pricing ----------
def comps(*ps, sold=True):
    return [Comp(price=p, sold=sold) for p in ps]


def test_pricing_ordering_and_outlier():
    info = compute_pricing(comps(24, 29.5, 32, 35, 38, 41, 45, 400), "working")
    p = info.estimated_price_usd
    assert p.low <= p.avg <= p.high
    assert p.high < 100          # the $400 outlier was dropped
    assert info.sample_size == 7


def test_pricing_condition_multiplier():
    base = compute_pricing(comps(20, 30, 40, 50, 60), "working").estimated_price_usd.avg
    parts = compute_pricing(comps(20, 30, 40, 50, 60), "for_parts").estimated_price_usd.avg
    assert parts == pytest.approx(base * 0.5, abs=0.01)


def test_pricing_empty_and_tiny():
    assert compute_pricing([], "working").confidence == "none"
    one = compute_pricing(comps(50), "working")
    assert one.confidence == "low" and one.estimated_price_usd.avg == 50


def test_pricing_confidence_levels():
    assert compute_pricing(comps(*range(20, 32)), "working").confidence == "high"
    assert compute_pricing(comps(20, 25, 30, 35, 40, sold=False), "working").confidence == "medium"


# ---------- listing ----------
V = VisionResult(part_name="Drain Pump", brand="Bosch", part_number="00145787",
                 part_number_confidence=.96, legibility="clear", appliance_type="washing machine",
                 visual_condition="clean, no cracks")


def test_title_limit_and_required_parts():
    t = build_title(V, "working", "00145787")
    assert len(t) <= 80 and "Bosch" in t and "00145787" in t
    long_v = V.model_copy(update={"part_name": "Heavy Duty Drain Pump Motor Assembly With Housing Kit",
                                  "brand": "Whirlpool Corporation"})
    t2 = build_title(long_v, "working", "W10876546ABC")
    assert len(t2) <= 80 and "W10876546ABC" in t2


def test_description_length_and_no_false_claims():
    note = NoteFacts(functional_status="untested")
    d = build_description(V, note, "00145787", [], "untested")
    words = len(d.split())
    assert 250 <= words <= 400, words
    assert "tested and is working" not in d      # never claim working when untested
    d2 = build_description(V, NoteFacts(functional_status="working"), "00145787", ["WAE24060UC"], "working")
    assert "tested and is working" in d2 and "WAE24060UC" in d2
    assert "guarantee" not in d.lower()


def test_weight_fallback():
    assert "lb" in shipping_weight(VisionResult(part_name="Control Board"))
    assert "unknown" in shipping_weight(VisionResult(part_name="Gizmo"))


# ---------- pipeline ----------
def providers(vision=None, prices=None, stt=None, demo=False):
    return Providers(vision or DemoVision(), stt or DemoTranscriber(), prices or DemoPrices(), demo=demo)


def test_pipeline_happy_path():
    r = run(analyze_part(IMG, b"aud", "n.webm", "", providers(), CFG))
    assert r.part_number == "00145787" and r.brand == "Bosch"
    assert r.condition == "Used - Tested Working"       # from transcript
    assert not r.needs_review and r.flags == []
    assert r.estimated_price_usd.low <= r.estimated_price_usd.avg <= r.estimated_price_usd.high
    assert r.timings_ms["total"] < 15000
    d = r.model_dump()
    for k in ("listing_title", "brand", "part_number", "compatible_models", "condition",
              "suggested_description", "estimated_price_usd", "shipping_weight_estimate"):
        assert k in d


def test_blurry_part_number_is_flagged_not_guessed():
    blurry = VisionResult(part_name="Control Board", brand="Samsung", part_number="DC92-0O3?7A",
                          part_number_confidence=0.3, legibility="blurry")
    r = run(analyze_part(IMG, None, "", "", providers(DemoVision(blurry)), CFG))
    assert r.part_number == "" and r.needs_review
    assert "part_number_unreadable" in r.flags
    assert "price_from_fuzzy_query" in r.flags


def test_low_confidence_clear_label_flagged():
    v = V.model_copy(update={"part_number_confidence": 0.6})
    r = run(analyze_part(IMG, None, "", "", providers(DemoVision(v)), CFG))
    assert "part_number_low_confidence" in r.flags and r.needs_review


class SlowPrices:
    async def search(self, q):
        await asyncio.sleep(5)
        return []


class BoomPrices:
    async def search(self, q):
        raise ProviderError("serpapi HTTP 500")


class BoomVision:
    async def analyze(self, *a):
        raise ProviderError("vision HTTP 500")


class BoomSTT:
    async def transcribe(self, *a):
        raise ProviderError("whisper HTTP 500")


def test_pricing_timeout_returns_partial_listing():
    cfg = Settings(pricing_timeout=0.2)
    t = time.perf_counter()
    r = run(analyze_part(IMG, None, "", "", providers(prices=SlowPrices()), cfg))
    assert time.perf_counter() - t < 2
    assert "pricing_unavailable_timeout" in r.flags and r.estimated_price_usd.avg == 0
    assert r.listing_title  # still usable


def test_pricing_error_returns_partial_listing():
    r = run(analyze_part(IMG, None, "", "", providers(prices=BoomPrices()), CFG))
    assert "pricing_unavailable_error" in r.flags


def test_vision_failure_raises():
    with pytest.raises(AnalysisError):
        run(analyze_part(IMG, None, "", "", providers(vision=BoomVision()), CFG))


def test_stt_failure_degrades_gracefully():
    r = run(analyze_part(IMG, b"a", "n.webm", "tested working", providers(stt=BoomSTT()), CFG))
    assert "voice_note_unavailable" in r.flags and r.condition == "Used - Tested Working"


def test_total_under_deadline_with_realistic_latency():
    p = providers(DemoVision(delay=0.4), prices=DemoPrices(delay=0.3))
    t = time.perf_counter()
    r = run(analyze_part(IMG, b"a", "n.webm", "", p, CFG))
    assert time.perf_counter() - t < 15 and r.timings_ms["total"] < 15000


# ---------- OpenAI / SerpApi clients via mock transport ----------
VISION_JSON = {"part_name": "Drain Pump", "brand": "Bosch", "part_number": "00145787",
               "part_number_confidence": 0.95, "legibility": "clear", "model_numbers": ["WAE24060UC"]}


def mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_openai_vision_parses_and_sends_expected_request():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(VISION_JSON)}}]})

    v = OpenAIVision("sk-test", client=mock_client(handler))
    res = run(v.analyze(b"x" * 10, "note"))
    assert res.part_number == "00145787" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "gpt-4o" and seen["body"]["temperature"] == 0
    assert "NEVER guess" in seen["body"]["messages"][0]["content"]
    assert seen["body"]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("resp", [
    httpx.Response(500, text="err"),
    httpx.Response(200, json={"oops": 1}),
    httpx.Response(200, json={"choices": [{"message": {"content": "not json at all"}}]}),
    httpx.Response(200, json={"choices": [{"message": {"content": '{"part_number_confidence": 7}'}}]}),
])
def test_openai_vision_errors_are_provider_errors(resp):
    v = OpenAIVision("k", client=mock_client(lambda r: resp))
    with pytest.raises(ProviderError):
        run(v.analyze(b"x"))


def test_openai_vision_extracts_json_wrapped_in_prose():
    wrapped = "Here you go:\n" + json.dumps(VISION_JSON)
    v = OpenAIVision("k", client=mock_client(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": wrapped}}]})))
    assert run(v.analyze(b"x")).brand == "Bosch"


def test_whisper():
    v = OpenAIWhisper("k", client=mock_client(lambda r: httpx.Response(200, json={"text": " tested working "})))
    assert run(v.transcribe(b"a", "n.webm")) == "tested working"
    bad = OpenAIWhisper("k", client=mock_client(lambda r: httpx.Response(429)))
    with pytest.raises(ProviderError):
        run(bad.transcribe(b"a", "n.webm"))


SERP = {"organic_results": [
    {"title": "Bosch pump", "price": {"raw": "$32.50", "extracted": 32.5}, "shipping": {"extracted": 8}},
    {"title": "range", "price": {"raw": "$20.00 to $30.00", "from": 20.0}},
    {"title": "no price"},
    {"title": "string price", "price": "$41.00"},
    {"title": "zero", "price": {"extracted": 0}},
]}


def test_serpapi_parse():
    c = parse_comps(SERP, sold=True)
    assert [x.price for x in c] == [32.5, 20.0, 41.0] and c[0].shipping == 8 and c[0].sold


def test_serpapi_sold_then_active_fallback():
    calls = []

    def handler(req):
        calls.append(dict(req.url.params))
        return httpx.Response(200, json={"organic_results": [{"title": "a", "price": {"extracted": 30}}]})

    s = SerpApiEbay("k", client=mock_client(handler))
    out = run(s.search("Bosch 00145787"))
    assert calls[0].get("LH_Sold") == "1" and "LH_Sold" not in calls[1]   # thin sold set -> add active
    assert len(out) == 2 and calls[0]["_nkw"] == "Bosch 00145787"
    with pytest.raises(ProviderError):
        run(SerpApiEbay("k", client=mock_client(lambda r: httpx.Response(401))).search("x"))


# ---------- exports ----------
def test_export_csv_formula_injection_neutralised():
    r = run(analyze_part(IMG, None, "", "", providers(), CFG)).model_dump()
    r["brand"] = '=HYPERLINK("http://evil")'
    r["id"], r["created_at"] = 1, "now"
    inv = to_inventory_csv([r])
    assert "'=HYPERLINK" in inv and ",=HYPERLINK" not in inv
    assert "Add" in to_ebay_csv([r])


# ---------- ambiguity gate + local OCR parsing ----------
def test_ambiguous_chars_flagged_unless_very_confident():
    sus = V.model_copy(update={"part_number": "W1I81723", "part_number_confidence": 0.8})
    r = run(analyze_part(IMG, None, "", "", providers(DemoVision(sus)), CFG))
    assert "part_number_ambiguous_chars" in r.flags and r.needs_review
    sure = sus.model_copy(update={"part_number_confidence": 0.97})
    assert "part_number_ambiguous_chars" not in run(analyze_part(IMG, None, "", "", providers(DemoVision(sure)), CFG)).flags
    clean = V.model_copy(update={"part_number": "DC92-58887", "part_number_confidence": 0.8})
    assert "part_number_ambiguous_chars" not in run(analyze_part(IMG, None, "", "", providers(DemoVision(clean)), CFG)).flags


def test_missing_part_name_is_warning_not_blocker():
    v = V.model_copy(update={"part_name": ""})
    r = run(analyze_part(IMG, None, "", "", providers(DemoVision(v)), CFG))
    assert "part_name_not_detected" in r.flags and not r.needs_review


def test_local_ocr_parse():
    from app.providers.local_ocr import parse_ocr
    lines = [("BOSCH", .99), ("Drain Pmp", .95), ("P/N: 00145787", .93), ("MODEL: WAE24060UC", .9)]
    v = parse_ocr(lines, blur=200)
    assert (v.brand, v.part_name, v.part_number) == ("Bosch", "Drain Pump", "00145787")
    assert v.model_numbers == ["WAE24060UC"] and v.legibility == "clear"
    split = parse_ocr([("P/N", .9), ("DC92-00001A", .88)], blur=200)   # label and value on separate lines
    assert split.part_number == "DC92-00001A"
    none = parse_ocr([], blur=0)
    assert none.part_number == "" and none.legibility == "none"
    blurry = parse_ocr([("P/N: 00145787", .93)], blur=5)
    assert blurry.part_number_confidence < 0.75   # sharpness discount pushes it to review
