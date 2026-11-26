from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings
from .export import to_ebay_csv, to_ebay_json, to_inventory_csv
from .imageprep import BadImage, prepare_image
from .pipeline import AnalysisError, Providers, analyze_part
from .providers.demo import DemoPrices, DemoTranscriber, DemoVision
from .providers.openai_client import OpenAIVision, OpenAIWhisper
from .providers.serpapi_client import SerpApiEbay
from .schemas import AnalysisResult
from .store import Store

STATIC = Path(__file__).parent / "static"
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")


def build_providers(cfg: Settings) -> Providers:
    if cfg.mode == "live":
        if not (cfg.openai_api_key and cfg.serpapi_api_key):
            raise RuntimeError("live mode needs OPENAI_API_KEY and SERPAPI_API_KEY")
        return Providers(OpenAIVision(cfg.openai_api_key, cfg.vision_model, cfg.vision_timeout),
                         OpenAIWhisper(cfg.openai_api_key, cfg.whisper_model),
                         SerpApiEbay(cfg.serpapi_api_key, cfg.pricing_timeout), demo=False)
    if cfg.mode == "local":  # free offline OCR for labels; prices still demo-grade
        from .providers.local_ocr import LocalOCRVision
        return Providers(LocalOCRVision(), None, DemoPrices(), demo=True)
    return Providers(DemoVision(), DemoTranscriber(), DemoPrices(), demo=True)


class SignupBody(BaseModel):
    email: str


def create_app(cfg: Settings | None = None, providers: Providers | None = None,
               store: Store | None = None) -> FastAPI:
    cfg = cfg or Settings.from_env()
    providers = providers or build_providers(cfg)
    store = store or Store(cfg.db_path, cfg.free_monthly_limit)
    app = FastAPI(title="PartAi", version="0.1.0")

    def tenant(x_api_key: str = Header(default="")) -> dict:
        t = store.tenant_by_key(x_api_key) if x_api_key else None
        if not t:
            raise HTTPException(401, "missing or invalid X-API-Key")
        return t

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": cfg.mode, "demo": providers.demo}

    @app.post("/api/v1/signup", status_code=201)
    def signup(body: SignupBody):
        if not _EMAIL.match(body.email):
            raise HTTPException(422, "invalid email")
        key = store.create_tenant(body.email)
        if key is None:
            raise HTTPException(409, "email already registered")
        return {"api_key": key, "plan": "free", "monthly_limit": cfg.free_monthly_limit,
                "note": "Store this key now; it cannot be shown again."}

    @app.get("/api/v1/me")
    def me(t: dict = Depends(tenant)):
        return store.usage(t)

    @app.post("/api/v1/parts/analyze", response_model=AnalysisResult)
    async def analyze(image: UploadFile = File(...), audio: UploadFile | None = File(None),
                      note: str = Form(""), t: dict = Depends(tenant)):
        u = store.usage(t)
        if u["remaining"] == 0:
            raise HTTPException(402, f"Free plan limit reached ({u['monthly_limit']} parts/month). Upgrade to Pro.")
        raw = await image.read(cfg.max_image_bytes + 1)
        if len(raw) > cfg.max_image_bytes:
            raise HTTPException(413, "image too large (max 10 MB)")
        try:
            jpeg = await asyncio.to_thread(prepare_image, raw)  # CPU-bound: keep the event loop free
        except BadImage as exc:
            raise HTTPException(415, str(exc))
        audio_bytes, audio_name = None, ""
        if audio is not None:
            audio_bytes = await audio.read(cfg.max_audio_bytes + 1)
            if len(audio_bytes) > cfg.max_audio_bytes:
                raise HTTPException(413, "audio too large (max 5 MB)")
            audio_name = audio.filename or "note.webm"
        try:
            result = await analyze_part(jpeg, audio_bytes, audio_name, note[:500], providers, cfg)
        except AnalysisError as exc:
            raise HTTPException(502, f"analysis unavailable: {exc}")
        store.add_item(t, result.model_dump())  # quota is consumed only on success
        return result

    @app.get("/api/v1/parts")
    def parts(t: dict = Depends(tenant)):
        return store.list_items(t)

    @app.get("/api/v1/parts/export")
    def export(fmt: str = Query("inventory_csv", pattern="^(inventory_csv|ebay_csv|ebay_json)$"),
               t: dict = Depends(tenant)):
        items = store.list_items(t, limit=1000)
        if fmt == "ebay_json":
            return JSONResponse([to_ebay_json(i) for i in items])
        body = to_ebay_csv(items) if fmt == "ebay_csv" else to_inventory_csv(items)
        return Response(body, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="partai_{fmt}.csv"'})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/", StaticFiles(directory=STATIC), name="static")
    return app


def app_factory() -> FastAPI:  # uvicorn --factory app.main:app_factory
    return create_app()
