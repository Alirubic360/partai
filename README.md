# PartAi — The 90-Second Inventory

Photo + 10-second voice note of a used appliance part -> eBay-ready listing JSON/CSV with three price points.
Separate product from the AutoRankLab WordPress site; nothing here touches the site.

## Status (read this first)
| Piece | State |
|---|---|
| FastAPI backend, quota/plans, exports, PWA | built, tested (46 tests) |
| OpenAI Vision / Whisper / SerpApi clients | built, tested against **mocked** HTTP only |
| Live accuracy (the "99%" goal) | **not measured** — needs real keys + labeled photos (`eval/run_eval.py`) |
| Billing (Stripe), email verification, rate limiting | **not built** |

## Run
```bash
pip install -r requirements.txt
PARTAI_MODE=demo uvicorn app.main:app_factory --factory --port 8000   # canned results, flagged demo=true
# live: export PARTAI_MODE=live OPENAI_API_KEY=... SERPAPI_API_KEY=...
pytest -q
```
Open http://localhost:8000 on a phone (HTTPS required for camera/mic outside localhost).

## API (header `X-API-Key`)
- `POST /api/v1/signup {email}` -> key (shown once) · `GET /api/v1/me` -> plan + usage
- `POST /api/v1/parts/analyze` multipart: `image`, optional `audio`, optional `note` -> listing JSON
- `GET /api/v1/parts` · `GET /api/v1/parts/export?fmt=inventory_csv|ebay_csv|ebay_json`
- Set a plan: `Store(path).set_plan("email", "pro")` (until Stripe is wired)

## Design rules that protect sellers
- Part number is never guessed: low confidence or blurry label -> blank + `needs_review` + flags.
- "Tested working" appears only if the technician said so; otherwise "Used - Untested".
- `compatible_models` lists only models printed on the label or spoken, not model-memory guesses.
- Pricing uses sold+completed eBay comps first, IQR outlier removal; confidence = sample size.
- 15 s end-to-end budget: vision+Whisper parallel; slow pricing returns a partial listing, flagged.
- Free plan: 10 parts/month, consumed only on success. CSV export neutralises formula injection.

## Measure real accuracy
Put 100+ real photos + `labels.json` in a folder, then `python -m eval.run_eval DIR --live`.
Watch `wrong_but_published`: unflagged wrong part numbers are the metric that costs money.
