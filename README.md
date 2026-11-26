# PartAi — The 90-Second Inventory

Photo + 10-second voice note of a used appliance part -> eBay-ready listing JSON/CSV with three price points.
Separate product from the AutoRankLab WordPress site; nothing here touches the site.

## Status (read this first)
| Piece | State |
|---|---|
| FastAPI backend, quota/plans, exports, PWA | built, tested (46 tests) |
| OpenAI Vision / Whisper / SerpApi clients | built, tested against **mocked** HTTP only |
| Live accuracy with GPT-4o (the "99%" goal) | **not measured** — needs real keys + real labeled photos |
| Offline OCR mode (`PARTAI_MODE=local`) | built; measured on **synthetic** labels only (see below) |
| Technician confirm/correct step (PWA + `PATCH /api/v1/parts/{id}`), rate limits | built, tested |
| Dockerfile, GitHub Actions CI | written; **not run** (no Docker here; CI runs on first push) |
| Gemini vision provider, in-browser dictation, `/landing` waitlist page | built; Gemini tested with **mocked** HTTP only; dictation not testable headless (try in phone Chrome) |
| Billing (Stripe), email verification, HTTPS/hosting | **not built** |

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
- `PATCH /api/v1/parts/{id}` correct part_number/brand/title/price_avg; marks the item confirmed
- Limits: 5 signups/hour per IP, 30 analyses/minute per key (in-memory; use Redis if you scale out)
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

## Offline OCR mode (free, no API keys)
`pip install -r requirements-local.txt` then `PARTAI_MODE=local`. Reads printed label text only
(brand, P/N, model, part name); cannot judge condition. Prices are still demo until SerpApi is set.

Benchmark on **synthetic** label images (`python -m eval.make_synthetic DIR N SEED`, then
`PARTAI_MODE=local python -m eval.run_eval DIR`). Seed 99 / 200 images was never used for tuning:

| Difficulty | P/N read correctly | Sent to review | Wrong P/N published |
|---|---|---|---|
| clean | 100% | 6% | 0 |
| slight blur | 100% | 4% | 0 |
| heavy blur | 52% | 72% | 0 |
| unreadable | 0% | 100% | 0 |

Latency ~0.2 s/image. Synthetic labels are cleaner than real parts (no glare, grime, curved or
embossed text), so real-world numbers will be lower. Measure on real photos before trusting it.

### Hard mode (field-like damage: glare, tilt, low-res, heavy JPEG) — 200 synthetic images
`python -m eval.make_synthetic DIR 200 5 --hard`. Local OCR reads each image twice with different
preprocessing and publishes a part number only when both passes agree (cut wrong-but-published from 7 to 3).

| Difficulty | P/N correct | Sent to review | Wrong P/N published |
|---|---|---|---|
| clean | 68% | 40% | 3 |
| slight blur | 84% | 42% | 0 |
| heavy blur / unreadable | 10% / 0% | 100% | 0 |

Remaining failures are consistent truncations/digit slips that survive both passes. That is why the
PWA shows the part number in an editable field with a "Confirm" button: **a human must check the P/N
against the label before listing.** Do not auto-publish from OCR output alone.

## Zero-cost setup
| Need | Free option |
|---|---|
| Read labels | `PARTAI_MODE=local` (offline OCR, no key) |
| Vision model | `PARTAI_MODE=live PARTAI_VISION=gemini GEMINI_API_KEY=...` (free key at aistudio.google.com/apikey; check current quota) |
| Voice note | Browser dictation button (Chrome) fills the note field; no Whisper needed |
| Prices | none yet: with no `SERPAPI_API_KEY` live mode returns **no price + a flag**, never invented numbers |
| Early-access page | `/landing?src=reddit`; list emails with `python -m app.admin waitlist` |

Live mode picks OpenAI if `OPENAI_API_KEY` is set, else Gemini; override with `PARTAI_VISION=openai|gemini|local`.
Live mode with no vision key refuses to start instead of silently degrading.
