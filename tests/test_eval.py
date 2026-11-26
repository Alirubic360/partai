import asyncio
import io
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.pipeline import Providers
from app.providers.demo import DemoPrices
from app.schemas import VisionResult
from eval.run_eval import evaluate


class Scripted:
    """Vision stub returning a scripted answer per call, to validate the metrics math."""
    def __init__(self, answers):
        self.answers, self.i = answers, 0

    async def analyze(self, img, hint=""):
        a = self.answers[self.i]
        self.i += 1
        return a


def make_dir(tmp_path: Path, n: int, truth="00145787"):
    labels = []
    for i in range(n):
        Image.new("RGB", (400, 300), (i * 10, 50, 50)).save(tmp_path / f"{i}.jpg")
        labels.append({"image": f"{i}.jpg", "part_number": truth, "brand": "Bosch"})
    return labels


def vr(pn, conf, leg="clear"):
    return VisionResult(part_name="Drain Pump", brand="Bosch", part_number=pn,
                        part_number_confidence=conf, legibility=leg)


def test_eval_metrics(tmp_path):
    labels = make_dir(tmp_path, 4)
    answers = [vr("00145787", .97),            # correct, published
               vr("00145787", .97),            # correct, published
               vr("00145788", .97),            # WRONG but confident -> wrong_but_published
               vr("0014?787", .30, "blurry")]  # unreadable -> flagged, not published
    prov = Providers(Scripted(answers), None, DemoPrices())
    m = asyncio.run(evaluate(labels, tmp_path, prov, Settings()))
    assert m["n"] == 4 and m["pn_exact_match"] == 0.5
    assert m["flagged_rate"] == 0.25
    assert m["wrong_but_published"] == 1
    assert m["published_precision"] == round(2 / 3, 4)
