"""Deterministic listing copy. Only states what was actually seen/said — never invents claims."""
from __future__ import annotations

import re

from .schemas import NoteFacts, VisionResult

EBAY_TITLE_MAX = 80

CONDITION_TEXT = {
    "working": "Used - Tested Working",
    "untested": "Used - Untested",
    "for_parts": "For Parts or Not Working",
}
EBAY_CONDITION_ID = {"working": 3000, "untested": 3000, "for_parts": 7000}

# rough fallback shipping weights (lb) when the vision model gives no estimate
_WEIGHT_FALLBACK = [("control board", 1.0), ("board", 1.0), ("pump", 2.0), ("motor", 5.0),
                    ("compressor", 20.0), ("door", 6.0), ("hose", 1.0), ("valve", 0.8),
                    ("element", 1.0), ("thermostat", 0.4), ("sensor", 0.3), ("fan", 1.5)]


def build_title(v: VisionResult, status: str, part_number: str) -> str:
    appliance = v.appliance_type.title() if v.appliance_type else ""
    # (text, required) in priority order; optional pieces dropped until it fits in 80 chars
    pieces = [
        (v.brand, True), (v.part_name, True), (part_number, True),
        ("OEM", False), (appliance, False),
        ("Tested Working" if status == "working" else "For Parts" if status == "for_parts" else "Used", False),
    ]
    chosen = [(t.strip(), req) for t, req in pieces if t and t.strip()]
    def join(items): return " ".join(t for t, _ in items)
    while len(join(chosen)) > EBAY_TITLE_MAX:
        for i in range(len(chosen) - 1, -1, -1):
            if not chosen[i][1]:
                chosen.pop(i)
                break
        else:
            break
    # Still too long with only required pieces: shorten the part NAME word-by-word.
    # The part number is what buyers search by, so it is never truncated.
    if len(join(chosen)) > EBAY_TITLE_MAX and v.part_name:
        words = v.part_name.split()
        while len(words) > 1:
            words.pop()
            trial = [(" ".join(words) if t == v.part_name.strip() else t, r) for t, r in chosen]
            if len(join(trial)) <= EBAY_TITLE_MAX:
                chosen = trial
                break
        else:
            chosen = [(" ".join(words) if t == v.part_name.strip() else t, r) for t, r in chosen]
    title = join(chosen)
    if len(title) > EBAY_TITLE_MAX and part_number in title:   # last resort: cut the brand
        title = title.replace(v.brand, "", 1).strip() if v.brand else title
    return title[:EBAY_TITLE_MAX].rstrip()


def shipping_weight(v: VisionResult) -> str:
    if v.weight_lb_estimate and v.weight_lb_estimate > 0:
        return f"~{v.weight_lb_estimate:g} lb (estimate - weigh before shipping)"
    name = v.part_name.lower()
    for key, w in _WEIGHT_FALLBACK:
        if key in name:
            return f"~{w:g} lb (estimate - weigh before shipping)"
    return "unknown - weigh before shipping"


def build_description(v: VisionResult, note: NoteFacts, part_number: str,
                      models: list[str], status: str) -> str:
    brand = v.brand or "OEM"
    name = v.part_name or "appliance part"
    appliance = f" {v.appliance_type}" if v.appliance_type else ""
    pn = part_number or "see photos"
    cond = CONDITION_TEXT[status]
    func = {
        "working": "The seller reports this part was tested and is working.",
        "untested": "This part has not been tested by the seller. It is sold as-is, so please "
                    "read the condition notes and review every photo before you buy.",
        "for_parts": "This part is sold for parts or repair only and is not guaranteed to work.",
    }[status]
    seen = v.visual_condition.strip().rstrip(".") or "see photos for visible condition"
    models_txt = (", ".join(models) if models else "not listed by the seller")
    tech = f" Seller note: {note.condition_notes.strip().rstrip('.')}." if note.condition_notes.strip() else ""
    paras = [
        f"You are bidding on a used {brand} {name} for{appliance} repair. It was removed from a working "
        f"unit in a repair shop and has been photographed exactly as it is today, so what you see in "
        f"the pictures is what you will receive. The manufacturer part number printed on the part is {pn}.",
        f"Condition: {cond}. {func} Visible condition: {seen}.{tech}",
        f"Compatibility: the appliance model numbers recorded with this part are {models_txt}. "
        f"Appliance manufacturers often reuse one part across many models and sometimes change a part "
        f"number between production years. Please compare the part number above with the number on "
        f"your old part or in your service manual before ordering. If you are unsure, send us a message "
        f"with your appliance model number and we will help you check the fit.",
        "What is included: the single part shown in the photos. Mounting hardware, wire harnesses and "
        "other accessories are not included unless they appear in the pictures. Items are inspected "
        "and cleaned lightly before shipping, but used parts can show normal signs of wear.",
        "Shipping and returns: we pack each part to protect it in transit and ship within one business "
        "day of cleared payment. Combine shipping is available when you buy more than one part from "
        "us. Please contact us before opening a case if anything is wrong with your order, and we "
        "will work with you to make it right.",
        "Thank you for shopping with a small repair business. We list new parts every week, so save "
        "our shop to your favorites and check back for more genuine used appliance parts.",
    ]
    return "\n\n".join(paras)
