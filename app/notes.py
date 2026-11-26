"""Rule-based extraction from the technician's note. Deterministic, testable, free."""
from __future__ import annotations

import re

from .schemas import NoteFacts

_PARTS = re.compile(r"\b(for parts|parts only|not working|doesn'?t work|does not work|"
                    r"broken|faulty|dead|burnt|burned|cracked|shorted)\b", re.I)
_NEGATED_WORKING = re.compile(r"\b(not|n't|never|untested|un-tested|not tested)\b.{0,12}\b(tested|working)\b", re.I)
_WORKING = re.compile(r"\b(tested (and )?(working|good|ok)|works? (fine|great|perfectly)|"
                      r"working( fine| perfectly| great)?|in working order|pulled (from|out of) (a )?working)\b", re.I)
_UNTESTED = re.compile(r"\b(untested|not tested|haven'?t tested|unknown condition|as[- ]is)\b", re.I)
# e.g. WAE24060UC, WM3500CW, RF28R7351SR  (letters+digits, 6-16 chars, at least 3 digits)
_MODEL = re.compile(r"\b(?=[A-Z0-9]*\d{3})[A-Z]{1,5}\d{2,}[A-Z0-9]{0,8}\b")


def parse_note(text: str) -> NoteFacts:
    t = (text or "").strip()
    if not t:
        return NoteFacts()
    if _PARTS.search(t):
        status = "for_parts"
    elif _UNTESTED.search(t) or _NEGATED_WORKING.search(t):
        status = "untested"
    elif _WORKING.search(t):
        status = "working"
    else:
        status = "untested"
    models = sorted(set(m.upper() for m in _MODEL.findall(t.upper())))
    return NoteFacts(functional_status=status, model_numbers=models, condition_notes=t[:300])
