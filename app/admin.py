"""Owner CLI.  python -m app.admin waitlist | plan EMAIL free|pro|enterprise"""
from __future__ import annotations

import sys

from .config import Settings
from .store import Store


def main(argv: list[str]) -> int:
    cfg = Settings.from_env()
    st = Store(cfg.db_path, cfg.free_monthly_limit)
    if argv[:1] == ["waitlist"]:
        for r in st.waitlist():
            print(f"{r['created_at'][:19]}\t{r['email']}\t{r['source']}")
        return 0
    if len(argv) == 3 and argv[0] == "plan":
        print("updated" if st.set_plan(argv[1], argv[2]) else "no such user")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
