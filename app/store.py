"""SQLite persistence: tenants (API keys hashed), plans, monthly quota, saved items."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, key_hash TEXT UNIQUE NOT NULL,
  plan TEXT NOT NULL DEFAULT 'free', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items(
  id INTEGER PRIMARY KEY, tenant_id INTEGER NOT NULL, created_at TEXT NOT NULL,
  month TEXT NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_items_tenant_month ON items(tenant_id, month);
"""
PLANS = {"free": 10, "pro": None, "enterprise": None}  # None = unlimited


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


class Store:
    def __init__(self, path: str, free_limit: int = 10):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.free_limit = free_limit
        with self._tx() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def create_tenant(self, email: str) -> str | None:
        key = "pk_" + secrets.token_urlsafe(24)
        try:
            with self._tx() as db:
                db.execute("INSERT INTO tenants(email,key_hash,created_at) VALUES(?,?,?)",
                           (email.lower(), _hash(key), datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError:
            return None
        return key

    def tenant_by_key(self, key: str) -> dict | None:
        with self._tx() as db:
            r = db.execute("SELECT * FROM tenants WHERE key_hash=?", (_hash(key),)).fetchone()
        return dict(r) if r else None

    def set_plan(self, email: str, plan: str) -> bool:
        if plan not in PLANS:
            raise ValueError(f"unknown plan {plan}")
        with self._tx() as db:
            return db.execute("UPDATE tenants SET plan=? WHERE email=?", (plan, email.lower())).rowcount > 0

    def usage(self, tenant: dict) -> dict:
        with self._tx() as db:
            used = db.execute("SELECT COUNT(*) FROM items WHERE tenant_id=? AND month=?",
                              (tenant["id"], _month())).fetchone()[0]
        limit = self.free_limit if tenant["plan"] == "free" else PLANS[tenant["plan"]]
        return {"plan": tenant["plan"], "used_this_month": used, "monthly_limit": limit,
                "remaining": None if limit is None else max(0, limit - used)}

    def add_item(self, tenant: dict, data: dict) -> int:
        with self._tx() as db:
            cur = db.execute("INSERT INTO items(tenant_id,created_at,month,data) VALUES(?,?,?,?)",
                             (tenant["id"], datetime.now(timezone.utc).isoformat(), _month(), json.dumps(data)))
            return cur.lastrowid

    def list_items(self, tenant: dict, limit: int = 200) -> list[dict]:
        with self._tx() as db:
            rows = db.execute("SELECT id,created_at,data FROM items WHERE tenant_id=? ORDER BY id DESC LIMIT ?",
                              (tenant["id"], limit)).fetchall()
        return [{"id": r["id"], "created_at": r["created_at"], **json.loads(r["data"])} for r in rows]
