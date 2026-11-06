"""Consent-based NANNET device leases and issue reporting.

Device identity is a random client-held secret, never a MAC/IMEI/serial.
A copied credential cannot obtain a second simultaneous lease. Slots are
released transactionally and become available to the next installation.
"""
from __future__ import annotations
import hashlib, json, secrets, sqlite3, time, uuid
from pathlib import Path

TERMS_VERSION = "nannet-device-registration-v1"
LEASE_SECONDS = 3600
MAX_REPORT = 8192

def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

class DeviceRegistry:
    def __init__(self, database: str | Path, lease_seconds: int = LEASE_SECONDS, authority=None):
        self.database = str(database)
        self.lease_seconds = max(300, min(int(lease_seconds), 86400))
        self.authority = authority
        Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY, secret_hash TEXT UNIQUE NOT NULL,
                created_at INTEGER NOT NULL, last_seen INTEGER NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS device_leases (
                lease_id TEXT PRIMARY KEY, device_id TEXT NOT NULL,
                hw_slot INTEGER UNIQUE, token_hash TEXT UNIQUE NOT NULL,
                issued_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
                released_at INTEGER, FOREIGN KEY(device_id) REFERENCES devices(device_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS device_reports (
                report_id TEXT PRIMARY KEY, device_id TEXT NOT NULL,
                report_type TEXT NOT NULL, summary TEXT NOT NULL,
                details TEXT NOT NULL, system_report TEXT NOT NULL,
                created_at INTEGER NOT NULL)""")

    def _new_token(self, device_id: str, now: int | None = None) -> str:
        if self.authority is None:
            return "nnt_" + secrets.token_urlsafe(32)
        issued = self.authority.issue_identity_token(
            "guest", device_id,
            ["universe:read", "snapshot:read", "corpus:write", "training:execute"],
            ttl_seconds=self.lease_seconds, now=now)
        return "nnt_" + issued["token"]

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=15, isolation_level="IMMEDIATE")
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def register(self, payload: dict, now: int | None = None) -> dict:
        now = int(time.time()) if now is None else int(now)
        if not isinstance(payload, dict) or payload.get("consent") is not True:
            raise ValueError("explicit consent is required")
        if payload.get("terms_version") != TERMS_VERSION:
            raise ValueError("unsupported terms version")
        secret = payload.get("device_secret")
        report = payload.get("system_report", {})
        if not isinstance(secret, str) or not 32 <= len(secret) <= 256:
            raise ValueError("invalid device credential")
        if not isinstance(report, dict) or len(json.dumps(report)) > 4096:
            raise ValueError("invalid system report")
        secret_hash = _digest(secret)
        lease_id = str(uuid.uuid4())
        placeholder_hash = _digest("pending:" + secrets.token_urlsafe(32))
        with self._connect() as db:
            db.execute("UPDATE device_leases SET hw_slot=NULL,released_at=? WHERE released_at IS NULL AND expires_at<=?", (now, now))
            row = db.execute("SELECT device_id FROM devices WHERE secret_hash=?", (secret_hash,)).fetchone()
            if row:
                device_id = row["device_id"]
                active = db.execute("SELECT 1 FROM device_leases WHERE device_id=? AND released_at IS NULL AND expires_at>?", (device_id, now)).fetchone()
                if active:
                    raise ValueError("device already has an active lease; cloned credentials are not eligible")
                db.execute("UPDATE devices SET last_seen=? WHERE device_id=?", (now, device_id))
            else:
                device_id = uuid.uuid4().hex
                db.execute("INSERT INTO devices VALUES (?,?,?,?)", (device_id, secret_hash, now, now))
            used = {r[0] for r in db.execute("SELECT hw_slot FROM device_leases WHERE hw_slot IS NOT NULL AND released_at IS NULL AND expires_at>?", (now,))}
            slot = next((n for n in range(4, 256) if n not in used), None)
            if slot is None:
                raise ValueError("no hardware slots currently available")
            expires = now + self.lease_seconds
            db.execute("INSERT INTO device_leases VALUES (?,?,?,?,?,?,NULL)", (lease_id, device_id, slot, placeholder_hash, now, expires))
        try:
            token = self._new_token(device_id)
            with self._connect() as db:
                updated = db.execute("UPDATE device_leases SET token_hash=? WHERE lease_id=? AND token_hash=?", (_digest(token), lease_id, placeholder_hash))
                if updated.rowcount != 1:
                    raise ValueError("device lease changed during credential issuance")
        except Exception:
            with self._connect() as db:
                db.execute("UPDATE device_leases SET hw_slot=NULL,released_at=? WHERE lease_id=?", (int(time.time()), lease_id))
            raise
        return {"device_id": device_id, "address": f"nan://GM0:SC0:MC0:BS0:HW{slot}", "lease_id": lease_id, "lease_expires_at": expires, "token": token, "head_count": 256}

    def _active(self, db, token: str, now: int):
        if not isinstance(token, str) or not token.startswith("nnt_"):
            raise ValueError("device bearer required")
        row = db.execute("""SELECT l.*,d.device_id FROM device_leases l
            JOIN devices d ON d.device_id=l.device_id
            WHERE l.token_hash=? AND l.released_at IS NULL AND l.expires_at>?""", (_digest(token), now)).fetchone()
        if not row:
            raise ValueError("device token invalid, expired, or released")
        if self.authority is not None:
            try:
                claims = self.authority.verify_access_token(token[4:], required_scope="training:execute", now=now)
                if claims.get("role") != "guest" or claims.get("sub") != row["device_id"]:
                    raise ValueError("guest identity mismatch")
            except Exception as exc:
                raise ValueError("device token invalid, expired, revoked, or out of scope") from exc
        return row

    def heartbeat(self, token: str, now: int | None = None) -> dict:
        now = int(time.time()) if now is None else int(now)
        with self._connect() as db:
            row = self._active(db, token, now)
            lease_id, device_id, slot = row["lease_id"], row["device_id"], row["hw_slot"]
        expires = now + self.lease_seconds
        replacement = self._new_token(device_id)
        with self._connect() as db:
            updated = db.execute(
                "UPDATE device_leases SET token_hash=?,expires_at=? WHERE lease_id=? AND token_hash=? AND released_at IS NULL AND expires_at>?",
                (_digest(replacement), expires, lease_id, _digest(token), now))
            if updated.rowcount != 1:
                raise ValueError("device lease changed during heartbeat")
            db.execute("UPDATE devices SET last_seen=? WHERE device_id=?", (now, device_id))
        return {"lease_id": lease_id, "lease_expires_at": expires,
                "address": f"nan://GM0:SC0:MC0:BS0:HW{slot}", "token": replacement}

    def release(self, token: str, now: int | None = None) -> dict:
        now = int(time.time()) if now is None else int(now)
        with self._connect() as db:
            row = self._active(db, token, now)
            address = f"nan://GM0:SC0:MC0:BS0:HW{row['hw_slot']}"
            db.execute("UPDATE device_leases SET hw_slot=NULL,released_at=? WHERE lease_id=?", (now, row["lease_id"]))
        return {"released": True, "address": address}

    def rotate(self, token: str, now: int | None = None) -> dict:
        """Rotate a live bearer without creating a second device or lease."""
        now = int(time.time()) if now is None else int(now)
        with self._connect() as db:
            row = self._active(db, token, now)
            lease_id, device_id, slot = row["lease_id"], row["device_id"], row["hw_slot"]
        replacement = self._new_token(device_id)
        expires = now + self.lease_seconds
        with self._connect() as db:
            updated = db.execute(
                "UPDATE device_leases SET token_hash=?,expires_at=? WHERE lease_id=? AND token_hash=? AND released_at IS NULL AND expires_at>?",
                (_digest(replacement), expires, lease_id, _digest(token), now))
            if updated.rowcount != 1:
                raise ValueError("device lease changed during token rotation")
            db.execute("UPDATE devices SET last_seen=? WHERE device_id=?", (now, device_id))
        return {"lease_id": lease_id, "lease_expires_at": expires,
                "address": f"nan://GM0:SC0:MC0:BS0:HW{slot}", "token": replacement}

    def report(self, token: str, payload: dict, now: int | None = None) -> dict:
        now = int(time.time()) if now is None else int(now)
        if not isinstance(payload, dict): raise ValueError("invalid report")
        kind, summary, details = payload.get("report_type"), payload.get("summary"), payload.get("details", "")
        allowed = {"device_error", "system_error", "network_error", "security_concern", "other"}
        if kind not in allowed or not isinstance(summary, str) or not 1 <= len(summary) <= 240 or not isinstance(details, str) or len(details) > MAX_REPORT:
            raise ValueError("invalid report fields")
        system = payload.get("system_report", {})
        if not isinstance(system, dict) or len(json.dumps(system)) > 4096: raise ValueError("invalid system report")
        report_id = str(uuid.uuid4())
        with self._connect() as db:
            row = self._active(db, token, now)
            db.execute("INSERT INTO device_reports VALUES (?,?,?,?,?,?,?)", (report_id, row["device_id"], kind, summary, details, json.dumps(system, sort_keys=True), now))
        return {"accepted": True, "report_id": report_id}
