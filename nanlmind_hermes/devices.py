"""Consent-based NANNET device leases and issue reporting.

Device identity is a random client-held secret, never a MAC/IMEI/serial.
A copied credential cannot obtain a second simultaneous lease. BS0:0 is the
shared runner-ring NANNET port; HW0..HW3 are fixed hardware endpoints, HW4
belongs to WordNet, and HW5..HW255 are available device leases. A hardware
lease and a separately router-authorized guest universe are different records.
"""
from __future__ import annotations
import hashlib, json, secrets, sqlite3, time, uuid
from pathlib import Path

TERMS_VERSION = "nannet-device-registration-v1"
LEASE_SECONDS = 3600
MAX_REPORT = 8192

# Fixed reserved HERMES hardware endpoint. These values are authority-owned:
# callers may not choose or override the address/node point. Dynamic device
# registration remains limited to HW5..HW255 and never allocates this slot.
FIXED_HW2_ENDPOINT = "HW2"
FIXED_HW2_SLOT = 2
FIXED_HW2_NODE_POINT = 3
FIXED_HW2_ADDRESS = "nan://GM0:SC0:MC0:BS0:0:HW2:0"
FIXED_HW2_SCOPE = "installation:admin"
FIXED_HW2_ALT_SCOPE = "hermes:lease"
FIXED_HW2_SCOPES = ("corpus:write", "snapshot:read", "training:execute", "universe:read")
FIXED_HW2_BIND_PATH = "/v1/hermes/hw2/bind"
FIXED_HW2_STATUS_PATH = "/v1/hermes/hw2/status"
FIXED_HW2_REPORT_PATH = "/v1/hermes/hw2/report"
FIXED_HW2_REVOKE_PATH = "/v1/hermes/hw2/revoke"

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
            # Permanent fixed-endpoint bindings are separate from the dynamic
            # HW5..HW255 participant lease pool. The partial unique indexes let
            # an explicitly revoked binding be replaced without allowing two
            # installations to hold HW2 simultaneously.
            db.execute("""CREATE TABLE IF NOT EXISTS fixed_hw_bindings (
                lease_id TEXT PRIMARY KEY, hw_slot INTEGER NOT NULL,
                device_id TEXT NOT NULL, address TEXT NOT NULL,
                node_point INTEGER NOT NULL, token_hash TEXT NOT NULL,
                token_jti TEXT, scopes TEXT NOT NULL,
                bound_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL, revoked_at INTEGER,
                revoked_by TEXT, revoked_reason TEXT)""")
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS fixed_hw_bindings_active_slot
                ON fixed_hw_bindings(hw_slot) WHERE revoked_at IS NULL""")
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS fixed_hw_bindings_active_address
                ON fixed_hw_bindings(address) WHERE revoked_at IS NULL""")
        # Upgrade active legacy participant leases out of the fixed endpoints.
        # HW0..HW3 and WordNet HW4 are reserved; only HW5+ is allocatable.
        # Preserve credentials and lease IDs; only the participant address moves.
        now = int(time.time())
        with self._connect() as db:
            legacy = db.execute(
                "SELECT lease_id FROM device_leases WHERE hw_slot IS NOT NULL "
                "AND hw_slot<5 AND released_at IS NULL AND expires_at>?",
                (now,)).fetchall()
            used = {row[0] for row in db.execute(
                "SELECT hw_slot FROM device_leases WHERE hw_slot>=5 "
                "AND released_at IS NULL AND expires_at>?", (now,))}
            for row in legacy:
                slot = next((n for n in range(5, 256) if n not in used), None)
                if slot is None:
                    raise RuntimeError("cannot migrate reserved HW0-HW4 lease: no HW5+ hardware address")
                db.execute("UPDATE device_leases SET hw_slot=? WHERE lease_id=?",
                           (slot, row["lease_id"]))
                used.add(slot)

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
            # BS0:0 is the shared runner-ring NANNET port. HW0..HW3 are
            # reserved hardware nodes, HW4 is WordNet; HW5+ is allocatable.
            # A leased hardware node alone does not grant guest universe access.
            used = {r[0] for r in db.execute(
                "SELECT hw_slot FROM device_leases WHERE hw_slot IS NOT NULL "
                "AND released_at IS NULL AND expires_at>?", (now,))}
            slot = next((n for n in range(5, 256) if n not in used), None)
            if slot is None:
                raise ValueError("no participant addresses currently available")
            expires = now + self.lease_seconds
            db.execute("INSERT INTO device_leases VALUES (?,?,?,?,?,?,NULL)",
                       (lease_id, device_id, slot, placeholder_hash, now, expires))
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
        return {"device_id": device_id, "participant_id": device_id,
                "address": f"nan://GM0:SC0:MC0:BS0:HW{slot}",
                "lease_id": lease_id, "lease_expires_at": expires,
                "token": token, "head_count": 256}

    def bind_fixed_hw2(self, installation_id: str, payload: dict | None = None,
                       now: int | None = None) -> dict:
        """Bind the authority-owned permanent HW2 endpoint to one installation.

        This is intentionally not part of dynamic HW5..HW255 allocation. The
        address and node point are constants defined by the authority; a caller
        can only confirm them, never select another value.
        """
        now = int(time.time()) if now is None else int(now)
        if (not isinstance(installation_id, str) or not installation_id
                or len(installation_id) > 256):
            raise ValueError("invalid installation identity")
        payload = {} if payload is None else payload
        if not isinstance(payload, dict):
            raise ValueError("invalid fixed HW2 binding request")
        if "address" in payload and payload.get("address") != FIXED_HW2_ADDRESS:
            raise ValueError("fixed HW2 address is assigned by the authority")
        if "node_point" in payload:
            point = payload.get("node_point")
            if isinstance(point, bool) or point != FIXED_HW2_NODE_POINT:
                raise ValueError("fixed HW2 node point is assigned by the authority")
        if self.authority is not None and hasattr(self.authority, "is_paired_installation"):
            if not self.authority.is_paired_installation(installation_id):
                raise ValueError("fixed HW2 binding requires a paired installation")

        scopes = list(FIXED_HW2_SCOPES)
        expires = now + self.lease_seconds
        placeholder_hash = _digest("fixed-hw2-pending:" + secrets.token_urlsafe(32))
        previous = None
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM fixed_hw_bindings WHERE hw_slot=? AND revoked_at IS NULL "
                "ORDER BY bound_at DESC LIMIT 1", (FIXED_HW2_SLOT,)).fetchone()
            if row is not None and row["device_id"] != installation_id:
                raise ValueError("fixed HW2 identity is already bound to another installation")
            if row is not None:
                lease_id = row["lease_id"]
                bound_at = row["bound_at"]
                previous = (row["token_hash"], row["token_jti"],
                            row["expires_at"], row["scopes"])
                db.execute(
                    "UPDATE fixed_hw_bindings SET token_hash=?,token_jti=NULL,scopes=?,"
                    "updated_at=?,expires_at=? WHERE lease_id=? AND revoked_at IS NULL",
                    (placeholder_hash, json.dumps(scopes), now, expires, lease_id))
                created = False
            else:
                lease_id = str(uuid.uuid4())
                bound_at = now
                db.execute(
                    "INSERT INTO fixed_hw_bindings "
                    "(lease_id,hw_slot,device_id,address,node_point,token_hash,"
                    "token_jti,scopes,bound_at,updated_at,expires_at,"
                    "revoked_at,revoked_by,revoked_reason) "
                    "VALUES (?,?,?,?,?,?,NULL,?,?,?,?,NULL,NULL,NULL)",
                    (lease_id, FIXED_HW2_SLOT, installation_id, FIXED_HW2_ADDRESS,
                     FIXED_HW2_NODE_POINT, placeholder_hash, json.dumps(scopes),
                     bound_at, now, expires))
                created = True
        try:
            token = self._new_token(installation_id, now=now)
            jti = None
            if self.authority is not None:
                try:
                    jti = self.authority._decode_claims(token[4:]).get("jti")
                except Exception:
                    jti = None
            with self._connect() as db:
                updated = db.execute(
                    "UPDATE fixed_hw_bindings SET token_hash=?,token_jti=?,"
                    "updated_at=?,expires_at=? WHERE lease_id=? AND token_hash=? "
                    "AND revoked_at IS NULL",
                    (_digest(token), jti, now, expires, lease_id, placeholder_hash))
                if updated.rowcount != 1:
                    raise ValueError("fixed HW2 binding changed during credential issuance")
        except Exception:
            with self._connect() as db:
                if created:
                    db.execute(
                        "DELETE FROM fixed_hw_bindings WHERE lease_id=? AND token_hash=?",
                        (lease_id, placeholder_hash))
                elif previous is not None:
                    db.execute(
                        "UPDATE fixed_hw_bindings SET token_hash=?,token_jti=?,"
                        "scopes=?,expires_at=? WHERE lease_id=? AND token_hash=?",
                        (previous[0], previous[1], previous[3], previous[2],
                         lease_id, placeholder_hash))
            raise
        return {"binding_id": lease_id, "installation_id": installation_id,
                "device_id": installation_id, "participant_id": installation_id,
                "address": FIXED_HW2_ADDRESS, "node_point": FIXED_HW2_NODE_POINT,
                "hw_slot": FIXED_HW2_SLOT, "fixed_endpoint": FIXED_HW2_ENDPOINT,
                "network_port": "BS0:0", "allocation": "fixed-reserved",
                "permanent": True, "revocable": True,
                "lease_id": lease_id, "lease_expires_at": expires,
                "expires_at": expires, "scopes": scopes, "role": "guest",
                "token": token, "token_type": "Bearer",
                "bound_at": bound_at, "updated_at": now}

    @staticmethod
    def _fixed_public_receipt(row) -> dict:
        """Return binding metadata without the bearer or any key material."""
        return {"fixed_hw2_bound": True, "bound": True,
                "installation_id": row["device_id"],
                "participant_id": row["device_id"],
                "address": row["address"], "node_point": row["node_point"],
                "hw_slot": row["hw_slot"], "fixed_endpoint": FIXED_HW2_ENDPOINT,
                "network_port": "BS0:0", "allocation": "fixed-reserved",
                "permanent": True, "lease_id": row["lease_id"],
                "scopes": json.loads(row["scopes"]),
                "bound_at": row["bound_at"], "updated_at": row["updated_at"],
                "lease_expires_at": row["expires_at"],
                "expires_at": row["expires_at"]}

    def _fixed_active(self, db, token: str, now: int):
        if not isinstance(token, str) or not token.startswith("nnt_"):
            raise ValueError("fixed HW2 bearer required")
        row = db.execute(
            "SELECT * FROM fixed_hw_bindings WHERE token_hash=? AND revoked_at IS NULL",
            (_digest(token),)).fetchone()
        if not row:
            raise ValueError("fixed HW2 token invalid, revoked, or not bound")
        if row["expires_at"] <= now:
            raise ValueError("fixed HW2 token expired; rebind required")
        if self.authority is not None:
            try:
                claims = self.authority.verify_access_token(
                    token[4:], required_scope="training:execute", now=now)
                if claims.get("role") != "guest" or claims.get("sub") != row["device_id"]:
                    raise ValueError("fixed HW2 guest identity mismatch")
            except Exception as exc:
                raise ValueError(
                    "fixed HW2 token invalid, expired, revoked, or out of scope") from exc
        return row

    def fixed_hw2_status(self, token: str | None = None,
                         installation_id: str | None = None,
                         now: int | None = None) -> dict:
        """Report a fixed binding without ever returning the bearer."""
        now = int(time.time()) if now is None else int(now)
        with self._connect() as db:
            if token is not None:
                row = self._fixed_active(db, token, now)
            else:
                if (not isinstance(installation_id, str) or not installation_id
                        or len(installation_id) > 256):
                    raise ValueError("installation identity or fixed HW2 bearer required")
                row = db.execute(
                    "SELECT * FROM fixed_hw_bindings WHERE device_id=? "
                    "AND hw_slot=? AND revoked_at IS NULL "
                    "ORDER BY bound_at DESC LIMIT 1",
                    (installation_id, FIXED_HW2_SLOT)).fetchone()
                if row is None:
                    return {"fixed_hw2_bound": False, "bound": False,
                            "installation_id": installation_id,
                            "address": FIXED_HW2_ADDRESS,
                            "node_point": FIXED_HW2_NODE_POINT,
                            "hw_slot": FIXED_HW2_SLOT,
                            "fixed_endpoint": FIXED_HW2_ENDPOINT,
                            "network_port": "BS0:0",
                            "allocation": "fixed-reserved", "permanent": True}
        return self._fixed_public_receipt(row)

    def report_fixed_hw2(self, token: str, payload: dict,
                         now: int | None = None) -> dict:
        now = int(time.time()) if now is None else int(now)
        if not isinstance(payload, dict):
            raise ValueError("invalid report")
        kind = payload.get("report_type")
        summary = payload.get("summary")
        details = payload.get("details", "")
        allowed = {"device_error", "system_error", "network_error",
                   "security_concern", "other"}
        if (kind not in allowed or not isinstance(summary, str)
                or not 1 <= len(summary) <= 240
                or not isinstance(details, str) or len(details) > MAX_REPORT):
            raise ValueError("invalid report fields")
        system = payload.get("system_report", {})
        if not isinstance(system, dict) or len(json.dumps(system)) > 4096:
            raise ValueError("invalid system report")
        report_id = str(uuid.uuid4())
        with self._connect() as db:
            row = self._fixed_active(db, token, now)
            db.execute("INSERT INTO device_reports VALUES (?,?,?,?,?,?,?)",
                       (report_id, row["device_id"], kind, summary, details,
                        json.dumps(system, sort_keys=True), now))
        return {"accepted": True, "report_id": report_id,
                "address": FIXED_HW2_ADDRESS,
                "node_point": FIXED_HW2_NODE_POINT,
                "fixed_hw2_bound": True}

    def revoke_fixed_hw2(self, installation_id: str | None = None,
                         lease_id: str | None = None,
                         revoked_by: str | None = None,
                         reason: str | None = None,
                         now: int | None = None) -> dict:
        """Explicitly revoke the permanent binding so it can be rebound."""
        now = int(time.time()) if now is None else int(now)
        if not installation_id and not lease_id:
            raise ValueError("fixed HW2 installation or lease identity required")
        with self._connect() as db:
            if lease_id is not None:
                row = db.execute(
                    "SELECT * FROM fixed_hw_bindings WHERE lease_id=? "
                    "AND revoked_at IS NULL", (lease_id,)).fetchone()
            else:
                row = db.execute(
                    "SELECT * FROM fixed_hw_bindings WHERE device_id=? "
                    "AND hw_slot=? AND revoked_at IS NULL "
                    "ORDER BY bound_at DESC LIMIT 1",
                    (installation_id, FIXED_HW2_SLOT)).fetchone()
            if row is None:
                raise ValueError("fixed HW2 binding not found or already revoked")
            db.execute(
                "UPDATE fixed_hw_bindings SET revoked_at=?,revoked_by=?,"
                "revoked_reason=? WHERE lease_id=? AND revoked_at IS NULL",
                (now, revoked_by, (reason or "")[:240], row["lease_id"]))
        if row["token_jti"]:
            with sqlite3.connect(self.database) as db:
                db.execute("CREATE TABLE IF NOT EXISTS revoked_tokens "
                           "(jti TEXT PRIMARY KEY,revoked_at INTEGER NOT NULL)")
                db.execute("INSERT OR IGNORE INTO revoked_tokens VALUES (?,?)",
                           (row["token_jti"], now))
        return {"revoked": True, "installation_id": row["device_id"],
                "lease_id": row["lease_id"], "address": row["address"],
                "node_point": row["node_point"], "revoked_by": revoked_by}

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
        if self.authority is not None:
            self.authority.revoke_token(token[4:])
        return {"lease_id": lease_id, "lease_expires_at": expires,
                "participant_id": device_id,
                "address": f"nan://GM0:SC0:MC0:BS0:HW{slot}", "token": replacement}

    def release(self, token: str, now: int | None = None) -> dict:
        now = int(time.time()) if now is None else int(now)
        with self._connect() as db:
            row = self._active(db, token, now)
            address = f"nan://GM0:SC0:MC0:BS0:HW{row['hw_slot']}"
            db.execute("UPDATE device_leases SET hw_slot=NULL,released_at=? WHERE lease_id=?", (now, row["lease_id"]))
            device_id = row["device_id"]
        if self.authority is not None:
            self.authority.revoke_token(token[4:])
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
        if self.authority is not None:
            self.authority.revoke_token(token[4:])
        return {"lease_id": lease_id, "lease_expires_at": expires,
                "participant_id": device_id,
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
            try:
                row = self._active(db, token, now)
            except ValueError as exc:
                try:
                    row = self._fixed_active(db, token, now)
                except ValueError:
                    raise exc
            db.execute("INSERT INTO device_reports VALUES (?,?,?,?,?,?,?)", (report_id, row["device_id"], kind, summary, details, json.dumps(system, sort_keys=True), now))
        return {"accepted": True, "report_id": report_id}
