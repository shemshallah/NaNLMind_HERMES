"""One-time pairing authority for NANNET_API_HERMES."""
from __future__ import annotations
import hashlib, json, os, secrets, sqlite3, time, uuid
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from .auth import ALLOWED_SCOPES, AuthError, _b64, _unb64, _derive, _json_bytes, issue_token, derive_identity_private_key, verify_request

class PairingAuthority:
    def __init__(self, database: str | Path, master_key: bytes):
        self.database=str(database); self.master_key=master_key
        Path(self.database).parent.mkdir(parents=True,exist_ok=True)
        if Path(self.database).parent != Path("."):
            try: os.chmod(Path(self.database).parent,0o700)
            except OSError: pass
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE IF NOT EXISTS authority_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS pairing_codes (code_hash TEXT PRIMARY KEY,scopes TEXT NOT NULL,expires_at INTEGER NOT NULL,used_at INTEGER)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(pairing_codes)")}
            if "role" not in columns:
                db.execute("ALTER TABLE pairing_codes ADD COLUMN role TEXT NOT NULL DEFAULT 'guest'")
            db.execute("CREATE TABLE IF NOT EXISTS installations (installation_id TEXT PRIMARY KEY,public_key TEXT NOT NULL,paired_at INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS identity_status (identity TEXT NOT NULL,role TEXT NOT NULL,created_at INTEGER NOT NULL,last_seen INTEGER NOT NULL DEFAULT 0,revoked_at INTEGER,PRIMARY KEY(identity,role))")
            db.execute("CREATE TABLE IF NOT EXISTS oidc_jti (jti TEXT PRIMARY KEY,identity TEXT NOT NULL,expires_at INTEGER NOT NULL,consumed_at INTEGER NOT NULL)")
            identity_columns = {row[1] for row in db.execute("PRAGMA table_info(identity_status)")}
            if "last_seen" not in identity_columns:
                db.execute("ALTER TABLE identity_status ADD COLUMN last_seen INTEGER NOT NULL DEFAULT 0")
        try: os.chmod(self.database,0o600)
        except OSError: pass
        self._ensure_signer()

    def _ensure_signer(self):
        with sqlite3.connect(self.database) as db:
            row=db.execute("SELECT value FROM authority_meta WHERE key='issuer_private'").fetchone()
            if row: return
            private=Ed25519PrivateKey.generate()
            raw=private.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption())
            import os
            salt,nonce=secrets.token_bytes(16),secrets.token_bytes(12)
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            key=_derive(self.master_key,salt,b"authority-signing-key")
            envelope={"salt":_b64(salt),"nonce":_b64(nonce),"ciphertext":_b64(AESGCM(key).encrypt(nonce,raw,b"nannet-authority-v1"))}
            db.execute("INSERT INTO authority_meta VALUES ('issuer_private',?)",(json.dumps(envelope,separators=(",",":")),))

    def _legacy_private_key(self):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        with sqlite3.connect(self.database) as db:
            row=db.execute("SELECT value FROM authority_meta WHERE key='issuer_private'").fetchone()
        if not row: raise AuthError("authority signing key is missing")
        env=json.loads(row[0]); key=_derive(self.master_key,_unb64(env["salt"]),b"authority-signing-key")
        from cryptography.exceptions import InvalidTag
        try: raw=AESGCM(key).decrypt(_unb64(env["nonce"]),_unb64(env["ciphertext"]),b"nannet-authority-v1")
        except InvalidTag as exc: raise AuthError("authority master key cannot unlock signing key") from exc
        return Ed25519PrivateKey.from_private_bytes(raw)

    def _private_key(self, role="guest", identity="legacy"):
        if identity == "legacy":
            return self._legacy_private_key()
        return derive_identity_private_key(self.master_key, role, identity)

    def issuer_public_key(self, role="guest", identity="legacy") -> str:
        return _b64(self._private_key(role, identity).public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw))

    def consume_oidc_jti(self, jti: str, identity: str, expires_at: int, now=None) -> None:
        if not isinstance(jti, str) or not jti or len(jti) > 256 or not isinstance(identity, str) or not identity:
            raise AuthError("invalid OIDC token identity")
        stamp = int(time.time()) if now is None else int(now)
        try:
            with sqlite3.connect(self.database) as db:
                db.execute("INSERT INTO oidc_jti VALUES (?,?,?,?)", (jti, identity, int(expires_at), stamp))
                db.execute("DELETE FROM oidc_jti WHERE expires_at<=?", (stamp,))
        except sqlite3.IntegrityError as exc:
            raise AuthError("OIDC token has already been consumed") from exc

    def register_identity(self, role: str, identity: str, now=None) -> None:
        from .auth import IDENTITY_DOMAINS
        if role not in IDENTITY_DOMAINS or not isinstance(identity, str) or not identity or len(identity) > 256:
            raise AuthError("invalid identity")
        stamp = int(time.time()) if now is None else int(now)
        with sqlite3.connect(self.database) as db:
            db.execute("INSERT OR IGNORE INTO identity_status (identity,role,created_at,last_seen,revoked_at) VALUES (?,?,?,?,NULL)", (identity, role, stamp, stamp))
            row = db.execute("SELECT revoked_at FROM identity_status WHERE identity=? AND role=?", (identity, role)).fetchone()
            if row and row[0] is not None:
                raise AuthError("identity has been revoked")
            db.execute("UPDATE identity_status SET last_seen=? WHERE identity=? AND role=? AND revoked_at IS NULL", (stamp, identity, role))

    def revoke_identity(self, role: str, identity: str, now=None) -> dict:
        stamp = int(time.time()) if now is None else int(now)
        with sqlite3.connect(self.database) as db:
            cur = db.execute("UPDATE identity_status SET revoked_at=? WHERE identity=? AND role=? AND revoked_at IS NULL", (stamp, identity, role))
            if cur.rowcount != 1:
                raise AuthError("identity not found or already revoked")
        return {"revoked": True, "identity": identity, "role": role}

    def issue_identity_token(self, role: str, identity: str, scopes, ttl_seconds=3600, now=None) -> dict:
        from .auth import identity_scope_ceiling
        requested = sorted(set(scopes))
        if role not in ("admin", "guest") or not requested or not set(requested) <= identity_scope_ceiling(role):
            raise AuthError("identity token scopes exceed role ceiling")
        self.register_identity(role, identity, now)
        token = issue_token(self._private_key(role, identity), identity, requested, ttl_seconds, now, role=role)
        return {"token": token, "role": role, "identity": identity, "scopes": requested}

    def create_pairing_code(self, scopes, ttl_seconds=600, now=None, role="guest") -> dict:
        from .auth import identity_scope_ceiling
        scopes=sorted(set(scopes))
        if role not in ("admin", "guest") or not scopes or not set(scopes)<=identity_scope_ceiling(role):
            raise AuthError("pairing scopes exceed role ceiling")
        if isinstance(ttl_seconds,bool) or not isinstance(ttl_seconds,int) or not 60<=ttl_seconds<=3600: raise AuthError("pairing code lifetime must be 60–3600 seconds")
        now=int(time.time()) if now is None else int(now)
        code="nnh_"+secrets.token_urlsafe(24)
        digest=hashlib.sha256(code.encode()).hexdigest()
        with sqlite3.connect(self.database) as db:
            db.execute("INSERT INTO pairing_codes (code_hash,scopes,expires_at,used_at,role) VALUES (?,?,?,NULL,?)",(digest,json.dumps(scopes),now+ttl_seconds,role))
        return {"pairing_code":code,"expires_at":now+ttl_seconds,"scopes":scopes,"role":role}

    def pair(self, code: str, installation_id: str, public_key: str, token_ttl_seconds=900, now=None) -> dict:
        now=int(time.time()) if now is None else int(now)
        try:
            uuid.UUID(installation_id); raw=_unb64(public_key)
        except (ValueError,TypeError) as exc: raise AuthError("invalid installation identity") from exc
        if len(raw)!=32: raise AuthError("installation public key must be 32 bytes")
        digest=hashlib.sha256(code.encode()).hexdigest()
        with sqlite3.connect(self.database,timeout=10,isolation_level="IMMEDIATE") as db:
            row=db.execute("SELECT scopes,expires_at,used_at,role FROM pairing_codes WHERE code_hash=?",(digest,)).fetchone()
            if not row or row[2] is not None or now>=row[1]: raise AuthError("pairing code is invalid, expired, or already used")
            existing=db.execute("SELECT public_key FROM installations WHERE installation_id=?",(installation_id,)).fetchone()
            if existing and existing[0]!=public_key: raise AuthError("installation ID is already bound to another key")
            db.execute("UPDATE pairing_codes SET used_at=? WHERE code_hash=? AND used_at IS NULL",(now,digest))
            if db.total_changes<1: raise AuthError("pairing code was already consumed")
            db.execute("INSERT OR REPLACE INTO installations VALUES (?,?,?)",(installation_id,public_key,now))
            scopes=json.loads(row[0]); role=row[3]
        self.register_identity(role, installation_id, now)
        token=issue_token(self._private_key(role, installation_id),installation_id,scopes,token_ttl_seconds,now,role=role)
        return {"token":token,"issuer_public_key":self.issuer_public_key(role, installation_id),"installation_id":installation_id,"scopes":scopes,"role":role,"expires_at":now+token_ttl_seconds}


    def is_paired_installation(self, installation_id: str) -> bool:
        with sqlite3.connect(self.database) as db:
            return db.execute("SELECT 1 FROM installations WHERE installation_id=?", (installation_id,)).fetchone() is not None

    def verify_signed_request(self, token: str, method: str, path: str, body: bytes,
                              envelope: dict, now=None) -> dict:
        """Verify an installation's Ed25519 request signature and consume its nonce."""
        claims = self.verify_access_token(token, now=now)
        stamp = int(time.time()) if now is None else int(now)
        with sqlite3.connect(self.database) as db:
            row = db.execute("SELECT public_key FROM installations WHERE installation_id=?",
                             (claims["sub"],)).fetchone()
        if not row:
            raise AuthError("signed installation identity is not registered")
        public = Ed25519PublicKey.from_public_bytes(_unb64(row[0]))
        nonce = verify_request(public, method, path, body, envelope, now=stamp)
        try:
            with sqlite3.connect(self.database, timeout=10) as db:
                db.execute("CREATE TABLE IF NOT EXISTS request_nonces (installation_id TEXT NOT NULL, nonce TEXT NOT NULL, seen_at INTEGER NOT NULL, PRIMARY KEY(installation_id,nonce))")
                db.execute("DELETE FROM request_nonces WHERE seen_at<?", (stamp - 600,))
                db.execute("INSERT INTO request_nonces VALUES (?,?,?)", (claims["sub"], nonce, stamp))
        except sqlite3.IntegrityError as exc:
            raise AuthError("replayed signed request nonce") from exc
        return claims

    def revoke_token(self, token: str, now=None) -> dict:
        from .auth import verify_token
        claims = self._decode_claims(token)
        try:
            verify_token(token, self._private_key(claims.get("role", "guest"), claims["sub"]).public_key(), now=now)
        except AuthError:
            claims = verify_token(token, self._legacy_private_key().public_key(), now=now)
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE IF NOT EXISTS revoked_tokens (jti TEXT PRIMARY KEY,revoked_at INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO revoked_tokens VALUES (?,?)",(claims["jti"],int(time.time()) if now is None else int(now)))
        return {"revoked":True,"token_id":claims["jti"],"installation_id":claims["sub"]}

    def verify_access_token(self, token: str, required_scope=None, now=None) -> dict:
        from .auth import verify_token
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE IF NOT EXISTS revoked_tokens (jti TEXT PRIMARY KEY,revoked_at INTEGER NOT NULL)")
            revoked={row[0] for row in db.execute("SELECT jti FROM revoked_tokens")}
        claims = self._decode_claims(token)
        role = claims.get("role", "guest")
        try:
            verified = verify_token(token, self._private_key(role, claims["sub"]).public_key(), required_scope, now, revoked)
        except AuthError:
            verified = verify_token(token, self._legacy_private_key().public_key(), required_scope, now, revoked)
        with sqlite3.connect(self.database) as db:
            status = db.execute("SELECT revoked_at FROM identity_status WHERE identity=? AND role=?", (verified["sub"], verified.get("role", "guest"))).fetchone()
        if status and status[0] is not None:
            raise AuthError("identity has been revoked")
        return verified

    @staticmethod
    def _decode_claims(token):
        try:
            payload = token.split(".")[1]
            claims = json.loads(_unb64(payload))
            if not isinstance(claims, dict) or not isinstance(claims.get("sub"), str) or not claims["sub"] or len(claims["sub"]) > 256:
                raise AuthError("invalid NANNET token claims")
            if claims.get("role", "guest") not in ("admin", "guest"):
                raise AuthError("invalid NANNET token role")
            return claims
        except (IndexError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise AuthError("invalid NANNET token") from exc
