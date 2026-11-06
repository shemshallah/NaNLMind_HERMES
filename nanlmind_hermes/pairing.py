"""One-time pairing authority for NANNET_API_HERMES."""
from __future__ import annotations
import hashlib, json, os, secrets, sqlite3, time, uuid
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .auth import ALLOWED_SCOPES, AuthError, _b64, _unb64, _derive, _json_bytes, issue_token

class PairingAuthority:
    def __init__(self, database: str | Path, master_key: bytes):
        self.database=str(database); self.master_key=master_key
        Path(self.database).parent.mkdir(parents=True,exist_ok=True)
        try: os.chmod(Path(self.database).parent,0o700)
        except OSError: pass
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE IF NOT EXISTS authority_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS pairing_codes (code_hash TEXT PRIMARY KEY,scopes TEXT NOT NULL,expires_at INTEGER NOT NULL,used_at INTEGER)")
            db.execute("CREATE TABLE IF NOT EXISTS installations (installation_id TEXT PRIMARY KEY,public_key TEXT NOT NULL,paired_at INTEGER NOT NULL)")
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

    def _private_key(self):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        with sqlite3.connect(self.database) as db:
            row=db.execute("SELECT value FROM authority_meta WHERE key='issuer_private'").fetchone()
        if not row: raise AuthError("authority signing key is missing")
        env=json.loads(row[0]); key=_derive(self.master_key,_unb64(env["salt"]),b"authority-signing-key")
        from cryptography.exceptions import InvalidTag
        try: raw=AESGCM(key).decrypt(_unb64(env["nonce"]),_unb64(env["ciphertext"]),b"nannet-authority-v1")
        except InvalidTag as exc: raise AuthError("authority master key cannot unlock signing key") from exc
        return Ed25519PrivateKey.from_private_bytes(raw)

    def issuer_public_key(self) -> str:
        return _b64(self._private_key().public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw))

    def create_pairing_code(self, scopes, ttl_seconds=600, now=None) -> dict:
        scopes=sorted(set(scopes))
        if not scopes or not set(scopes)<=ALLOWED_SCOPES: raise AuthError("invalid pairing scopes")
        if isinstance(ttl_seconds,bool) or not isinstance(ttl_seconds,int) or not 60<=ttl_seconds<=3600: raise AuthError("pairing code lifetime must be 60–3600 seconds")
        now=int(time.time()) if now is None else int(now)
        code="nnh_"+secrets.token_urlsafe(24)
        digest=hashlib.sha256(code.encode()).hexdigest()
        with sqlite3.connect(self.database) as db:
            db.execute("INSERT INTO pairing_codes VALUES (?,?,?,NULL)",(digest,json.dumps(scopes),now+ttl_seconds))
        return {"pairing_code":code,"expires_at":now+ttl_seconds,"scopes":scopes}

    def pair(self, code: str, installation_id: str, public_key: str, token_ttl_seconds=900, now=None) -> dict:
        now=int(time.time()) if now is None else int(now)
        try:
            uuid.UUID(installation_id); raw=_unb64(public_key)
        except (ValueError,TypeError) as exc: raise AuthError("invalid installation identity") from exc
        if len(raw)!=32: raise AuthError("installation public key must be 32 bytes")
        digest=hashlib.sha256(code.encode()).hexdigest()
        with sqlite3.connect(self.database,timeout=10,isolation_level="IMMEDIATE") as db:
            row=db.execute("SELECT scopes,expires_at,used_at FROM pairing_codes WHERE code_hash=?",(digest,)).fetchone()
            if not row or row[2] is not None or now>=row[1]: raise AuthError("pairing code is invalid, expired, or already used")
            existing=db.execute("SELECT public_key FROM installations WHERE installation_id=?",(installation_id,)).fetchone()
            if existing and existing[0]!=public_key: raise AuthError("installation ID is already bound to another key")
            db.execute("UPDATE pairing_codes SET used_at=? WHERE code_hash=? AND used_at IS NULL",(now,digest))
            if db.total_changes<1: raise AuthError("pairing code was already consumed")
            db.execute("INSERT OR REPLACE INTO installations VALUES (?,?,?)",(installation_id,public_key,now))
            scopes=json.loads(row[0])
        token=issue_token(self._private_key(),installation_id,scopes,token_ttl_seconds,now)
        return {"token":token,"issuer_public_key":self.issuer_public_key(),"installation_id":installation_id,"scopes":scopes,"expires_at":now+token_ttl_seconds}

    def revoke_token(self, token: str, now=None) -> dict:
        from .auth import verify_token
        claims=verify_token(token,self._private_key().public_key(),now=now)
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE IF NOT EXISTS revoked_tokens (jti TEXT PRIMARY KEY,revoked_at INTEGER NOT NULL)")
            db.execute("INSERT OR IGNORE INTO revoked_tokens VALUES (?,?)",(claims["jti"],int(time.time()) if now is None else int(now)))
        return {"revoked":True,"token_id":claims["jti"],"installation_id":claims["sub"]}

    def verify_access_token(self, token: str, required_scope=None, now=None) -> dict:
        from .auth import verify_token
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE IF NOT EXISTS revoked_tokens (jti TEXT PRIMARY KEY,revoked_at INTEGER NOT NULL)")
            revoked={row[0] for row in db.execute("SELECT jti FROM revoked_tokens")}
        return verify_token(token,self._private_key().public_key(),required_scope,now,revoked)
