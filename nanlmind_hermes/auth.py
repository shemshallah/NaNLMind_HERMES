"""NANNET_API_HERMES installation identity, scoped tokens, and payload crypto."""
from __future__ import annotations
import base64, hashlib, json, os, secrets, sqlite3, time, uuid
from pathlib import Path
from typing import Iterable
from .crypto_compat import (
    InvalidSignature, InvalidTag, hashes, serialization,
    Ed25519PrivateKey, Ed25519PublicKey, AESGCM, HKDF,
)

TOKEN_ISSUER = "nannet-api-hermes"
TOKEN_AUDIENCE = "nanlmind-hermes"
ALLOWED_SCOPES = frozenset({"corpus:write","training:execute","universe:read","snapshot:read","snapshot:write","universe:register","installation:admin"})

class AuthError(ValueError):
    """Credential, token, signature, or authorization failure."""

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _derive(master_key: bytes, salt: bytes, purpose: bytes) -> bytes:
    if not isinstance(master_key, bytes) or len(master_key) < 32:
        raise AuthError("NANNET master key must contain at least 32 random bytes")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=b"NANNET_API_HERMES/" + purpose).derive(master_key)

def generate_master_key() -> str:
    """Return a base64url 256-bit key for an OS secret store or protected env var."""
    return _b64(secrets.token_bytes(32))

def create_installation(state_dir: str | Path, master_key: bytes) -> dict:
    """Create one persistent installation identity; private key is AES-GCM encrypted."""
    root = Path(state_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    try: os.chmod(root, 0o700)
    except OSError: pass
    path = root / "installation.json"
    if path.exists(): raise AuthError("installation identity already exists; refusing to overwrite it")
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    public_raw = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    salt, nonce = secrets.token_bytes(16), secrets.token_bytes(12)
    key = _derive(master_key, salt, b"installation-private-key")
    ciphertext = AESGCM(key).encrypt(nonce, private_raw, b"nanlmind-installation-v1")
    installation_id = str(uuid.uuid4())
    document = {"schema":"nannet-installation/1","installation_id":installation_id,"created_at":int(time.time()),"public_key":_b64(public_raw),
      "private_key_envelope":{"algorithm":"HKDF-SHA256+A256GCM","salt":_b64(salt),"nonce":_b64(nonce),"ciphertext":_b64(ciphertext)}}
    temp = path.with_suffix(".tmp")
    temp.write_bytes(_json_bytes(document) + b"\n")
    try: os.chmod(temp, 0o600)
    except OSError: pass
    os.replace(temp, path)
    return {"installation_id":installation_id,"public_key":_b64(public_raw),"path":str(path)}

def load_installation(state_dir: str | Path, master_key: bytes) -> tuple[dict, Ed25519PrivateKey]:
    path = Path(state_dir).expanduser() / "installation.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8")); env = doc["private_key_envelope"]
        key = _derive(master_key, _unb64(env["salt"]), b"installation-private-key")
        raw = AESGCM(key).decrypt(_unb64(env["nonce"]), _unb64(env["ciphertext"]), b"nanlmind-installation-v1")
        private = Ed25519PrivateKey.from_private_bytes(raw)
        public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        if _b64(public) != doc["public_key"]: raise AuthError("installation public/private key mismatch")
        return doc, private
    except (OSError, KeyError, ValueError, InvalidTag, json.JSONDecodeError) as exc:
        if isinstance(exc, AuthError): raise
        raise AuthError("cannot unlock installation identity; check NANNET master key and state") from exc

def issue_token(issuer_private_key: Ed25519PrivateKey, installation_id: str, scopes: Iterable[str], ttl_seconds: int = 900, now: int | None = None, role: str = "guest") -> str:
    requested = sorted(set(scopes))
    ceiling = identity_scope_ceiling(role)
    if not installation_id or not requested or not set(requested) <= ceiling: raise AuthError("invalid installation ID, role, or scope set")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or not 30 <= ttl_seconds <= 86400: raise AuthError("token lifetime must be 30–86400 seconds")
    issued = int(time.time()) if now is None else int(now)
    header = {"alg":"EdDSA","typ":"NANNET-HERMES"}
    claims = {"iss":TOKEN_ISSUER,"aud":TOKEN_AUDIENCE,"sub":installation_id,"role":role,"scope":requested,"iat":issued,"nbf":issued,"exp":issued+ttl_seconds,"jti":str(uuid.uuid4())}
    signing_input = _b64(_json_bytes(header)) + "." + _b64(_json_bytes(claims))
    return signing_input + "." + _b64(issuer_private_key.sign(signing_input.encode("ascii")))

def verify_token(token: str, issuer_public_key: Ed25519PublicKey, required_scope: str | None = None, now: int | None = None, revoked_ids: set[str] | None = None) -> dict:
    try:
        h,p,s = token.split("."); signing_input = h + "." + p
        issuer_public_key.verify(_unb64(s), signing_input.encode("ascii"))
        header, claims = json.loads(_unb64(h)), json.loads(_unb64(p))
    except (ValueError, KeyError, InvalidSignature, json.JSONDecodeError) as exc: raise AuthError("invalid NANNET token or signature") from exc
    current = int(time.time()) if now is None else int(now)
    if header != {"alg":"EdDSA","typ":"NANNET-HERMES"}: raise AuthError("unsupported token header")
    if claims.get("iss") != TOKEN_ISSUER or claims.get("aud") != TOKEN_AUDIENCE: raise AuthError("wrong token issuer or audience")
    if current < claims.get("nbf",0) or current >= claims.get("exp",0): raise AuthError("token is not yet valid or has expired")
    if claims.get("jti") in (revoked_ids or set()): raise AuthError("token has been revoked")
    scopes = claims.get("scope",[])
    role = claims.get("role", "guest")
    if role not in IDENTITY_DOMAINS: raise AuthError("invalid token role")
    if not isinstance(scopes,list) or not set(scopes) <= identity_scope_ceiling(role): raise AuthError("token scope exceeds role ceiling")
    if required_scope and required_scope not in scopes: raise AuthError("token lacks required scope: " + required_scope)
    return claims

def sign_request(private_key: Ed25519PrivateKey, method: str, path: str, body: bytes, timestamp: int | None = None, nonce: str | None = None) -> dict:
    timestamp = int(time.time()) if timestamp is None else int(timestamp); nonce = nonce or _b64(secrets.token_bytes(18))
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = "\n".join((method.upper(),path,str(timestamp),nonce,body_hash)).encode("utf-8")
    return {"timestamp":timestamp,"nonce":nonce,"body_sha256":body_hash,"signature":_b64(private_key.sign(canonical))}

def verify_request(public_key: Ed25519PublicKey, method: str, path: str, body: bytes, envelope: dict, now: int | None = None, max_skew_seconds: int = 300) -> str:
    timestamp,nonce = envelope.get("timestamp"),envelope.get("nonce")
    current = int(time.time()) if now is None else int(now)
    if isinstance(timestamp,bool) or not isinstance(timestamp,int) or abs(current-timestamp)>max_skew_seconds: raise AuthError("request timestamp outside allowed clock skew")
    if not isinstance(nonce,str) or not 16<=len(nonce)<=128: raise AuthError("invalid request nonce")
    body_hash = hashlib.sha256(body).hexdigest()
    if envelope.get("body_sha256") != body_hash: raise AuthError("request body digest mismatch")
    canonical = "\n".join((method.upper(),path,str(timestamp),nonce,body_hash)).encode("utf-8")
    try: public_key.verify(_unb64(envelope["signature"]),canonical)
    except (InvalidSignature,KeyError,ValueError) as exc: raise AuthError("request signature invalid") from exc
    return nonce

def encrypt_payload(master_key: bytes, plaintext: bytes, associated_data: bytes = b"") -> dict:
    salt,nonce = secrets.token_bytes(16),secrets.token_bytes(12); key = _derive(master_key,salt,b"payload-encryption")
    return {"alg":"HKDF-SHA256+A256GCM","salt":_b64(salt),"nonce":_b64(nonce),"ciphertext":_b64(AESGCM(key).encrypt(nonce,plaintext,associated_data))}

def decrypt_payload(master_key: bytes, envelope: dict, associated_data: bytes = b"") -> bytes:
    try:
        key = _derive(master_key,_unb64(envelope["salt"]),b"payload-encryption")
        return AESGCM(key).decrypt(_unb64(envelope["nonce"]),_unb64(envelope["ciphertext"]),associated_data)
    except (KeyError,ValueError,InvalidTag) as exc: raise AuthError("payload authentication/decryption failed") from exc

class TokenRegistry:
    """Persistent token revocation and signed-request replay protection."""
    def __init__(self, database: str | Path):
        self.path = str(database); Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS revoked (jti TEXT PRIMARY KEY, revoked_at INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS nonces (installation_id TEXT NOT NULL, nonce TEXT NOT NULL, seen_at INTEGER NOT NULL, PRIMARY KEY (installation_id,nonce))")
    def revoke(self,jti: str,now: int | None = None) -> None:
        with sqlite3.connect(self.path) as db: db.execute("INSERT OR IGNORE INTO revoked VALUES (?,?)",(jti,int(time.time()) if now is None else int(now)))
    def revoked_ids(self) -> set[str]:
        with sqlite3.connect(self.path) as db: return {r[0] for r in db.execute("SELECT jti FROM revoked")}
    def consume_nonce(self,installation_id: str,nonce: str,now: int | None = None) -> None:
        try:
            with sqlite3.connect(self.path) as db: db.execute("INSERT INTO nonces VALUES (?,?,?)",(installation_id,nonce,int(time.time()) if now is None else int(now)))
        except sqlite3.IntegrityError as exc: raise AuthError("replayed request nonce") from exc


# Root-derived identity keys are domain-separated: an admin identity can never
# derive or authenticate as a guest identity. The master key remains server-side.
IDENTITY_DOMAINS = {
    "admin": b"identity/admin/v1",
    "guest": b"identity/guest/v1",
}

def derive_identity_private_key(master_key: bytes, role: str, identity: str) -> Ed25519PrivateKey:
    """Derive a stable per-identity Ed25519 key from the authority root.

    Call only inside the authority service. Never expose master_key or the
    derived private seed to a runner/client. Revocation remains server-side.
    """
    if role not in IDENTITY_DOMAINS:
        raise AuthError("identity role must be admin or guest")
    if not isinstance(identity, str) or not identity or len(identity) > 256:
        raise AuthError("invalid identity")
    # Fixed salt + distinct role namespace + identity yields independent keys.
    seed = _derive(master_key, b"NANNET-IDENTITY-V1", IDENTITY_DOMAINS[role] + b"/" + identity.encode("utf-8"))
    return Ed25519PrivateKey.from_private_bytes(seed)

def identity_public_key(master_key: bytes, role: str, identity: str) -> str:
    """Return only the public half of a root-derived identity key."""
    private = derive_identity_private_key(master_key, role, identity)
    raw = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return _b64(raw)

def identity_scope_ceiling(role: str) -> frozenset[str]:
    """Hard role boundary; caller may further narrow scopes per identity."""
    if role == "admin":
        return frozenset({"training:execute", "universe:read", "universe:register",
                          "snapshot:read", "snapshot:write", "corpus:write",
                          "installation:admin"})
    if role == "guest":
        return frozenset({"universe:read", "snapshot:read", "corpus:write",
                          "training:execute"})
    raise AuthError("identity role must be admin or guest")
