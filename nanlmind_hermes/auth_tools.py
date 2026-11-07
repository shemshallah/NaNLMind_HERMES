"""Hermes-facing local installation identity and pairing handlers."""
from __future__ import annotations
import base64, json, os, urllib.parse, urllib.request, hmac
from .hermes_secrets import get_secret
from pathlib import Path
from .auth import create_installation, load_installation, _unb64, _b64, encrypt_payload, decrypt_payload, verify_token, sign_request

def _universe_dir(ctx):
    try: configured=ctx.get_config("state_dir",default="~/.local/share/nanlmind-hermes")
    except Exception: configured="~/.local/share/nanlmind-hermes"
    path=Path(str(configured)).expanduser()/"universe"; path.mkdir(parents=True,exist_ok=True)
    return path

def _config(ctx, key, default=""):
    try: return str(ctx.get_config(key, default=default) or "").strip()
    except Exception: return default


def _master_key(ctx):
    encoded=get_secret("NANNET_MASTER_KEY")
    if not encoded: raise ValueError("Configure NANNET_MASTER_KEY in Hermes-managed plugin secrets")
    try: raw=_unb64(encoded)
    except Exception as exc: raise ValueError("NANNET_MASTER_KEY must be base64url encoded") from exc
    if len(raw)!=32: raise ValueError("NANNET_MASTER_KEY must decode to exactly 32 bytes")
    return raw

def auth_init(ctx):
    identity=create_installation(_universe_dir(ctx),_master_key(ctx))
    return {"installation_id":identity["installation_id"],"public_key":identity["public_key"],
            "identity_path":identity["path"],"private_key_encrypted_at_rest":True,"master_key_returned":False}

def auth_status(ctx):
    root=_universe_dir(ctx)
    path=root/"installation.json"
    token_path=root/"nannet_token.json"
    result={"identity_present":path.exists(),"master_key_present":bool(get_secret("NANNET_MASTER_KEY")),
            "paired_token_present":token_path.exists(),
            "pairing_authority_configured":bool(_config(ctx,"nannet_auth_url") and _config(ctx,"nannet_issuer_public_key"))}
    if not path.exists(): return result
    try:
        doc,_=load_installation(root,_master_key(ctx))
        result.update({"installation_id":doc["installation_id"],"public_key":doc["public_key"],"identity_unlocked":True})
    except Exception:
        result["identity_unlocked"]=False
        return result
    if token_path.exists() and _config(ctx,"nannet_issuer_public_key"):
        try:
            envelope=json.loads(token_path.read_text(encoding="utf-8"))
            token=decrypt_payload(_master_key(ctx),envelope,("nannet-token-v1:"+doc["installation_id"]).encode()).decode()
            from .crypto_compat import Ed25519PublicKey
            claims=verify_token(token,Ed25519PublicKey.from_public_bytes(_unb64(_config(ctx,"nannet_issuer_public_key"))))
            result.update({"token_valid":True,"token_scopes":claims["scope"],"token_expires_at":claims["exp"]})
        except Exception:
            result["token_valid"]=False
    return result

def _signed_headers(private_key, method, path, body):
    envelope = sign_request(private_key, method, path, body)
    return {
        "X-NANNET-Timestamp": str(envelope["timestamp"]),
        "X-NANNET-Nonce": envelope["nonce"],
        "X-NANNET-Body-SHA256": envelope["body_sha256"],
        "X-NANNET-Signature": envelope["signature"],
    }


def auth_pair(ctx,pairing_code):
    """Pair against a trusted authority and pin its issuer before accepting its token."""
    import ipaddress
    from urllib.parse import urlsplit
    base=_config(ctx,"nannet_auth_url").rstrip("/")
    pinned=_config(ctx,"nannet_issuer_public_key")
    if not base or not pinned: raise ValueError("NANNET_AUTH_URL and pinned NANNET_ISSUER_PUBLIC_KEY are required")
    parsed=urlsplit(base)
    loopback=False
    try: loopback=ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError: loopback=(parsed.hostname or "").lower()=="localhost"
    if parsed.scheme!="https" and not (parsed.scheme=="http" and loopback):
        raise ValueError("pairing requires HTTPS; plain HTTP is allowed only for loopback development")
    root=_universe_dir(ctx); master=_master_key(ctx)
    doc,private=load_installation(root,master)
    payload={"pairing_code":pairing_code,"installation_id":doc["installation_id"],"public_key":doc["public_key"]}
    request=urllib.request.Request(base+"/v1/pair",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Cache-Control":"no-store"},method="POST")
    try:
        with urllib.request.urlopen(request,timeout=15) as response: result=json.loads(response.read(32768))
    except Exception as exc: raise ValueError("NANNET pairing request failed") from exc
    if not hmac.compare_digest(str(result.get("issuer_public_key","")),pinned):
        raise ValueError("pairing authority issuer key does not match pinned NANNET_ISSUER_PUBLIC_KEY")
    from .crypto_compat import Ed25519PublicKey
    issuer=Ed25519PublicKey.from_public_bytes(_unb64(pinned))
    claims=verify_token(result["token"],issuer,now=None)
    if claims.get("sub")!=doc["installation_id"] or sorted(claims.get("scope",[]))!=sorted(result.get("scopes",[])):
        raise ValueError("issued token identity or scopes do not match pairing response")
    envelope=encrypt_payload(master,result["token"].encode(),("nannet-token-v1:"+doc["installation_id"]).encode())
    target=root/"nannet_token.json"; temp=target.with_suffix(".tmp")
    temp.write_text(json.dumps(envelope,separators=(",",":"))+"\n",encoding="utf-8")
    try: os.chmod(temp,0o600)
    except OSError: pass
    os.replace(temp,target)
    return {"paired":True,"installation_id":doc["installation_id"],"scopes":claims["scope"],"expires_at":claims["exp"],"token_saved_encrypted":True,"issuer_key_pinned":True}

def auth_revoke(ctx):
    """Revoke this installation's active token at the pinned authority."""
    import urllib.request, ipaddress
    from urllib.parse import urlsplit
    base=_config(ctx,"nannet_auth_url").rstrip("/")
    pinned=_config(ctx,"nannet_issuer_public_key")
    if not base or not pinned:
        raise ValueError("NANNET_AUTH_URL and pinned issuer key are required")
    root=_universe_dir(ctx); master=_master_key(ctx)
    doc,_=load_installation(root,master)
    token_path=root/"nannet_token.json"
    if not token_path.exists():
        raise ValueError("no paired NANNET token is stored")
    envelope=json.loads(token_path.read_text(encoding="utf-8"))
    token=decrypt_payload(master,envelope,("nannet-token-v1:"+doc["installation_id"]).encode()).decode()
    parsed=urlsplit(base)
    try: loopback=ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError: loopback=(parsed.hostname or "").lower()=="localhost"
    if parsed.scheme!="https" and not (parsed.scheme=="http" and loopback):
        raise ValueError("token revocation requires HTTPS except for loopback development")
    _, private = load_installation(root, master)
    body = b"{}"
    headers = {"Authorization":"Bearer "+token,"Content-Type":"application/json","Cache-Control":"no-store"}
    headers.update(_signed_headers(private, "POST", "/v1/revoke", body))
    req=urllib.request.Request(base+"/v1/revoke",data=body,headers=headers,method="POST")
    try:
        with urllib.request.urlopen(req,timeout=15) as response:
            result=json.loads(response.read(8192))
    except Exception as exc:
        raise ValueError("NANNET token revocation failed") from exc
    token_path.unlink(missing_ok=True)
    return {"revoked":bool(result.get("revoked")),"installation_id":doc["installation_id"],"local_token_erased":True}

def require_scope(ctx, required_scope):
    """Enforce the paired token locally when the NANNET authority is configured."""
    root=_universe_dir(ctx)
    token_path=root/"nannet_token.json"
    configured=bool(_config(ctx,"nannet_auth_url") and _config(ctx,"nannet_issuer_public_key"))
    if not configured:
        raise ValueError("NANNET authority URL and pinned issuer key must be configured")
    if not token_path.exists():
        raise ValueError("this installation must pair with NANNET before this operation")
    if not _config(ctx,"nannet_issuer_public_key"):
        raise ValueError("NANNET_ISSUER_PUBLIC_KEY is required to validate the stored token")
    doc,_=load_installation(root,_master_key(ctx))
    envelope=json.loads(token_path.read_text(encoding="utf-8"))
    token=decrypt_payload(_master_key(ctx),envelope,("nannet-token-v1:"+doc["installation_id"]).encode()).decode()
    from .crypto_compat import Ed25519PublicKey
    claims=verify_token(token,Ed25519PublicKey.from_public_bytes(_unb64(_config(ctx,"nannet_issuer_public_key"))),required_scope=required_scope)
    if claims.get("sub")!=doc["installation_id"]:
        raise ValueError("NANNET token belongs to a different installation")
    authority_url=_config(ctx,"nannet_auth_url").rstrip("/")
    if authority_url:
        import ipaddress
        from urllib.parse import urlsplit
        parsed=urlsplit(authority_url)
        try: loopback=ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError: loopback=(parsed.hostname or "").lower()=="localhost"
        if parsed.scheme!="https" and not (parsed.scheme=="http" and loopback):
            raise ValueError("online scope verification requires HTTPS except for loopback development")
        body = json.dumps({"scope":required_scope}, separators=(",", ":")).encode()
        _, private = load_installation(root, _master_key(ctx))
        headers = {"Authorization":"Bearer "+token,"Content-Type":"application/json","Cache-Control":"no-store"}
        headers.update(_signed_headers(private, "POST", "/v1/authorize", body))
        request=urllib.request.Request(authority_url+"/v1/authorize",
            data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request,timeout=10) as response:
                remote=json.loads(response.read(8192))
        except Exception as exc:
            raise ValueError("NANNET authority rejected or could not verify this operation; failing closed") from exc
        if not remote.get("authorized") or remote.get("installation_id")!=doc["installation_id"]:
            raise ValueError("NANNET authority denied this installation scope")
    return True
