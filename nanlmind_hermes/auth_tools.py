"""Hermes-facing local installation identity and pairing handlers."""
from __future__ import annotations
import base64, json, os, urllib.parse, urllib.request, hmac
from pathlib import Path
from .auth import create_installation, load_installation, _unb64, _b64, encrypt_payload, verify_token

def _universe_dir(ctx):
    try: configured=ctx.get_config("state_dir",default="~/.local/share/nanlmind-hermes")
    except Exception: configured="~/.local/share/nanlmind-hermes"
    path=Path(str(configured)).expanduser()/"universe"; path.mkdir(parents=True,exist_ok=True)
    return path

def _master_key():
    encoded=os.environ.get("NANNET_MASTER_KEY","")
    if not encoded: raise ValueError("NANNET_MASTER_KEY is not set; provision an independent random 256-bit key in the host OS secret store")
    try: raw=_unb64(encoded)
    except Exception as exc: raise ValueError("NANNET_MASTER_KEY must be base64url encoded") from exc
    if len(raw)!=32: raise ValueError("NANNET_MASTER_KEY must decode to exactly 32 bytes")
    return raw

def auth_init(ctx):
    identity=create_installation(_universe_dir(ctx),_master_key())
    return {"installation_id":identity["installation_id"],"public_key":identity["public_key"],
            "identity_path":identity["path"],"private_key_encrypted_at_rest":True,"master_key_returned":False}

def auth_status(ctx):
    path=_universe_dir(ctx)/"installation.json"; result={"identity_present":path.exists(),
        "master_key_present":bool(os.environ.get("NANNET_MASTER_KEY")),"paired_token_present":(_universe_dir(ctx)/"nannet_token.json").exists()}
    if not path.exists(): return result
    try:
        doc,_=load_installation(path.parent,_master_key())
        result.update({"installation_id":doc["installation_id"],"public_key":doc["public_key"],"identity_unlocked":True})
    except Exception: result["identity_unlocked"]=False
    result["pairing_authority_configured"]=bool(os.environ.get("NANNET_AUTH_URL") and os.environ.get("NANNET_ISSUER_PUBLIC_KEY"))
    return result

def auth_pair(ctx,pairing_code):
    """Pair against a trusted authority and pin its issuer before accepting its token."""
    import ipaddress
    from urllib.parse import urlsplit
    base=os.environ.get("NANNET_AUTH_URL","").rstrip("/")
    pinned=os.environ.get("NANNET_ISSUER_PUBLIC_KEY","")
    if not base or not pinned: raise ValueError("NANNET_AUTH_URL and pinned NANNET_ISSUER_PUBLIC_KEY are required")
    parsed=urlsplit(base)
    loopback=False
    try: loopback=ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError: loopback=(parsed.hostname or "").lower()=="localhost"
    if parsed.scheme!="https" and not (parsed.scheme=="http" and loopback):
        raise ValueError("pairing requires HTTPS; plain HTTP is allowed only for loopback development")
    root=_universe_dir(ctx); master=_master_key()
    doc,private=load_installation(root,master)
    payload={"pairing_code":pairing_code,"installation_id":doc["installation_id"],"public_key":doc["public_key"]}
    request=urllib.request.Request(base+"/v1/pair",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Cache-Control":"no-store"},method="POST")
    try:
        with urllib.request.urlopen(request,timeout=15) as response: result=json.loads(response.read(32768))
    except Exception as exc: raise ValueError("NANNET pairing request failed") from exc
    if not hmac.compare_digest(str(result.get("issuer_public_key","")),pinned):
        raise ValueError("pairing authority issuer key does not match pinned NANNET_ISSUER_PUBLIC_KEY")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
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
