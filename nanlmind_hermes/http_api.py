"""HTTPS-capable NANNET pairing authority and device lease API."""
from __future__ import annotations
import json, ipaddress, ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .auth import AuthError
from .devices import DeviceRegistry
from .pairing import PairingAuthority

def make_server(authority: PairingAuthority, host="127.0.0.1", port=8765, tls_cert=None, tls_key=None):
    try: loopback=ipaddress.ip_address(host).is_loopback
    except ValueError: loopback=host.lower()=="localhost"
    if not loopback and not (tls_cert and tls_key):
        raise ValueError("non-loopback pairing listener requires TLS certificate and private key")
    devices = DeviceRegistry(authority.database, authority=authority)
    class Handler(BaseHTTPRequestHandler):
        server_version="NANNET-HERMES/1"
        def log_message(self, fmt, *args): return
        def _reply(self,status,payload):
            data=json.dumps(payload,separators=(",",":")).encode()
            self.send_response(status); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data))); self.send_header("Cache-Control","no-store")
            self.end_headers(); self.wfile.write(data)
        def _body(self, limit=16384):
            size=int(self.headers.get("Content-Length","0"))
            if size < 1 or size > limit: raise ValueError("invalid_body_size")
            value=json.loads(self.rfile.read(size))
            if not isinstance(value,dict): raise ValueError("invalid_request")
            return value
        def _bearer(self):
            header=self.headers.get("Authorization","")
            return header[7:] if header.startswith("Bearer ") else ""
        def do_POST(self):
            if self.path == "/v1/devices/register":
                try: return self._reply(201,devices.register(self._body(8192)))
                except (ValueError,TypeError,json.JSONDecodeError) as exc: return self._reply(400,{"error":str(exc)[:160]})
            if self.path in ("/v1/devices/heartbeat","/v1/devices/release","/v1/devices/report","/v1/devices/rotate"):
                token=self._bearer()
                if not token: return self._reply(401,{"error":"bearer_required"})
                try:
                    payload=self._body(8192) if self.path.endswith("/report") else self._body(1024)
                    if self.path.endswith("/heartbeat"): result=devices.heartbeat(token)
                    elif self.path.endswith("/release"): result=devices.release(token)
                    elif self.path.endswith("/rotate"): result=devices.rotate(token)
                    else: result=devices.report(token,payload)
                    return self._reply(200,result)
                except (ValueError,TypeError,json.JSONDecodeError) as exc:
                    return self._reply(401 if "token" in str(exc) or "bearer" in str(exc) else 400,{"error":str(exc)[:160]})
            if self.path == "/v1/authorize":
                token = self._bearer()
                if not token: return self._reply(401, {"error": "bearer_required"})
                try:
                    payload = self._body(4096); scope = payload.get("scope")
                    if not isinstance(scope, str): return self._reply(400, {"error": "scope_required"})
                    claims = authority.verify_access_token(token, required_scope=scope)
                    return self._reply(200, {"authorized": True, "installation_id": claims["sub"], "scope": scope, "expires_at": claims["exp"]})
                except AuthError: return self._reply(401, {"error": "token_invalid_or_scope_denied"})
                except (ValueError, TypeError, json.JSONDecodeError): return self._reply(400, {"error": "invalid_request"})
            if self.path == "/v1/admin/identities/revoke":
                token = self._bearer()
                if not token: return self._reply(401, {"error": "bearer_required"})
                try:
                    claims = authority.verify_access_token(token, required_scope="installation:admin")
                    if claims.get("role") != "admin": return self._reply(403, {"error": "admin_role_required"})
                    payload = self._body(4096)
                    role, identity = payload.get("role"), payload.get("identity")
                    if role not in ("admin", "guest") or not isinstance(identity, str):
                        return self._reply(400, {"error": "invalid_identity_target"})
                    return self._reply(200, authority.revoke_identity(role, identity))
                except AuthError: return self._reply(401, {"error": "token_invalid_or_scope_denied"})
                except (ValueError, TypeError, json.JSONDecodeError): return self._reply(400, {"error": "invalid_request"})
            if self.path == "/v1/revoke":
                token = self._bearer()
                if not token: return self._reply(401, {"error": "bearer_required"})
                try: return self._reply(200, authority.revoke_token(token))
                except AuthError: return self._reply(401, {"error": "token_invalid_or_revoked"})
            if self.path!="/v1/pair": return self._reply(404,{"error":"not_found"})
            try:
                payload=self._body(16384)
                result=authority.pair(payload["pairing_code"],payload["installation_id"],payload["public_key"])
                self._reply(200,result)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError,AuthError) as exc:
                self._reply(400,{"error":"pairing_rejected","message":str(exc)[:200]})
        def do_GET(self):
            if self.path=="/health": return self._reply(200,{"service":"nannet-api-hermes","status":"ready"})
            return self._reply(404,{"error":"not_found"})
    server=ThreadingHTTPServer((host,port),Handler)
    if tls_cert or tls_key:
        if not (tls_cert and tls_key): raise ValueError("both TLS certificate and key are required")
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.minimum_version=ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(tls_cert,tls_key); server.socket=context.wrap_socket(server.socket,server_side=True)
    return server
