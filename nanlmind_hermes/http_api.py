"""Minimal HTTPS-capable NANNET pairing endpoint."""
from __future__ import annotations
import json, ipaddress, ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .auth import AuthError
from .pairing import PairingAuthority

def make_server(authority: PairingAuthority, host="127.0.0.1", port=8765, tls_cert=None, tls_key=None):
    try: loopback=ipaddress.ip_address(host).is_loopback
    except ValueError: loopback=host.lower()=="localhost"
    if not loopback and not (tls_cert and tls_key):
        raise ValueError("non-loopback pairing listener requires TLS certificate and private key")
    class Handler(BaseHTTPRequestHandler):
        server_version="NANNET-HERMES/1"
        def log_message(self, fmt, *args): return
        def _reply(self,status,payload):
            data=json.dumps(payload,separators=(",",":")).encode()
            self.send_response(status); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(data))); self.send_header("Cache-Control","no-store")
            self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            if self.path=="/v1/revoke":\n                token=self.headers.get("Authorization","")\n                if not token.startswith("Bearer "): return self._reply(401,{"error":"bearer_required"})\n                try: return self._reply(200,authority.revoke_token(token[7:]))\n                except AuthError: return self._reply(401,{"error":"token_invalid_or_revoked"})\n            if self.path!="/v1/pair": return self._reply(404,{"error":"not_found"})
            try:
                size=int(self.headers.get("Content-Length","0"))
                if size<1 or size>16384: return self._reply(413,{"error":"invalid_body_size"})
                payload=json.loads(self.rfile.read(size))
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
