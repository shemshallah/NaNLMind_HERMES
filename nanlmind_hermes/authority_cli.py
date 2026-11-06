"""Admin CLI for local NANNET pairing authority."""
from __future__ import annotations
import argparse, base64, os
from .auth import _unb64, generate_master_key
from .pairing import PairingAuthority
from .http_api import make_server

def _master(name):
    value=os.environ.get(name,"")
    if not value: raise SystemExit(name+" must be provisioned in the host secret store")
    try: raw=_unb64(value)
    except Exception as exc: raise SystemExit(name+" must be base64url encoded") from exc
    if len(raw)!=32: raise SystemExit(name+" must decode to exactly 32 bytes")
    return raw

def main():
    parser=argparse.ArgumentParser(prog="nannet-authority")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("public-key")
    code=sub.add_parser("create-code"); code.add_argument("--scopes",required=True); code.add_argument("--ttl-seconds",type=int,default=600); code.add_argument("--role",choices=("guest","admin"),default="guest")
    serve=sub.add_parser("serve"); serve.add_argument("--host",default="127.0.0.1"); serve.add_argument("--port",type=int,default=8765); serve.add_argument("--tls-cert"); serve.add_argument("--tls-key")
    args=parser.parse_args()
    authority=PairingAuthority(os.environ.get("NANNET_AUTHORITY_DB",os.path.expanduser("~/.local/share/nannet-api-hermes/authority.sqlite")),_master("NANNET_AUTHORITY_MASTER_KEY"))
    if args.command=="public-key": print(authority.issuer_public_key()); return
    if args.command=="create-code":
        scopes=[s.strip() for s in args.scopes.split(",") if s.strip()]
        result=authority.create_pairing_code(scopes,args.ttl_seconds,role=args.role)
        print(result["pairing_code"]); return
    server=make_server(authority,args.host,args.port,args.tls_cert,args.tls_key)
    print("NANNET pairing endpoint listening on "+args.host+":"+str(args.port))
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=="__main__": main()
