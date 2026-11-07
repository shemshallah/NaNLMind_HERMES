import hashlib, json, sqlite3, tempfile, threading, time, unittest, urllib.request, urllib.error
from pathlib import Path
from nanlmind_hermes.auth import _unb64, generate_master_key
from nanlmind_hermes.pairing import PairingAuthority
from nanlmind_hermes.http_api import make_server
from nanlmind_hermes.devices import DeviceRegistry, TERMS_VERSION

def payload(secret):
    return {"device_secret":secret,"consent":True,"terms_version":TERMS_VERSION,
            "system_report":{"platform":"Android","abi":"aarch64"}}

class DeviceLeaseTests(unittest.TestCase):
    def test_one_active_lease_per_device_and_release_reuses_participant_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry=DeviceRegistry(Path(tmp)/"authority.sqlite")
            first=registry.register(payload("a"*43),now=1000)
            self.assertEqual(first["address"],"nan://GM0:SC0:MC0:BS0:HW4")
            with sqlite3.connect(Path(tmp)/"authority.sqlite") as db:
                self.assertEqual(db.execute(
                    "SELECT hw_slot FROM device_leases WHERE lease_id=?",
                    (first["lease_id"],)).fetchone()[0], 4)
            with self.assertRaisesRegex(ValueError,"active lease"):
                registry.register(payload("a"*43),now=1001)
            parallel=registry.register(payload("b"*43),now=1001)
            self.assertEqual(parallel["address"],"nan://GM0:SC0:MC0:BS0:HW5")
            hb=registry.heartbeat(first["token"],now=1100)
            self.assertGreater(hb["lease_expires_at"],first["lease_expires_at"])
            self.assertNotEqual(hb["token"], first["token"])
            released=registry.release(hb["token"],now=1200)
            self.assertTrue(released["released"])
            second=registry.register(payload("c"*43),now=1201)
            self.assertEqual(second["address"],released["address"])
            self.assertEqual(second["address"],"nan://GM0:SC0:MC0:BS0:HW4")
            self.assertNotEqual(second["participant_id"],first["participant_id"])
            with self.assertRaisesRegex(ValueError,"invalid, expired, or released"):
                registry.heartbeat(first["token"],now=1202)


    def test_token_rotation_keeps_same_participant_and_revokes_old_bearer(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry=DeviceRegistry(Path(tmp)/"authority.sqlite")
            first=registry.register(payload("r"*43),now=1000)
            rotated=registry.rotate(first["token"],now=1100)
            self.assertEqual(rotated["address"],first["address"])
            self.assertEqual(rotated["participant_id"],first["participant_id"])
            self.assertEqual(rotated["lease_id"],first["lease_id"])
            with self.assertRaisesRegex(ValueError,"invalid, expired, or released"):
                registry.heartbeat(first["token"],now=1101)
            self.assertEqual(registry.heartbeat(rotated["token"],now=1102)["lease_id"],first["lease_id"])

    def test_legacy_reserved_slot_leases_migrate_to_hw4_plus(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "authority.sqlite"
            DeviceRegistry(path)
            now = int(time.time())
            with sqlite3.connect(path) as db:
                db.execute("INSERT INTO devices VALUES (?,?,?,?)", (
                    "legacy-device", hashlib.sha256(b"legacy-secret").hexdigest(),
                    now, now))
                db.execute("INSERT INTO device_leases VALUES (?,?,?,?,?,?,NULL)", (
                    "legacy-lease", "legacy-device", 2,
                    hashlib.sha256(b"legacy-token").hexdigest(), now, now + 3600))
            migrated = DeviceRegistry(path)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute(
                    "SELECT hw_slot FROM device_leases WHERE lease_id='legacy-lease'"
                ).fetchone()[0], 4)
            next_lease = migrated.register(payload("m" * 43), now=now + 1)
            self.assertEqual(next_lease["address"], "nan://GM0:SC0:MC0:BS0:HW5")

    def test_registration_requires_consent_and_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry=DeviceRegistry(Path(tmp)/"authority.sqlite")
            bad=payload("x"*43); bad["consent"]=False
            with self.assertRaisesRegex(ValueError,"consent"): registry.register(bad)
            bad=payload("x"*43); bad["terms_version"]="old"
            with self.assertRaisesRegex(ValueError,"terms"): registry.register(bad)

    def test_expired_lease_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry=DeviceRegistry(Path(tmp)/"authority.sqlite",lease_seconds=300)
            a=registry.register(payload("a"*43),now=1000)
            b=registry.register(payload("b"*43),now=1301)
            self.assertEqual(a["address"],b["address"])

    def test_http_android_client_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            authority=PairingAuthority(Path(tmp)/"auth.sqlite",_unb64(generate_master_key()))
            server=make_server(authority,"127.0.0.1",0)
            thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            base="http://127.0.0.1:%d"%server.server_address[1]
            try:
                def post(path,body,token=""):
                    headers={"Content-Type":"application/json"}
                    if token: headers["Authorization"]="Bearer "+token
                    req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers=headers,method="POST")
                    with urllib.request.urlopen(req,timeout=5) as response: return response.status,json.loads(response.read())
                status,reg=post("/v1/devices/register",payload("z"*43))
                self.assertEqual(status,201)
                self.assertEqual(authority.verify_access_token(reg["token"][4:], "training:execute")["role"], "guest")
                _,hb=post("/v1/devices/heartbeat",{},reg["token"])
                self.assertIn("lease_expires_at",hb)
                self.assertNotEqual(hb["token"], reg["token"])
                from nanlmind_hermes.auth import AuthError
                with self.assertRaises(AuthError):
                    authority.verify_access_token(reg["token"][4:], "training:execute")
                with self.assertRaises(urllib.error.HTTPError):
                    post("/v1/devices/report",{"report_type":"device_error","summary":"old token","details":""},reg["token"])
                _,report=post("/v1/devices/report",{"report_type":"device_error","summary":"test","details":""},hb["token"])
                self.assertTrue(report["accepted"])
                _,released=post("/v1/devices/release",{},hb["token"])
                self.assertTrue(released["released"])
                with self.assertRaises(AuthError):
                    authority.verify_access_token(hb["token"][4:], "training:execute")
                with self.assertRaises(urllib.error.HTTPError):
                    post("/v1/devices/heartbeat",{},reg["token"])
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=2)

if __name__=="__main__": unittest.main()
