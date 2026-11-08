"""Independent HERMES compatibility tests for NaNROUTER's graph-page wire contract."""
import hashlib
import json
import unittest

from nanlmind_hermes.universe_chart import Chart8MiB
from nanlmind_hermes.universe_root import attach_universe_root
from nanlmind_hermes.mirror_topology import TOPOLOGY_DIGEST
from nanlmind_hermes.router_graph_page import (
    RouterGraphPageVerifier, RouterGraphPageError,
)


def canon(value):
    return json.dumps(value,sort_keys=True,separators=(",",":")).encode()


def edge(edge_id, src=7, dst=265, layer=4, source_head=7,
         relation="supports", status="candidate"):
    # MP9 is pair (0,10), canonical cell 265.
    return dict(edge_id=edge_id,src=src,dst=dst,layer=layer,
                relation=relation,evidence_id="nml:evidence:1",
                evidence_version=1,evidence_digest="f"*64,
                source_head=source_head,generation=1,tick=0,status=status,
                src_route="HW0:" + str(src) if src<256 else "HW0:MP"+str(src-256),
                dst_route="HW0:" + str(dst) if dst<256 else "HW0:MP"+str(dst-256))


def page(chart, records, offset=0, limit=100):
    records=sorted(records,key=lambda r:r["edge_id"])
    state=hashlib.sha256(b"nanrouter.nan-layer-state/1\0")
    layers=[0]*10
    active=retracted=0
    for record in records:
        wire={key:value for key,value in record.items()
              if key not in ("src_route","dst_route")}
        raw=canon(wire)
        state.update(len(raw).to_bytes(8,"big")+raw)
        layers[record["layer"]]+=1
        if record["status"]=="candidate":
            active+=1
        else:
            retracted+=1
    header=dict(schema="nanrouter.nan-layer-master-commitment/1",
                topology_sha256=TOPOLOGY_DIGEST,
                parent_root_sha256=chart.root_hash,
                chart_generation=chart.snapshot_generation,
                total_edges=len(records),candidate_edges=active,
                retracted_edges=retracted,layer_counts=layers,
                state_sha256=state.hexdigest(),
                history_sha256="b"*64)
    part=records[offset:offset+limit]
    return {
        **header,
        "graph_commitment_sha256":hashlib.sha256(
            b"nanrouter.nan-layer-commitment/1\0"+canon(header)).hexdigest(),
        "offset":offset,"entries":part,
        "next_offset":offset+len(part) if offset+len(part)<len(records) else None,
        "signed_by_router":False,"native_mth_receipt":False,
        "mathematical_truth_promoted":False,"snapshot_consistent":True}


class GraphPageVerifierTests(unittest.TestCase):
    def setUp(self):
        self.chart=Chart8MiB.genesis()
        attach_universe_root(self.chart)

    def test_empty_and_two_pages_complete_without_second_graph(self):
        empty=RouterGraphPageVerifier(self.chart)
        result=empty.add_page(page(self.chart,[]))
        self.assertTrue(result["state_verified"])
        self.assertFalse(result["sender_authenticated"])
        with self.assertRaises(RouterGraphPageError):
            empty.add_page(page(self.chart,[]))
        records=[edge("e1"),edge("e2",src=265,dst=10,layer=9,source_head=10)]
        verifier=RouterGraphPageVerifier(self.chart)
        first=verifier.add_page(page(self.chart,records,limit=1))
        self.assertFalse(first["state_verified"])
        self.assertEqual(first["verified_entries"],1)
        second=verifier.add_page(page(self.chart,records,offset=1,limit=1))
        self.assertTrue(second["state_verified"])
        self.assertTrue(second["chart_root_bound"])
        self.assertEqual(second["verified_entries"],2)

    def test_wrong_parent_epoch_topology_and_commitment_rejected(self):
        records=[edge("e1")]
        for field,value in (
                ("parent_root_sha256","0"*64),
                ("chart_generation",5),
                ("topology_sha256","0"*64),
                ("graph_commitment_sha256","0"*64),
                ("signed_by_router",True),
                ("native_mth_receipt",True)):
            with self.subTest(field=field):
                corrupted=page(self.chart,records)
                corrupted[field]=value
                verifier=RouterGraphPageVerifier(self.chart)
                with self.assertRaises(RouterGraphPageError):
                    verifier.add_page(corrupted)
                self.assertTrue(verifier.status()["failed"])

    def test_replay_mixed_revision_and_wrong_epoch_fail_closed(self):
        records=[edge("e1"),edge("e2",src=265,dst=10,layer=9,source_head=10)]
        verifier=RouterGraphPageVerifier(self.chart)
        first_page=page(self.chart,records,limit=1)
        verifier.add_page(first_page)
        with self.assertRaises(RouterGraphPageError):
            verifier.add_page(first_page)
        verifier=RouterGraphPageVerifier(self.chart)
        verifier.add_page(first_page)
        changed=page(self.chart,[edge("e1"),edge("e2",src=265,dst=10,layer=9,
                                                source_head=10,relation="refutes")],
                     offset=1,limit=1)
        with self.assertRaises(RouterGraphPageError):
            verifier.add_page(changed)
        verifier=RouterGraphPageVerifier(self.chart)
        verifier.add_page(first_page)
        self.chart.snapshot_generation+=1
        with self.assertRaises(RouterGraphPageError):
            verifier.add_page(page(self.chart,records,offset=1,limit=1))

    def test_untrusted_entry_rejected_even_with_matching_header(self):
        for changes in (
                dict(source_head=99),
                dict(src_route="HW0:MP9"),
                dict(dst=32896),
                dict(layer=10),
                dict(evidence_digest="bad"),
                dict(status="verified"),
                dict(relation="hallucination")):
            with self.subTest(changes=changes):
                bad=edge("e1")
                bad.update(changes)
                verifier=RouterGraphPageVerifier(self.chart)
                with self.assertRaises(RouterGraphPageError):
                    verifier.add_page(page(self.chart,[bad]))

    def test_altered_entry_fails_full_state_digest(self):
        valid=edge("e1")
        p=page(self.chart,[valid])
        p["entries"]=[{**valid,"relation":"refutes"}]
        verifier=RouterGraphPageVerifier(self.chart)
        with self.assertRaises(RouterGraphPageError):
            verifier.add_page(p)


if __name__=="__main__":
    unittest.main()
