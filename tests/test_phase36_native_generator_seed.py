import json
from pathlib import Path
R=Path(__file__).resolve().parents[1];C=R/'research/dynamic_retargeting_20260811/phase36_native_generator_seed_contract.json';O=R/'research/dynamic_retargeting_20260811/phase36_native_generator_seed_result.json'
def test_contract_forbids_contact_labels_and_training():
 c=json.loads(C.read_text());assert c['contact_labels_exported'] is False and c['training_authorized'] is False;assert c['resources']['physics_steps']==0 and c['resources']['optimizer_steps']==0
def test_result_if_present():
 if not O.exists():return
 r=json.loads(O.read_text());assert r['semantics']['contact_labels_present'] is False and r['semantics']['optimizer_eligible'] is False;assert r['decision']['training_unlocked'] is False;assert r['decision']['native_generator_seed_exported']==all(r['checks'].values())
