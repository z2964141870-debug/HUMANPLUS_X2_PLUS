#!/usr/bin/env python3
"""Zero-search correction: replay saved controls with active sole spheres only."""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import numpy as np

OUT=Path('/home/humanplus/projects/ZHY/dsms_workspace/phase5_ddr')
sys.path.insert(0,str(OUT))
from phase5_ddr_x2 import setup,replay,NF

def main():
    model,contract,X,kp,rq,bids=setup()
    bronze=np.array([X[i,contract.qpos_addresses] for i in range(NF)]); bronze[:,-2:]=0
    saved=np.load(OUT/'ddr_candidate.npz'); candidate=np.asarray(saved['targets'],dtype=np.float64)
    baseline,_=replay(model,contract,bids,kp,rq,X,bronze)
    corrected,_=replay(model,contract,bids,kp,rq,X,candidate)
    old_path=OUT/'result.json'; backup=OUT/'result_pre_active12_audit.json'
    if not backup.exists(): shutil.copy2(old_path,backup)
    old=json.loads(backup.read_text())
    preserved={k:old[k] for k in ('seed','population','elite','iterations','horizon_nodes','runtime_s','history')}
    corrected.update(preserved)
    corrected.update({
      'schema':'x2_ddr_phase5_result_active12_v2',
      'baseline':baseline,
      'audit_provenance':{
        'kind':'zero-new-search deterministic replay of saved targets',
        'candidate_artifact':str(OUT/'ddr_candidate.npz'),
        'excluded_visual_geoms':[14,37],
        'sole_filter':'model.geom_contype[g] != 0',
        'active_spheres_per_foot':12,
        'supersedes':str(backup),
        'reason':'old common penetration auditor treated contype=0 visual mesh size[0] as a sphere radius'
      }
    })
    old_path.write_text(json.dumps(corrected,indent=2)+'\n')
    audit={'schema':'x2_ddr_phase5_active12_audit_v1','new_searches':0,'baseline':baseline,'candidate':corrected,
           'candidate_targets_bit_equal_saved':bool(np.array_equal(candidate,saved['targets']))}
    (OUT/'active12_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'baseline_penetration':baseline['penetration_min_mm'],'candidate_penetration':corrected['penetration_min_mm'],
                      'candidate_gates':corrected['gates'],'candidate_pass':corrected['pass']},indent=2))

if __name__=='__main__':main()
