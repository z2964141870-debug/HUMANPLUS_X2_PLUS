#!/usr/bin/env python3
"""Export a regular 50 Hz state-only seed from the stable Phase34 move."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import mujoco,numpy as np
from official_x2.analyze_phase34_full_closed_trace import read_trace,root_score,sha256
from official_x2.replay_official_trace_direct_mujoco import yaw_tilt

REPO=Path(__file__).resolve().parents[2];MAN=REPO/'reports/official_x2/phase34_closed_full_trace_manifest.json';WBT=REPO/'reports/retarget/x2_faithful_zero_update_phase26.json';CONTRACT=REPO/'research/dynamic_retargeting_20260811/phase36_native_generator_seed_contract.json';OUT=REPO/'research/dynamic_retargeting_20260811/phase36_native_generator_seed.npz';RESULT=REPO/'research/dynamic_retargeting_20260811/phase36_native_generator_seed_result.json';SCENE=Path('/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml')

def run():
 c=json.loads(CONTRACT.read_text());m=json.loads(MAN.read_text());roll=Path(m['artifacts']['rollout']);mm=Path(m['artifacts']['mmap'])
 if sha256(roll)!=m['artifacts']['rollout_sha256'] or sha256(mm)!=m['artifacts']['mmap_sha256'] or sha256(SCENE)!=m['provenance']['scene_sha256']:raise RuntimeError('Phase34 asset drift')
 physical=read_trace(mm);tele=json.loads(roll.read_text())['trace'];ti=next(i for i,r in enumerate(tele) if r['stage']=='move' and abs(r['elapsed_s'])<1e-12);row=tele[ti]
 search=range(max(0,2100),min(len(physical['qpos']),2400));base=min(search,key=lambda i:root_score(physical['qpos'][i],row));score=root_score(physical['qpos'][base],row)
 idx=base+20*np.arange(200);qpos=physical['qpos'][idx];qvel=physical['qvel'][idx];time=np.arange(200)/50.;model=mujoco.MjModel.from_xml_path(str(SCENE));official31=json.loads(WBT.read_text())['contract']['official31_order'];qadr=np.asarray([int(model.joint(n).qposadr[0]) for n in official31]);dadr=np.asarray([int(model.joint(n).dofadr[0]) for n in official31]);ranges=model.jnt_range[[int(model.joint(n).id) for n in official31]]
 overshoot=float(max(0,np.max(np.maximum(ranges[:,0]-qpos[:,qadr],qpos[:,qadr]-ranges[:,1]))));tilt=np.asarray([yaw_tilt(q[3:7])[1] for q in qpos]);finite=bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)))
 np.savez_compressed(OUT,time_s=time,qpos=qpos,qvel=qvel,root_pose=qpos[:,:7],root_velocity=qvel[:,:6],joint_names=np.asarray(official31),joint_pos31=qpos[:,qadr],joint_vel31=qvel[:,dadr])
 metrics={'frames':len(qpos),'fps':50.,'first_move_telemetry_index':ti,'base_physics_index':base,'alignment_score':score,'root_z_min_m':float(np.min(qpos[:,2])),'root_tilt_max_rad':float(np.max(tilt)),'joint_limit_overshoot_rad':overshoot,'finite':finite}
 g=c['hard_gates'];checks={'frames':metrics['frames']==g['frames'],'fps':metrics['fps']==g['fps'],'alignment':score<=g['alignment_score_max'],'finite':finite,'root_z':metrics['root_z_min_m']>=g['root_z_min_m'],'root_tilt':metrics['root_tilt_max_rad']<=g['root_tilt_max_rad'],'limits':overshoot<=g['joint_limit_overshoot_rad']};passed=all(checks.values())
 return {'stage':'Dynamic Retargeting Phase36','scope':'offline regular-50Hz state seed; no contact labels/physics/optimizer/GPU','contract_sha256':sha256(CONTRACT),'assets':{'phase34_manifest_sha256':sha256(MAN),'output':str(OUT),'output_sha256':sha256(OUT)},'metrics':metrics,'checks':checks,
  'semantics':{'usable_as_native_success_basin_seed':passed,'joint_order_source':'Phase26 frozen official31','contact_labels_present':False,'optimizer_eligible':False},'execution':c['resources'],'decision':{'native_generator_seed_exported':passed,'live_generator_zero_update_authorized':False,'training_unlocked':False,'result':'NATIVE_GENERATOR_SEED_EXPORTED_ZERO_UPDATE_STILL_LOCKED' if passed else 'NATIVE_GENERATOR_SEED_REJECTED'}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=RESULT);a=ap.parse_args();r=run();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'metrics':r['metrics'],'checks':r['checks'],'decision':r['decision']},indent=2))
if __name__=='__main__':main()
