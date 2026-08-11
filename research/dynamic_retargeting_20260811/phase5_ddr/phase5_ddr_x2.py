#!/usr/bin/env python3
"""Phase5 route C: minimal DDR-style rolling CEM on raw official X2 MuJoCo."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path('/home/humanplus/projects/ZHY/dsms_workspace')
P3=ROOT/'phase3b'; OUT=ROOT/'phase5_ddr'
REPO=Path('/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim')
SCENE=Path('/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml')
CONTROL=Path('/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml')
LUNGE=Path('/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/silver_contact/phase30_time_dilation/x2_phase30_time_dilation_1p46.pkl')
sys.path[:0]=[str(P3),str(REPO/'tools'),str(REPO)]
import mujoco
import retarget.run_x2_forefoot_official_physics_screen as physics
from x2_lunge_dsms_phase3b_corrected import load_reference

SEED=5505; DT=.001; NODE_STEPS=20; NF=18; DECISIONS=17; H=4; POP=24; ELITE=6; ITERS=3
BODIES=['pelvis','left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def intended(t): return {'left':t<=10/30+1e-12,'right':t<=25/30+1e-12}

def setup():
    entry,adapted,ref,X,contract=load_reference(NF)
    model=mujoco.MjModel.from_xml_path(str(SCENE)); model.opt.timestep=DT
    bids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in BODIES]
    # Reference semantics are FK keypoints/orientation only; joint angles do not enter task cost.
    kp=np.zeros((NF,len(bids),3)); torso_q=np.zeros((NF,4)); d=mujoco.MjData(model)
    for i in range(NF):
        d.qpos[:]=X[i,:model.nq]; d.qvel[:]=X[i,model.nq:model.nq+model.nv]; mujoco.mj_forward(model,d)
        kp[i]=d.xpos[bids]; torso_q[i]=d.xquat[bids[0]]
    return model,contract,X[:NF],kp,torso_q,bids

def qangle(q1,q2): return 2*np.arccos(np.clip(abs(float(np.dot(q1,q2))),0,1))

def contacts(model,data,floor,geom_side):
    c={'left':set(),'right':set()}
    for ci in range(data.ncon):
        x=data.contact[ci]; a,b=int(x.geom1),int(x.geom2); other=b if a==floor else a if b==floor else -1
        if other in geom_side:c[geom_side[other]].add(other)
    return c

def segment(model,contract,bids,kpref,qref,data0,qstart,targets,start_node,record=False):
    d=mujoco.MjData(model); d.qpos[:]=data0.qpos; d.qvel[:]=data0.qvel; d.act[:]=data0.act; d.time=data0.time; mujoco.mj_forward(model,d)
    floor,feet_all=physics.foot_geom_contract(model)
    feet={s:tuple(g for g in v if model.geom_contype[g]!=0) for s,v in feet_all.items()}
    gs={g:s for s,v in feet.items() for g in v}
    prev=d.geom_xpos.copy(); prevv=d.qvel[:2].copy(); cost=0.; trace=[]; sat=0; nsat=0
    for j,tgt1 in enumerate(targets):
        tgt0=qstart if j==0 else targets[j-1]
        for k in range(NODE_STEPS):
            a=(k+1)/NODE_STEPS; tgt=(1-a)*tgt0+a*tgt1
            q=d.qpos[contract.qpos_addresses]; qd=d.qvel[contract.qvel_addresses]
            raw=contract.kp*(tgt-q)-contract.kd*qd; sat+=np.sum((raw<contract.torque_low)|(raw>contract.torque_high)); nsat+=len(raw)
            d.ctrl[:]=np.clip(raw,contract.torque_low,contract.torque_high); mujoco.mj_step(model,d)
            phase=start_node+j+a; ri=min(int(np.floor(phase)),NF-1); rj=min(ri+1,NF-1); ra=phase-ri
            rk=(1-ra)*kpref[ri]+ra*kpref[rj]
            pos=d.xpos[bids]-rk
            # pelvis/ankles dominate semantics; wrists preserve upper-body lunge intent.
            cost += 4*np.dot(pos[0],pos[0])+10*np.sum(pos[1:3]**2)+1.5*np.sum(pos[3:]**2)
            cost += 2*qangle(d.xquat[bids[0]],qref[ri])**2
            cs=contacts(model,d,floor,gs); tm=(phase*0.02)
            for side,geoms in feet.items():
                if intended(tm)[side]:
                    if not cs[side]: cost+=2.0
                    for g in cs[side]:
                        v=np.linalg.norm((d.geom_xpos[g,:2]-prev[g,:2])/DT); cost+=5*v*v
            if not cs['left'] and not cs['right']:cost+=4.0
            acc=np.linalg.norm((d.qvel[:2]-prevv)/DT); cost+=.01*max(0,acc-4)**2
            cost+=1e-7*np.dot(d.ctrl,d.ctrl)
            if record: trace.append((d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()))
            prev[:]=d.geom_xpos; prevv=d.qvel[:2].copy()
    cost += .02*np.sum(np.diff(np.vstack([qstart,targets]),axis=0)**2)
    return cost,d,trace,sat/max(nsat,1)

def replay(model,contract,bids,kpref,qref,X,targets):
    d=mujoco.MjData(model); d.qpos[:]=X[0,:model.nq]; d.qvel[:]=X[0,model.nq:]; mujoco.mj_forward(model,d)
    _,d,tr,sat=segment(model,contract,bids,kpref,qref,d,targets[0],targets[1:],0,True)
    floor,feet_all=physics.foot_geom_contract(model)
    feet={s:tuple(g for g in v if model.geom_contype[g]!=0) for s,v in feet_all.items()}
    gs={g:s for s,v in feet.items() for g in v}
    # Re-evaluate saved states without trusting optimizer-side diagnostics.
    prev=None; init={}; realized={s:[] for s in feet}; slips={s:[] for s in feet}; exc={s:[] for s in feet}; pen={s:np.inf for s in feet}
    rv=[]; rz=[]; tilt=[]; sem=[]
    for k,(qp,qv,u) in enumerate(tr):
        d.qpos[:]=qp; d.qvel[:]=qv; mujoco.mj_forward(model,d)
        if prev is None:
            prev=d.geom_xpos.copy(); init={s:np.mean(d.geom_xpos[list(v),:2],0) for s,v in feet.items()}
        cs=contacts(model,d,floor,gs); tm=(k+1)*DT
        for s,v in feet.items():
            realized[s].append(bool(cs[s])); exc[s].append(np.linalg.norm(np.mean(d.geom_xpos[list(v),:2],0)-init[s]))
            for g in cs[s]: slips[s].append(np.linalg.norm((d.geom_xpos[g,:2]-prev[g,:2])/DT))
            pen[s]=min(pen[s],min(float(d.geom_xpos[g,2]-model.geom_size[g,0]-d.geom_xpos[floor,2]) for g in v))
        ri=min(int(round(tm/.02)),NF-1); sem.append(np.sqrt(np.mean((d.xpos[bids]-kpref[ri])**2)))
        rv.append(qv[:2]); rz.append(qp[2]); tilt.append(physics.root_tilt(qp[3:7])); prev[:]=d.geom_xpos
    rv=np.asarray(rv); acc=np.linalg.norm(np.diff(rv,axis=0)/DT,axis=1); times=(np.arange(len(tr))+1)*DT
    result={'duration_s':len(tr)*DT,'survive_full_prefix':bool(min(rz)>=.42 and max(tilt)<=.90),'root_z_min_m':float(min(rz)),'tilt_max_rad':float(max(tilt)),
      'torque_saturation_fraction':float(sat),'penetration_min_mm':{s:float(1000*pen[s]) for s in feet},'contact':{},
      'unintended_flight_fraction':float(np.mean(~(np.asarray(realized['left'])|np.asarray(realized['right'])))),
      'root_horiz_accel_p95_mps2':float(np.percentile(acc,95)),'head_q_max_abs_rad':float(np.max(np.abs(np.stack([x[0] for x in tr])[:,contract.qpos_addresses[-2:]]))),
      'semantic_keypoint_rmse_m':float(np.mean(sem))}
    for s in feet:
        mask=np.array([intended(t)[s] for t in times]); actual=np.asarray(realized[s])
        result['contact'][s]={'contradiction_fraction':float(np.mean(~actual[mask])),'stance_speed_p95_mps':float(np.percentile(slips[s],95)) if slips[s] else None,'stance_excursion_max_m':float(np.max(np.asarray(exc[s])[mask]))}
    result['gates']={'survive':result['survive_full_prefix'],'penetration':min(result['penetration_min_mm'].values())>=-.5,
      'stance_speed':all(result['contact'][s]['stance_speed_p95_mps'] is not None and result['contact'][s]['stance_speed_p95_mps']<=.10 for s in feet),
      'stance_excursion':all(result['contact'][s]['stance_excursion_max_m']<=.03 for s in feet),'flight':result['unintended_flight_fraction']<=.02,
      'root_accel':result['root_horiz_accel_p95_mps2']<=4,'head':result['head_q_max_abs_rad']<=.02,'semantic':result['semantic_keypoint_rmse_m']<=.08}
    result['pass']=all(result['gates'].values()); return result,tr

def main():
    OUT.mkdir(parents=True,exist_ok=True); model,c,X,kp,rq,bids=setup(); bronze=np.array([X[i,c.qpos_addresses] for i in range(NF)]); bronze[:,-2:]=0
    # Preflight: raw baseline and zero-offset replay must be bit deterministic.
    b1,_=replay(model,c,bids,kp,rq,X,bronze); b2,_=replay(model,c,bids,kp,rq,X,bronze)
    pre={'schema':'x2_ddr_phase5_preflight_v1','scene_sha256':sha(SCENE),'control_sha256':sha(CONTROL),'motion_sha256':sha(LUNGE),'mujoco':mujoco.__version__,
      'raw_motor':bool(np.all(model.actuator_gaintype==mujoco.mjtGain.mjGAIN_FIXED) and np.all(model.actuator_biasprm[:,1:3]==0)),
      'official_euler':int(model.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_EULER),'baseline_1':b1,'zero_candidate_deterministic':b1==b2}
    pre['pass']=pre['mujoco']=='3.3.7' and pre['raw_motor'] and pre['official_euler'] and pre['zero_candidate_deterministic']
    (OUT/'preflight.json').write_text(json.dumps(pre,indent=2)+'\n'); print(json.dumps({'preflight_pass':pre['pass'],'baseline_pass':b1['pass']},indent=2),flush=True)
    if not pre['pass']: raise SystemExit(2)
    rng=np.random.default_rng(SEED); d=mujoco.MjData(model); d.qpos[:]=X[0,:model.nq]; d.qvel[:]=X[0,model.nq:]; mujoco.mj_forward(model,d)
    committed=[bronze[0].copy()]; history=[]; t0=time.time()
    sigma0=np.full(model.nu,.08); sigma0[:12]=.12; sigma0[12:15]=.05; sigma0[-2:]=0
    for t in range(DECISIONS):
        h=min(H,DECISIONS-t); mean=np.zeros((h,model.nu)); sd=np.tile(sigma0,(h,1)); base=bronze[t+1:t+1+h]
        for it in range(ITERS):
            eps=rng.normal(size=(POP,h,model.nu))*sd+mean; eps[:,:,-2:]=0
            costs=[]
            for p in range(POP):
                tg=np.clip(base+eps[p],model.jnt_range[model.actuator_trnid[:,0],0],model.jnt_range[model.actuator_trnid[:,0],1]); tg[:,-2:]=0
                co,_,_,_=segment(model,c,bids,kp,rq,d,committed[-1],tg,t,False); costs.append(co)
            elite=np.argsort(costs)[:ELITE]; mean=np.mean(eps[elite],0); sd=np.maximum(np.std(eps[elite],0),.015); sd[:,-2:]=0
        chosen=np.clip(base[0]+mean[0],model.jnt_range[model.actuator_trnid[:,0],0],model.jnt_range[model.actuator_trnid[:,0],1]); chosen[-2:]=0
        co,d,_,_=segment(model,c,bids,kp,rq,d,committed[-1],[chosen],t,False); committed.append(chosen); history.append({'node':t+1,'best_rollout_cost':float(min(costs)),'executed_cost':float(co)})
    targets=np.asarray(committed); result,tr=replay(model,c,bids,kp,rq,X,targets); result.update({'schema':'x2_ddr_phase5_result_v1','seed':SEED,'population':POP,'elite':ELITE,'iterations':ITERS,'horizon_nodes':H,'runtime_s':time.time()-t0,'history':history,'baseline':b1})
    np.savez_compressed(OUT/'ddr_candidate.npz',targets=targets,qpos=np.asarray([x[0] for x in tr]),qvel=np.asarray([x[1] for x in tr]),ctrl=np.asarray([x[2] for x in tr]))
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:result[k] for k in ('pass','runtime_s','semantic_keypoint_rmse_m','gates')},indent=2))

if __name__=='__main__':main()
