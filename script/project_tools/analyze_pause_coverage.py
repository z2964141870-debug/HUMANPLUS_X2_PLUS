#!/usr/bin/env python3
"""Offline verdict for captured no-custom pause. No live APIs."""
import json,argparse
from pathlib import Path
KEYS=('state_leg','state_waist','state_arm','state_head','imu_torso')
def verdict(data):
 samples=data['samples'];end=samples[-1]['streams'];pause=next(x for x in data['pause'] if x['event']=='resume_confirmed')
 # Cumulative receive-gap covers the complete capture, not just final fresh sample.
 gaps={k:end[k]['max_rx_gap_s'] for k in KEYS}
 final_fresh=all(end[k]['age_s'] is not None and 0<=end[k]['age_s']<=.5 for k in KEYS+('cmd_leg','cmd_waist','cmd_arm','cmd_head'))
 continuous=all(0<=v<=.5 for v in gaps.values())
 return {'verdict':'PASS' if continuous and final_fresh else 'FAIL','feedback_continuous':continuous,'final_flow_fresh':final_fresh,'feedback_max_gap_s':gaps,'pause_seconds':pause['stop_to_cont_s'],'reason':'Final recovery does not erase blackout. 500ms is offline classification against native state-age threshold, not a new robot threshold. Whole-capture gaps do not alone prove cause.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('capture');a=p.parse_args();result=verdict(json.loads(Path(a.capture).read_text()));print(json.dumps(result,indent=2))
