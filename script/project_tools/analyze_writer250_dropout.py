#!/usr/bin/env python3
"""Offline receipt chronology; never connects to robot or changes control."""
import json,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def analyze(case):
    d=ROOT/'logs/x2_ssh_takeover_20261020'
    rows=[json.loads(x) for x in (d/(case+'_supervisor.jsonl')).read_text().splitlines()]
    trigger=next(x for x in rows if x['event']=='triggered')
    samples=[x for x in rows if x['event']=='sample']
    tags=['state_leg','state_waist','state_arm','state_head','imu']
    frozen=next((i for i in range(len(samples)-1) if all(samples[i]['counts'][k]==samples[i+1]['counts'][k] for k in tags)),None)
    assert frozen is not None
    first=samples[frozen];end=frozen
    while end+1<len(samples) and all(samples[end+1]['counts'][k]==first['counts'][k] for k in tags):end+=1
    offsets=[x['wall_ns']/1e9-x['mono_s'] for x in rows]
    offset=statistics.median(offsets)
    runtime=(d/(case+'_runtime.log')).read_text()
    def stamp(marker):
        return float(next(re.search(r'\[(\d+\.\d+)\]',s).group(1) for s in runtime.splitlines() if marker in s))
    control=stamp('-> CONTROL');fatal=stamp('E-STOP -> PURE DAMPING')
    last_rx={k:first['mono_s']-first['age_s'][k] for k in tags}
    return dict(case=case,trigger_mono_s=trigger['mono_s'],wall_minus_mono_median_s=offset,wall_minus_mono_range_ms=1000*(max(offsets)-min(offsets)),last_receipt_after_trigger_s={k:v-trigger['mono_s'] for k,v in last_rx.items()},last_receipt_spread_ms=1000*(max(last_rx.values())-min(last_rx.values())),frozen_samples=end-frozen+1,minimum_observed_no_receipt_s={k:samples[end]['mono_s']-v for k,v in last_rx.items()},receipt_resumed_by_after_trigger_s=samples[end+1]['mono_s']-trigger['mono_s'] if end+1<len(samples) else None,command_counts_increase={k:samples[end]['counts'][k]-first['counts'][k] for k in first['counts'] if k.startswith('command_')},control_after_trigger_s=control-(trigger['wall_ns']/1e9),fatal_after_control_s=fatal-control,approx_fatal_since_observer_last_receipt_s={k:fatal-(v+offset) for k,v in last_rx.items()},vendor_trigger_time_trace_available=False,limits='Same-host wall/mono mapping only; receipt blackout does not prove sensor source stopped; vendor initial-trigger Publish trace missing, no unique DDS cause inferred.')
if __name__=='__main__':
    print(json.dumps([analyze(c) for c in ['resume_powered3','settle_powered3']],indent=2))
