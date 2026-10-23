"""Pure evidence predicate, not a robot controller or authorization."""
import math
GROUPS=('leg','waist','arm','head')
def recovered(before_counts,after_counts,owners,ages,now,window_started,window_s=2.):
    """Require sole official endpoint + new command receipt throughout observation.

    Call once per sample for at least window_s with external continuous success latch.
    This alone cannot establish GID attribution, MC mode, torque or physical posture.
    """
    if not all(math.isfinite(x) for x in (now,window_started,window_s)) or window_s<=0 or now-window_started<window_s:return False
    for g in GROUPS:
        if owners.get(g)!=['mc_ros2_node']:return False
        if after_counts.get(g,0)-before_counts.get(g,0)<2:return False
        age=ages.get('command_'+g)
        if age is None or not 0<=age<=.5:return False
    for key in ['state_'+g for g in GROUPS]+['imu']:
        age=ages.get(key)
        if age is None or not 0<=age<=.5:return False
    return True
