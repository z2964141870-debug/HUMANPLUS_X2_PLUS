"""Pure nonactuating guard for a fixed-duration remote observation window."""
import math

def check_window(*,observer_running,ready,observer_started_at,now,duration,pause_s=2.,recovery_s=10.,margin_s=5.):
    """Times are local monotonic stamps, not cross-host sensor timestamps.

    started_at must be recorded BEFORE remote observer launch (conservative).
    Recheck immediately before action. A successful result is not authorization.
    """
    values=(observer_started_at,now,duration,pause_s,recovery_s,margin_s)
    if not all(math.isfinite(x) for x in values):return False
    if min(duration,pause_s,recovery_s,margin_s)<=0 or now<observer_started_at:return False
    return bool(observer_running and ready and duration-(now-observer_started_at)>=pause_s+recovery_s+margin_s)
