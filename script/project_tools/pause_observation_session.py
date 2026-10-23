"""Dependency-injected, nonactuating orchestration of one observation/pause session.

No CLI or live adapters are provided. Callbacks are synchronous and must return
promptly: deadlines bound polling, not a blocking callback or the target pause.
The target owns its unconditional CONT cleanup, including if observation dies
during its call. This module never attempts resume, retries, or a second pause.

Results are observational evidence, NOT a safety guarantee or authorization.
In particular, target process status does not establish DDS writer ownership.
"""

import math
from collections.abc import Mapping
from numbers import Real

try:
    from . import observation_window_gate
except ImportError:  # Also support importing from the project_tools directory.
    import observation_window_gate


class SessionError(RuntimeError):
    """The session was rejected or failed to collect the required evidence."""


def session(observer, target, clock, sleep, log):
    """Collect evidence around at most one externally implemented two-second pause.

    ``observer`` supplies start(seconds=60), running(), ready(), fresh(),
    sole_official(), no_custom_writer(), and finish() (an arbitrary result).
    ``target`` supplies pin_identity() and pause_two_seconds(identity), returning
    a mapping with resumed (literal bool True) and stop_to_cont_s (1.9..2.5).
    ``clock`` is a finite, nondecreasing local monotonic seconds callable;
    ``sleep(seconds)`` must advance it. ``log(event)`` receives dict snapshots.

    Readiness has a 12-second budget measured conservatively from BEFORE start;
    admission is rechecked immediately before the window gate and target call.
    Recovery has a 10-second budget after pause returns and needs two continuous
    seconds of fresh, sole-official, no-custom-writer observation. Freshness loss
    resets that interval; observer loss or writer ownership failure rejects it.
    No target status is substituted for observer ownership checks.

    Return a dict containing timings, target result, recovery evidence, events,
    and observer_result from finish(). Raise SessionError on failed checks.
    Dependency exceptions propagate. finish() runs exactly once whenever start
    is attempted, including partial start failure. Cleanup failure propagates
    unless another error is already active (then attached as cleanup_error).
    """
    duration = 60.0
    events = []
    last_time = None
    started = False
    failure = None
    evidence = None

    def now():
        nonlocal last_time
        stamp = clock()
        if (isinstance(stamp, bool) or not isinstance(stamp, Real)
                or not math.isfinite(stamp)):
            raise SessionError("clock must return finite real seconds")
        if last_time is not None and stamp < last_time:
            raise SessionError("clock moved backwards")
        last_time = stamp
        return stamp

    def emit(state, **fields):
        event = {"state": state, **fields}
        events.append(event)
        log(dict(event))

    def snapshot():
        # Evaluate every predicate, rather than short-circuiting telemetry.
        return {
            "running": bool(observer.running()),
            "ready": bool(observer.ready()),
            "fresh": bool(observer.fresh()),
            "sole_official": bool(observer.sole_official()),
            "no_custom_writer": bool(observer.no_custom_writer()),
        }

    def poll_until(deadline, stamp):
        delay = min(0.1, deadline - stamp)
        if delay <= 0:
            raise SessionError("polling deadline exhausted")
        sleep(delay)
        after = now()
        if after <= stamp:
            raise SessionError("clock did not advance during sleep")

    observer_started_at = now()  # Conservative: precedes even observer startup.
    try:
        started = True
        observer.start(seconds=60)
        emit("started", at=observer_started_at, duration_s=duration)
        ready_deadline = observer_started_at + 12.0
        while True:
            status = snapshot()
            stamp = now()
            emit("waiting_ready", at=stamp, **status)
            if not status["running"]:
                raise SessionError("observer stopped before readiness")
            if stamp > ready_deadline:
                raise SessionError("observer readiness exceeded 12 seconds")
            if all(status.values()):
                break
            if stamp >= ready_deadline:
                raise SessionError("observer readiness timed out")
            poll_until(ready_deadline, stamp)

        identity = target.pin_identity()
        # Logging and identity pinning happen before the final admission checks.
        emit("checking_admission", at=now())
        admission = snapshot()
        pause_requested_at = now()
        if not all(admission.values()):
            raise SessionError("observer admission recheck rejected pause")
        if not observation_window_gate.check_window(
                observer_running=admission["running"], ready=admission["ready"],
                observer_started_at=observer_started_at, now=pause_requested_at,
                duration=duration, pause_s=2.0, recovery_s=10.0, margin_s=5.0):
            raise SessionError("observation window rejected pause")
        pause_result = target.pause_two_seconds(identity)
        # The target must finish its own finally/CONT before this postcheck.
        pause_returned_at = now()
        post_pause = snapshot()
        emit("pause_returned", at=pause_returned_at, **post_pause)
        if not post_pause["running"]:
            raise SessionError("observer disappeared during pause")
        if not isinstance(pause_result, Mapping):
            raise SessionError("target pause result must be a mapping")
        elapsed = pause_result.get("stop_to_cont_s")
        if pause_result.get("resumed") is not True:
            raise SessionError("target did not report resumed=True")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, Real)
                or not math.isfinite(elapsed) or not 1.9 <= elapsed <= 2.5):
            raise SessionError("target stop-to-CONT interval is outside 1.9..2.5s")

        recovery_deadline = pause_returned_at + 10.0
        stable_since = None
        status = post_pause
        while True:
            stamp = now()
            if not status["running"]:
                raise SessionError("observer stopped during recovery")
            if not status["sole_official"] or not status["no_custom_writer"]:
                raise SessionError("writer ownership check failed during recovery")
            if status["fresh"]:
                if stable_since is None:
                    stable_since = stamp
            else:
                stable_since = None
            stable_s = 0.0 if stable_since is None else stamp - stable_since
            emit("recovering", at=stamp, stable_s=stable_s, **status)
            if stamp > recovery_deadline:
                raise SessionError("recovery exceeded 10 seconds")
            if stable_s >= 2.0:
                evidence = {
                    "kind": "observation_evidence", "safety_guarantee": False,
                    "observer_started_at": observer_started_at,
                    "pause_requested_at": pause_requested_at,
                    "pause_returned_at": pause_returned_at,
                    "pause_result": dict(pause_result),
                    "recovery_started_at": stable_since,
                    "recovery_completed_at": stamp,
                    "recovery_stable_s": stable_s, "events": events,
                }
                emit("complete", at=stamp, stable_s=stable_s)
                break
            if stamp >= recovery_deadline:
                raise SessionError("continuous two-second recovery timed out")
            poll_until(recovery_deadline, stamp)
            status = snapshot()
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if started:
            try:
                observer_result = observer.finish()
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                # Preserve the primary failure without hiding cleanup evidence.
                failure.cleanup_error = cleanup_error
            else:
                if evidence is not None:
                    evidence["observer_result"] = observer_result
    return evidence
