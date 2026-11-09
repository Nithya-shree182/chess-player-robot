"""Slow, contact-sensing gripper closing and releasing (verified on the rook, 2026-11-06).

A fast swing of the moving jaw flicks a narrow piece sideways before it is held, and a closed-loop 'go to 1 %' cannot tell
contact from air. slow_close() steps the COMMAND down in small increments (about one per second at the arm's speed setting) and
compares the MEASURED jaw with the command: when the jaw can no longer follow, the pads touch something. It then squeezes lightly.
slow_release() opens in steps the same way so the piece is not dragged or knocked as the pad leaves it.
"""
import time

from chess_player import config as C

STEP_PCT = 0.5  # command step while closing / opening
LAG_PCT = 1.0  # the jaw is this far behind the command: contact
# Speed knobs (2026-11-09: user wanted faster open/close, same distances). Before: close wait 0.1 s / move 0.35 s, release step 0.5 % / wait 0.05 s / move 0.35 s.
CLOSE_WAIT_S = 0.1  # back to the original: at 0.05/0.2 the jaw read 'contact' too early (10.7 % vs 8.9 % taught) and the grip was rejected
CLOSE_MOVE_S = 0.35
FAST_MARGIN_PCT = 4.0  # closing is fast down to (taught contact + this), then slow and contact-sensing
RELEASE_STEP_PCT = 1.0
RELEASE_WAIT_S = 0.02
RELEASE_MOVE_S = 0.2


def slow_close(arm, squeeze=C.HOLD_SQUEEZE_PCT, floor=1.0, fast_to=None):
    """Close in small steps until the pads touch, then squeeze lightly. Returns (contact_pct or None, final_pct).
    None means the jaws closed all the way without meeting anything (an empty hand)."""
    cmd = float(arm.joints()["gripper"])
    if fast_to is not None and cmd > fast_to + 0.5:
        arm.gripper(fast_to, 0.5)  # quick part of the closing, well before the pads can touch
        time.sleep(0.7)
        cmd = float(fast_to)
    while cmd > floor + 0.5:
        cmd = max(floor, cmd - STEP_PCT)
        arm.gripper(cmd, CLOSE_MOVE_S)
        time.sleep(CLOSE_WAIT_S)
        p = float(arm.present()["gripper"])
        if p > cmd + LAG_PCT:
            arm.gripper(max(0.0, p - squeeze), 0.8)
            arm.dwell()
            return p, float(arm.present()["gripper"])
    return None, float(arm.present()["gripper"])


def slow_release(arm, to_pct, step=RELEASE_STEP_PCT):
    """Open in small steps to `to_pct`."""
    cmd = float(arm.joints()["gripper"])
    while cmd < to_pct - 1e-6:
        cmd = min(to_pct, cmd + step)
        arm.gripper(cmd, RELEASE_MOVE_S)
        time.sleep(RELEASE_WAIT_S)
    arm.dwell()
    return float(arm.present()["gripper"])
