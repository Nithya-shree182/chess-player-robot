"""Square-centre formula fitted from the hand-taught centres (data/center_teach.json, made by scripts/center_teach.sh).

MODEL (board frame, mm; +X toward the h-file = the robot's right, +Y toward rank 8 = forward):
    where the tip must go for `piece` on a square with nominal centre n = (x, y) from the grid:
        target(piece, n) = n + c[piece] + M @ (n - n0)         n0 = board centre
    c[piece] = constant offset per piece type (the grip puts the fixed finger beside the centre, so it differs per type),
    M = one 2x2 matrix shared by all pieces (the leftover scale/rotation/perspective error of the board->robot calibration across the board).
Everything is in BOARD coordinates, so it travels with whatever the grid/calibration (regrid + camcal/recalibrate) gives in a new scene: a
new setup is calibrated the usual way and the SAME formula (this file's model.json) is applied on top. Validate it in a new scene with
`--check` (teach 2-3 centres there, compare with the prediction).

  python -m chess_player.calib.centre_model            fit + leave-one-out validation -> data/centre_model/model.json, report.txt
  python -m chess_player.calib.centre_model --check    compare the saved teach file with the saved model (no refit)
  centre_offset(piece, "e2") -> np.array([dx, dy]) to ADD to the nominal square centre (mm); None if there is no model.
"""
import argparse
import json

import numpy as np

from chess_player import config as C

OUT = C.DATA / "centre_model"
TEACH = C.DATA / "center_teach.json"
N0 = np.array([C.PLAY_MM / 2.0, C.PLAY_MM / 2.0])
MIN_FOR_M = 6  # fewer entries than this: constants only (a 2x2 matrix from 3-5 points would just chase noise)
WARN_MM = 2.0


def load_teach(path=TEACH):
    d = json.loads(path.read_text())
    return [(e.get("piece", "*"), np.array(e["square_centre_board_mm"], float),
             np.array([e["offset_from_centre_mm"]["x"], e["offset_from_centre_mm"]["y"]], float)) for e in d.values()]


def fit(samples, use_m=None):
    """samples: [(piece, nominal_xy, offset_xy)]. Returns {'c': {piece: [dx, dy]}, 'M': 2x2, 'use_m': bool}."""
    pieces = sorted({p for p, _, _ in samples})
    use_m = len(samples) >= MIN_FOR_M if use_m is None else use_m
    y = np.zeros(2 * len(samples))
    # separate constants for x and y: unknowns = cx[p], cy[p], M(4)
    cols = 2 * len(pieces) + (4 if use_m else 0)
    A = np.zeros((2 * len(samples), cols))
    for i, (p, n, d) in enumerate(samples):
        k = pieces.index(p)
        A[2 * i, k] = 1.0
        A[2 * i + 1, len(pieces) + k] = 1.0
        if use_m:
            u = n - N0
            A[2 * i, 2 * len(pieces):2 * len(pieces) + 2] = u
            A[2 * i + 1, 2 * len(pieces) + 2:2 * len(pieces) + 4] = u
        y[2 * i], y[2 * i + 1] = d
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    c = {p: [float(sol[k]), float(sol[len(pieces) + k])] for k, p in enumerate(pieces)}
    M = np.zeros((2, 2))
    if use_m:
        M[0], M[1] = sol[2 * len(pieces):2 * len(pieces) + 2], sol[2 * len(pieces) + 2:]
    return {"c": c, "M": M.tolist(), "use_m": bool(use_m)}


def predict(model, piece, nominal_xy):
    c = np.array(model["c"].get(piece, model["c"].get("*", [0.0, 0.0])))
    return c + np.array(model["M"]) @ (np.asarray(nominal_xy, float) - N0)


def validate(samples):
    """Leave-one-out: fit without one entry, predict it. Returns (resid_in_sample, loo) lists of error vectors (mm)."""
    m = fit(samples)
    ins = [predict(m, p, n) - d for p, n, d in samples]
    loo = []
    for i, (p, n, d) in enumerate(samples):
        rest = samples[:i] + samples[i + 1:]
        if not any(q == p for q, _, _ in rest):
            loo.append(None)  # the only sample of that piece: nothing to predict it from
            continue
        loo.append(predict(fit(rest), p, n) - d)
    return m, ins, loo


def _rms(v):
    v = [x for x in v if x is not None]
    return float(np.sqrt(np.mean([float(x @ x) for x in v]))) if v else float("nan")


def report(samples, names, m, ins, loo):
    L = [f"{len(samples)} taught centres, pieces {sorted(m['c'])}, shared matrix {'ON' if m['use_m'] else 'OFF (too few entries)'}"]
    for p, c in m["c"].items():
        L.append(f"  c[{p}] = {c[0]:+.2f} mm X, {c[1]:+.2f} mm Y")
    if m["use_m"]:
        L.append(f"  M = {np.round(m['M'], 4).tolist()}  (offset change per mm of board position)")
    L.append(f"in-sample RMS {_rms(ins):.2f} mm, leave-one-out RMS {_rms(loo):.2f} mm (the honest number)")
    for nm, a, b in zip(names, ins, loo):
        L.append(f"  {nm:8s} fit error {a[0]:+.2f},{a[1]:+.2f}   predicted-from-the-others error " + ("n/a" if b is None else f"{b[0]:+.2f},{b[1]:+.2f}"))
    bad = [x for x in loo if x is not None and np.linalg.norm(x) > WARN_MM]
    L.append("OK: every held-out prediction within %.1f mm" % WARN_MM if not bad else
             f"WARNING: {len(bad)} held-out predictions are off by more than {WARN_MM} mm: teach more squares, or the taught centres are inconsistent (re-teach those)")
    return "\n".join(L)


def centre_offset(piece, square):
    """Offset (mm, board frame) to add to the nominal centre of `square` for `piece`, from data/centre_model/model.json (None = no model)."""
    f = OUT / "model.json"
    if not f.exists():
        return None
    import chess

    from chess_player.geometry import square_center_mm

    return predict(json.loads(f.read_text()), piece.lower(), square_center_mm(chess.parse_square(square)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare the teach file with the SAVED model (use in a new scene after teaching 2-3 centres)")
    ap.add_argument("--teach", default=str(TEACH))
    a = ap.parse_args()
    raw = json.loads(open(a.teach).read())
    names = list(raw)
    samples = load_teach(C.Path(a.teach))
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check:
        m = json.loads((OUT / "model.json").read_text())
        errs = [predict(m, p, n) - d for p, n, d in samples]
        for nm, e in zip(names, errs):
            print(f"  {nm:8s} model minus taught: {e[0]:+.2f}, {e[1]:+.2f} mm")
        print(f"RMS {_rms(errs):.2f} mm -> " + ("model holds in this scene" if _rms(errs) <= WARN_MM else "model does NOT hold: the scene's calibration differs, re-run regrid/recalibrate/camcal and teach again"))
        return
    m, ins, loo = validate(samples)
    txt = report(samples, names, m, ins, loo)
    print(txt)
    (OUT / "model.json").write_text(json.dumps(m, indent=1))
    (OUT / "report.txt").write_text(txt + "\n")
    (OUT / "teach_used.json").write_text(json.dumps(raw, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
