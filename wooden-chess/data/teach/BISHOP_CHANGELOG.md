# Bishop (from scripts/move.sh sessions)
- 2026-11-09: problem: bishop h1 -> f3 grabbed the base at the corner of h1, not the centre. Cause (likely): bishop numbers were taught at c1 (shift_x -10, y -4.3) and it has no camera reference (no cam_ref), unlike the pawn.
- 2026-11-09: user wants the grip at the thick upper part of the bishop, whose diameter equals the pawn's base. So the bishop now uses the PAWN's gripping numbers
  (shift_x -13, shift_y +1.7, contact 8.9 %, squeeze 1.8, place dx +3 / dy -2) and grip_tip 25.05 -> 45.05 mm (+20 mm, as asked). Backup of the old json in data/teach/backups/.
  The pawn contact tolerance (3.5 %) also applies to the bishop (wooden_pick_place.py, PAWN_CONTACT_TOL_PCT).
- Still missing: cam_ref (camera centring) for the bishop: needs a re-teach.
- 2026-11-09: h1 pick aimed at the left corner of the square: +X 2 mm, +Y 2 mm (robot frame) -> shift_x -11.0, shift_y +3.7. grip_tip unchanged at 45.05.
