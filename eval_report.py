"""Print the tables from one or more finished eval runs, without re-running anything.

    source .venv/bin/activate
    python eval_report.py v3_45k                          # a name under outputs/eval/
    python eval_report.py v3_30k v3_45k                   # two runs, with a side-by-side comparison
    python eval_report.py v3_45k --failures               # list every failed trial
"""

import argparse
from pathlib import Path

from axibo.report import load_run, print_comparison, print_run


def _f(v) -> str:
    """Format a possibly-missing numeric cell (release columns are blank when there was no release)."""
    return "-" if v in (None, "") else f"{float(v):.1f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+",
                   help="run names under outputs/eval/, or explicit dirs / eval_log.csv paths")
    p.add_argument("--failures", action="store_true", help="also list every non-success trial with its metrics")
    args = p.parse_args()

    loaded = []
    for arg in args.runs:
        path = Path(arg)
        if not path.exists():  # a bare name refers to outputs/eval/<name>
            path = Path("outputs/eval") / arg
        if not path.exists():
            raise SystemExit(f"no such run: {arg} (looked for {arg} and outputs/eval/{arg})")
        rows, meta = load_run(path)
        name = path.name if path.suffix != ".csv" else path.parent.name
        loaded.append((name, rows, meta))
        print_run(rows, title=name, meta=meta)

        if args.failures:
            bad = [r for r in rows if r["outcome"] != "success"]
            # The trial index is what replay_trace.py --trial takes, and the release geometry is what the label
            # was decided from, so the two are printed together: that is what makes a label checkable by eye.
            print(f"\n  {len(bad)} failed trials  (replay: python replay_trace.py {name} --trial N)")
            for r in bad:
                rel, ctr = r.get("release_step") or "-", r.get("centered_at_release", "")
                print(f"    trial {r['trial']:<4} L{r['layout_id']:<3} {r['source']:13}-> {r['destination']:13} "
                      f"{r['outcome']:22} rel@{rel:<4} "
                      f"dz {_f(r.get('release_dz_mm')):>7} xy {_f(r.get('release_xy_offset_mm')):>6} "
                      f"ctr {(ctr or '-')[:1]:1} | final xy {r['final_xy_offset_mm']:6.1f} "
                      f"dz {r['final_dz_error_mm']:+7.1f} grip {r['final_gripper_mm']:5.1f} | "
                      f"lift@{r['lifted_step'] or '-'} transp@{r['transported_step'] or '-'}")

    if len(loaded) > 1:
        print_comparison([(n, rows) for n, rows, _ in loaded])


if __name__ == "__main__":
    main()
