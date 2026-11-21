"""One command, question to verdicts (T32): retrieve → generate → detect.

    python scripts/run_pipeline.py --lang hi --split test --sample 10 --embedder me5-small

Runs scripts/generate_answers.py's retrieval and generation (cached) inside one run folder,
then the detector (``configs/detection/<name>.yaml``) on every answer. Thresholds are the
val-tuned ones from ``results/m0/pilot.json`` when it exists (recorded in the run), else
the config's. Writes to ``results/runs/<run_id>/``: records.jsonl, retrieval.jsonl,
detected.jsonl and spotcheck.md (the T36 review sheet).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_answers as ga  # noqa: E402  (sibling script)

from indichallu.annotate.spotcheck import spotcheck_markdown  # noqa: E402
from indichallu.data.schema import write_jsonl  # noqa: E402
from indichallu.detection.aggregate import Thresholds  # noqa: E402
from indichallu.detection.detector import Detector, load_detection_config, load_nli  # noqa: E402
from indichallu.retrieval.embed import REPO_ROOT  # noqa: E402
from indichallu.runs import start_run  # noqa: E402

PILOT = REPO_ROOT / "results" / "m0" / "pilot.json"


def pilot_thresholds(path: Path | None = None) -> Thresholds | None:
    """The val-tuned thresholds of our detector from the pilot summary, if committed.

    ``PILOT`` is read at call time (not bound as a default), so it can be redirected."""
    path = PILOT if path is None else path
    if not path.exists():
        return None
    return Thresholds(**json.loads(path.read_text("utf-8"))["detection_test"]["ours"]["thresholds"])


def main(argv: list[str] | None = None) -> int:
    p = ga.build_parser(__doc__.splitlines()[0])
    p.add_argument("--detection", default="m0", help="configs/detection/<name>.yaml")
    p.set_defaults(name="pipeline")
    args = p.parse_args(argv)
    ga.check_args(p, args)
    det_cfg = load_detection_config(args.detection)

    with start_run(ga.run_config(args), name=args.name) as run:
        records = ga.generate(args, run)
        th = pilot_thresholds()
        source = (
            "results/m0/pilot.json (val-tuned)"
            if th
            else f"configs/detection/{args.detection}.yaml"
        )
        th = th or Thresholds(**det_cfg["thresholds"])
        n = det_cfg["nli"]
        run.add_model("nli", n["model_id"], revision=n["revision"])
        run.meta["thresholds"] = {**asdict(th), "source": source}
        run._save()
        detector = Detector.from_config(det_cfg, args.lang, nli=load_nli(det_cfg), thresholds=th)
        detected = [detector.detect(r) for r in records]
        write_jsonl(detected, run.path("detected.jsonl"))
        run.path("spotcheck.md").write_text(
            spotcheck_markdown(detected, f"T36 spot-check · run {run.id}"), "utf-8"
        )
        verdicts = [c.verdict.value for r in detected for c in r.claims]
        print(
            f"detected {len(detected)} answers, {len(verdicts)} claims "
            f"(thresholds from {source}); sheet: {run.path('spotcheck.md')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
