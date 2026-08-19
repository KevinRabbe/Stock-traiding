from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .forward_outcomes import refresh_forward_outcome_scorecard


def _parse_as_of(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh market series for persisted forward decision candidates and "
            "materialize realized 5/20/60-session labels without trading authority."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/runtime"))
    parser.add_argument("--as-of")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = refresh_forward_outcome_scorecard(
        data_root=args.data_root,
        runtime_dir=args.runtime_dir,
        as_of=_parse_as_of(args.as_of),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
