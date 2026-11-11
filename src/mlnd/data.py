from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from mlnd.config import ROOT, load_config


class DataValidationError(ValueError):
    """Raised when the raw data breaks the expected schema or value rules."""


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    path = Path(path) if path else ROOT / cfg["paths"]["raw_data"]
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    return pd.read_excel(path)


def validate(df: pd.DataFrame, cfg: dict | None = None) -> None:
    """Raise DataValidationError listing every problem found."""
    cfg = cfg or load_config()
    label, county = cfg["label"], cfg["county"]
    feats, counties = cfg["features"], cfg["counties"]
    expected = [label, county, *feats]
    problems: list[str] = []

    if len(df) == 0:
        raise DataValidationError("Data validation failed:\n - dataset is empty")

    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if missing:
        problems.append(f"missing columns: {missing}")
        raise DataValidationError("Data validation failed:\n - " + "\n - ".join(problems))

    nulls = df[expected].isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{col}: {n} missing values")

    bad_label = ~df[label].isin([0, 1])
    if bad_label.any():
        found = sorted(map(str, df.loc[bad_label, label].unique()))
        problems.append(f"{label}: {int(bad_label.sum())} values not in [0, 1] (found {found})")

    bad_county = ~df[county].isin(counties)
    if bad_county.any():
        found = sorted(map(str, df.loc[bad_county, county].unique()))
        problems.append(f"{county}: {int(bad_county.sum())} rows with unknown county {found}")

    for col in feats:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col}: not numeric (dtype {df[col].dtype})")
            continue
        n_out = int(((df[col] < 0) | (df[col] > 1)).sum())
        if n_out:
            problems.append(f"{col}: {n_out} values outside [0, 1]")

    if problems:
        raise DataValidationError("Data validation failed:\n - " + "\n - ".join(problems))


def clipping_report(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Per feature: compare the count exactly on 0/1 with the neighbouring strip.

    If readings were genuinely continuous, the count exactly on a bound would be
    similar to the count in a thin strip next to it. A large ratio means values
    beyond the bound were clipped onto it.
    """
    cfg = cfg or load_config()
    w = cfg["clipping"]["edge_width"]
    thr = cfg["clipping"]["ratio_threshold"]
    rows = []
    for col in cfg["features"]:
        x = df[col]
        at0, at1 = int((x == 0).sum()), int((x == 1).sum())
        near0 = int(((x > 0) & (x <= w)).sum())
        near1 = int(((x < 1) & (x >= 1 - w)).sum())
        r0, r1 = at0 / max(near0, 1), at1 / max(near1, 1)
        rows.append(
            {
                "feature": col,
                "at_0": at0,
                "near_0": near0,
                "ratio_0": round(r0, 1),
                "at_1": at1,
                "near_1": near1,
                "ratio_1": round(r1, 1),
                "clipped": bool(r0 >= thr or r1 >= thr),
            }
        )
    return pd.DataFrame(rows)


def audit(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    label, county, feats = cfg["label"], cfg["county"], cfg["features"]

    counts = df[label].value_counts().sort_index()
    by_county = pd.crosstab(df[county], df[label])
    by_county["total"] = by_county.sum(axis=1)
    by_county["pct_infected"] = (by_county[1] / by_county["total"] * 100).round(1)

    on_bound = ((df[feats] == 0) | (df[feats] == 1)).mean(axis=1)
    bound_share = on_bound.groupby(df[label]).mean().round(4)

    clip = clipping_report(df, cfg)
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "pct_infected": round(float(df[label].mean() * 100), 2),
        "imbalance_ratio": round(float(counts.max() / counts.min()), 2),
        "by_county": by_county.reset_index(),
        "feature_summary": df[feats]
        .agg(["mean", "std", "min", "max"])
        .T.round(3)
        .rename_axis("feature")
        .reset_index(),
        "clipping": clip,
        "n_features_clipped": int(clip["clipped"].sum()),
        "share_values_on_bound_by_class": {int(k): float(v) for k, v in bound_share.items()},
    }


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def write_report(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_feat = len(result["clipping"])
    if result["n_features_clipped"] > n_feat / 2:
        verdict = (
            f"{result['n_features_clipped']} of {n_feat} features show a pile-up of values exactly "
            "on 0 and/or 1 that is far larger than the density just inside the bound. That pattern "
            "is consistent with clipping of out-of-range readings, not with genuine continuous "
            "readings."
        )
    else:
        verdict = (
            "Mass on the bounds is not clearly larger than the density next to them; "
            "clipping is not indicated."
        )
    md = f"""# Data audit

- Rows: {result['n_rows']}  |  Columns: {result['n_columns']}  |  Duplicate rows: {result['duplicate_rows']}
- Class counts (0 healthy, 1 infected): {result['class_counts']}  |  Infected: {result['pct_infected']}%  |  Imbalance ratio: {result['imbalance_ratio']} : 1
- Validation: schema, label values, counties and [0, 1] ranges all passed.

## Records and infection rate by county
{_md_table(result['by_county'])}

## Feature summary
{_md_table(result['feature_summary'])}

## Clipping check (values exactly on 0 or 1)
`ratio` = count exactly on the bound / count in the neighbouring strip.

{_md_table(result['clipping'])}

**Verdict:** {verdict}

Share of values sitting exactly on 0 or 1, by class: {result['share_values_on_bound_by_class']}
"""
    md_path, js_path = out_dir / "data_audit.md", out_dir / "data_audit.json"
    md_path.write_text(md, encoding="utf-8")
    serial = {
        k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v)
        for k, v in result.items()
    }
    js_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    return md_path, js_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate and audit the raw MLND data.")
    ap.add_argument("--path", help="Excel file to audit (default: path in config)")
    args = ap.parse_args(argv)

    cfg = load_config()
    try:
        df = load_raw(args.path)
        validate(df, cfg)
    except (FileNotFoundError, DataValidationError) as e:
        print(f"AUDIT FAILED: {e}", file=sys.stderr)
        return 1

    result = audit(df, cfg)
    md_path, js_path = write_report(result, ROOT / cfg["paths"]["reports_dir"])
    print(
        f"Audit OK: {result['n_rows']} rows, {result['pct_infected']}% infected, "
        f"{result['n_features_clipped']}/{len(cfg['features'])} features show clipping."
    )
    print(f"Report written to {md_path} and {js_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())