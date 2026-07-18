"""Shared paths and helpers for pipeline scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATH_INPUT_CSV = PROJECT_ROOT / "data" / "input" / "labels_core_cleaned.csv"
PATH_ANALYSIS_BASE = PROJECT_ROOT / "data" / "processed" / "analysis_ready_base.csv"
PATH_ANALYSIS_TOPICS = PROJECT_ROOT / "data" / "processed" / "analysis_ready_with_topics.csv"
PATH_ANALYSIS_ENTROPY = PROJECT_ROOT / "data" / "processed" / "analysis_ready_with_entropy.csv"
PATH_MODEL_DATA_FINAL = PROJECT_ROOT / "data" / "processed" / "model_data_final.csv"
PATH_TOPIC_UNIQUE = PROJECT_ROOT / "data" / "processed" / "topic_entropy_unique_texts.csv"
PATH_TOPIC_EMBEDDINGS = PROJECT_ROOT / "data" / "processed" / "topic_embeddings_unique.npz"

EXPECTED_ROW_COUNT = 17_143
EXPECTED_INCLUDE_MAIN = 17_143

# Observation-window starts (Appendix A); relative_window_day is days since these dates.
WINDOW_START_T1 = pd.Timestamp("2021-07-18").date()
WINDOW_START_T2 = pd.Timestamp("2025-08-04").date()

REF_NARRATIVE = "Trauma-Help-Loss"
REF_EMOTION = "Neutral-Informational"
REF_TOPIC = "0"

LLM_LABEL_COLS = (
    "llm_label_narrative",
    "llm_label_emotion",
    "llm_label_expression",
)
CLEAN_LABEL_COLS = (
    "label_narrative_clean",
    "label_emotion_clean",
    "label_expression_clean",
)
REQUIRED_INPUT_COLS = (
    *LLM_LABEL_COLS,
    *CLEAN_LABEL_COLS,
    "mid",
    "text_hash",
    "include_main",
    "indirect_clean",
    "needs_manual_review",
    "high_clarity",
)

FULL_SAMPLE_CONTROLS = (
    "log_followers",
    "verified",
    "post_hour_cst",
    "hashtag_count",
    "text_length",
)

PERIPHERAL_SAMPLE_CONTROLS = (
    "log_followers",
    "post_hour_cst",
    "hashtag_count",
    "text_length",
)

# D1-M1: total association — no verified / log_followers (those define peripheral).
D1_M1_CONTROLS = (
    "post_hour_cst",
    "hashtag_count",
    "text_length",
)

# Backward-compatible alias (descriptive tables, etc.).
CONTROL_COLS = list(FULL_SAMPLE_CONTROLS)

TRAUMA_NARRATIVES_CLEAN = frozenset({"Trauma-Help-Loss", "Memory-Reactivation"})
DISASTER_IMPACT_NARRATIVES = TRAUMA_NARRATIVES_CLEAN

FitterName = Literal["ols", "glm"]
SampleKey = Literal[
    "h1",
    "h2",
    "d1",
    "peripheral",
    "trauma_peripheral",
    "full_main",
    "full_indirect",
]

LEGACY_MODEL_IDS = {
    "h2_indirect_period": "h2b_indirect_reactivation",
    "d1_indirect_peripheral": "e1b_indirect_peripheral",
}


class ClusterCovarianceError(RuntimeError):
    """Raised when account-clustered covariance cannot be computed."""


class ZeroVarianceDiagnosticError(ValueError):
    """Raised when the model frame contains zero-variance predictors."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    module: str
    y_col: str
    x_cols: tuple[str, ...]
    sample_key: SampleKey
    fitter: FitterName
    focal_terms: tuple[str, ...]
    control_cols: tuple[str, ...]
    use_topic_fe: bool = False
    extra_report_terms: tuple[str, ...] = ()
    formal_label: str = ""
    legacy_model_id: str = ""
    # Categorical columns entered as C(col, Treatment(reference=...))
    categorical_cols: tuple[tuple[str, str], ...] = ()
    # Extra numeric RHS terms (e.g. relative_window_day)
    extra_controls: tuple[str, ...] = ()
    default_spec_id: str = "main"


def controls_formula(control_cols: tuple[str, ...] | None = None) -> str:
    cols = control_cols if control_cols is not None else FULL_SAMPLE_CONTROLS
    return " + ".join(cols)


def _categorical_term(col: str, reference: str) -> str:
    return f"C({col}, Treatment(reference='{reference}'))"


def build_formula(spec: ModelSpec, *, y_col: str | None = None) -> str:
    y = y_col or spec.y_col
    parts: list[str] = list(spec.x_cols)
    if spec.control_cols:
        parts.append(controls_formula(spec.control_cols))
    parts.extend(spec.extra_controls)
    for col, ref in spec.categorical_cols:
        parts.append(_categorical_term(col, ref))
    if spec.use_topic_fe:
        parts.append(_categorical_term("topic_id", REF_TOPIC))
    rhs = " + ".join(p for p in parts if p)
    return f"{y} ~ {rhs}"


def apply_sample_mask(df: pd.DataFrame, sample_key: SampleKey) -> pd.DataFrame:
    """Prefer frozen eligible flags from model_data_final; fall back to legacy masks."""
    if sample_key == "h1":
        if "sample_h1_eligible" in df.columns:
            return df.loc[df["sample_h1_eligible"]].copy()
        return df.loc[df["model_sample_h2"]].copy()
    if sample_key == "h2":
        if "sample_h2_eligible" in df.columns:
            return df.loc[df["sample_h2_eligible"]].copy()
        impact = (
            df["disaster_impact_related"]
            if "disaster_impact_related" in df.columns
            else df["narrative_trauma_clean"]
        )
        return df.loc[(df["include_main"] == 1) & (df["peripheral"] == 1) & (impact == 1)].copy()
    if sample_key == "d1":
        if "sample_d1_eligible" in df.columns:
            return df.loc[df["sample_d1_eligible"]].copy()
        return df.loc[df["model_sample_h1_indirect"]].copy()
    # Legacy keys
    if sample_key == "peripheral":
        if "sample_h1_eligible" in df.columns:
            return df.loc[df["sample_h1_eligible"]].copy()
        return df.loc[df["model_sample_h2"]].copy()
    if sample_key == "trauma_peripheral":
        return apply_sample_mask(df, "h2")
    if sample_key == "full_main":
        return df.loc[df["include_main"] == 1].copy()
    if sample_key == "full_indirect":
        return apply_sample_mask(df, "d1")
    msg = f"unknown sample_key: {sample_key}"
    raise ValueError(msg)


def _h1_spec() -> ModelSpec:
    return ModelSpec(
        model_id="h1_engagement_indirect",
        module="H1",
        y_col="log_engagement",
        x_cols=("indirect_clean", "t2"),
        sample_key="h1",
        fitter="ols",
        focal_terms=("indirect_clean",),
        control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        use_topic_fe=True,
        extra_report_terms=("t2",),
        formal_label="H1",
        default_spec_id="main",
    )


def _h2_m1_spec() -> ModelSpec:
    return ModelSpec(
        model_id="h2_indirect_period",
        module="H2",
        y_col="indirect_clean",
        x_cols=("t2",),
        sample_key="h2",
        fitter="glm",
        focal_terms=("t2",),
        control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        use_topic_fe=True,
        formal_label="H2",
        legacy_model_id="h2b_indirect_reactivation",
        default_spec_id="h2_m1_period",
    )


def _h2_m2_spec() -> ModelSpec:
    return ModelSpec(
        model_id="h2_indirect_period",
        module="H2",
        y_col="indirect_clean",
        x_cols=("t2",),
        sample_key="h2",
        fitter="glm",
        focal_terms=("t2",),
        control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        use_topic_fe=True,
        categorical_cols=(
            ("label_narrative_clean", REF_NARRATIVE),
            ("label_emotion_clean", REF_EMOTION),
        ),
        extra_controls=("relative_window_day",),
        formal_label="H2",
        legacy_model_id="h2b_indirect_reactivation",
        default_spec_id="h2_m2_composition",
    )


def _d1_m1_spec() -> ModelSpec:
    return ModelSpec(
        model_id="d1_indirect_peripheral",
        module="D1",
        y_col="indirect_clean",
        x_cols=("peripheral", "t2"),
        sample_key="d1",
        fitter="glm",
        focal_terms=("peripheral",),
        control_cols=D1_M1_CONTROLS,
        use_topic_fe=True,
        extra_report_terms=("t2",),
        formal_label="D1",
        legacy_model_id="e1b_indirect_peripheral",
        default_spec_id="d1_m1_total_association",
    )


def _d1_m2_spec() -> ModelSpec:
    return ModelSpec(
        model_id="d1_indirect_peripheral",
        module="D1",
        y_col="indirect_clean",
        x_cols=("peripheral", "t2"),
        sample_key="d1",
        fitter="glm",
        focal_terms=("peripheral",),
        control_cols=FULL_SAMPLE_CONTROLS,
        use_topic_fe=True,
        extra_report_terms=("t2",),
        formal_label="D1",
        legacy_model_id="e1b_indirect_peripheral",
        default_spec_id="d1_m2_conditional_threshold",
    )


def main_model_specs() -> list[ModelSpec]:
    """Confirmatory / primary diagnostic specs: H1, H2-M1, D1-M1."""
    return [_h1_spec(), _h2_m1_spec(), _d1_m1_spec()]


def composition_or_conditional_specs() -> list[ModelSpec]:
    """H2-M2 composition adjustment and D1-M2 conditional threshold."""
    return [_h2_m2_spec(), _d1_m2_spec()]


def legacy_model_specs() -> list[ModelSpec]:
    """Excluded entropy / old-id specs (not run by default)."""
    return [
        ModelSpec(
            model_id="h2a_entropy_reactivation",
            module="legacy_H2a",
            y_col="entropy_norm",
            x_cols=("t2",),
            sample_key="h2",
            fitter="ols",
            focal_terms=("t2",),
            control_cols=PERIPHERAL_SAMPLE_CONTROLS,
            formal_label="",
            default_spec_id="legacy",
        ),
        ModelSpec(
            model_id="e2_entropy_increment",
            module="legacy_E2",
            y_col="log_engagement",
            x_cols=("indirect_clean", "entropy_norm", "t2"),
            sample_key="h1",
            fitter="ols",
            focal_terms=("indirect_clean", "entropy_norm"),
            control_cols=PERIPHERAL_SAMPLE_CONTROLS,
            use_topic_fe=True,
            extra_report_terms=("t2",),
            default_spec_id="legacy",
        ),
        ModelSpec(
            model_id="e1a_entropy_peripheral",
            module="legacy_E1a",
            y_col="entropy_norm",
            x_cols=("peripheral", "t2"),
            sample_key="full_main",
            fitter="ols",
            focal_terms=("peripheral",),
            control_cols=FULL_SAMPLE_CONTROLS,
            extra_report_terms=("t2",),
            default_spec_id="legacy",
        ),
        ModelSpec(
            model_id="h2b_indirect_reactivation",
            module="legacy_H2b",
            y_col="indirect_clean",
            x_cols=("t2",),
            sample_key="h2",
            fitter="glm",
            focal_terms=("t2",),
            control_cols=PERIPHERAL_SAMPLE_CONTROLS,
            legacy_model_id="h2b_indirect_reactivation",
            default_spec_id="legacy",
        ),
        ModelSpec(
            model_id="e1b_indirect_peripheral",
            module="legacy_E1b",
            y_col="indirect_clean",
            x_cols=("peripheral", "t2"),
            sample_key="d1",
            fitter="glm",
            focal_terms=("peripheral",),
            control_cols=FULL_SAMPLE_CONTROLS,
            extra_report_terms=("t2",),
            legacy_model_id="e1b_indirect_peripheral",
            default_spec_id="legacy",
        ),
    ]


def all_known_specs() -> list[ModelSpec]:
    return [
        *main_model_specs(),
        *composition_or_conditional_specs(),
        *legacy_model_specs(),
    ]


def appendix_without_t2_specs() -> list[ModelSpec]:
    """H1 / D1 variants omitting t2 (legacy appendix path)."""
    out: list[ModelSpec] = []
    for spec in (_h1_spec(), _d1_m1_spec(), _d1_m2_spec()):
        x_no_t2 = tuple(c for c in spec.x_cols if c != "t2")
        extra = tuple(t for t in spec.extra_report_terms if t != "t2")
        out.append(
            ModelSpec(
                model_id=spec.model_id,
                module=spec.module,
                y_col=spec.y_col,
                x_cols=x_no_t2,
                sample_key=spec.sample_key,
                fitter=spec.fitter,
                focal_terms=spec.focal_terms,
                control_cols=spec.control_cols,
                use_topic_fe=spec.use_topic_fe,
                extra_report_terms=extra,
                formal_label=spec.formal_label,
                legacy_model_id=spec.legacy_model_id,
                categorical_cols=spec.categorical_cols,
                extra_controls=spec.extra_controls,
                default_spec_id=f"{spec.default_spec_id}_without_t2",
            )
        )
    return out


def spec_by_model_id(model_id: str, *, prefer_main: bool = True) -> ModelSpec:
    """Resolve a ModelSpec by model_id (main/composition first, then legacy)."""
    # Old ids resolve to current primary models before matching legacy stubs.
    for new_id, old_id in LEGACY_MODEL_IDS.items():
        if model_id == old_id:
            return spec_by_model_id(new_id, prefer_main=prefer_main)
    pools = (
        [*main_model_specs(), *composition_or_conditional_specs(), *legacy_model_specs()]
        if prefer_main
        else [*legacy_model_specs(), *main_model_specs(), *composition_or_conditional_specs()]
    )
    # Prefer default_spec_id that is the primary for that id
    primary = {
        "h1_engagement_indirect": "main",
        "h2_indirect_period": "h2_m1_period",
        "d1_indirect_peripheral": "d1_m1_total_association",
    }
    want = primary.get(model_id)
    if want:
        for spec in pools:
            if spec.model_id == model_id and spec.default_spec_id == want:
                return spec
    for spec in pools:
        if spec.model_id == model_id:
            return spec
    msg = f"unknown model_id: {model_id}"
    raise KeyError(msg)


def spec_by_spec_id(spec_id: str) -> ModelSpec:
    if spec_id == "main":
        return _h1_spec()
    for spec in all_known_specs():
        if spec.default_spec_id == spec_id:
            return spec
    msg = f"unknown spec_id: {spec_id}"
    raise KeyError(msg)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_input_csv(path: Path | None = None) -> pd.DataFrame:
    p = path or PATH_INPUT_CSV
    return pd.read_csv(p, dtype={"mid": str})


def analysis_text_series(df: pd.DataFrame) -> pd.Series:
    if "analysis_text_final" in df.columns:
        s = df["analysis_text_final"]
        if "text" in df.columns:
            return s.fillna(df["text"])
        return s
    if "text" in df.columns:
        return df["text"]
    msg = "missing analysis_text_final and text"
    raise KeyError(msg)


def count_hashtags(text: object) -> int:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0
    return len(re.findall(r"#([^#]+)#", str(text)))


def narrative_trauma_clean(series: pd.Series) -> pd.Series:
    return series.isin(TRAUMA_NARRATIVES_CLEAN).astype(int)


def disaster_impact_related(series: pd.Series) -> pd.Series:
    """Alias for disaster-impact / trauma-help narratives used in H2 sample."""
    return narrative_trauma_clean(series)


def relative_window_day_series(
    post_dates: pd.Series,
    t2: pd.Series,
) -> pd.Series:
    """Days since wave observation-window start (0-based); NA if date missing."""
    dates = pd.to_datetime(post_dates, errors="coerce")
    # Normalize to date
    if hasattr(dates.dt, "tz_localize"):
        try:
            dates = dates.dt.tz_convert(None)
        except (TypeError, AttributeError):
            pass
    day = dates.dt.date
    t2_num = pd.to_numeric(t2, errors="coerce").fillna(0).astype(int)
    out = []
    for d, is_t2 in zip(day, t2_num, strict=False):
        if d is None or (isinstance(d, float) and np.isnan(d)) or pd.isna(d):
            out.append(np.nan)
            continue
        start = WINDOW_START_T2 if int(is_t2) == 1 else WINDOW_START_T1
        out.append(float((d - start).days))
    return pd.Series(out, index=post_dates.index, dtype="float64")


def prepare_model_frame(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: list[str],
    control_cols: tuple[str, ...],
    extra_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    use = [y_col, *x_cols, *control_cols, "account_id"]
    if extra_cols:
        use.extend(extra_cols)
    if categorical_cols:
        use.extend(categorical_cols)
    use = list(dict.fromkeys(c for c in use if c in df.columns))
    sub = df[use].copy()
    sub = sub.dropna(subset=[y_col, "account_id"])
    numeric = [c for c in [*x_cols, *control_cols, *(extra_cols or [])] if c in sub.columns]
    # topic_id / labels are categorical — do not coerce to float
    cat_set = set(categorical_cols or [])
    numeric = [c for c in numeric if c not in cat_set and c != "topic_id"]
    for c in numeric:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    drop_subset = [c for c in numeric if c in sub.columns]
    if categorical_cols:
        drop_subset.extend(c for c in categorical_cols if c in sub.columns)
    sub = sub.dropna(subset=list(dict.fromkeys(drop_subset)))
    return sub


def _zero_variance_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out: list[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            out.append(c)
    return out


def _quasi_separation_hints(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> list[str]:
    hints: list[str] = []
    y = pd.to_numeric(df[y_col], errors="coerce")
    for c in x_cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.nunique(dropna=True) > 2:
            continue
        for val in x.dropna().unique():
            mask = x == val
            if int(mask.sum()) == 0:
                continue
            if y[mask].nunique(dropna=True) <= 1:
                hints.append(f"{c}={val} -> {y_col} constant")
    return hints


def diagnose_model_frame(
    df: pd.DataFrame,
    formula: str,
    *,
    fitter: FitterName,
    y_col: str,
    x_cols: list[str],
    control_cols: tuple[str, ...],
    cluster_col: str = "account_id",
) -> dict[str, Any]:
    import patsy

    check_cols = list(dict.fromkeys([*x_cols, *control_cols]))
    zero_var = _zero_variance_cols(df, check_cols)
    n_obs = len(df)
    n_accounts = int(df[cluster_col].nunique()) if cluster_col in df.columns else 0

    y_design, x_design = patsy.dmatrices(formula, df, return_type="dataframe")
    x_np = x_design.to_numpy()
    rank = int(np.linalg.matrix_rank(x_np))
    n_params = int(x_design.shape[1])
    try:
        cond = float(np.linalg.cond(x_np))
    except (np.linalg.LinAlgError, FloatingPointError):
        cond = float("inf")

    quasi = _quasi_separation_hints(df, y_col, x_cols) if fitter == "glm" else []

    return {
        "zero_variance_cols": zero_var,
        "n_obs": n_obs,
        "n_accounts": n_accounts,
        "design_matrix_rank": rank,
        "n_params": n_params,
        "rank_deficient": rank < n_params,
        "condition_number": cond,
        "covariance_estimator_expected": "cluster",
        "quasi_separation_hint": quasi or None,
    }


def format_diagnostic_header(diag: dict[str, Any], *, formula: str) -> str:
    lines = [
        "=== Pre-fit diagnostics ===",
        f"formula: {formula}",
        f"n_obs={diag['n_obs']}, n_accounts={diag['n_accounts']}",
        f"design_matrix_rank={diag['design_matrix_rank']}, n_params={diag['n_params']}",
        f"condition_number={diag['condition_number']:.4g}",
        f"zero_variance_cols={diag['zero_variance_cols'] or 'none'}",
        f"rank_deficient={diag['rank_deficient']}",
    ]
    if diag.get("quasi_separation_hint"):
        lines.append(f"quasi_separation_hint={diag['quasi_separation_hint']}")
    lines.append("=== Model summary ===")
    return "\n".join(lines) + "\n\n"


def _assert_cluster_covariance(fit: Any, *, formula: str, n: int, n_accounts: int) -> None:
    cov = getattr(fit, "cov_type", None)
    if cov == "cluster":
        return
    msg = (
        f"Expected cluster covariance but got {cov!r}. "
        f"formula={formula!r}, n={n}, n_accounts={n_accounts}"
    )
    raise ClusterCovarianceError(msg)


def fit_ols_cluster(df: pd.DataFrame, formula: str) -> tuple[Any, int]:
    n = len(df)
    n_accounts = int(df["account_id"].nunique())
    model = sm.OLS.from_formula(formula, data=df)
    try:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df["account_id"]})
    except (np.linalg.LinAlgError, ValueError) as exc:
        msg = (
            f"OLS cluster covariance failed: {exc}. "
            f"formula={formula!r}, n={n}, n_accounts={n_accounts}"
        )
        raise ClusterCovarianceError(msg) from exc
    _assert_cluster_covariance(fit, formula=formula, n=n, n_accounts=n_accounts)
    return fit, n


def fit_glm_binomial_cluster(df: pd.DataFrame, formula: str) -> tuple[Any, int]:
    n = len(df)
    n_accounts = int(df["account_id"].nunique())
    model = sm.GLM.from_formula(formula, data=df, family=Binomial())
    try:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df["account_id"]})
    except (np.linalg.LinAlgError, ValueError) as exc:
        msg = (
            f"GLM cluster covariance failed: {exc}. "
            f"formula={formula!r}, n={n}, n_accounts={n_accounts}"
        )
        raise ClusterCovarianceError(msg) from exc
    _assert_cluster_covariance(fit, formula=formula, n=n, n_accounts=n_accounts)
    return fit, n


DEFAULT_EMBED_MODEL = "shibing624/text2vec-base-chinese"
DEFAULT_K = 10
DEFAULT_TAU = 0.10


def softmax_rows(sim: np.ndarray, tau: float) -> np.ndarray:
    scaled = sim / tau
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy_norm_from_probs(probs: np.ndarray, k: int) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    return ent / np.log(k)


def compute_topic_entropy(
    texts: list[str],
    *,
    k: int = DEFAULT_K,
    tau: float = DEFAULT_TAU,
    random_state: int = 42,
    embeddings: np.ndarray | None = None,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if embeddings is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.asarray(emb, dtype=np.float64)
    else:
        embeddings = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(embeddings)
    centers = km.cluster_centers_
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    center_norms = np.where(center_norms == 0, 1.0, center_norms)
    centers = centers / center_norms

    sim = embeddings @ centers.T
    probs = softmax_rows(sim, tau)
    ent = entropy_norm_from_probs(probs, k)
    topic_id = probs.argmax(axis=1)

    result = pd.DataFrame(
        {
            "topic_id": topic_id.astype(int),
            "entropy_norm": ent,
            "k": k,
            "tau": tau,
        }
    )
    meta = {
        "n_texts": len(texts),
        "k": k,
        "tau": tau,
        "entropy_min": float(ent.min()),
        "entropy_max": float(ent.max()),
        "entropy_mean": float(ent.mean()),
        "topic_counts": {str(i): int((topic_id == i).sum()) for i in range(k)},
    }
    return result, meta


def build_unique_corpus(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.loc[df["include_main"] == 1].copy()
    text = analysis_text_series(sub)
    sub["_text"] = text
    return (
        sub.groupby("text_hash", as_index=False)
        .first()[["text_hash", "_text"]]
        .rename(columns={"_text": "text"})
    )


def ensure_embedding_cache(
    base_df: pd.DataFrame,
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    cache_path: Path = PATH_TOPIC_EMBEDDINGS,
) -> np.ndarray:
    """Build or load unique-text embeddings (for entropy robustness specs)."""
    uniq = build_unique_corpus(base_df)
    cached = load_embedding_cache_arrays(uniq, cache_path=cache_path)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    texts = uniq["text"].fillna("").astype(str).tolist()
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(emb, dtype=np.float64)
    cache_path = cache_path.expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        text_hash=uniq["text_hash"].to_numpy(),
        embeddings=embeddings,
        model=model_name,
    )
    return embeddings


def load_embedding_cache_arrays(
    uniq: pd.DataFrame,
    *,
    cache_path: Path = PATH_TOPIC_EMBEDDINGS,
) -> np.ndarray | None:
    cache = cache_path.expanduser().resolve()
    if not cache.is_file():
        return None
    data = np.load(cache, allow_pickle=True)
    hashes = data["text_hash"]
    emb = data["embeddings"]
    order = {h: i for i, h in enumerate(hashes)}
    try:
        idx = [order[h] for h in uniq["text_hash"]]
    except KeyError:
        return None
    return emb[idx]


def merge_entropy_to_posts(df: pd.DataFrame, unique_entropy: pd.DataFrame) -> pd.DataFrame:
    cols = ["text_hash", "topic_id", "entropy_norm"]
    ent = unique_entropy[[c for c in cols if c in unique_entropy.columns]].copy()
    return df.merge(ent, on="text_hash", how="left")


def merge_topics_to_posts(df: pd.DataFrame, unique_topics: pd.DataFrame) -> pd.DataFrame:
    """Merge topic_id only (formal modeling path; no entropy_norm)."""
    cols = ["text_hash", "topic_id"]
    top = unique_topics[cols].copy()
    return df.merge(top, on="text_hash", how="left")


def topics_frame_from_entropy_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop entropy_norm from a with_entropy posts table for the formal topics path."""
    out = df.copy()
    if "entropy_norm" in out.columns:
        out = out.drop(columns=["entropy_norm"])
    return out


def coef_row(
    model_id: str,
    fit: Any,
    term: str,
    n: int,
    spec_id: str = "main",
    *,
    module: str = "",
    notes: str = "",
) -> dict[str, Any]:
    base = {
        "model_id": model_id,
        "module": module,
        "spec_id": spec_id,
        "term": term,
        "notes": notes,
        "n": n,
    }
    if term not in fit.params.index:
        return {**base, "coef": np.nan, "se": np.nan, "pvalue": np.nan}
    return {
        **base,
        "coef": float(fit.params[term]),
        "se": float(fit.bse[term]),
        "pvalue": float(fit.pvalues[term]),
    }


def coef_rows_from_fit(
    spec: ModelSpec,
    fit: Any,
    n: int,
    spec_id: str,
    *,
    notes: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for term in spec.focal_terms:
        rows.append(
            coef_row(
                spec.model_id,
                fit,
                term,
                n,
                spec_id,
                module=spec.module,
                notes=notes,
            )
        )
    for term in spec.extra_report_terms:
        rows.append(
            coef_row(
                spec.model_id,
                fit,
                term,
                n,
                spec_id,
                module=spec.module,
                notes=(notes + "; control term").strip("; "),
            )
        )
    return rows


def run_model_spec(
    df: pd.DataFrame,
    spec: ModelSpec,
    spec_id: str,
    *,
    y_col: str | None = None,
    notes: str = "",
    meta_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    eligible = apply_sample_mask(df, spec.sample_key)
    n_eligible = len(eligible)
    sub = eligible
    cat_cols = [c for c, _ in spec.categorical_cols]
    extra_cols: list[str] = list(spec.extra_controls)
    if spec.use_topic_fe:
        sub = sub.copy()
        sub["topic_id"] = sub["topic_id"].astype("Int64").astype(str)
        cat_cols = [*cat_cols, "topic_id"]
    y = y_col or spec.y_col
    x_cols = list(spec.x_cols)
    sub = prepare_model_frame(
        sub,
        y_col=y,
        x_cols=x_cols,
        control_cols=spec.control_cols,
        extra_cols=extra_cols or None,
        categorical_cols=cat_cols or None,
    )
    n_estimation = len(sub)
    formula = build_formula(spec, y_col=y)
    diagnostics = diagnose_model_frame(
        sub,
        formula,
        fitter=spec.fitter,
        y_col=y,
        x_cols=x_cols,
        control_cols=spec.control_cols,
    )
    if diagnostics["zero_variance_cols"]:
        msg = (
            f"Zero-variance predictors in {spec.model_id} ({spec_id}): "
            f"{diagnostics['zero_variance_cols']}. formula={formula!r}"
        )
        raise ZeroVarianceDiagnosticError(msg)

    if spec.fitter == "glm":
        fit, n = fit_glm_binomial_cluster(sub, formula)
    else:
        fit, n = fit_ols_cluster(sub, formula)

    # Contigency helpers for H2
    contig: dict[str, Any] = {}
    if "t2" in sub.columns and "topic_id" in sub.columns:
        ct = pd.crosstab(sub["topic_id"], sub["t2"])
        contig["topic_x_t2"] = ct.to_dict()
        topics = set(sub["topic_id"].unique())
        t1_only = []
        t2_only = []
        both = []
        for tid in topics:
            row = ct.loc[tid] if tid in ct.index else None
            if row is None:
                continue
            n0 = int(row.get(0, row.get(0.0, 0)) if hasattr(row, "get") else 0)
            n1 = int(row.get(1, row.get(1.0, 0)) if hasattr(row, "get") else 0)
            # crosstab columns may be int
            try:
                n0 = int(ct.loc[tid, 0]) if 0 in ct.columns else 0
            except (KeyError, TypeError):
                n0 = 0
            try:
                n1 = int(ct.loc[tid, 1]) if 1 in ct.columns else 0
            except (KeyError, TypeError):
                n1 = 0
            if n0 > 0 and n1 > 0:
                both.append(tid)
            elif n0 > 0:
                t1_only.append(tid)
            elif n1 > 0:
                t2_only.append(tid)
        contig["topics_t1_only"] = t1_only
        contig["topics_t2_only"] = t2_only
        contig["topics_both_periods"] = both
        contig["topic_min_cell"] = int(ct.to_numpy().min()) if ct.size else None
    if "label_narrative_clean" in sub.columns and "t2" in sub.columns:
        contig["narrative_x_t2"] = pd.crosstab(sub["label_narrative_clean"], sub["t2"]).to_dict()
    if "label_emotion_clean" in sub.columns and "t2" in sub.columns:
        contig["emotion_x_t2"] = pd.crosstab(sub["label_emotion_clean"], sub["t2"]).to_dict()

    # Dropped columns from patsy (singular)
    dropped = []
    try:
        import patsy

        _, x_design = patsy.dmatrices(formula, sub, return_type="dataframe")
        # Compare to fit.params
        design_cols = list(x_design.columns)
        fit_cols = list(fit.params.index)
        dropped = [c for c in design_cols if c not in fit_cols]
    except Exception:  # noqa: BLE001
        dropped = []

    n_t1 = int((pd.to_numeric(sub["t2"], errors="coerce") == 0).sum()) if "t2" in sub else None
    n_t2 = int((pd.to_numeric(sub["t2"], errors="coerce") == 1).sum()) if "t2" in sub else None
    n_direct = (
        int((pd.to_numeric(sub[y], errors="coerce") == 0).sum()) if spec.fitter == "glm" else None
    )
    n_indirect = (
        int((pd.to_numeric(sub[y], errors="coerce") == 1).sum()) if spec.fitter == "glm" else None
    )

    meta: dict[str, Any] = {
        "model_id": spec.model_id,
        "legacy_model_id": spec.legacy_model_id or None,
        "formal_label": spec.formal_label or None,
        "spec_id": spec_id,
        "module": spec.module,
        "formula": formula,
        "control_cols": list(spec.control_cols),
        "extra_controls": list(spec.extra_controls),
        "categorical_cols": [{"col": c, "reference": r} for c, r in spec.categorical_cols],
        "topic_fe_reference": REF_TOPIC if spec.use_topic_fe else None,
        "sample_key": spec.sample_key,
        "n_eligible": n_eligible,
        "n_estimation": n_estimation,
        "n_dropped_complete_case": int(n_eligible - n_estimation),
        "n_t1": n_t1,
        "n_t2": n_t2,
        "n_direct": n_direct,
        "n_indirect": n_indirect,
        "diagnostics": diagnostics,
        "dropped_design_columns": dropped,
        "contingency": contig or None,
        "covariance_estimator": getattr(fit, "cov_type", None),
        "built_at": utc_now_iso(),
        "notes": notes,
    }
    if meta_dir is not None:
        meta_path = meta_dir / f"{spec.model_id}_{spec_id}_meta.json"
        write_json_report(meta_path, meta)

    header = format_diagnostic_header(diagnostics, formula=formula)
    summary = header + fit.summary().as_text()
    rows = coef_rows_from_fit(spec, fit, n, spec_id, notes=notes)
    for row in rows:
        row["formal_label"] = spec.formal_label
        row["legacy_model_id"] = spec.legacy_model_id
    return rows, summary, meta
