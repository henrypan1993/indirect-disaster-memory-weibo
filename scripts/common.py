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
PATH_ANALYSIS_ENTROPY = PROJECT_ROOT / "data" / "processed" / "analysis_ready_with_entropy.csv"
PATH_TOPIC_UNIQUE = PROJECT_ROOT / "data" / "processed" / "topic_entropy_unique_texts.csv"
PATH_TOPIC_EMBEDDINGS = PROJECT_ROOT / "data" / "processed" / "topic_embeddings_unique.npz"

EXPECTED_ROW_COUNT = 17_143

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

# Backward-compatible alias (descriptive tables, etc.).
CONTROL_COLS = list(FULL_SAMPLE_CONTROLS)

TRAUMA_NARRATIVES_CLEAN = frozenset({"Trauma-Help-Loss", "Memory-Reactivation"})

FitterName = Literal["ols", "glm"]
SampleKey = Literal["peripheral", "trauma_peripheral", "full_main", "full_indirect"]


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


def controls_formula(control_cols: tuple[str, ...] | None = None) -> str:
    cols = control_cols if control_cols is not None else FULL_SAMPLE_CONTROLS
    return " + ".join(cols)


def build_formula(spec: ModelSpec, *, y_col: str | None = None) -> str:
    y = y_col or spec.y_col
    rhs = " + ".join(spec.x_cols) + f" + {controls_formula(spec.control_cols)}"
    if spec.use_topic_fe:
        rhs += " + C(topic_id)"
    return f"{y} ~ {rhs}"


def apply_sample_mask(df: pd.DataFrame, sample_key: SampleKey) -> pd.DataFrame:
    if sample_key == "peripheral":
        return df.loc[df["model_sample_h2"]].copy()
    if sample_key == "trauma_peripheral":
        return df.loc[
            (df["include_main"] == 1)
            & (df["peripheral"] == 1)
            & (df["narrative_trauma_clean"] == 1)
        ].copy()
    if sample_key == "full_main":
        return df.loc[df["include_main"] == 1].copy()
    if sample_key == "full_indirect":
        return df.loc[df["model_sample_h1_indirect"]].copy()
    msg = f"unknown sample_key: {sample_key}"
    raise ValueError(msg)


def _main_specs_with_t2() -> list[ModelSpec]:
    return [
        ModelSpec(
            model_id="h1_engagement_indirect",
            module="H1",
            y_col="log_engagement",
            x_cols=("indirect_clean", "t2"),
            sample_key="peripheral",
            fitter="ols",
            focal_terms=("indirect_clean",),
            control_cols=PERIPHERAL_SAMPLE_CONTROLS,
            use_topic_fe=True,
            extra_report_terms=("t2",),
        ),
        ModelSpec(
            model_id="h2a_entropy_reactivation",
            module="H2a",
            y_col="entropy_norm",
            x_cols=("t2",),
            sample_key="trauma_peripheral",
            fitter="ols",
            focal_terms=("t2",),
            control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        ),
        ModelSpec(
            model_id="h2b_indirect_reactivation",
            module="H2b",
            y_col="indirect_clean",
            x_cols=("t2",),
            sample_key="trauma_peripheral",
            fitter="glm",
            focal_terms=("t2",),
            control_cols=PERIPHERAL_SAMPLE_CONTROLS,
        ),
        ModelSpec(
            model_id="e1a_entropy_peripheral",
            module="E1a",
            y_col="entropy_norm",
            x_cols=("peripheral", "t2"),
            sample_key="full_main",
            fitter="ols",
            focal_terms=("peripheral",),
            control_cols=FULL_SAMPLE_CONTROLS,
            extra_report_terms=("t2",),
        ),
        ModelSpec(
            model_id="e1b_indirect_peripheral",
            module="E1b",
            y_col="indirect_clean",
            x_cols=("peripheral", "t2"),
            sample_key="full_indirect",
            fitter="glm",
            focal_terms=("peripheral",),
            control_cols=FULL_SAMPLE_CONTROLS,
            extra_report_terms=("t2",),
        ),
        ModelSpec(
            model_id="e2_entropy_increment",
            module="E2",
            y_col="log_engagement",
            x_cols=("indirect_clean", "entropy_norm", "t2"),
            sample_key="peripheral",
            fitter="ols",
            focal_terms=("indirect_clean", "entropy_norm"),
            control_cols=PERIPHERAL_SAMPLE_CONTROLS,
            use_topic_fe=True,
            extra_report_terms=("t2",),
        ),
    ]


def main_model_specs() -> list[ModelSpec]:
    return _main_specs_with_t2()


def appendix_without_t2_specs() -> list[ModelSpec]:
    """Same model_ids as main, but formulas omit t2."""
    out: list[ModelSpec] = []
    for spec in _main_specs_with_t2():
        if spec.module.startswith("H2"):
            continue
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
            )
        )
    return out


def spec_by_model_id(model_id: str) -> ModelSpec:
    for spec in _main_specs_with_t2():
        if spec.model_id == model_id:
            return spec
    msg = f"unknown model_id: {model_id}"
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


def prepare_model_frame(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: list[str],
    control_cols: tuple[str, ...],
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    use = [y_col, *x_cols, *control_cols, "account_id"]
    if extra_cols:
        use.extend(extra_cols)
    use = list(dict.fromkeys(c for c in use if c in df.columns))
    sub = df[use].copy()
    sub = sub.dropna(subset=[y_col, "account_id"])
    numeric = [c for c in x_cols + list(control_cols) if c in sub.columns]
    for c in numeric:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=[c for c in numeric if c in sub.columns])
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
    ent = unique_entropy[cols].copy()
    return df.merge(ent, on="text_hash", how="left")


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
    sub = apply_sample_mask(df, spec.sample_key)
    extra_cols: list[str] | None = None
    if spec.use_topic_fe:
        sub["topic_id"] = sub["topic_id"].astype("Int64").astype(str)
        extra_cols = ["topic_id"]
    y = y_col or spec.y_col
    x_cols = list(spec.x_cols)
    sub = prepare_model_frame(
        sub,
        y_col=y,
        x_cols=x_cols,
        control_cols=spec.control_cols,
        extra_cols=extra_cols,
    )
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

    meta: dict[str, Any] = {
        "model_id": spec.model_id,
        "spec_id": spec_id,
        "module": spec.module,
        "formula": formula,
        "control_cols": list(spec.control_cols),
        "sample_key": spec.sample_key,
        "diagnostics": diagnostics,
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
    return rows, summary, meta
