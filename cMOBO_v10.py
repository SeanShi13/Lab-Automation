# -*- coding: utf-8 -*-
"""
Merged v9: three-panel figure set for constrained MOBO.

  (a) 3D design space projections        (from v8 Fig_c)
  (b) Ms–Yield objective space           (from v8 Fig_b)
  (c) 3D surrogate landscape + learning trajectory

Variables (x):
  1) Fe3+|Fe2+
  2) FeCl3·6H2O (g)   (if missing, computed from Fe3+ (mol))
  3) Surfactant|Fe3+

Objectives (maximize):
  - Ms  (prefer "Ms (emu/g)" if present, else "Ms (%)")
  - Yield (%)

Constraint / feasibility:
  - Surface Instability (0|1)  -> interpret via FEASIBLE_VALUE

Models and recommendation:
  - LogisticRegression for P(feasible|x), with a constant-probability fallback
  - Two independent SingleTaskGP models for standardized Ms and Yield
  - ModelListGP trained with SumMarginalLogLikelihood
  - qNEHVI evaluated independently for each LHS candidate (q = 1)
  - Candidate score: qNEHVI(x) * P(feasible|x)^gamma
  - The TOP_K highest-scoring individual candidates form the next batch

Visualization:
  - The final surrogate landscape uses all valid observations
  - The third design coordinate is interpolated using a thin-plate-spline RBF
  - The displayed landscape is not the qNEHVI acquisition function
"""

import os
import inspect
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from sklearn.linear_model import LogisticRegression


# ===================== USER SETTINGS =====================
FILE_NAME = r"xxx"
OUT_DIR = r".\bo_out_10-1"
os.makedirs(OUT_DIR, exist_ok=True)

FEASIBLE_VALUE = 1  # 0 or 1 (match your dataset meaning)

SEED = 42
TOP_K = 5
N_CANDIDATES = 8000
FEAS_WEIGHT_POWER = 1.0

# Reproducibility settings used by the objective models and qNEHVI
OBSERVATION_VARIANCE = 1e-4
N_QMC_SAMPLES = 128
REF_POINT_MARGIN = 0.2
FEAS_PROB_FLOOR = 1e-6

# Landscape visualization settings
N_GRID = 140
PCT_LOW, PCT_HIGH = 40, 100
RBF_SMOOTHING = 0.035

# ===================== COLUMN NAMES =====================
COL_RATIO_FE3_FE2 = "Fe3+|Fe2+"
COL_FECL3_G = "FeCl3·6H2O (g)"
COL_FE3_MOL = "Fe3+ (mol)"
COL_RATIO_SURF_FE3 = "Surfactant|Fe3+"

COL_FEAS = "Surface Instability (0|1)"
COL_YIELD = "Yield (%)"
COL_MS_EMUG = "Ms (emu/g)"
COL_MS_PERCENT = "Ms (%)"

NOTE_KEYS = {"1stBO": 1, "2ndBO": 2, "3rdBO": 3, "4thBO": 4, "5thBO": 5}

MOLAR_MASS_FECL3_6H2O = 270.3
MOLAR_MASS_FESO4_7H2O = 278.02
MOLAR_MASS_FE3O4 = 231.533

EXPORT_DPI = 1200
EXPORT_PNG = True
EXPORT_SVG = True

# SVG: keep text as text elements (not paths) for Illustrator
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---------- Shared visual style for (b) and (c) ----------
FONT_FAMILY = "Arial"
FONT_WEIGHT = "bold"
FONTSIZE_TICK = 15
FONTSIZE_LABEL = 15
FONTSIZE_LEGEND = 15
FONTSIZE_NUMBER = 13      # numbers inside data points
FONTSIZE_REGION = 12       # region / annotation text
FONTSIZE_COLORBAR = 14

SPINE_LW = 2.0             # frame linewidth
SCATTER_EDGE_LW = 1.5      # scatter point edge linewidth
SCATTER_SIZE = 190          # unified scatter point size across all figures
LEGEND_MARKER_SIZE = 10    # legend handle marker size
LEGEND_MARKER_EDGE_LW = 1.5
# =========================================================


# ===================== UTILITIES =====================
def standardize_1d(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    mu = np.nanmean(y)
    sd = np.nanstd(y)
    if (not np.isfinite(sd)) or sd < 1e-12:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def lhs(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return X


def theoretical_magnetite_g(fe3_mol, fe2_mol, ratio_fe3_fe2):
    if ratio_fe3_fe2 >= 2.0:
        m_mol = fe2_mol
    else:
        init_m = fe3_mol / 2.0
        fe2_rem = fe2_mol - (fe3_mol / 2.0)
        if fe2_rem > 0:
            fe2_ox = fe2_rem * (2.0 / 3.0)
            fe2_new = fe2_rem * (1.0 / 3.0)
            extra_m = fe2_new if fe2_ox >= 2.0 * fe2_new else fe2_ox / 2.0
            m_mol = init_m + extra_m
        else:
            m_mol = init_m
    return m_mol * MOLAR_MASS_FE3O4


def find_note_column(df: pd.DataFrame):
    for cc in df.columns:
        if str(cc).strip().lower() == "note":
            return cc
    for cc in df.columns:
        if "note" in str(cc).strip().lower():
            return cc
    return None


def import_botorch():
    import gpytorch
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.fit import fit_gpytorch_mll

    try:
        from botorch.acquisition.multi_objective.monte_carlo import (
            qNoisyExpectedHypervolumeImprovement as qNEHVI
        )
    except ImportError as exc:
        raise ImportError(
            "This script requires BoTorch with "
            "qNoisyExpectedHypervolumeImprovement. "
            "Install a compatible BoTorch release before running it."
        ) from exc

    return dict(
        gpytorch=gpytorch,
        SingleTaskGP=SingleTaskGP,
        ModelListGP=ModelListGP,
        qNEHVI=qNEHVI,
        SobolQMCNormalSampler=SobolQMCNormalSampler,
        fit_gpytorch_mll=fit_gpytorch_mll,
    )


def build_qnehvi_version_robust(qNEHVI_cls, *, model, ref_point, train_X, sampler):
    sig = inspect.signature(qNEHVI_cls.__init__)
    allowed = set(sig.parameters.keys())
    kw = {
        "model": model, "ref_point": ref_point, "sampler": sampler,
        "X_baseline": train_X, "X_observed": train_X,
        "prune_baseline": True, "cache_root": True,
    }
    kw = {k: v for k, v in kw.items() if k in allowed}
    return qNEHVI_cls(**kw)


def _save_fig(fig, path_base):
    if EXPORT_PNG:
        fig.savefig(path_base + ".png", dpi=EXPORT_DPI, bbox_inches="tight")
    if EXPORT_SVG:
        fig.savefig(path_base + ".svg", bbox_inches="tight")


# ===================== MAIN =====================
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------- Load ----------
    df = pd.read_excel(FILE_NAME)

    if COL_MS_EMUG in df.columns:
        col_ms = COL_MS_EMUG
    elif COL_MS_PERCENT in df.columns:
        col_ms = COL_MS_PERCENT
    else:
        raise RuntimeError(f"Missing Ms column. Expected '{COL_MS_EMUG}' or '{COL_MS_PERCENT}'.")

    if COL_FECL3_G not in df.columns:
        if COL_FE3_MOL in df.columns:
            df[COL_FECL3_G] = df[COL_FE3_MOL].astype(float) * MOLAR_MASS_FECL3_6H2O
        else:
            raise RuntimeError(f"Missing '{COL_FECL3_G}'. Provide it or '{COL_FE3_MOL}'.")

    required = [COL_RATIO_FE3_FE2, COL_FECL3_G, COL_RATIO_SURF_FE3, col_ms, COL_YIELD, COL_FEAS]
    dfv = df.dropna(subset=required).copy()
    if len(dfv) < 6:
        raise RuntimeError(f"Too few valid rows: {len(dfv)}")

    # batch column
    COL_BATCH = "batch"
    if COL_BATCH not in dfv.columns:
        for cc in dfv.columns:
            if str(cc).strip().lower() == "batch":
                COL_BATCH = cc
                break
        else:
            raise RuntimeError("Missing 'batch' column.")
    batch = dfv[COL_BATCH].astype(int).to_numpy()
    init_mask = batch <= 11
    ref_mask = batch >= 12

    # ---------- X raw -> normalized ----------
    X_raw = dfv[[COL_RATIO_FE3_FE2, COL_FECL3_G, COL_RATIO_SURF_FE3]].to_numpy(dtype=float)
    X_raw += np.random.normal(0, 1e-12, size=X_raw.shape)
    x_min = X_raw.min(axis=0)
    x_max = X_raw.max(axis=0)
    x_rng = np.where((x_max - x_min) > 0, (x_max - x_min), 1.0)
    X = (X_raw - x_min) / x_rng

    # ---------- y ----------
    y_ms = dfv[col_ms].to_numpy(dtype=float)
    y_yield = pd.to_numeric(dfv[COL_YIELD], errors="coerce").to_numpy(dtype=float)
    yy_max = float(np.nanmax(y_yield)) if np.isfinite(np.nanmax(y_yield)) else np.nan
    if np.isfinite(yy_max) and yy_max <= 1.5:
        y_yield = y_yield * 100.0
    y_ms_std, ms_mu, ms_sd = standardize_1d(y_ms)
    y_y_std, y_mu, y_sd = standardize_1d(y_yield)

    # ---------- constraint ----------
    c_raw = dfv[COL_FEAS].astype(int).to_numpy()
    if set(np.unique(c_raw)) - {0, 1}:
        raise RuntimeError(f"'{COL_FEAS}' must be 0/1.")
    c = (c_raw == int(FEASIBLE_VALUE)).astype(int)

    # ---------- feasibility model ----------
    unique_c = np.unique(c)
    if unique_c.size < 2:
        const_p = float(unique_c[0])
        def p_feas_fn(Xq):
            return np.full((Xq.shape[0],), const_p, dtype=float)
        feas_clf = None
    else:
        feas_clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=SEED)
        feas_clf.fit(X, c)
        def p_feas_fn(Xq):
            return feas_clf.predict_proba(Xq)[:, 1]

    # ---------- BoTorch ----------
    bt = import_botorch()
    SingleTaskGP = bt["SingleTaskGP"]
    ModelListGP = bt["ModelListGP"]
    qNEHVI = bt["qNEHVI"]
    SobolQMCNormalSampler = bt["SobolQMCNormalSampler"]
    fit_gpytorch_mll = bt["fit_gpytorch_mll"]

    try:
        from gpytorch.mlls import SumMarginalLogLikelihood
    except Exception:
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    dtype = torch.double
    train_X = torch.tensor(X, dtype=dtype)
    train_Y1 = torch.tensor(y_ms_std.reshape(-1, 1), dtype=dtype)
    train_Y2 = torch.tensor(y_y_std.reshape(-1, 1), dtype=dtype)

    def make_gp(tx, ty):
        observation_variance = torch.full_like(ty, OBSERVATION_VARIANCE)
        try:
            return SingleTaskGP(tx, ty, train_Yvar=observation_variance)
        except TypeError as exc:
            raise TypeError(
                "The installed BoTorch version does not support train_Yvar. "
                "Use a compatible version to preserve the fixed-noise model."
            ) from exc

    model_ms = make_gp(train_X, train_Y1)
    model_y = make_gp(train_X, train_Y2)
    model = ModelListGP(model_ms, model_y)

    mll = SumMarginalLogLikelihood(model.likelihood, model)
    model.train(); mll.train()
    fit_gpytorch_mll(mll)
    model.eval()

    # ---------- acquisition ----------
    Y_base = torch.cat([train_Y1, train_Y2], dim=-1)
    ref_point = (Y_base.min(dim=0).values - REF_POINT_MARGIN).tolist()

    try:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([N_QMC_SAMPLES]))
    except TypeError:
        sampler = SobolQMCNormalSampler(num_samples=N_QMC_SAMPLES)

    acq = build_qnehvi_version_robust(
        qNEHVI_cls=qNEHVI, model=model, ref_point=ref_point,
        train_X=train_X, sampler=sampler,
    )

    print("Objective models: two fixed-noise SingleTaskGP models in ModelListGP")
    print(f"Observation variance: {OBSERVATION_VARIANCE:g}")
    print(f"Acquisition: {qNEHVI.__name__}, evaluated with q = 1")
    print(f"Sobol QMC samples: {N_QMC_SAMPLES}")
    print(f"Reference point: {ref_point}")
    print(f"LHS candidates: {N_CANDIDATES}; selected candidates: {TOP_K}")

    # ---------- Candidates ----------
    # Each candidate is evaluated independently with q = 1.
    # The top-ranked individual candidates are then selected as one batch.
    Xcand = lhs(N_CANDIDATES, d=3, seed=SEED + 1)
    p_feas_raw = p_feas_fn(Xcand)
    p_feas_pow = (
        np.clip(p_feas_raw, FEAS_PROB_FLOOR, 1.0) ** FEAS_WEIGHT_POWER
    )

    Xcand_t = torch.tensor(Xcand, dtype=dtype)
    with torch.no_grad():
        a_vals = acq(Xcand_t.unsqueeze(1)).view(-1).cpu().numpy()

    score = a_vals * p_feas_pow
    top_idx = np.argsort(-score)[:TOP_K]

    Xnext = Xcand[top_idx]
    Xnext_raw = x_min + Xnext * x_rng

    next_ratio_fe3_fe2 = Xnext_raw[:, 0]
    next_fecl3_g = Xnext_raw[:, 1]
    next_ratio_surf_fe3 = Xnext_raw[:, 2]
    next_fe3_mol = next_fecl3_g / MOLAR_MASS_FECL3_6H2O
    next_fe2_mol = next_fe3_mol / next_ratio_fe3_fe2
    next_feso4_g = next_fe2_mol * MOLAR_MASS_FESO4_7H2O
    next_theory_fe3o4_g = np.array([
        theoretical_magnetite_g(next_fe3_mol[i], next_fe2_mol[i], next_ratio_fe3_fe2[i])
        for i in range(TOP_K)
    ])

    Xnext_t = torch.tensor(Xnext, dtype=dtype)
    with torch.no_grad():
        ms_pred_std = model_ms.posterior(Xnext_t).mean.view(-1).cpu().numpy()
        yd_pred_std = model_y.posterior(Xnext_t).mean.view(-1).cpu().numpy()
    ms_pred = ms_pred_std * ms_sd + ms_mu
    yd_pred = yd_pred_std * y_sd + y_mu

    # ---------- Tables ----------
    table1 = dfv[[COL_RATIO_FE3_FE2, COL_FECL3_G, COL_RATIO_SURF_FE3, col_ms, COL_YIELD, COL_FEAS]].copy()
    table1 = table1.rename(columns={col_ms: "Ms", COL_FEAS: f"{COL_FEAS} (raw)"})
    table1.to_excel(os.path.join(OUT_DIR, "Table1_Experiments.xlsx"), index=False)
    table1.to_latex(os.path.join(OUT_DIR, "Table1_Experiments.tex"), index=False, float_format="%.4g")

    table2 = pd.DataFrame([{
        "Rank": i + 1,
        "Fe3+|Fe2+": float(next_ratio_fe3_fe2[i]),
        "FeCl3·6H2O (g)": float(next_fecl3_g[i]),
        "FeSO4·7H2O (g)": float(next_feso4_g[i]),
        "Surfactant|Fe3+": float(next_ratio_surf_fe3[i]),
        "P(feasible)": float(p_feas_raw[top_idx[i]]),
        "P^power": float(p_feas_pow[top_idx[i]]),
        "qNEHVI": float(a_vals[top_idx[i]]),
        "qNEHVI×P^power": float(score[top_idx[i]]),
        "Theoretical Fe3O4 (g)": float(next_theory_fe3o4_g[i]),
        "Pred Ms (GP mean)": float(ms_pred[i]),
        "Pred Yield (GP mean)": float(yd_pred[i]),
    } for i in range(TOP_K)])
    table2.to_excel(os.path.join(OUT_DIR, "Table2_BO_Suggestions.xlsx"), index=False)
    table2.to_latex(os.path.join(OUT_DIR, "Table2_BO_Suggestions.tex"), index=False, float_format="%.4g")

    # =====================================================================
    # (a) 3D design space – main + inset (figure-in-figure)
    # =====================================================================
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import MaxNLocator

    _fa = FontProperties(family=FONT_FAMILY, weight=FONT_WEIGHT, size=FONTSIZE_TICK)

    x1 = dfv[COL_RATIO_FE3_FE2].to_numpy(float)
    x2 = dfv[COL_FECL3_G].to_numpy(float)
    x3 = dfv[COL_RATIO_SURF_FE3].to_numpy(float)

    FE3_CONC_MIN, FE3_CONC_MAX = 0.261, 0.555
    x2_min_a, x2_max_a = float(np.nanmin(x2)), float(np.nanmax(x2))
    denom_a = (x2_max_a - x2_min_a) if (x2_max_a - x2_min_a) > 0 else 1.0
    fe3_M = FE3_CONC_MIN + (x2 - x2_min_a) / denom_a * (FE3_CONC_MAX - FE3_CONC_MIN)

    def _auto_eps(v, frac, mult):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if v.size < 2: return 1e-12
        u = np.sort(np.unique(v))
        if u.size < 2: return 1e-12
        span = float(u[-1] - u[0])
        diffs = np.diff(u); diffs = diffs[diffs > 0]
        eps_frac = frac * (span if span > 0 else 1.0)
        if diffs.size == 0: return max(1e-12, eps_frac)
        q10 = float(np.quantile(diffs, 0.10)); med = float(np.median(diffs))
        return max(1e-12, eps_frac) if q10 > 0.5 * med else max(eps_frac, mult * q10)

    def _cc_3d(x, y, z, ex, ey, ez):
        n = len(x)
        if n == 0: return []
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if abs(x[i]-x[j])<=ex and abs(y[i]-y[j])<=ey and abs(z[i]-z[j])<=ez:
                    adj[i].append(j); adj[j].append(i)
        comps = []; seen = np.zeros(n, dtype=bool)
        for i in range(n):
            if seen[i]: continue
            stack=[i]; seen[i]=True; comp=[]
            while stack:
                u=stack.pop(); comp.append(u)
                for v in adj[u]:
                    if not seen[v]: seen[v]=True; stack.append(v)
            comps.append(comp)
        return comps

    def _group_3d(x, y, z, batch_arr, *, ex, ey, ez):
        comps = _cc_3d(x, y, z, ex, ey, ez)
        xs,ys,zs,labels,colors = [],[],[],[],[]
        for comp in comps:
            comp = np.asarray(comp, int)
            xm,ym,zm = float(np.mean(x[comp])),float(np.mean(y[comp])),float(np.mean(z[comp]))
            blist = sorted(batch_arr[comp].tolist())
            xs.append(xm); ys.append(ym); zs.append(zm)
            labels.append("\n".join(str(b) for b in blist))
            if all(b<=11 for b in blist): colors.append("tab:blue")
            elif all(b>=12 for b in blist): colors.append("tab:orange")
            else: colors.append("0.5")
        return np.asarray(xs),np.asarray(ys),np.asarray(zs),labels,np.asarray(colors,object)

    S_C_MAIN = SCATTER_SIZE
    S_C_INSET = SCATTER_SIZE

    eps_x = _auto_eps(x1, 0.02, 4.0)
    eps_y = _auto_eps(fe3_M, 0.03, 4.0)
    eps_z = _auto_eps(x3, 0.05, 4.0)

    gx, gy, gz, glabels, gcolors = _group_3d(
        x1, fe3_M, x3, batch, ex=eps_x, ey=eps_y, ez=eps_z)

    _ginit = np.array([("tab:blue" in c) for c in gcolors])

    def _setup_3d(ax, n_ticks=5, show_labels=True):
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.set_facecolor((1,1,1,1))
                axis.pane.set_edgecolor("k")
                axis.pane.set_linewidth(2.0)
            except Exception: pass
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(n_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
        ax.zaxis.set_major_locator(MaxNLocator(n_ticks))
        ax.tick_params(axis='both', direction='in')
        for label in ax.get_xticklabels()+ax.get_yticklabels()+ax.get_zticklabels():
            label.set_fontproperties(_fa)
        if show_labels:
            ax.set_xlabel(r"Fe$^{3+}$/Fe$^{2+}$", labelpad=12, fontproperties=_fa, fontsize=FONTSIZE_LABEL)
            ax.set_ylabel(r"Fe$^{3+}$ (mol/L)", labelpad=12, fontproperties=_fa, fontsize=FONTSIZE_LABEL)
            ax.set_zlabel(r"Surfactant/Fe$^{3+}$", labelpad=18, fontproperties=_fa, fontsize=FONTSIZE_LABEL)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
        try:
            ax.zaxis.set_tick_params(pad=4)
            ax.zaxis.label.set_rotation(90)
            ax.zaxis._axinfo["label"]["space_factor"] = 1.8
        except Exception: pass

    def _draw_pts(ax, mask, pt_size, num_fs, drop_lw, drop_alpha):
        ix = np.where(mask)[0]
        if ix.size == 0: return
        sub_x, sub_y, sub_z = gx[ix], gy[ix], gz[ix]
        z_floor = float(np.nanmin(gz))
        for xi, yi, zi in zip(sub_x, sub_y, sub_z):
            ax.plot([xi,xi],[yi,yi],[z_floor,zi], linestyle="--", linewidth=drop_lw,
                    color="k", alpha=drop_alpha, zorder=1)
        ax.scatter(sub_x, sub_y, sub_z, s=pt_size, c=gcolors[ix], edgecolors="k",
                   linewidths=SCATTER_EDGE_LW, alpha=1.0, depthshade=False, zorder=2)
        for i in ix:
            ax.text(gx[i], gy[i], gz[i], glabels[i], ha="center", va="center",
                    fontsize=num_fs, fontproperties=_fa, color="black",
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    # ---- (a1) initialization figure ----
    fig_a1 = plt.figure(figsize=(9.0, 7.0), facecolor="none")
    ax_a1 = fig_a1.add_subplot(111, projection="3d")
    _setup_3d(ax_a1)
    for axis in (ax_a1.xaxis, ax_a1.yaxis, ax_a1.zaxis):
        try:
            axis.pane.set_facecolor((1, 1, 1, 0))
        except Exception: pass
    fig_a1.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax_a1.view_init(elev=13, azim=-75)

    # draw init points (grouped, no numbers)
    _draw_pts(ax_a1, _ginit, S_C_MAIN, FONTSIZE_NUMBER, 1.0, 0.5)
    # clear text labels from init (no numbers on init circles)
    for txt in ax_a1.texts:
        txt.remove()

    # draw raw refinement points individually (no numbers, all 5 visible)
    _ref_x = x1[ref_mask]
    _ref_y = fe3_M[ref_mask]
    _ref_z = x3[ref_mask]
    _z_floor_a1 = float(np.nanmin(gz))
    for xi, yi, zi in zip(_ref_x, _ref_y, _ref_z):
        ax_a1.plot([xi,xi],[yi,yi],[_z_floor_a1,zi], linestyle="--",
                   linewidth=1.0, color="k", alpha=0.5, zorder=1)
    ax_a1.scatter(_ref_x, _ref_y, _ref_z, s=S_C_MAIN, c="tab:orange", edgecolors="k",
                  linewidths=SCATTER_EDGE_LW, alpha=1.0, depthshade=False, zorder=2)

    leg_a1 = ax_a1.legend(
        handles=[
            Line2D([0],[0], marker="o", linestyle="None", markersize=LEGEND_MARKER_SIZE,
                   markerfacecolor="tab:blue", markeredgecolor="k",
                   markeredgewidth=LEGEND_MARKER_EDGE_LW, label="Initialization"),
            Line2D([0],[0], marker="o", linestyle="None", markersize=LEGEND_MARKER_SIZE,
                   markerfacecolor="tab:orange", markeredgecolor="k",
                   markeredgewidth=LEGEND_MARKER_EDGE_LW, label="Refinement"),
        ],
        frameon=True, loc="upper left",
        bbox_to_anchor=(0.05, 0.85), bbox_transform=ax_a1.transAxes,
        facecolor="white", edgecolor="none", framealpha=1.0, prop=_fa)
    for t in leg_a1.get_texts(): t.set_fontsize(FONTSIZE_LEGEND)

    _extra_a1 = [ax_a1.xaxis.label, ax_a1.yaxis.label, ax_a1.zaxis.label, leg_a1]
    _extra_a1 = [a for a in _extra_a1 if a is not None]
    out_a1 = os.path.join(OUT_DIR, "Fig_a1_Initialization_3D")
    if EXPORT_PNG:
        fig_a1.savefig(out_a1 + ".png", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.10,
                       facecolor="none", bbox_extra_artists=_extra_a1, transparent=True)
    if EXPORT_SVG:
        fig_a1.savefig(out_a1 + ".svg", bbox_inches="tight", pad_inches=0.10,
                       facecolor="none", bbox_extra_artists=_extra_a1, transparent=True)
    plt.close(fig_a1)

    # ---- (a2) refinement figure ----
    ref_raw = ref_mask
    if np.any(ref_raw):
        _rx = x1[ref_raw]
        _ry = fe3_M[ref_raw]
        _rz = x3[ref_raw]
        _pad = 0.10
        _rxl = (float(_rx.min()) - _pad*abs(_rx.max()-_rx.min()+1e-12),
                float(_rx.max()) + _pad*abs(_rx.max()-_rx.min()+1e-12))
        _ryl = (float(_ry.min()) - _pad*abs(_ry.max()-_ry.min()+1e-12),
                float(_ry.max()) + _pad*abs(_ry.max()-_ry.min()+1e-12))
        _rzl = (float(_rz.min()) - _pad*abs(_rz.max()-_rz.min()+1e-12),
                float(_rz.max()) + _pad*abs(_rz.max()-_rz.min()+1e-12))

        fig_a2 = plt.figure(figsize=(5.5, 4.2), facecolor="white")
        ax_a2 = fig_a2.add_subplot(111, projection="3d")
        _setup_3d(ax_a2, n_ticks=2, show_labels=False)
        fig_a2.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        ax_a2.view_init(elev=13, azim=-75)

        _z_floor_a2 = float(_rz.min())
        for xi, yi, zi in zip(_rx, _ry, _rz):
            ax_a2.plot([xi,xi],[yi,yi],[_z_floor_a2,zi], linestyle="--",
                       linewidth=1.2, color="k", alpha=0.7, zorder=1)
        ax_a2.scatter(_rx, _ry, _rz, s=S_C_INSET, c="tab:orange", edgecolors="k",
                      linewidths=SCATTER_EDGE_LW, alpha=1.0, depthshade=False, zorder=2)
        for rank, (xi, yi, zi) in enumerate(zip(_rx, _ry, _rz), start=1):
            ax_a2.text(xi, yi, zi, str(rank), ha="center", va="center",
                       fontsize=FONTSIZE_NUMBER + 1, fontproperties=_fa, color="black",
                       path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

        ax_a2.set_xlim3d(*_rxl)
        ax_a2.set_ylim3d(*_ryl)
        ax_a2.set_zlim3d(*_rzl)

        _extra_a2 = [ax_a2.xaxis.label, ax_a2.yaxis.label, ax_a2.zaxis.label]
        _extra_a2 = [a for a in _extra_a2 if a is not None]
        out_a2 = os.path.join(OUT_DIR, "Fig_a2_Refinement_3D")
        if EXPORT_PNG:
            fig_a2.savefig(out_a2 + ".png", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.10,
                           facecolor="white", bbox_extra_artists=_extra_a2)
        if EXPORT_SVG:
            fig_a2.savefig(out_a2 + ".svg", bbox_inches="tight", pad_inches=0.10,
                           facecolor="white", bbox_extra_artists=_extra_a2)
        plt.close(fig_a2)

    # =====================================================================
    # (b) Ms–Yield objective space  [from v8 Fig_b]
    # =====================================================================
    ms = y_ms; yd = y_yield
    S_PT = SCATTER_SIZE
    SPLIT_MS = 15.44
    OBJ_MS = 30.0; OBJ_YD = 85.0

    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import FontProperties
    _fb = FontProperties(family=FONT_FAMILY, weight=FONT_WEIGHT, size=FONTSIZE_TICK - 1.5)

    fig_b = plt.figure(figsize=(6.5, 5.0))
    axb = fig_b.add_subplot(111)
    for spine in axb.spines.values():
        spine.set_linewidth(SPINE_LW)
    axb.tick_params(direction='in', labelsize=FONTSIZE_TICK - 1.5)
    for label in axb.get_xticklabels() + axb.get_yticklabels():
        label.set_fontproperties(_fb)

    xmin_b, xmax_b = float(np.nanmin(ms)), float(np.nanmax(ms))
    ymin_b, ymax_b = float(np.nanmin(yd)), float(np.nanmax(yd))
    xmax_b = max(xmax_b, OBJ_MS); ymax_b = max(ymax_b, OBJ_YD)
    xpad = 0.03 * (xmax_b - xmin_b + 1e-12)
    ypad = 0.06 * (ymax_b - ymin_b + 1e-12)
    xmin_b, xmax_b = xmin_b - xpad, xmax_b + xpad
    ymin_b, ymax_b = ymin_b - ypad, ymax_b + ypad

    axb.axvspan(xmin_b, SPLIT_MS, facecolor="0.88", zorder=0)
    axb.axvspan(SPLIT_MS, xmax_b, facecolor="0.97", zorder=0)
    axb.axvline(SPLIT_MS, color="k", linestyle="--", linewidth=2.0, zorder=1)

    OBJ_FACE = "#CDEFD6"; OBJ_EDGE = "#2E8B57"
    rect = Rectangle((OBJ_MS, OBJ_YD), (xmax_b - OBJ_MS), (ymax_b - OBJ_YD),
                      facecolor=OBJ_FACE, alpha=1, edgecolor=OBJ_EDGE, linewidth=2.6, zorder=1)
    axb.add_patch(rect)

    base_color = np.where(init_mask, "tab:blue", "tab:orange")
    special_blue = init_mask & (ms < SPLIT_MS)
    feasible_pts = (c == 1); infeasible_pts = ~feasible_pts

    if np.any(special_blue):
        axb.scatter(ms[special_blue], yd[special_blue], s=S_PT, c="tab:blue",
                    edgecolors="k", linewidths=SCATTER_EDGE_LW, zorder=3, alpha=1.0, rasterized=False)
    m_feas = feasible_pts & (~special_blue)
    if np.any(m_feas):
        axb.scatter(ms[m_feas], yd[m_feas], s=S_PT, c=base_color[m_feas],
                    edgecolors="k", linewidths=SCATTER_EDGE_LW, zorder=3, alpha=1.0, rasterized=False)
    m_inf = infeasible_pts & (~special_blue)
    if np.any(m_inf):
        axb.scatter(ms[m_inf], yd[m_inf], s=S_PT, facecolors="none",
                    edgecolors="k", linewidths=SCATTER_EDGE_LW, zorder=3, alpha=0.25, rasterized=False)

    opt_idx = np.where(ref_mask)[0]
    if opt_idx.size > 0:
        opt_idx = opt_idx[np.argsort(batch[opt_idx])]
        k = min(TOP_K, opt_idx.size)
        for j in range(k):
            i = opt_idx[j]
            axb.text(float(ms[i]), float(yd[i]), str(j + 1), ha="center", va="center",
                     fontsize=FONTSIZE_NUMBER - 1.5, fontweight=FONT_WEIGHT, color="black", zorder=4,
                     path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    _bbox_below = dict(facecolor="0.88", edgecolor="none", alpha=1.0, boxstyle="round,pad=0.2")
    _bbox_above = dict(facecolor="0.97", edgecolor="none", alpha=1.0, boxstyle="round,pad=0.2")
    axb.text(xmin_b + 0.5*(SPLIT_MS - xmin_b), ymin_b + 0.5*(ymax_b - ymin_b),
             "Below constraint\nthreshold", ha="center", va="center", fontsize=FONTSIZE_REGION,
             fontproperties=_fb, color="0.15", zorder=5, bbox=_bbox_below)
    axb.text(SPLIT_MS + 0.5*(xmax_b - SPLIT_MS), ymin_b + 0.5*(ymax_b - ymin_b),
             "Above constraint threshold", ha="center", va="center", fontsize=FONTSIZE_REGION,
             fontproperties=_fb, color="0.15", zorder=5, bbox=_bbox_above)

    # Objectives annotation with arrow
    _obj_x = OBJ_MS + 0.25 * (xmax_b - OBJ_MS)
    _obj_y = OBJ_YD + 0.5 * (ymax_b - OBJ_YD)
    _bbox_obj = dict(facecolor=OBJ_FACE, edgecolor="none", alpha=1.0, boxstyle="round,pad=0.2")
    axb.annotate("Objectives", xy=(OBJ_MS + 0.5*(xmax_b - OBJ_MS), OBJ_YD + 0.3*(ymax_b - OBJ_YD)),
                 xytext=(_obj_x, _obj_y),
                 fontsize=FONTSIZE_REGION, fontproperties=_fb, color="0.15",
                 ha="center", va="center", zorder=5, bbox=_bbox_obj,
                 arrowprops=dict(arrowstyle="->", color="0.15", lw=1.5))

    axb.set_xlim(xmin_b, xmax_b); axb.set_ylim(ymin_b, ymax_b)
    axb.set_xlabel("Ms (emu/g)", fontproperties=_fb, fontsize=FONTSIZE_LABEL - 1.5)
    axb.set_ylabel("Yield (%)", fontproperties=_fb, fontsize=FONTSIZE_LABEL - 1.5)
    axb.grid(False)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements_b = [
        Line2D([0],[0], marker="o", linestyle="None", markersize=LEGEND_MARKER_SIZE,
               markerfacecolor="tab:blue", markeredgecolor="k", markeredgewidth=LEGEND_MARKER_EDGE_LW, label="Initialization"),
        Line2D([0],[0], marker="o", linestyle="None", markersize=LEGEND_MARKER_SIZE,
               markerfacecolor="tab:orange", markeredgecolor="k", markeredgewidth=LEGEND_MARKER_EDGE_LW, label="Refinement"),
    ]
    leg_b = axb.legend(handles=legend_elements_b, loc="upper left", frameon=True,
                       facecolor="white", edgecolor="none", framealpha=1.0, prop=_fb)
    for text in leg_b.get_texts():
        text.set_fontsize(FONTSIZE_LEGEND - 1.5)

    _save_fig(fig_b, os.path.join(OUT_DIR, "Fig_b_ObjectiveSpace_Pareto"))
    plt.close(fig_b)

    # =====================================================================
    # (c) 3D surrogate landscape + learning trajectory
    # =====================================================================
    from scipy.interpolate import RBFInterpolator

    def x2raw_to_fe3conc(x2_raw):
        return FE3_CONC_MIN + ((x2_raw - x_min[1]) / x_rng[1]) * (FE3_CONC_MAX - FE3_CONC_MIN)

    note_col = find_note_column(dfv)
    if note_col is None:
        raise RuntimeError("Cannot find a 'note' column (needed for BO trajectory).")

    note_series = dfv[note_col].astype(str).str.strip()
    refine_iter = note_series.map(NOTE_KEYS)
    refine_mask_c = refine_iter.notna().to_numpy()
    init_mask_c = ~refine_mask_c

    refine_idx = np.where(refine_mask_c)[0]
    refine_idx_sorted = refine_idx[np.argsort(refine_iter.to_numpy()[refine_mask_c].astype(int))]
    init_idx = np.where(init_mask_c)[0]
    if init_idx.size == 0:
        raise RuntimeError("No initialization points found.")

    p_obs = p_feas_fn(X)
    score_obs = y_ms_std + y_y_std
    z_obs = score_obs * (
        np.clip(p_obs, FEAS_PROB_FLOOR, 1.0) ** FEAS_WEIGHT_POWER
    )
    start_idx = init_idx[np.argmax(z_obs[init_mask_c])]
    path_idx = np.concatenate([[start_idx], refine_idx_sorted])

    x3_raw_all = dfv[COL_RATIO_SURF_FE3].to_numpy(float)
    x3_norm_all = (x3_raw_all - x_min[2]) / x_rng[2]

    x1n = np.linspace(0.0, 1.0, N_GRID)
    x2n = np.linspace(0.0, 1.0, N_GRID)
    X1n, X2n = np.meshgrid(x1n, x2n)

    # A lightly regularized RBF gives a continuous third-variable slice over
    # the rectangular plotting window without convex-hull seams in 3D.
    grid_xy = np.column_stack([X1n.ravel(), X2n.ravel()])
    x3_rbf = RBFInterpolator(
        X[:, :2], x3_norm_all, kernel="thin_plate_spline", smoothing=RBF_SMOOTHING)
    x3_interp = x3_rbf(grid_xy)
    x3_interp = np.clip(x3_interp, 0.0, 1.0)
    grid_norm = np.column_stack([X1n.ravel(), X2n.ravel(), x3_interp])
    grid_t = torch.tensor(grid_norm, dtype=dtype)

    with torch.no_grad():
        mu_ms = model_ms.posterior(grid_t).mean.squeeze(-1).cpu().numpy()
        mu_y = model_y.posterior(grid_t).mean.squeeze(-1).cpu().numpy()

    score_mu = mu_ms + mu_y
    p_grid = p_feas_fn(grid_norm)
    Z = (
        score_mu
        * (np.clip(p_grid, FEAS_PROB_FLOOR, 1.0) ** FEAS_WEIGHT_POWER)
    ).reshape(N_GRID, N_GRID)

    X1_raw = X1n * x_rng[0] + x_min[0]
    X2_raw = X2n * x_rng[1] + x_min[1]
    X2_fe3c = x2raw_to_fe3conc(X2_raw)

    x1_raw_obs = dfv[COL_RATIO_FE3_FE2].to_numpy(float)
    x2_raw_obs = dfv[COL_FECL3_G].to_numpy(float)
    x2_fe3c_obs = x2raw_to_fe3conc(x2_raw_obs)

    roi_x = x1_raw_obs[path_idx]
    xmin_c, xmax_c = float(np.min(roi_x)), float(np.max(roi_x))
    mx = 0.22 * (xmax_c - xmin_c + 1e-12)
    gxmin, gxmax = float(np.min(X1_raw)), float(np.max(X1_raw))
    xlim_c = (max(gxmin, xmin_c - mx), min(gxmax, xmax_c + mx))

    vmin_c = float(np.nanpercentile(Z, PCT_LOW))
    vmax_c = float(np.nanpercentile(Z, PCT_HIGH))
    if (not np.isfinite(vmin_c)) or (not np.isfinite(vmax_c)) or vmin_c >= vmax_c:
        vmin_c, vmax_c = float(np.nanmin(Z)), float(np.nanmax(Z))
    levels_c = np.linspace(vmin_c, vmax_c, 35)

    from matplotlib.font_manager import FontProperties
    arial_bold = FontProperties(family=FONT_FAMILY, weight=FONT_WEIGHT, size=FONTSIZE_TICK)

    # Retain the focused 2D panel window, but render it as a 3D response surface.
    xlim_c = (0.925, 1.075)
    ylim_c = (0.27, 0.41)
    x_vec = X1_raw[0, :]
    y_vec = X2_fe3c[:, 0]
    x_keep = (x_vec >= xlim_c[0]) & (x_vec <= xlim_c[1])
    y_keep = (y_vec >= ylim_c[0]) & (y_vec <= ylim_c[1])
    X1_plot = X1_raw[np.ix_(y_keep, x_keep)]
    X2_plot = X2_fe3c[np.ix_(y_keep, x_keep)]
    Z_plot = Z[np.ix_(y_keep, x_keep)]

    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import proj3d
    from scipy.interpolate import RegularGridInterpolator

    class Arrow3D(mpatches.FancyArrowPatch):
        """A 2D FancyArrowPatch projected cleanly between two 3D points."""
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(
                xs3d, ys3d, zs3d, self.axes.get_proj())
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return float(np.min(zs))

    norm_c = Normalize(vmin=vmin_c, vmax=vmax_c, clip=True)
    cmap_c = plt.get_cmap("viridis")
    z_min_plot = float(np.nanmin(Z_plot))
    z_max_plot = float(np.nanmax(Z_plot))
    z_span = max(z_max_plot - z_min_plot, 1e-12)
    z_floor = z_min_plot - 0.22 * z_span
    z_lift = 0.035 * z_span

    # Bilinear interpolation places the learning path exactly on the displayed surface.
    z_interp = RegularGridInterpolator(
        (y_vec, x_vec), Z, method="linear", bounds_error=False, fill_value=None)
    path_xy = np.column_stack([
        x2_fe3c_obs[path_idx],
        x1_raw_obs[path_idx],
    ])
    z_path_surface = z_interp(path_xy)
    z_path = z_path_surface + z_lift

    fig_c = plt.figure(figsize=(8.5, 6.3), facecolor="white")
    axc = fig_c.add_subplot(111, projection="3d", computed_zorder=False)
    fig_c.subplots_adjust(left=0.00, right=0.87, bottom=0.04, top=0.98)

    surf = axc.plot_surface(
        X1_plot, X2_plot, Z_plot,
        facecolors=cmap_c(norm_c(Z_plot)),
        rstride=2, cstride=2,
        linewidth=0.22, edgecolor=(1, 1, 1, 0.20),
        antialiased=True, shade=False, alpha=0.98, zorder=1)

    # A contour "map" on the floor makes the surface topology legible at a glance.
    axc.contourf(
        X1_plot, X2_plot, Z_plot, zdir="z", offset=z_floor,
        levels=levels_c, cmap=cmap_c, norm=norm_c, alpha=0.58, zorder=0)
    axc.contour(
        X1_plot, X2_plot, Z_plot, zdir="z", offset=z_floor,
        levels=10, colors="0.18", linewidths=0.55, alpha=0.34, zorder=0)

    # Draw a clean, elevated learning trajectory with arrowheads and drop lines.
    path_x = x1_raw_obs[path_idx]
    path_y = x2_fe3c_obs[path_idx]
    for i in range(len(path_idx)):
        axc.plot(
            [path_x[i], path_x[i]], [path_y[i], path_y[i]],
            [z_floor, z_path_surface[i]],
            color="0.18", linestyle=":", linewidth=1.0, alpha=0.38, zorder=4)

    for i in range(len(path_idx) - 1):
        ls = "--" if i == 0 else "-"
        alpha = 0.78 if i == 0 else 0.98
        lw = 1.9 if i == 0 else 2.5
        arrow = Arrow3D(
            [path_x[i], path_x[i + 1]],
            [path_y[i], path_y[i + 1]],
            [z_path[i], z_path[i + 1]],
            arrowstyle="-|>", mutation_scale=14,
            linewidth=lw, linestyle=ls, color="k", alpha=alpha, zorder=7)
        axc.add_artist(arrow)

    S_CIRC = SCATTER_SIZE * 0.90
    axc.scatter(
        [path_x[0]], [path_y[0]], [z_path[0]],
        s=S_CIRC, marker="o", c=["tab:blue"], edgecolors="k",
        linewidths=SCATTER_EDGE_LW, depthshade=False, zorder=9)

    for j, idx in enumerate(refine_idx_sorted, start=1):
        pi = j  # path position 0 is the best initialization
        axc.scatter(
            [path_x[pi]], [path_y[pi]], [z_path[pi]],
            s=S_CIRC, marker="o", c=["tab:orange"], edgecolors="k",
            linewidths=SCATTER_EDGE_LW, depthshade=False, zorder=9)
        axc.text(
            float(path_x[pi]), float(path_y[pi]), float(z_path[pi] + 0.012 * z_span),
            str(j), ha="center", va="center",
            fontsize=FONTSIZE_NUMBER - 1, fontproperties=arial_bold, color="k", zorder=10,
            path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])

    # Subtle halo around the final experiment emphasizes the destination.
    if len(path_idx) > 1:
        axc.scatter(
            [path_x[-1]], [path_y[-1]], [z_path[-1]],
            s=S_CIRC * 1.55, marker="o", facecolors="none", edgecolors="#F5C242",
            linewidths=3.0, depthshade=False, zorder=8)

    axc.set_xlim(*xlim_c)
    axc.set_ylim(*ylim_c)
    axc.set_zlim(z_floor, z_max_plot + 0.15 * z_span)
    axc.view_init(elev=30, azim=-58)
    axc.set_box_aspect((1.32, 1.0, 0.72))

    axc.set_xlabel(
        r"Fe$^{3+}$/Fe$^{2+}$", labelpad=9,
        fontproperties=arial_bold, fontsize=FONTSIZE_LABEL - 2.5)
    axc.set_ylabel(
        r"Fe$^{3+}$ (mol/L)", labelpad=10,
        fontproperties=arial_bold, fontsize=FONTSIZE_LABEL - 2.5)
    # The colorbar carries the z-axis definition and avoids duplicated vertical text.
    axc.set_zlabel("")
    axc.xaxis.set_major_locator(plt.MaxNLocator(5))
    axc.yaxis.set_major_locator(plt.MaxNLocator(5))
    axc.zaxis.set_major_locator(plt.MaxNLocator(4))
    axc.tick_params(axis="x", labelsize=FONTSIZE_TICK - 3.5, pad=0)
    axc.tick_params(axis="y", labelsize=FONTSIZE_TICK - 3.5, pad=0)
    axc.tick_params(axis="z", labelsize=FONTSIZE_TICK - 3.5, pad=2)
    for label in axc.get_xticklabels() + axc.get_yticklabels() + axc.get_zticklabels():
        label.set_fontproperties(arial_bold)

    for axis in (axc.xaxis, axc.yaxis, axc.zaxis):
        axis.pane.set_facecolor((0.97, 0.98, 0.99, 0.30))
        axis.pane.set_edgecolor((0.20, 0.20, 0.20, 0.75))
        axis.pane.set_linewidth(1.3)
        axis._axinfo["grid"]["color"] = (0.25, 0.25, 0.25, 0.15)
        axis._axinfo["grid"]["linewidth"] = 0.65

    sm = cm.ScalarMappable(norm=norm_c, cmap=cmap_c)
    sm.set_array([])
    cbar = fig_c.colorbar(
        sm, ax=axc, pad=0.035, shrink=0.70, aspect=24, extend="both")
    cbar.set_label(
        r"$(\mu_{\mathrm{Ms}} + \mu_{\mathrm{Yield}})\times P(\mathrm{feasible})$",
        fontproperties=arial_bold, fontsize=FONTSIZE_COLORBAR - 2, labelpad=7)
    cbar.locator = plt.MaxNLocator(4)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK - 3.5, direction="in")
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(arial_bold)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markersize=LEGEND_MARKER_SIZE, markerfacecolor="tab:blue",
               markeredgecolor="k", markeredgewidth=LEGEND_MARKER_EDGE_LW,
               label="Best initialization"),
        Line2D([0], [0], marker="o", linestyle="None",
               markersize=LEGEND_MARKER_SIZE, markerfacecolor="tab:orange",
               markeredgecolor="k", markeredgewidth=LEGEND_MARKER_EDGE_LW,
               label="Refinement"),
        Line2D([0], [0], color="k", linewidth=2.6, marker=">",
               markersize=7, markevery=[1], label="Learning trajectory"),
    ]
    legend_c = axc.legend(
        handles=legend_handles, frameon=True, loc="upper left",
        bbox_to_anchor=(0.015, 0.96), facecolor="white", edgecolor="none",
        framealpha=0.90, prop=arial_bold)
    for text in legend_c.get_texts():
        text.set_fontsize(FONTSIZE_LEGEND - 3.5)

    out_c = os.path.join(OUT_DIR, "Fig_c_SurrogateLandscape_Trajectory")
    if EXPORT_PNG:
        fig_c.savefig(out_c + ".png", dpi=EXPORT_DPI, bbox_inches="tight")
    if EXPORT_SVG:
        fig_c.savefig(out_c + ".svg", bbox_inches="tight")
    plt.close(fig_c)


if __name__ == "__main__":
    main()
