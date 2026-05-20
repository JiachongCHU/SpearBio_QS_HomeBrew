import streamlit as st
import pandas as pd
import string
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm as scnorm
import matplotlib.pyplot as plt
import io, os
from typing import List
import matplotlib.lines as mlines
import glob
import matplotlib as mpl
from matplotlib.colors import PowerNorm
import re
from pathlib import Path
from PIL import Image



version = "demo v1.1.0"

@st.cache_data(show_spinner=False)
def load_quantstudio(uploaded_file) -> pd.DataFrame:
    """
    Read QuantStudio exports from CSV or Excel (xlsx/xls) and standardize columns.
    Removes comment lines for CSV; for Excel, reads the first sheet and
    auto-recovers header if needed.
    """
    from pathlib import Path

    suffix = Path(uploaded_file.name).suffix.lower()

    def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if cl == "well position": rename[c] = "Well Position"
            elif cl == "cycle number": rename[c] = "Cycle"
            elif cl == "well":         rename[c] = "Index"  # numeric index, not A1 label
            elif cl == "stage number": rename[c] = "Stage"
            elif cl == "step number":  rename[c] = "Step"
        if rename:
            df = df.rename(columns=rename)
        if "Index" in df.columns and "Well" in df.columns:
            df = df.drop(columns=["Index"])
        return df

    if suffix == ".csv":
        txt = uploaded_file.getvalue().decode("utf-8", errors="replace")
        # strip blank lines and comment-like prefaces
        lines = [ln for ln in txt.splitlines()
                 if (s := ln.strip()) and not s.startswith("#") and s not in {"...", "…"}]
        df = pd.read_csv(io.StringIO("\n".join(lines)), engine="python")
        return _standardize_cols(df)

    elif suffix in (".xlsx", ".xls"):
        # try normal read first
        uploaded_file.seek(0)
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0)  # engine auto
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, sheet_name=0, engine="openpyxl")

        # if header wasn't detected, try to find it
        cols_str = pd.Index([str(c).lower() for c in df.columns])
        if not any(k in " ".join(cols_str) for k in ["well position", "cycle number", "stage number", "step number", "well"]):
            uploaded_file.seek(0)
            tmp = pd.read_excel(uploaded_file, sheet_name=0, header=None)
            # find a row that contains likely header tokens
            header_row = None
            tokens = {"well position", "cycle number", "stage number", "step number", "well"}
            for i in range(min(len(tmp), 30)):  # scan first 30 rows
                row_vals = tmp.iloc[i].astype(str).str.lower().str.strip()
                if any(v in tokens for v in row_vals):
                    header_row = i
                    break
            uploaded_file.seek(0)
            if header_row is not None:
                df = pd.read_excel(uploaded_file, sheet_name=0, header=header_row)
        return _standardize_cols(df)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")
        
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl == "well position": rename[c] = "Well Position"
        elif cl == "cycle number": rename[c] = "Cycle"
        elif cl == "well":         rename[c] = "Index"
        elif cl == "stage number": rename[c] = "Stage"
        elif cl == "step number":  rename[c] = "Step"
    return df.rename(columns=rename) if rename else df

def _load_combined_xlsx(file_like):
    # Always try to read all three sheets with header row = 25th row (0-indexed 24)
    df    = pd.read_excel(file_like, sheet_name="Raw Data",       header=24, engine="openpyxl")
    return (_standardize_columns(df))

def _guess_runname(filename: str) -> str:
    return Path(filename).stem


# Define 4PL function
def four_param_logistic(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c)**b)

# Define inverse function to calculate Ct
def inverse_four_pl(threshold, a, b, c, d):
    try:
        return c * ((a - d) / (threshold - d) - 1)**(1 / b)
    except:
        return None
        

import numpy as np
from scipy.optimize import curve_fit

def calculate_ct(x, y, threshold, startpoint=10, use_4pl=False, return_std=False, scale='log'):
    x = np.asarray(x); y = np.asarray(y)

    # drop NaNs
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return (None, None) if return_std else None

    # ensure x ascending
    order = np.argsort(x)
    x, y = x[order], y[order]

    # restrict to x >= startpoint (hard guarantee)
    post = x >= startpoint
    if np.count_nonzero(post) < 2:
        return (None, None) if return_std else None
    x_fit, y_fit = x[post], y[post]

    # 4PL first (optional)
    if use_4pl:
        try:
            if x_fit.size >= 5:
                popt, pcov = curve_fit(four_param_logistic, x_fit, y_fit, maxfev=10000)
                ct = inverse_four_pl(threshold, *popt)
                if (ct is not None) and (x_fit[0] <= ct <= x_fit[-1]):
                    if return_std:
                        eps = 1e-8
                        grads = np.zeros(4)
                        for i in range(4):
                            p_hi = np.array(popt); p_hi[i] += eps
                            p_lo = np.array(popt); p_lo[i] -= eps
                            ct_hi = inverse_four_pl(threshold, *p_hi)
                            ct_lo = inverse_four_pl(threshold, *p_lo)
                            grads[i] = (ct_hi - ct_lo) / (2*eps)
                        ct_var = float(np.dot(grads.T, np.dot(pcov, grads)))
                        ct_std = np.sqrt(ct_var) if ct_var >= 0 else np.nan
                        return float(ct), float(ct_std)
                    return float(ct)
        except Exception:
            pass  # fall through to interpolation

    # Fallback: interpolate crossing within [x_fit[0], x_fit[-1]]
    if scale == 'linear':
        above = y_fit > threshold
        if not np.any(above):
            return (None, None) if return_std else None
        idx = int(np.argmax(above))
        if idx == 0:
            ct = float(x_fit[0])
        else:
            x1, x2 = x_fit[idx-1], x_fit[idx]
            y1, y2 = y_fit[idx-1], y_fit[idx]
            if y2 == y1:
                return (None, None) if return_std else None
            ct = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
        return (float(ct), None) if return_std else float(ct)

    elif scale == 'log':
        if (threshold is None) or (not np.isfinite(threshold)) or (threshold <= 0):
            return (None, None) if return_std else None
        pos = y_fit > 0
        if np.count_nonzero(pos) < 2:
            return (None, None) if return_std else None
        xf = x_fit[pos]
        yf_log = np.log10(y_fit[pos])
        thr_log = np.log10(threshold)
        above = yf_log > thr_log
        if not np.any(above):
            return (None, None) if return_std else None
        idx = int(np.argmax(above))
        if idx == 0:
            ct = float(xf[0])
        else:
            x1, x2 = xf[idx-1], xf[idx]
            y1, y2 = yf_log[idx-1], yf_log[idx]
            if y2 == y1:
                return (None, None) if return_std else None
            ct = x1 + (thr_log - y1) * (x2 - x1) / (y2 - y1)
        return (float(ct), None) if return_std else float(ct)

    else:
        raise ValueError("scale must be 'linear' or 'log'")


def find_threshold_for_target_ct_multi(
    x,                 # 1D array of cycles (shared by all wells)
    ybg_list,          # list/tuple of 1–4 background-corrected y arrays (same length as x)
    target_ct,         # desired average Ct (float)
    calculate_ct_func, # callable: (x, y, thr) -> ct OR (ct, ...)
    ct_tol=0.01,
    max_iter=60,
    eps=1e-12
):
    """
    Find a single fluorescence threshold such that the *average* Ct across
    multiple wells equals target_ct (within ct_tol).

    Returns
    -------
    threshold : float
        Threshold giving average Ct ≈ target_ct.
    ct_avg : float
        Average Ct at that threshold (over valid wells).
    ct_list : list[float or None]
        Per-well Ct at that threshold (None for wells that failed Ct).
    """
    import numpy as np

    x = np.asarray(x, dtype=float)

    # Combine all wells' y to set a robust search bracket
    y_all = np.concatenate([np.asarray(y, dtype=float) for y in ybg_list])
    lo = max(np.nanmin(y_all) + eps, eps)
    hi = np.nanmax(y_all) - eps
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("Invalid combined y range for threshold search.")

    def _ct_list(thr):
        out = []
        for y in ybg_list:
            res = calculate_ct_func(x, y, thr)
            ct = res[0] if isinstance(res, (tuple, list)) else res
            out.append(ct if (ct is not None and np.isfinite(ct)) else None)
        return out

    def _ct_avg(thr):
        cts = [ct for ct in _ct_list(thr) if ct is not None]
        if len(cts) == 0:
            return None
        return float(np.mean(cts))

    def _valid(v):
        return v is not None and np.isfinite(v)

    # Evaluate ends; if invalid, nudge to percentiles
    ct_lo = _ct_avg(lo)
    ct_hi = _ct_avg(hi)
    if not _valid(ct_lo):
        lo = np.nanpercentile(y_all, 5) + eps
        ct_lo = _ct_avg(lo)
    if not _valid(ct_hi):
        hi = np.nanpercentile(y_all, 95) - eps
        ct_hi = _ct_avg(hi)

    if not (_valid(ct_lo) and _valid(ct_hi)):
        raise RuntimeError("Could not evaluate average Ct at search bounds.")

    # Ensure monotonic direction (Ct increases with threshold in typical qPCR)
    if ct_lo > ct_hi:
        lo, hi = hi, lo
        ct_lo, ct_hi = ct_hi, ct_lo

    # Feasibility check
    if target_ct < ct_lo - 1e-9 or target_ct > ct_hi + 1e-9:
        raise ValueError(
            f"Target Ct {target_ct:.2f} is outside achievable average range "
            f"[{ct_lo:.2f}, {ct_hi:.2f}]"
        )

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ct_mid = _ct_avg(mid)

        if not _valid(ct_mid):
            mid = np.nextafter(mid, hi)  # nudge upward
            ct_mid = _ct_avg(mid)
            if not _valid(ct_mid):
                hi = mid
                continue

        err = ct_mid - target_ct
        if abs(err) <= ct_tol:
            cts_at_mid = _ct_list(mid)
            return mid, ct_mid, cts_at_mid

        if err < 0:   # need later Ct -> increase threshold
            lo, ct_lo = mid, ct_mid
        else:         # need earlier Ct -> decrease threshold
            hi, ct_hi = mid, ct_mid

    # Fallback: best midpoint
    mid = 0.5 * (lo + hi)
    return mid, _ct_avg(mid), _ct_list(mid)


    
def spr_QSqpcr_dY_calculation(df, selected_wells):
    fam_raw_ch = 'X1_M1'   # pre: FAM raw
    rox_raw_ch = 'X4_M4'   # pre: ROX raw
    Rn = []
    for rows in selected_wells:
        for well in rows:
            sub = df[df["Well Position"].astype(str) == str(well)]
            fam_y = sub[fam_raw_ch].astype(float)
            rox_y = sub[rox_raw_ch].astype(float)
            fRn = fam_y/rox_y
            Rn.extend(fRn.to_numpy())
    sorted_Rn = sorted(Rn)                     
    p5, p95 = np.percentile(sorted_Rn, [5, 95])
    DeltaY = p95 - p5
    return DeltaY
    
def linear_exp_fit(x,a,b,c):
    return a*x + b*(2**x) + c

def linear_fit(x,d,e):
    return d*x + e
    
def SPR_fitbackground(median_intercept,fam_y,rox_y,start,end,cycles,plot = False):
    window = np.arange(start,end)
    fam_y_window = fam_y[start:end]
    rox_y_window = rox_y[start:end]

    popt, pcov = curve_fit(linear_exp_fit, window, fam_y_window, p0=[1, 2, 0.1])  # p0 = initial guess
    a,b,c = popt

    popt, pcov = curve_fit(linear_fit, window, rox_y_window, p0=[1, 0.1])  # p0 = initial guess
    d,e = popt
    # Use 0-indexed positions k'=0..n-1 matching the fit coordinate system.
    # c and e are intercepts at k'=0; using 1-indexed cycles would introduce a
    # constant offset of (a - mi*d) in every bkg value.
    kp = np.arange(len(fam_y))
    fam_bkg_fit = a*kp + c
    rox_bkg_fit = d*kp + e
    if plot:
        plt.plot(cycles,fam_y,label = f'fam')
        plt.plot(cycles,rox_y,label = f'rox')
        plt.plot(cycles,fam_bkg_fit,'--',label = f'fam* fit')
        plt.plot(cycles,rox_bkg_fit,'--',label = f'rox fit')
        plt.legend()
        plt.show()
    bkg = (a - median_intercept*d) * kp + (c - median_intercept*e)
    if np.mean(bkg)>0:
        return (fam_y - bkg) / rox_y
    else:
        bkg = (d - a/median_intercept) * kp + (e - c/median_intercept)
        return fam_y / (rox_y - bkg)

def spr_QSqpcr_background_dY_residue(df, selected_wells, startcycle = 6, window_size = 6, StepIndY = 100):
    fam_raw_ch = 'X1_M1'   # pre: FAM raw
    rox_raw_ch = 'X4_M4'   # pre: ROX raw
     
    residue = []
    for ii, rows in enumerate(selected_wells):
        for jj,well in enumerate(rows):
                sub = df[df["Well Position"].astype(str) == str(well)]
                fam_y = sub[fam_raw_ch].astype(float)
                rox_y = sub[rox_raw_ch].astype(float)
                y_norm = fam_y/rox_y
                
                test_signal = y_norm
                y = np.asarray(test_signal, dtype=float)
                n = len(y)
                A = np.arange(n)
                p5, p95 = np.percentile(y, [5, 95])
                DeltaY = p95 - p5
                threshold = DeltaY/StepIndY
                Sn = np.full(n,np.nan)
                Sn[startcycle] = y[startcycle+window_size] - y[startcycle]
                detected = False
                start_point = -1
                
                for i in range(startcycle + 1, n - window_size-2):
                    Sn[i] = y[i+window_size] - y[i]
                    Sn[i+1] = y[i+1+window_size] - y[i+1]
                    Sn[i+2] = y[i+2+window_size] - y[i+2]
                    cond1 = (Sn[i] - Sn[i-1] > threshold)
                    cond2 = (Sn[i+1] - Sn[i] > threshold)
                    cond3 = (Sn[i+2] - Sn[i+1] > threshold)
                    if (cond1 & cond2 & cond3).all():
                        # if window_size % 2 == 0:
                        #     start = i - window_size // 2
                        #     end = i + window_size // 2
                        # else:
                        #     start = i - (window_size - 1) // 2 - 1
                        #     end = i + (window_size - 1) // 2 - 1
                        
                        start = i - 1 
                        end = i + window_size
                        xf = np.arange(start, end)
                        yy = y[start:end]
                        popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])  # p0 = initial guess
                        a,b,c = popt
                        baseline = a * A + c
                        # baseline = np.polyval(p, A)
                        E = (y - baseline)/baseline
                        start_point = end-1
                        detected = True
                        for xx in xf:
                            residue.append(y[xx] -  (a * xx + c))
    
    x = np.asarray(residue, dtype=float).ravel()
    mean, std = scnorm.fit(x)
    return residue,mean,std
    
def spr_QSqpcr_background_dY_v5(std, test_signal, sigma_mult=2.0, min_points=4, max_refit_iter = 3, startcycle = 6, window_size = 6, StepIndY = 40, returnbase = False):
    y = np.asarray(test_signal, dtype=float)
    n = len(y)
    A = np.arange(n)
    p5, p95 = np.percentile(y, [5, 95])
    DeltaY = p95 - p5
    threshold = DeltaY/StepIndY
    # print (threshold)
    Sn = np.full(n,np.nan)
    Sn[startcycle] = y[startcycle+window_size] - y[startcycle]
    detected = False
    start_point = -1
    
    sigma = np.full(n, float(std))

    
    for i in range(startcycle + 1, n - window_size - 2):
        # print (f'\n')
        Sn[i] = y[i+window_size] - y[i]
        Sn[i+1] = y[i+1+window_size] - y[i+1]
        Sn[i+2] = y[i+2+window_size] - y[i+2]
        
        cond1 = (Sn[i] - Sn[i-1] > threshold)
        cond2 = (Sn[i+1] - Sn[i] > threshold)
        cond3 = (Sn[i+2] - Sn[i+1] > threshold)
        # print (f'cycle = {i}')
        # print (Sn[i] - Sn[i-1])
        # print (Sn[i+1] - Sn[i])
        # print (Sn[i+2] - Sn[i+1])
        if (cond1 & cond2 & cond3).all():

            start = i - 1 
            end = i + window_size
            # if window_size % 2 == 0:
            #     start = i - 1 - window_size // 2
            #     end = i + 2 + window_size // 2
            # else:
            #     start = i - 1 - (window_size - 1) // 2 - 1
            #     end = i + 2 + (window_size - 1) // 2 - 1
                
            xf = np.arange(start, end)
            yy = y[start:end].copy()
            sig = sigma[start:end]

            # Initial fit
            mask = np.isfinite(yy) & np.isfinite(sig)
            if mask.sum() < min_points:
                popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])  # p0 = initial guess
                # a, b = np.polyfit(xf, yy, 1)
                a,b,c = popt
            else:
                # a, b = np.polyfit(xf[mask], yy[mask], 1)
                popt, pcov = curve_fit(linear_exp_fit, xf[mask], yy[mask], p0=[1, 2, 0.1])  # p0 = initial guess
                a,b,c = popt
                # Iteratively drop > sigma_mult * sigma residuals and refit
                for _ in range(max_refit_iter):
                    
                    res  = yy - linear_exp_fit(xf,a,b,c)
                    # print (f'res : {res}')
                    keep = (np.abs(res) <= sigma_mult * sig) & mask
                    if keep.sum() < min_points or keep.sum() == mask.sum():
                        break
                    popt, pcov = curve_fit(linear_exp_fit, xf, yy, p0=[1, 2, 0.1])  # p0 = initial guess
                    a,b,c = popt
                    mask = keep  # tighten for the next pass
                    yy = yy[keep]
                    xf = xf[keep]
                    sig = sig[keep]
                    mask = mask[keep]
            # plt.plot(xf,yy)
            # plt.plot(xf, linear_exp_fit(xf,a,b,c),'r--')
            # plt.plot(A,y)
            # plt.plot(A, linear_exp_fit(A,a,b,c),'r--')
            # print (f'fitted line = {a}x + {b}2^x + {c}')       
            # baseline = a * A + b
            baseline = a * A + c
            # print (f'baseline = {a}x + {c}')
            E = (y - baseline) / baseline
            start_point = end - 1 - 2
            detected = True
            expcurve = linear_exp_fit(A,a,b,c)
            intercept = c
            if returnbase:
                return E, start_point, start, end, intercept, baseline, expcurve
            else: return E, start_point, start, end, intercept

    # if break 
    if returnbase:
        return y - np.nanmean(y[startcycle:startcycle+window_size]), -1, startcycle, startcycle+window_size, np.nanmean(y[startcycle:startcycle+window_size]), np.full(n, float(np.nanmean(y[startcycle:startcycle+window_size]))), y - np.nanmean(y[startcycle:startcycle+window_size])
    else:
        return y - np.nanmean(y[startcycle:startcycle+window_size]), -1, startcycle, startcycle+window_size, np.nanmean(y[startcycle:startcycle+window_size])

def calc_ct_func(x, y, thr):
    return calculate_ct(x, y,threshold=thr,startpoint=startcycle_to_use,use_4pl=False,return_std=False)

    
fam_raw_ch = 'X1_M1'   # pre: FAM raw
rox_raw_ch = 'X4_M4'   # pre: ROX raw
FAM_post_ch = 'FAM'    # post: FAM multicomponent
ROX_post_ch = 'ROX'





# ---- Global Matplotlib sizing (tweak as you like) ----
# FIGSIZE = (7,5)   # inches
DPI = 110              # pixels per inch

mpl.rcParams.update({
    
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

# === Here we GO! ===
st.set_page_config(
    page_title="QuantStudio HomeBrew Analysis Tool",
    page_icon="assets/SpearLogo.png",    # or "🧪" or a URL
    layout="wide",
)


col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("assets/thumbnail_image001.png")
with col_title:
    st.title("QuantStudio HomeBrew Analysis Tool")
    st.caption(f"Version {version} • Contact: Jiachong Chu")





# st.subheader("Data Upload")
st.markdown("Please import the exported file from Design and Analysis (combined in one: xlsx; separated file: csv)")
uploaded_files = []
uploaded_files = st.file_uploader("Design and Analysis exported files",type=["csv", "xlsx", "xls"],accept_multiple_files=True)


def make_plate_df(plate_format: str) -> pd.DataFrame:
    """Create a boolean DataFrame for the plate shape."""
    if plate_format.startswith("384"):
        rows = list(string.ascii_uppercase[:16])  # A–P
        cols = list(range(1, 24+1))              # 1–24
    else:  # 96
        rows = list(string.ascii_uppercase[:8])  # A–H
        cols = list(range(1, 12+1))              # 1–12
    df = pd.DataFrame(False, index=rows, columns=cols)
    return df

def wells_from_df(df: pd.DataFrame) -> list[str]:
    """Extract selected wells from a boolean DataFrame in A1 style."""
    wells = []
    for r in df.index:
        for c in df.columns:
            if bool(df.loc[r, c]):
                wells.append(f"{r}{c}")
    return wells

# --- UI: Plate format ---
plate_format = st.radio(
    "Plate format",
    ["384-well (16×24)", "96-well (8×12)"],
    horizontal=True,
)

# Initialize plate grid
grid_key = f"well_grid_{'384' if plate_format.startswith('384') else '96'}"
if grid_key not in st.session_state:
    st.session_state[grid_key] = make_plate_df(plate_format)

plate_df = st.session_state[grid_key]

def full_plate_select(df: pd.DataFrame,
                      row_rule: str = "All rows",
                      col_rule: str = "All cols",
                      select: bool = True) -> pd.DataFrame:
    """
    Apply a bulk selection to the entire plate with row/col filters.
    - df: boolean DataFrame; index are row letters (A..), columns are ints (1..)
    - row_rule: "All rows" | "Odd rows only" | "Even rows only"
    - col_rule: "All cols" | "Odd cols only" | "Even cols only"
    - select: True -> set checked; False -> uncheck
    Returns a modified copy of df.
    """
    out = df.copy()

    def row_ok(r_label: str) -> bool:
        # 1-based row position: A=1, B=2, ...
        pos = df.index.get_loc(r_label) + 1
        if row_rule == "Odd rows only":
            return (pos % 2) == 1
        if row_rule == "Even rows only":
            return (pos % 2) == 0
        return True  # All rows

    def col_ok(c_label) -> bool:
        c = int(c_label)
        if col_rule == "Odd cols only":
            return (c % 2) == 1
        if col_rule == "Even cols only":
            return (c % 2) == 0
        return True  # All cols

    for r in df.index:
        if not row_ok(r):
            continue
        # vectorized assignment for allowed columns
        allowed_cols = [c for c in df.columns if col_ok(c)]
        if allowed_cols:
            out.loc[r, allowed_cols] = select
    return out


# Auto-select the whole plate when a new file is uploaded
if uploaded_files:
    _upload_key = tuple(f.name for f in uploaded_files)
    if st.session_state.get('_last_upload_key') != _upload_key:
        st.session_state['_last_upload_key'] = _upload_key
        st.session_state[grid_key] = full_plate_select(make_plate_df(plate_format))
        plate_df = st.session_state[grid_key]

# ---- UI controls for full-plate selection ----
cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])

with cc1:
    row_rule = st.selectbox(
        "Row rule",
        ["All rows", "Odd rows only", "Even rows only"],
        index=0,
        help="Choose which rows to affect when bulk selecting."
    )
with cc2:
    col_rule = st.selectbox(
        "Column rule",
        ["All cols", "Odd cols only", "Even cols only"],
        index=0,
        help="Choose which columns to affect when bulk selecting."
    )
with cc3:
    mode = st.radio("Mode", ["Select", "Deselect"], horizontal=True)
with cc4:
    applied = st.button("Apply full-plate selection", use_container_width=True)

if applied:
    st.session_state[grid_key] = full_plate_select(
        st.session_state[grid_key],
        row_rule=row_rule,
        col_rule=col_rule,
        select=(mode == "Select")
    )
    st.success(f"Applied: {mode.lower()} | {row_rule} × {col_rule}")
    plate_df = st.session_state[grid_key]
# --- Row/Column selectors ---
row_choice = st.multiselect("Select entire rows", plate_df.index)
col_choice = st.multiselect("Select entire columns", plate_df.columns)

if st.button("Apply row/col selection"):
    for r in row_choice:
        plate_df.loc[r, :] = True
    for c in col_choice:
        plate_df.loc[:, c] = True
    st.session_state[grid_key] = plate_df

# --- Grid editor ---
column_config = {c: st.column_config.CheckboxColumn() for c in plate_df.columns}
edited_grid = st.data_editor(
    plate_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=False,
    column_config=column_config,
    key=f"editor_{grid_key}",
)

st.session_state[grid_key] = edited_grid
plate_df = st.session_state[grid_key]
# --- Extract selected wells ---
selected_wells = wells_from_df(edited_grid)

with st.expander(f"Selected wells ({len(selected_wells)})", expanded=False):
    st.write(", ".join(selected_wells) if selected_wells else "None")

st.info(f"{len(selected_wells)} wells selected.")

cycles = np.arange(1,41)

# ===== qPOS (reference) well selector =====
def _well_col(df: pd.DataFrame) -> str:
    """Return the column name used for wells in Results (handles 'Well' vs 'Well Position')."""
    for cand in ("Well", "Well Position"):
        if cand in df.columns:
            return cand
    raise KeyError("Neither 'Well' nor 'Well Position' found in Results dataframe.")

def plate_info_auto(plate_format: str) -> tuple[list[str], float]:
    """
    UI block: allow user to pick multiple qPOS wells.
    Defaults:
      - 384 well: O24, P24
      - 96 well:  H6, H12
    Returns (qpos_wells, ref_cq) where ref_cq is mean/median of selected qPOS Cq.
    """

    # Build all wells from plate format
    if plate_format.startswith("384"):
        rows = list(string.ascii_uppercase[:16])  # A..P
        cols = list(range(1, 24 + 1))            # 1..24
        default_qpos = ["O24", "P24"]
    else:
        rows = list(string.ascii_uppercase[:8])   # A..H
        cols = list(range(1, 12 + 1))            # 1..12
        default_qpos = ["H6", "H12"]

    all_wells = [f"{r}{c}" for r in rows for c in cols]
    default_qpos = [w for w in default_qpos if w in all_wells]
    if not default_qpos:
        default_qpos = [all_wells[-1]]  # very defensive fallback
    # Multiselect qPOS wells
    qpos_wells = st.multiselect(
        "qPOS (reference) wells",
        options=all_wells,
        default=default_qpos,
        help="Pick one or more wells to serve as qPOS."
    )

    if len(qpos_wells) == 0:
        st.warning("Please select at least one qPOS well.")
        return [], np.nan

    return qpos_wells, all_wells

results_file = None    
raw_file = None
combined_file = None

for f in uploaded_files:  # from st.file_uploader
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        # candidate for combined export
        combined_file = f
    elif "raw data" in name:
        raw_file = f

if combined_file is not None:
    try:
        df = _load_combined_xlsx(combined_file)
        runname = _guess_runname(combined_file.name)
        st.success(f"Loaded combined workbook: {combined_file.name}")
    except Exception as e:
        st.error(f"Failed to read {combined_file.name}: {e}")
        st.stop()
else:
    if not raw_file:
        st.error("No Raw Data file found (expected *_Raw Data_*.csv)")
        st.stop()

    df      = load_quantstudio(raw_file)
    runname = Path(raw_file.name).stem.split('_Raw Data', 1)[0]
    st.success("All key files loaded successfully!")



qPOSwells, all_wells = plate_info_auto(plate_format)


target_ct = st.number_input("target ct", value=25.0, step=0.1,key = 'target ct value', help="Desired average Ct for qPOS")


c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    window_size_to_use = st.number_input("background screening window", value=8, step=1)
with c2:
    StepIndY_to_use = st.number_input("shannon limit", value=50, step=5)
with c3:
    startcycle_to_use = st.number_input("skipped cycles", value=5, step=1)

mask = edited_grid.astype(bool)
selected_rows = [r for r in mask.index if mask.loc[r].any()]
selected_cols = [c for c in mask.columns if mask[c].any()]

selected_wells = []

FRC = {w: np.nan for w in all_wells}
FRC_homebrew = {w: np.nan for w in all_wells}

residue,mean,res_std = spr_QSqpcr_background_dY_residue(df, selected_wells, startcycle = startcycle_to_use, window_size = window_size_to_use, StepIndY = StepIndY_to_use)

intercept_all = {}
startpoint_all = {}
bg_subtracked_y = {}
start_i = {}
end_i = {}
_rox_means_ic = {}


for i, r in enumerate(selected_rows):
    for j, c in enumerate(selected_cols):
        if not bool(mask.loc[r, c]):
            continue  # not selected, leave NaN
        well = f"{r}{c}"
        selected_wells.append(well)

        sub = df[df["Well Position"].astype(str) == str(well)]
        rox_y = sub[rox_raw_ch].astype(float).to_numpy()
        fam_y = sub[fam_raw_ch].astype(float).to_numpy()
        _rox_means_ic[well] = np.nanmean(rox_y)
        y_norm = fam_y/rox_y
        y_bg,start_point,start,end,intercept = spr_QSqpcr_background_dY_v5(res_std, y_norm, sigma_mult=2, min_points=4, max_refit_iter = 3, startcycle = startcycle_to_use, window_size = window_size_to_use, StepIndY = StepIndY_to_use,returnbase = False)
        if start_point == -1:
            print (f'{well}')
        start_i[well] = start
        end_i[well] = end
        startpoint_all[well] = start_point
        intercept_all[well] = intercept

for well in qPOSwells:
    sub = df[df["Well Position"].astype(str) == str(well)]
    rox_y = sub[rox_raw_ch].astype(float).to_numpy()
    fam_y = sub[fam_raw_ch].astype(float).to_numpy()
    _rox_means_ic[well] = np.nanmean(rox_y)
    y_norm = fam_y/rox_y
    y_bg,start_point,start,end,intercept = spr_QSqpcr_background_dY_v5(res_std, y_norm, sigma_mult=2, min_points=4, max_refit_iter = 3, startcycle = startcycle_to_use, window_size = window_size_to_use, StepIndY = StepIndY_to_use,returnbase = False)
    if start_point == -1:
        print (f'{well}')
    start_i[well] = start
    end_i[well] = end
    startpoint_all[well] = start_point
    intercept_all[well] = intercept

# Exclude low-ROX buffer/empty wells (< 20% of plate max) from intercept
# estimation. Even-column buffer wells (~13% of max ROX) would otherwise skew
# the plate reference intercept on full-plate runs.
_max_rox_ic   = max(_rox_means_ic.values(), default=1.0)
_rox_floor_ic = _max_rox_ic * 0.20
_wells_ic = [w for w in intercept_all if _rox_means_ic.get(w, 0) >= _rox_floor_ic]
x = np.array([intercept_all[w] for w in _wells_ic], dtype=float).ravel()
Q1,Q3 = np.percentile(x, (25,75))
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR

_ic_mask = np.isfinite(x) & (x<upper) & (x>lower)
idxs = np.where(_ic_mask)[0]

median_intercept = np.median(x[idxs])

# print (median_intercept)
def _get_fit_window(well, startcycle):
    liftoff = startpoint_all[well]
    bs = start_i[well]
    be = end_i[well]
    if 0 < liftoff <= 15 and liftoff - startcycle >= 3:
        return startcycle, liftoff
    elif bs - startcycle >= 3:
        return startcycle, bs
    else:
        return bs, be

ref_y_bg = []
for refwell in qPOSwells:
    sub = df[df["Well Position"].astype(str) == str(refwell)]
    fam_y = sub[fam_raw_ch].astype(float)
    rox_y = sub[rox_raw_ch].astype(float)
    fit_start, fit_end = _get_fit_window(refwell, startcycle_to_use)
    y_norm = SPR_fitbackground(median_intercept,fam_y,rox_y,fit_start,fit_end,cycles)
    y_bg,start_point,start,end,intercept = spr_QSqpcr_background_dY_v5(res_std, y_norm, sigma_mult=2, min_points=4, max_refit_iter = 3, startcycle = startcycle_to_use, window_size = window_size_to_use, StepIndY = StepIndY_to_use)
    ref_y_bg.append(y_bg)
    
thr_opt, ct_HB_ref_avg, ct_HB_ref = find_threshold_for_target_ct_multi(x=cycles,ybg_list=ref_y_bg,target_ct=target_ct,calculate_ct_func=calc_ct_func,ct_tol=0.01)

threshold_Ct = thr_opt
Ct_homebrew = {w: np.nan for w in all_wells}
baseline_start = {w: np.nan for w in all_wells}
baseline_end = {w: np.nan for w in all_wells}

# Plate-level empty-well detection: wells with both FAM and ROX below 5% of
# plate maximum are considered empty (no reaction mix loaded) → Ct = NaN.
_mean_fam = {w: df[df["Well Position"].astype(str)==w][fam_raw_ch].astype(float).mean()
             for w in selected_wells}
_mean_rox = {w: df[df["Well Position"].astype(str)==w][rox_raw_ch].astype(float).mean()
             for w in selected_wells}
_max_fam = max(_mean_fam.values(), default=1.0)
_max_rox = max(_mean_rox.values(), default=1.0)
_fam_floor = _max_fam * 0.05
_rox_floor = _max_rox * 0.05

fig, ax = plt.subplots(figsize=(6,4))
for well in selected_wells:
    sub = df[df["Well Position"].astype(str) == str(well)]
    fam_y = sub[fam_raw_ch].astype(float)
    rox_y = sub[rox_raw_ch].astype(float)

    # Skip empty wells
    if _mean_fam[well] < _fam_floor and _mean_rox[well] < _rox_floor:
        baseline_start[well] = np.nan
        baseline_end[well] = np.nan
        continue

    fit_start, fit_end = _get_fit_window(well, startcycle_to_use)
    y_norm = SPR_fitbackground(median_intercept,fam_y,rox_y,fit_start,fit_end,cycles)
    y_bg,start_point,start,end,_ = spr_QSqpcr_background_dY_v5(res_std, y_norm, sigma_mult=2, min_points=4, max_refit_iter = 3, startcycle = startcycle_to_use, window_size = window_size_to_use, StepIndY = StepIndY_to_use)
    ct, _ = calculate_ct(cycles, y_bg,  threshold=threshold_Ct, return_std=True,use_4pl=False,startpoint = start_point)
    Ct_homebrew[well] = ct
    baseline_start[well] = start
    baseline_end[well] = end
    ax.plot(cycles, y_bg, linewidth=1, label=f'{well}')
            # print (f'{well}, {ct:.2f}')
ax.set_xlabel('cycles')
ax.set_ylabel('X1_M1/X4_M4 normlaized and background subtracted')
ax.set_yscale ('log')
ax.hlines(y=threshold_Ct, xmin=cycles.min(), xmax=cycles.max(), colors='black', linestyles='--')
plt.grid(True, alpha=0.3)
# plt.ylim(1e-1,10)
st.pyplot(fig, use_container_width=False)



# === EXPORT: QS-style single-sheet "Results" with header + baseline start/end ===
from datetime import datetime

def _list_plate_wells_for_export(plate_format: str) -> list[str]:
    if plate_format.startswith("384"):
        rows = list(string.ascii_uppercase[:16])  # A..P
        cols = list(range(1, 24+1))              # 1..24
    else:
        rows = list(string.ascii_uppercase[:8])   # A..H
        cols = list(range(1, 12+1))              # 1..12
    return [f"{r}{c}" for r in rows for c in cols]

def _well_numeric_index(all_wells: list[str]) -> dict[str, int]:
    # 1-based like QS "Well" column
    return {w: i + 1 for i, w in enumerate(all_wells)}

def _qs_metadata_block(runname: str) -> list[tuple[str, str]]:
    now = datetime.now()
    return [
        ("File Name", runname),
        ("Comment", ""),
        ("Operator", "DEFAULT"),
        ("Barcode", ""),
        ("Instrument", "QuantStudio™ 5 System"),
        ("Block Type", "384-Well Block" if plate_format.startswith("384") else "96-Well Block"),
        ("Instrument", "QS5"),
        ("Instrument SN", ""),
        ("Heated Cover Serial Number", ""),
        ("Block serial", ""),
        ("Run Start", now.strftime("%Y-%m-%d %I:%M:%S %p EDT")),
        ("Run End",   now.strftime("%Y-%m-%d %I:%M:%S %p EDT")),
        ("Run Duration", ""),
        ("Sample Volume", "20.0"),
        ("Cover Temp", "105.0"),
        ("Passive Ref", "ROX"),
        ("PCR Stage", "Stage 2 Step 2"),
        ("Quantification", "CT"),
        ("Analysis Date", now.strftime("%Y-%m-%d %I:%M:%S %p EDT")),
        ("Software", "Design & Analysis Software v2.8.0"),
        ("Plugin Name", "Primary Analysis v1.8.1, Standard Curve v1.8.0"),
        ("Reduce Cq", "No (Default)"),
        ("Exported", now.strftime("%Y-%m-%d %I:%M:%S %p EDT")),
    ]

def _qs_results_table(
    plate_format: str,
    ct_map: dict[str, float],
    baseline_start: dict[str, float],
    baseline_end: dict[str, float],
    threshold_val: float,
    selected_wells: list[str],
) -> pd.DataFrame:
    all_wells = _list_plate_wells_for_export(plate_format)
    idx_map = _well_numeric_index(all_wells)

    # Limit to selected wells; if you want all, replace with: wells = all_wells
    wells = [w for w in all_wells if w in selected_wells]

    rows_out = []
    for w in wells:
        rows_out.append({
            "Well": idx_map.get(w, np.nan),
            "Well Position": w,
            "Omit": "FALSE",
            "Sample": "Sample 1",
            "Target": "Target 1",
            "Task": "UNKNOWN",
            "Reporter": "FAM",
            "Quencher": "NFQ-MGB",
            "Amp Status": "AMP",
            "Amp Score": np.nan,
            "Curve Quality": np.nan,
            "Result": "AMP",
            "Cq": ct_map.get(w, np.nan),
            "Cq Conf": np.nan,
            "Cq Mean": ct_map.get(w, np.nan),
            "Cq SD": np.nan,
            "Auto Th": "FALSE",
            "Thresh": float(threshold_val) if np.isfinite(threshold_val) else np.nan,
            "Auto B": "TRUE",
            "Baseline Start": baseline_start.get(w, np.nan),
            "Baseline End": baseline_end.get(w, np.nan),
        })

    cols = [
        "Well", "Well Position", "Omit", "Sample", "Target", "Task",
        "Reporter", "Quencher", "Amp Status", "Amp Score", "Curve Quality",
        "Result Quality Issues", "Cq", "Cq Confidence", "Cq Mean", "Cq SD",
        "Auto Threshold", "Threshold", "Auto Baseline", "Baseline Start", "Baseline End"
    ]
    df_out = pd.DataFrame(rows_out, columns=cols)
    with np.errstate(invalid="ignore"):
        df_out = df_out.sort_values(by=["Well", "Well Position"], kind="mergesort")
    return df_out

def export_qs_like_results_single_sheet(
    runname: str,
    plate_format: str,
    ct_map: dict[str, float],
    baseline_start: dict[str, float],
    baseline_end: dict[str, float],
    threshold_val: float,
    selected_wells: list[str],
):
    # 1) Build metadata + table
    meta = _qs_metadata_block(runname)
    table = _qs_results_table(
        plate_format=plate_format,
        ct_map=ct_map,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        threshold_val=threshold_val,
        selected_wells=selected_wells,
    )

    # 2) Write to one Excel sheet named "Results"
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            # header block (2 columns, no headers)
            pd.DataFrame(meta, columns=["", ""]).to_excel(
                writer, sheet_name="Results", index=False, header=False, startrow=0, startcol=0
            )
            # table starting at row 26 (zero-indexed 25)
            table.to_excel(writer, sheet_name="Results", index=False, startrow=24, startcol=0)
    except Exception:
        # fall back to openpyxl if xlsxwriter isn't available
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame(meta, columns=["", ""]).to_excel(
                writer, sheet_name="Results", index=False, header=False, startrow=0, startcol=0
            )
            table.to_excel(writer, sheet_name="Results", index=False, startrow=24, startcol=0)

    buf.seek(0)
    export_name = f"{runname}_Results_SpearBio_HomeBrew.xlsx"
    st.download_button(
        "⬇️ Download QS-style Results (HomeBrew Cq)",
        data=buf,
        file_name=export_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ---- Call it:
export_qs_like_results_single_sheet(
    runname=runname,
    plate_format=plate_format,
    ct_map=Ct_homebrew,
    baseline_start=baseline_start,
    baseline_end=baseline_end,
    threshold_val=threshold_Ct,
    selected_wells=selected_wells,
)
