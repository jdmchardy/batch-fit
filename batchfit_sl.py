
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import importlib.util
import zipfile, os
from typing import List, Tuple, Dict

# ---------------------------
# Load the user's functions from batchfit_core.py dynamically
# ---------------------------
spec = importlib.util.spec_from_file_location("batchfit_mod", "batchfit_core.py")
batchfit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batchfit_mod)

# Expose the functions we need
background_subtraction = batchfit_mod.background_subtraction
find_peaks_above_threshold = batchfit_mod.find_peaks_above_threshold
multi_gaussian = batchfit_mod.multi_gaussian
multi_pseudoVoigt = batchfit_mod.multi_pseudoVoigt
generate_peak_dataframe = batchfit_mod.generate_peak_dataframe
fit_peak_to_data = batchfit_mod.fit_peak_to_data

# ---------------------------
# Utility helpers
# ---------------------------

# Simple wrapper to mimic UploadedFile
class UploadedFileWrapper:
    def __init__(self, file_path):
        self.name = os.path.basename(file_path)
        self._file_path = file_path
    def read(self):
        with open(self._file_path, "rb") as f:
            return f.read()
    def seek(self):
        """Return a seekable file-like object"""
        return io.BytesIO(self.read())
            
def try_load_xy(file) -> Tuple[np.ndarray, np.ndarray]:
    """Attempt to load a two-column (x,y) dataset from an uploaded file-like object."""
    name = file.name
    data = None
    content = file.read()
    # Reset pointer for any subsequent reads elsewhere
    file.seek(0)
    # Try common formats
    for kwargs in [
        {"delimiter": None},              # whitespace
        {"delimiter": ","},
        {"delimiter": "\t"},
        {"delimiter": ";"},
    ]:
        try:
            data = np.loadtxt(io.BytesIO(content), **kwargs)
            if data.ndim == 1 and data.size > 2:
                # sometimes a single row — reshape to (n,2) if even length
                if data.size % 2 == 0:
                    data = data.reshape(-1, 2)
                else:
                    raise ValueError("Could not infer 2 columns from 1D data")
            if data.shape[1] >= 2:
                x = data[:, 0]
                y = data[:, 1]
                return x, y
        except Exception:
            pass
    # Fallback via pandas parser
    try:
        df = pd.read_csv(io.BytesIO(content))
        # pick first two numeric columns
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) >= 2:
            return df[num_cols[0]].to_numpy(), df[num_cols[1]].to_numpy()
    except Exception:
        pass
    raise ValueError(f"Could not parse two columns from file: {name}")

def plot_signal(x, y, title='', peaks=None, background=None, x_range_ROI=None, x_range_view=None,
                peaks_x=None, peaks_y=None, fit_x=None, fit_y=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st

    # Create figure with two rows: main plot + residuals
    fig, (ax, ax_resid) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    # --- Main plot ---
    ax.plot(x, y, lw=0.8, label="Data", color="black")
    if background is not None:
        ax.plot(x, background, lw=0.8, linestyle="--", label="Background")
    if x_range_ROI is not None:
        ax.axvspan(x_range_ROI[0], x_range_ROI[1], alpha=0.1, label="Fit window")
    if peaks is not None and len(peaks) > 0:
        ax.scatter(peaks, np.interp(peaks, x, y), s=20, marker="x", label="Detected peaks")
    if (fit_x is not None) and (fit_y is not None):
        ax.plot(fit_x, fit_y, lw=0.8, ls="dashed", color="red", label="Peak fit")

    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title)
    ax.legend(loc="best")

    # --- Residuals plot ---
    if (fit_x is not None) and (fit_y is not None):
        fit_interp = np.interp(x, fit_x, fit_y)
        residuals = y - fit_interp
        ax_resid.plot(x, residuals, lw=0.8, color="blue")
        ax_resid.axhline(0, color="gray", lw=0.8, ls="--")
        ax_resid.set_ylabel("Residuals")
    else:
        ax_resid.text(0.5, 0.5, "No fit → no residuals", ha="center", va="center",
                      transform=ax_resid.transAxes, fontsize=9, color="gray")
        ax_resid.set_ylabel("Residuals")

    ax_resid.set_xlabel("2θ (deg)")

    # --- X range and dynamic Y range scaling ---
    if x_range_view is not None:
        ax.set_xlim(x_range_view[0], x_range_view[1])

        # mask data to only what's inside x_range_view
        mask = (x >= x_range_view[0]) & (x <= x_range_view[1])
        if np.any(mask):
            # Main plot y limits
            y_min, y_max = np.min(y[mask]), np.max(y[mask])
            ax.set_ylim(y_min - 0.05*(y_max - y_min), y_max + 0.05*(y_max - y_min))

            # Residuals y limits
            if (fit_x is not None) and (fit_y is not None):
                resid_mask = (x >= x_range_view[0]) & (x <= x_range_view[1])
                resid_vals = residuals[resid_mask]
                if resid_vals.size > 0:
                    rmin, rmax = np.min(resid_vals), np.max(resid_vals)
                    ax_resid.set_ylim(rmin - 0.05*(rmax - rmin), rmax + 0.05*(rmax - rmin))

    plt.tight_layout()
    st.pyplot(fig)

def plot_intensity_map(xs: List[np.ndarray], ys: List[np.ndarray], overlay_centers: Dict[int, np.ndarray] = None):
    # Interpolate/stack onto a common x if necessary (assume xs are identical; if not, interpolate to first grid)
    x0 = xs[0]
    Y = []
    for x, y in zip(xs, ys):
        if np.array_equal(x, x0):
            Y.append(y)
        else:
            Y.append(np.interp(x0, x, y))
    data_matrix = np.vstack(Y)
    vmin, vmax = np.percentile(data_matrix, [25, 99])
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(data_matrix, aspect="auto", extent=[x0[0], x0[-1], 0, len(ys)], origin="lower", vmin=vmin, vmax=vmax, cmap="gray")
    ax.set_xlabel("X")
    ax.set_ylabel("Dataset #")
    fig.colorbar(im, ax=ax, label="Intensity")
    if overlay_centers:
        for idx, centers in overlay_centers.items():
            if centers is None or len(centers)==0: 
                continue
            yline = np.full_like(centers, idx+0.5, dtype=float)
            ax.scatter(centers, yline, s=8)
    ax.set_title("Intensity map (stacked datasets)")
    st.pyplot(fig)

def peak_params_to_dataframe(fit_results: Dict[int, dict]) -> pd.DataFrame:
    """Create a tidy table of peak parameters from our fit_results dict."""
    # Reuse the user's helper if available; otherwise build our own
    try:
        return generate_peak_dataframe(fit_results, None)
    except Exception:
        rows = []
        for idx, fr in fit_results.items():
            row = {"index": idx, "Filename": fr.get("Filename", f"dataset_{idx}")}
            centers = fr.get("peak_centers", [])
            params = fr.get("params", [])
            func = fr.get("peak_function", "")
            row["n_peaks"] = len(centers) if centers is not None else 0
            row["func"] = func
            if params is not None:
                for j, p in enumerate(np.array(params).reshape(-1, 4)):
                    amp, mu, sigma, gl = p
                    row[f"p{j+1}_amp"] = amp
                    row[f"p{j+1}_mu"] = mu
                    row[f"p{j+1}_sigma"] = sigma
                    row[f"p{j+1}_gl"] = gl
            rows.append(row)
        return pd.DataFrame(rows)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Batch Peak Fitting (Streamlit)", layout="wide")
st.title("Batch Peak Fitting")
st.caption("Converted from your original interactive matplotlib/Tkinter tool to Streamlit.")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Sidebar: upload and global controls
with st.sidebar:
    st.header("1) Upload zip file of .xys")
    uploaded_zip = st.file_uploader("Upload a ZIP of xy files (two columns: x, y)", 
                                    type=["zip"],
                                   key=f"uploader_{st.session_state.uploader_key}")

    uploaded_files = []
    if uploaded_zip:
        extract_dir = "uploaded_files"
        os.makedirs(extract_dir, exist_ok=True)
    
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
            
    
        for file_name in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, file_name)
            if os.path.isfile(file_path):
                uploaded_files.append(UploadedFileWrapper(file_path))
    
    if uploaded_zip:
        # Clear button
        if st.button("Clear uploaded files"):
            st.session_state.uploader_key += 1
            st.rerun()

    st.header("2) Peak model & background")
    peak_function = st.radio("Peak function", ["Gaussian", "pseudo-Voigt"], index=0)
    # important: the original code checks a global 'selected_function'
    batchfit_mod.selected_function = "Gaussian" if peak_function == "Gaussian" else "PseudoVoigt"
    # Background options
    poly_order = st.number_input("Background Chebyshev order (int)", min_value=0, max_value=12, value=6, step=1)
    threshold = st.number_input("Background intensity threshold (0-1 scale of max)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f")
    bkg_peak_threshold = st.number_input("Background peak threshold (fraction of max)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f")
    bkg_peak_prominence = st.number_input("Bkg peak prominence (fraction of max)", min_value=0.0, max_value=1.0, value=0.08, step=0.01, format="%.3f")
    bkg_peak_width_multiplier = st.number_input("Bkg exclusion width multiplier", min_value=0.0, max_value=10.0, value=1.5, step=0.1, format="%.2f")

    st.header("3) Peak detection")
    peak_width = st.number_input("Approx peak FWHM (same units as x)", min_value=0.0, value=0.1, step=0.01, format="%.3f")
    peak_prominence = st.number_input("Peak prominence (fraction of max)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.3f")
    max_peaks = st.number_input("Max peaks to fit per ROI", min_value=0, max_value=50, value=1, step=1)

    st.header("4) Options")
    background_enabled = st.checkbox("Show background-subtracted data (toggle view)", value=True)
    st.session_state["background_enabled"] = background_enabled

# Parse files
datasets = []
if uploaded_files:
    for f in uploaded_files:
        try:
            x, y = try_load_xy(f)
            datasets.append((f.name, x, y))
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")

if not datasets:
    st.info("Upload at least one data file to begin.")
    st.stop()

# Build common x-range for UI
x_global_min = float(min(x[0] for _, x, _ in datasets))
x_global_max = float(max(x[-1] for _, x, _ in datasets))
x_range_ROI = st.slider("Region of Interest (x-range)", min_value=x_global_min, max_value=x_global_max, value=(x_global_min, x_global_max))
x_range_view = st.slider("Plot range (x-range)", min_value=x_global_min, max_value=x_global_max, value=(x_global_min, x_global_max))

# Navigation
colA, colB, colC, colD = st.columns([1,1,1,2])
with colA:
    st.write(" ")
    current_index = st.number_input("Dataset index", min_value=0, max_value=len(datasets)-1, value=0, step=1)
with colB:
    if st.button("◀ Prev", use_container_width=True) and current_index > 0:
        current_index -= 1
with colC:
    if st.button("Next ▶", use_container_width=True) and current_index < len(datasets)-1:
        current_index += 1
with colD:
    batch_action = st.selectbox("Action", ["Preview only", "Fit current", "Fit all & export"], index=1)

st.session_state["current_index"] = current_index

# Run fit(s)
def run_single_fit(idx: int) -> dict:
    name, x, y = datasets[idx]
    # First compute background and corrected data for plotting regardless of fit
    y_corrected, background, bkg_non_x, bkg_non_y, bkg_peaks_x, bkg_peaks_y = background_subtraction(
        x=x, y=y, poly_order=poly_order, threshold=threshold, bkg_peak_prominence=bkg_peak_prominence,
        bkg_peak_width_multiplier=bkg_peak_width_multiplier, peak_distance=int(max(1, np.round(peak_width / max(np.diff(x))))) if len(x)>1 else 1
    )
    # Now perform the peak fit in the ROI
    x_fit, y_fit, background_all, background_fit, bkg_non_px, bkg_non_py, bkg_pkx, bkg_pky, y_fit_curve, params_opt, pcov, peak_centers, y_subtracted = fit_peak_to_data(
        x=x, y=y, x_range=x_range_ROI, max_peaks=int(max_peaks), threshold=threshold, bkg_peak_threshold=bkg_peak_threshold,
        bkg_peak_prominence=bkg_peak_prominence, bkg_peak_width_multiplier=bkg_peak_width_multiplier,
        peak_width=peak_width, peak_prominence=peak_prominence
    )
    return {
        "Filename": name,
        "x": x, "y": y,
        "background": background_all,
        "x_fit": x_fit, "y_fit": y_fit,
        "background_fit": background_fit,
        "bkg_non_peaks_x": bkg_non_px, "bkg_non_peaks_y": bkg_non_py,
        "bkg_peaks_x": bkg_pkx, "bkg_peaks_y": bkg_pky,
        "params": params_opt, "pcov": pcov,
        "peak_centers": peak_centers,
        "y_fit_curve": y_fit_curve,
        "y_subtracted": y_subtracted,
        "peak_function": peak_function,
        "groups": [],  # placeholder if user uses grouping in original tool
    }

fit_results: Dict[int, dict] = st.session_state.get("fit_results", {})

if batch_action == "Fit current":
    fit_results[current_index] = run_single_fit(current_index)
elif batch_action == "Fit all & export":
    for i in range(len(datasets)):
        fit_results[i] = run_single_fit(i)

st.session_state["fit_results"] = fit_results

# Layout: left plots, right tables/info
left, right = st.columns([3,2])

with left:
    name, x, y = datasets[current_index]
    # quick background to show
    y_corr, background, *_ = background_subtraction(
        x=x, y=y, poly_order=poly_order, threshold=threshold, bkg_peak_prominence=bkg_peak_prominence,
        bkg_peak_width_multiplier=bkg_peak_width_multiplier, peak_distance=int(max(1, np.round(peak_width / max(np.diff(x))))) if len(x)>1 else 1
    )
    if st.session_state.get("background_enabled", True):
        display_y = y - background
        bg_for_plot = None
    else:
        display_y = y
        bg_for_plot = background
    fr = fit_results.get(current_index)
    y_fit_curve = None
    x_fit_for_plot = None
    if fr is not None:
        x_fit_for_plot = fr["x_fit"]
        y_fit_curve = fr["y_fit_curve"]
    # Show signal plot
    plot_signal(
        x, display_y, title=f"{name}",
        background=bg_for_plot,
        x_range_ROI=x_range_ROI,
        x_range_view= x_range_view, 
        peaks=(fr["peak_centers"] if fr is not None else None),
        fit_x=x_fit_for_plot,
        fit_y=y_fit_curve
    )
    # Show intensity map if multiple datasets
    if len(datasets) > 1:
        overlay = {i: fr["peak_centers"] for i, fr in fit_results.items() if fr is not None}
        plot_intensity_map([x for _, x, _ in datasets], [y for _, _, y in datasets], overlay_centers=overlay)

with right:
    st.subheader("Fit parameters")
    if fit_results.get(current_index) is not None and fit_results[current_index].get("params") is not None:
        if peak_function == "Gaussian":
            params = np.array(fit_results[current_index]["params"]).reshape(-1, 3)
            df_params = pd.DataFrame(params, columns=["amp", "mu", "sigma"])
            df_params.index = [f"Peak {i+1}" for i in range(len(df_params))]
            st.dataframe(df_params, use_container_width=True)
        else:
            params = np.array(fit_results[current_index]["params"]).reshape(-1, 4)
            df_params = pd.DataFrame(params, columns=["amp", "mu", "sigma", "gl"])
            df_params.index = [f"Peak {i+1}" for i in range(len(df_params))]
            st.dataframe(df_params, use_container_width=True)
    else:
        st.info("No fit for current dataset yet. Click **Fit current**.")

    if fit_results:
        tidy = peak_params_to_dataframe(fit_results)
        st.markdown("---")
        st.subheader("Results table (all fits)")
        st.dataframe(tidy, use_container_width=True)
        # Provide CSV download
        csv = tidy.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="fit_results.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: adjust ROI and parameters, then click **Fit current** (or **Fit all & export**) to update results.")
