"""
app.py  -  Smart Image Enhancer Using Digital Image Processing
==============================================================
Streamlit user interface. All image-processing maths lives in
image_processing.py and frequency_filters.py; this file only builds the screen.

Run with:   streamlit run app.py
"""

import numpy as np
import streamlit as st

import explanations as ex
import frequency_filters as ff
import image_processing as ip
import utils

PAGES = [
    "1. Upload & Preview",
    "2. Histogram Analysis",
    "3. Image Enhancement",
    "4. Spatial Filtering & Noise",
    "5. Edge Detection & Sharpening",
    "6. Frequency-Domain Filtering",
    "7. Quality Metrics (MSE / PSNR)",
]


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
def key(name):
    """Widget keys contain a reset counter; pressing Reset bumps it, so every widget returns to its default."""
    return f"{name}_{st.session_state.get('reset_count', 0)}"


def do_reset():
    st.session_state["reset_count"] = st.session_state.get("reset_count", 0) + 1


def show_pair(original, processed, label="Processed image"):
    c1, c2 = st.columns(2)
    c1.subheader("Original")
    c1.image(original, use_container_width=True, clamp=True)
    c2.subheader(label)
    c2.image(processed, use_container_width=True, clamp=True)


def show_histograms(before, after, title_after="After"):
    c1, c2 = st.columns(2)
    c1.pyplot(utils.histogram_figure(before, "Histogram - before"))
    c2.pyplot(utils.histogram_figure(after, f"Histogram - {title_after.lower()}"))


def download_button(img, name, widget_key):
    st.download_button("⬇️ Download processed image (PNG)", data=utils.to_png_bytes(img),
                       file_name=f"{name}.png", mime="image/png", key=key(widget_key))


def explain(text, title="📖 How it works"):
    with st.expander(title, expanded=True):
        st.markdown(text)


def gray_notice(work):
    """Return a gray version of the working image, telling the user if a conversion happened."""
    if work.ndim == 3:
        st.info("This tool works on intensity values, so your colour image was converted to grayscale automatically.")
    return ip.to_gray(work)


def safe_run(func, *args, **kwargs):
    """Run an algorithm; show a friendly error instead of a crash if the parameters are invalid."""
    try:
        return func(*args, **kwargs)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        st.stop()


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_upload(base, work, meta):
    st.subheader("Upload & Preview")
    c1, c2 = st.columns(2)
    c1.markdown("**Original image (never modified)**")
    c1.image(base, use_container_width=True)
    c2.markdown("**Working image** (gray if the sidebar option is ticked)")
    c2.image(work, use_container_width=True, clamp=True)

    st.markdown("#### Image information")
    h, w = work.shape[:2]
    rows = {
        "File name": meta["filename"],
        "File format / mode": f"{meta['format'] or 'n/a'} / {meta['mode']}",
        "Original file size": utils.human_size(meta["file_size"]),
        "Dimensions used (width x height)": f"{w} x {h} pixels",
        "Colour channels in the file": meta["bands"],
        "Channels in the working image": 1 if work.ndim == 2 else 3,
        "Looks gray already?": "Yes" if utils.looks_grayscale(base) else "No",
    }
    if meta.get("resized_from"):
        rows["Resized for speed"] = f"from {meta['resized_from'][1]} x {meta['resized_from'][0]} (longest side limit {utils.MAX_SIDE}px)"
    st.table({"Property": list(rows.keys()), "Value": [str(v) for v in rows.values()]})
    download_button(work, "working_image", "dl_preview")


def page_histogram(base, work, meta):
    st.subheader("Histogram Analysis")
    explain(ex.HISTOGRAM, "📖 Why histograms matter")
    gray = ip.to_gray(work)
    st.markdown(utils.describe_histogram(work))

    if work.ndim == 3:
        t1, t2 = st.tabs(["RGB channel histograms", "Grayscale intensity histogram"])
        t1.pyplot(utils.histogram_figure(work, "RGB histograms"))
        t2.pyplot(utils.histogram_figure(gray, "Gray-level histogram"))
    else:
        st.pyplot(utils.histogram_figure(gray, "Gray-level histogram"))

    st.markdown("#### Before / after a processing step")
    choice = st.selectbox("Apply a quick enhancement and compare histograms",
                          ["Histogram equalization", "Contrast stretching", "Gamma 0.5 (brighten)",
                           "Gamma 2.0 (darken)", "Image negative"], key=key("hist_choice"))
    funcs = {
        "Histogram equalization": ip.histogram_equalization,
        "Contrast stretching": ip.contrast_stretch,
        "Gamma 0.5 (brighten)": lambda im: ip.gamma_transform(im, 0.5),
        "Gamma 2.0 (darken)": lambda im: ip.gamma_transform(im, 2.0),
        "Image negative": ip.negative,
    }
    out = funcs[choice](work)
    show_pair(work, out, choice)
    show_histograms(work, out, choice)
    st.caption(utils.describe_histogram(out))


def page_enhancement(base, work, meta):
    st.subheader("Image Enhancement")
    tech = st.selectbox("Technique", list(ex.ENHANCEMENT.keys()), key=key("enh_tech"))
    show_hist = True

    if ip.is_constant(work) and tech in ("Contrast stretching", "Histogram equalization"):
        st.warning("This image has a single intensity value (constant image), so there is no contrast to stretch or "
                   "equalise. The output will be identical to the input.")

    if tech == "Image negative":
        out = ip.negative(work)
    elif tech == "Log transformation":
        c_scale = st.slider("c scale (1.0 = maps 255 -> 255)", 0.1, 2.0, 1.0, 0.05, key=key("log_c"))
        out = safe_run(ip.log_transform, work, c_scale)
    elif tech == "Gamma (power-law) transformation":
        gamma = st.slider("Gamma (γ)", 0.1, 5.0, 1.0, 0.05, key=key("gamma"))
        out = safe_run(ip.gamma_transform, work, gamma)
    elif tech == "Contrast stretching":
        c1, c2 = st.columns(2)
        low = c1.slider("Lower percentile (0 = true minimum)", 0.0, 49.0, 0.0, 0.5, key=key("cs_low"))
        high = c2.slider("Upper percentile (100 = true maximum)", 51.0, 100.0, 100.0, 0.5, key=key("cs_high"))
        out = safe_run(ip.contrast_stretch, work, low, high)
    elif tech == "Histogram equalization":
        if work.ndim == 3:
            st.caption("Colour image: only the luminance (Y) channel is equalised so colours are preserved.")
        out = ip.histogram_equalization(work)
    elif tech == "Gray-level slicing":
        gray = gray_notice(work)
        lo_hi = st.slider("Intensity range [A, B] to highlight", 0, 255, (100, 180), key=key("slice_rng"))
        c1, c2 = st.columns(2)
        highlight = c1.slider("Highlight value", 0, 255, 255, key=key("slice_hl"))
        keep = c2.radio("Rest of the image", ["Keep original values", "Set to black"], key=key("slice_bg")) == "Keep original values"
        work = gray
        out = safe_run(ip.gray_level_slicing, gray, lo_hi[0], lo_hi[1], highlight, keep)
    else:  # Bit-plane slicing
        gray = gray_notice(work)
        bit = st.select_slider("Bit plane (7 = most significant, 0 = least significant)", options=list(range(8)),
                               value=7, key=key("bit"))
        work = gray
        out = safe_run(ip.bit_plane_slice, gray, bit)
        show_hist = False

    show_pair(work, out, f"{tech}")
    if show_hist:
        if st.checkbox("Show histograms before / after", value=True, key=key("enh_hist")):
            show_histograms(work, out)
    if tech == "Bit-plane slicing":
        with st.expander("Show all 8 bit planes"):
            cols = st.columns(4)
            for b in range(7, -1, -1):
                cols[(7 - b) % 4].image(ip.bit_plane_slice(work, b), caption=f"Bit plane {b}", use_container_width=True)
        with st.expander("Rebuild the image from selected bit planes"):
            planes = st.multiselect("Planes to keep", list(range(7, -1, -1)), default=[7, 6, 5], key=key("bit_keep"))
            rebuilt = ip.bit_plane_reconstruct(work, planes)
            st.image(rebuilt, caption=f"Rebuilt from planes {sorted(planes, reverse=True)}", use_container_width=True)
    explain(ex.ENHANCEMENT[tech])
    download_button(out, tech.lower().replace(" ", "_"), "dl_enh")


def _alpha_input(k, widget_key):
    """Number input for d with validation against the kernel size."""
    max_d = ip.max_alpha_trim(k)
    d = st.number_input(f"Values discarded d (even, 0 to {max_d})", min_value=-100, max_value=1000,
                        value=min(2, max_d), step=1, key=key(f"{widget_key}_{k}"))
    return int(d)


def page_spatial(base, work, meta):
    st.subheader("Spatial Filtering & Noise Reduction")
    c1, c2, c3 = st.columns(3)
    add_noise = c1.checkbox("Add salt-and-pepper noise", key=key("noise_on"))
    amount = c2.slider("Fraction of pixels corrupted", 0.01, 0.50, 0.05, 0.01, key=key("noise_amt"))
    seed = int(c3.number_input("Noise seed (same seed = same noise)", 0, 10_000, 42, key=key("noise_seed")))
    noisy = ip.add_salt_and_pepper(work, amount, seed=seed) if add_noise else work

    c1, c2 = st.columns(2)
    name = c1.selectbox("Filter", ip.SPATIAL_FILTERS, key=key("sp_filter"))
    k = int(c2.radio("Kernel size", [3, 5, 7], horizontal=True, format_func=lambda v: f"{v} x {v}", key=key("sp_k")))
    d = 2
    if name == "Alpha-trimmed mean":
        d = _alpha_input(k, "alpha_d")
        if d < 0 or d > ip.max_alpha_trim(k) or d % 2:
            st.error(f"Invalid d = {d}. For a {k}x{k} kernel, d must be an EVEN number between 0 and {ip.max_alpha_trim(k)}.")
            st.stop()
    if name == "Weighted mean":
        kern = ip.weighted_mean_kernel(k)
        total = int(round(1 / kern[0, 0]))
        st.markdown("Weights used (they sum to 1 after dividing):")
        st.code(f"{np.rint(kern * total).astype(int)}\n\nall divided by {total}")

    out = safe_run(ip.apply_named_filter, noisy, name, k, d)

    if add_noise:
        a, b, c = st.columns(3)
        a.subheader("Original"); a.image(work, use_container_width=True, clamp=True)
        b.subheader("Noisy input"); b.image(noisy, use_container_width=True, clamp=True)
        c.subheader(f"{name} ({k}x{k})"); c.image(out, use_container_width=True, clamp=True)
        st.caption("Because you know the clean original, MSE/PSNR against it are meaningful here:")
        m1, m2 = st.columns(2)
        m1.metric("MSE: noisy vs original", f"{utils.mse(work, noisy):.1f}")
        m2.metric("MSE: filtered vs original", f"{utils.mse(work, out):.1f}")
    else:
        show_pair(work, out, f"{name} ({k}x{k})")
        st.info("Tip: tick **Add salt-and-pepper noise** to see which filters actually remove noise.")

    if st.checkbox("Compare ALL filters on this noisy image (needs noise ticked)", key=key("cmp_all")):
        if not add_noise:
            st.warning("Please tick 'Add salt-and-pepper noise' first, so that there is a clean reference image.")
        else:
            d_all = (k * k - 1) // 2          # always even; between mean (0) and median (k*k-1)
            names, results, rows = [], [], []
            for n in ip.SPATIAL_FILTERS:
                r = ip.apply_named_filter(noisy, n, k, d_all)
                names.append(n); results.append(r)
                rows.append({"Filter": n, "MSE vs original": round(utils.mse(work, r), 2),
                             "PSNR (dB)": round(utils.psnr(work, r), 2)})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(f"Alpha-trimmed mean used d = {d_all} here. Lower MSE = numerically closer to the clean original; "
                       "judge the pictures with your own eyes as well.")
            cols = st.columns(4)
            for i, (n, r) in enumerate(zip(names, results)):
                cols[i % 4].image(r, caption=n, use_container_width=True, clamp=True)

    explain(ex.SPATIAL[name])
    download_button(out, f"{name.lower().replace(' ', '_')}_{k}x{k}", "dl_sp")


def page_edges(base, work, meta):
    st.subheader("Edge Detection & Sharpening")
    method = st.selectbox("Method", list(ex.EDGE.keys()), key=key("edge_method"))
    source = work
    if method.startswith(("Sobel", "Laplacian (edges)")):
        source = gray_notice(work)

    if method == "Sobel X (horizontal gradient)":
        out = ip.sobel_x(source)
    elif method == "Sobel Y (vertical gradient)":
        out = ip.sobel_y(source)
    elif method == "Sobel gradient magnitude":
        out = ip.sobel_magnitude(source)
    elif method == "Laplacian (edges)":
        diag = st.radio("Neighbourhood", ["4-neighbour (centre -4)", "8-neighbour (centre -8)"], horizontal=True, key=key("lap_n")).startswith("8")
        out = ip.laplacian_edges(source, diag)
    elif method == "Laplacian sharpening":
        c1, c2 = st.columns(2)
        strength = c1.slider("Strength c", 0.0, 3.0, 1.0, 0.1, key=key("lap_c"))
        diag = c2.radio("Neighbourhood", ["4-neighbour", "8-neighbour"], horizontal=True, key=key("lap_n2")) == "8-neighbour"
        out = safe_run(ip.laplacian_sharpen, source, strength, diag)
    elif method == "Unsharp masking":
        c1, c2 = st.columns(2)
        sigma = c1.slider("Blur sigma (σ)", 0.5, 5.0, 1.5, 0.1, key=key("um_sigma"))
        amount = c2.slider("Amount k", 0.0, 1.0, 1.0, 0.05, key=key("um_k"))
        out = safe_run(ip.unsharp_mask, source, sigma, amount)
    else:  # High-boost
        c1, c2 = st.columns(2)
        sigma = c1.slider("Blur sigma (σ)", 0.5, 5.0, 1.5, 0.1, key=key("hb_sigma"))
        k = c2.slider("Boost factor k (must be > 1)", 1.0, 6.0, 2.5, 0.1, key=key("hb_k"))
        if k <= 1:
            st.error("High-boost needs k > 1. With k = 1 it is ordinary unsharp masking.")
            st.stop()
        out = safe_run(ip.high_boost, source, sigma, k)

    show_pair(source, out, method)
    if method.startswith("Sobel"):
        st.code(f"Kx =\n{ip.SOBEL_X.astype(int)}\n\nKy =\n{ip.SOBEL_Y.astype(int)}")
    explain(ex.EDGE[method])
    download_button(out, method.lower().replace(" ", "_").replace("(", "").replace(")", ""), "dl_edge")


def page_frequency(base, work, meta):
    st.subheader("Frequency-Domain Filtering (FFT)")
    explain(ex.FREQUENCY_GENERAL, "📖 Spatial vs frequency domain")

    c1, c2 = st.columns(2)
    family = c1.selectbox("Filter family", ["Ideal", "Butterworth", "Gaussian"], key=key("ff_family"))
    mode = c2.radio("Type", ["Low-pass", "High-pass"], horizontal=True, key=key("ff_mode"))
    mode_key = "lowpass" if mode == "Low-pass" else "highpass"

    h, w = work.shape[:2]
    max_cut = max(2, min(h, w))
    c1, c2 = st.columns(2)
    cutoff = c1.slider("Cutoff frequency D0 (distance from the spectrum centre)", 1, max_cut, max(1, max_cut // 8), key=key("ff_cut"))
    order = 2
    if family == "Butterworth":
        order = c2.slider("Butterworth order n", 1, 10, 2, key=key("ff_order"))
    c1, c2 = st.columns(2)
    pad = c1.checkbox("Zero-pad to 2M x 2N (avoids wrap-around errors)", value=True, key=key("ff_pad"))
    normalize = c2.checkbox("Min-max scale result for display", value=(mode_key == "highpass"), key=key(f"ff_norm_{mode_key}"))

    out = safe_run(ff.apply_frequency_filter, work, family.lower(), mode_key, cutoff, order, pad,
                   "normalize" if normalize else "clip")
    H = ff.transfer_function(ff.padded_shape(work.shape, pad), family.lower(), mode_key, cutoff, order)

    show_pair(work, out, f"{family} {mode.lower()} (D0 = {cutoff})")
    st.markdown("#### Frequency-domain view")
    a, b, c = st.columns(3)
    a.image(ff.fft_spectrum(work), caption="Spectrum of the input (log magnitude, centred)", use_container_width=True)
    b.image(ff.mask_to_image(H), caption="Filter H(u,v): white = kept, black = removed", use_container_width=True)
    c.image(ff.fft_spectrum(out), caption="Spectrum of the output", use_container_width=True)
    if work.ndim == 3:
        st.caption("Spectra are computed from the grayscale version. The filter itself is applied to each colour channel.")
    if mode_key == "highpass" and not normalize:
        st.warning("High-pass output has mean close to 0, so without scaling most of it clips to black.")

    err = ff.reconstruction_error(work)
    st.caption(f"Correctness check: IFFT(FFT(image)) differs from the image by at most {err:.2e} (floating-point rounding only).")
    explain(ex.FREQUENCY[family] + ("\n" + ex.FREQUENCY["highpass_note"] if mode_key == "highpass" else ""))
    download_button(out, f"{family.lower()}_{mode_key}", "dl_freq")


def page_metrics(base, work, meta):
    st.subheader("Image Quality Comparison (optional)")
    explain(ex.METRICS, "⚠️ Read this before using MSE and PSNR")
    mode = st.radio("How do you want to build a comparison?",
                    ["Quick experiment: add noise to my image, then filter it", "Compare my image with a second image I upload"],
                    key=key("m_mode"))
    if mode.startswith("Quick"):
        c1, c2, c3 = st.columns(3)
        amount = c1.slider("Noise amount", 0.01, 0.5, 0.05, 0.01, key=key("m_amt"))
        name = c2.selectbox("Filter", ip.SPATIAL_FILTERS, key=key("m_filter"))
        k = int(c3.radio("Kernel", [3, 5], horizontal=True, key=key("m_k")))
        noisy = ip.add_salt_and_pepper(work, amount, seed=7)
        out = safe_run(ip.apply_named_filter, noisy, name, k, (k * k - 1) // 2)
        a, b, c = st.columns(3)
        a.image(work, caption="Reference (clean)", use_container_width=True, clamp=True)
        b.image(noisy, caption="Noisy", use_container_width=True, clamp=True)
        c.image(out, caption=f"{name} {k}x{k}", use_container_width=True, clamp=True)
        st.table({"Comparison": ["Noisy vs reference", "Filtered vs reference"],
                  "MSE": [f"{utils.mse(work, noisy):.2f}", f"{utils.mse(work, out):.2f}"],
                  "PSNR (dB)": [f"{utils.psnr(work, noisy):.2f}", f"{utils.psnr(work, out):.2f}"]})
    else:
        up = st.file_uploader("Second image (same width and height as the working image)", type=["jpg", "jpeg", "png"], key=key("m_up"))
        if up is None:
            st.info("Upload a second image to compare with the current working image (the reference).")
            return
        try:
            second, _ = utils.load_image(up.getvalue(), up.name)
            second, _ = utils.limit_size(second)
        except ValueError as exc:
            st.error(str(exc))
            return
        if work.ndim == 2:
            second = ip.to_gray(second)
        if second.shape != work.shape:
            st.error(f"Size mismatch: reference is {work.shape[1]} x {work.shape[0]} but the second image is "
                     f"{second.shape[1]} x {second.shape[0]}. MSE and PSNR need two images of identical size.")
            return
        show_pair(work, second, "Second image")
        p = utils.psnr(work, second)
        m1, m2 = st.columns(2)
        m1.metric("MSE", f"{utils.mse(work, second):.3f}")
        m2.metric("PSNR (dB)", "∞ (identical)" if np.isinf(p) else f"{p:.2f}")
    st.caption("Reminder: these numbers measure numerical difference from the reference, not visual quality.")


PAGE_FUNCS = [page_upload, page_histogram, page_enhancement, page_spatial, page_edges, page_frequency, page_metrics]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smart Image Enhancer", page_icon="🖼️", layout="wide")
    st.title("🖼️ Smart Image Enhancer")
    st.caption("Digital Image Processing mini-project: histograms, enhancement, spatial filters, edge detection, FFT filtering.")

    st.sidebar.header("Image")
    uploaded = st.sidebar.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    use_sample = st.sidebar.checkbox("Use built-in sample image", key="use_sample") if uploaded is None else False
    to_gray_on = st.sidebar.checkbox("Convert to grayscale", key=key("to_gray"))
    st.sidebar.button("🔄 Reset (original image + default settings)", on_click=do_reset)
    page = st.sidebar.radio("Tool", PAGES)
    st.sidebar.caption("Your original image is never overwritten - every tool works on a copy.")

    base, meta = None, None
    if uploaded is not None:
        try:
            base, info = utils.load_image(uploaded.getvalue(), uploaded.name)
        except ValueError as exc:
            st.error(f"❌ {exc}")
            st.stop()
        if utils.looks_grayscale(base):          # a gray picture saved as colour: treat it as a 1-channel image
            base = np.ascontiguousarray(base[..., 0])
        full_shape = base.shape
        base, resized = utils.limit_size(base)
        meta = dict(info, resized_from=full_shape if resized else None)
    elif use_sample:
        base = utils.make_sample_image()
        meta = {"filename": "built-in sample", "format": "generated", "mode": "RGB", "bands": 3,
                "file_size": base.nbytes, "resized_from": None}

    if base is None:
        st.info("👈 Upload a JPG/PNG image in the sidebar (or tick **Use built-in sample image**) to begin.")
        st.markdown("""
**What you can do here**
1. Look at **histograms** to understand brightness and contrast.
2. **Enhance** images: negative, log, gamma, contrast stretching, equalisation, gray-level and bit-plane slicing.
3. **Remove noise** with seven spatial filters and compare them on simulated salt-and-pepper noise.
4. Find **edges** (Sobel, Laplacian) and **sharpen** (unsharp masking, high-boost).
5. Filter in the **frequency domain** with FFT (ideal, Butterworth, Gaussian; low-pass and high-pass).
6. Measure similarity with **MSE** and **PSNR**.
""")
        return

    work = ip.to_gray(base) if to_gray_on else base
    PAGE_FUNCS[PAGES.index(page)](base, work, meta)


if __name__ == "__main__":
    main()
