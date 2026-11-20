import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from api_client import check_api_health, predict_via_api
from cnn_model import load_model as load_classification_model
from config import API_URL, CLASS_LABELS, DEVICE, NUM_CLASSES, PREDICTION_THRESHOLD
from preprocessing import decode_channel_bytes, predict_from_channels

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Protein Atlas · Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --surface2:  #181c24;
    --border:    #1e2530;
    --accent:    #00e5ff;
    --accent2:   #7b61ff;
    --red:       #ff4b6e;
    --green:     #00e676;
    --blue:      #448aff;
    --yellow:    #ffd740;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --radius:    12px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

/* ── Hero title ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.6rem, 4vw, 2.8rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 60%, var(--red) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.7rem;
}

/* ── Channel upload labels ── */
.ch-red    { color: var(--red);    font-weight: 600; }
.ch-green  { color: var(--green);  font-weight: 600; }
.ch-blue   { color: var(--blue);   font-weight: 600; }
.ch-yellow { color: var(--yellow); font-weight: 600; }

/* ── Prediction pill ── */
.pill {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.2rem 0.15rem;
    border: 1px solid;
}
.pill-active {
    background: rgba(0,229,255,0.08);
    border-color: var(--accent);
    color: var(--accent);
}
.pill-inactive {
    background: rgba(107,114,128,0.06);
    border-color: var(--border);
    color: var(--muted);
}

/* ── Metric boxes ── */
.metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
.metric-box {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Threshold slider ── */
[data-testid="stSlider"] > div > div > div { background: var(--accent2) !important; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.6rem !important;
    width: 100%;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Progress / spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHANNEL_COLORS = {
    "Red (Microtubule)":   ("#ff4b6e", "red"),
    "Green (Protein)":     ("#00e676", "green"),
    "Blue (Nucleus)":      ("#448aff", "blue"),
    "Yellow (ER)":         ("#ffd740", "yellow"),
}

@st.cache_resource(show_spinner=False)
def load_model():
    return load_classification_model(device=DEVICE)

# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(uploaded_file):
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return decode_channel_bytes(uploaded_file.read())

def run_inference(channels, threshold, channel_files=None):
    if API_URL:
        return predict_via_api(API_URL, channel_files or {}, threshold)
    return predict_from_channels(model, channels, threshold=threshold, device=DEVICE)

def composite_rgb(r, g, b):
    """Merge R/G/B channels into a viewable RGB thumbnail."""
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb = (rgb / rgb.max() * 255).astype(np.uint8) if rgb.max() > 0 else rgb.astype(np.uint8)
    return rgb

def prob_bar_chart(probs, preds, threshold):
    fig, ax = plt.subplots(figsize=(7, 9))
    fig.patch.set_facecolor("#111318")
    ax.set_facecolor("#111318")

    colors = ["#00e5ff" if p == 1 else "#1e2530" for p in preds]
    bars   = ax.barh(CLASS_LABELS, probs, color=colors, edgecolor="none", height=0.6)
    ax.axvline(threshold, color="#7b61ff", linewidth=1.2, linestyle="--", alpha=0.8, label=f"Threshold {threshold:.2f}")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="#6b7280", fontsize=9, labelpad=8)
    ax.tick_params(colors="#a0aab4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2530")
    ax.invert_yaxis()

    legend = ax.legend(fontsize=8, facecolor="#181c24", edgecolor="#1e2530", labelcolor="#a0aab4")
    plt.tight_layout(pad=1.2)
    return fig

backend_name = "FastAPI" if API_URL else "local checkpoint"
try:
    if API_URL:
        model_ok = check_api_health(API_URL)
        if not model_ok:
            model_err = f"FastAPI health check failed at {API_URL}"
    else:
        model = load_model()
        model_ok = True
except Exception as e:
    model_ok = False
    model_err = str(e)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem'>
        <div style='font-family:Space Mono,monospace;font-size:0.65rem;
                    letter-spacing:0.14em;text-transform:uppercase;color:#6b7280;'>
            Model status
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model_ok:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:1.2rem'>
            <span style='width:8px;height:8px;background:#00e676;border-radius:50%;
                         display:inline-block;box-shadow:0 0 6px #00e676'></span>
            <span style='color:#00e676;font-size:0.82rem;font-weight:600'>{backend_name} · Ready</span>
        </div>
        <div style='color:#6b7280;font-size:0.75rem;line-height:1.5;margin-bottom:1.2rem'>
            Endpoint: <code style='color:#ffd740'>{API_URL or "local inference"}</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem'>
            <span style='width:8px;height:8px;background:#ff4b6e;border-radius:50%;display:inline-block'></span>
            <span style='color:#ff4b6e;font-size:0.82rem;font-weight:600'>Inference backend unavailable</span>
        </div>
        <div style='color:#6b7280;font-size:0.75rem;line-height:1.5'>
            {model_err}<br><code style='color:#ffd740'>{API_URL or "PROTEIN_API_URL is not set"}</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.65rem;
                letter-spacing:0.14em;text-transform:uppercase;color:#6b7280;margin-bottom:0.7rem'>
        Inference settings
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider("Prediction threshold", 0.1, 0.9, float(PREDICTION_THRESHOLD), 0.05,
                          help="Lower = more predictions; higher = more conservative.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.65rem;
                letter-spacing:0.14em;text-transform:uppercase;color:#6b7280;margin-bottom:0.8rem'>
        Model info
    </div>
    <div style='font-size:0.78rem;color:#a0aab4;line-height:1.9'>
        <b style='color:#e8eaf0'>Architecture</b><br>Custom 4-ch CNN<br>
        <b style='color:#e8eaf0'>Input</b><br>4 × 128 × 128<br>
        <b style='color:#e8eaf0'>Classes</b><br>28 localisations<br>
        <b style='color:#e8eaf0'>Best Micro F1</b><br>
        <span style='color:#00e5ff;font-family:Space Mono,monospace'>0.6518</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
    <div class='hero-title'>Human Protein<br>Atlas Classifier</div>
    <div class='hero-sub'>Multi-label subcellular localisation · 28 classes · 4-channel fluorescence</div>
</div>
<div class='divider'></div>
""", unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown("""
<div class='card-title' style='font-family:Space Mono,monospace;font-size:0.68rem;
     letter-spacing:0.12em;text-transform:uppercase;color:#6b7280;margin-bottom:0.8rem'>
    01 · Upload four-channel fluorescence images
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<p style='color:#ff4b6e;font-size:0.85rem;font-weight:600;margin-bottom:0.3rem'>● Red — Microtubule</p>", unsafe_allow_html=True)
    red_file = st.file_uploader("", type=["png","jpg","jpeg"], key="red", label_visibility="collapsed")
with col2:
    st.markdown("<p style='color:#00e676;font-size:0.85rem;font-weight:600;margin-bottom:0.3rem'>● Green — Protein of interest</p>", unsafe_allow_html=True)
    green_file = st.file_uploader("", type=["png","jpg","jpeg"], key="green", label_visibility="collapsed")
with col3:
    st.markdown("<p style='color:#448aff;font-size:0.85rem;font-weight:600;margin-bottom:0.3rem'>● Blue — Nucleus (DAPI)</p>", unsafe_allow_html=True)
    blue_file = st.file_uploader("", type=["png","jpg","jpeg"], key="blue", label_visibility="collapsed")
with col4:
    st.markdown("<p style='color:#ffd740;font-size:0.85rem;font-weight:600;margin-bottom:0.3rem'>● Yellow — ER</p>", unsafe_allow_html=True)
    yellow_file = st.file_uploader("", type=["png","jpg","jpeg"], key="yellow", label_visibility="collapsed")

# ── Preview ────────────────────────────────────────────────────────────────────
all_uploaded = all([red_file, green_file, blue_file, yellow_file])

if all_uploaded:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card-title' style='font-family:Space Mono,monospace;font-size:0.68rem;
         letter-spacing:0.12em;text-transform:uppercase;color:#6b7280;margin-bottom:0.8rem'>
        02 · Channel preview
    </div>
    """, unsafe_allow_html=True)

    red_arr    = preprocess(red_file)
    green_arr  = preprocess(green_file)
    blue_arr   = preprocess(blue_file)
    yellow_arr = preprocess(yellow_file)

    p1, p2, p3, p4, p5 = st.columns(5)
    for col, arr, label, color in zip(
        [p1, p2, p3, p4],
        [red_arr, green_arr, blue_arr, yellow_arr],
        ["Red", "Green", "Blue", "Yellow"],
        ["Reds", "Greens", "Blues", "YlOrBr"],
    ):
        with col:
            fig2, ax2 = plt.subplots(figsize=(2.5, 2.5))
            fig2.patch.set_facecolor("#111318")
            ax2.imshow(arr, cmap=color)
            ax2.set_title(label, color="#a0aab4", fontsize=8, pad=4)
            ax2.axis("off")
            plt.tight_layout(pad=0.2)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    with p5:
        composite = composite_rgb(red_arr, green_arr, blue_arr)
        fig3, ax3 = plt.subplots(figsize=(2.5, 2.5))
        fig3.patch.set_facecolor("#111318")
        ax3.imshow(composite)
        ax3.set_title("RGB Composite", color="#00e5ff", fontsize=8, pad=4)
        ax3.axis("off")
        plt.tight_layout(pad=0.2)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Run inference ──────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([2, 2, 2])
    with btn_col:
        run = st.button("🔬  Run Classification", disabled=not model_ok)

    if run and model_ok:
        try:
            with st.spinner("Running inference…"):
                probs, preds = run_inference(
                    [red_arr, green_arr, blue_arr, yellow_arr],
                    threshold,
                    {
                        "red": red_file,
                        "green": green_file,
                        "blue": blue_file,
                        "yellow": yellow_file,
                    },
                )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

        probs = np.asarray(probs, dtype=np.float32)
        preds = np.asarray(preds, dtype=np.int64)

        n_detected = int(preds.sum())
        top_idx    = int(np.argmax(probs))

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card-title' style='font-family:Space Mono,monospace;font-size:0.68rem;
             letter-spacing:0.12em;text-transform:uppercase;color:#6b7280;margin-bottom:0.8rem'>
            03 · Results
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-box'>
                <div class='metric-val'>{n_detected}</div>
                <div class='metric-label'>Localisations detected</div>
            </div>
            <div class='metric-box'>
                <div class='metric-val' style='color:#7b61ff'>{probs[top_idx]:.2f}</div>
                <div class='metric-label'>Top probability</div>
            </div>
            <div class='metric-box'>
                <div class='metric-val' style='color:#ffd740'>{threshold:.2f}</div>
                <div class='metric-label'>Threshold used</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Predicted class pills
        detected = [CLASS_LABELS[i] for i, p in enumerate(preds) if p == 1]
        if detected:
            pills_html = "".join(f"<span class='pill pill-active'>{l}</span>" for l in detected)
        else:
            pills_html = "<span style='color:#6b7280;font-size:0.85rem'>No classes detected above threshold.</span>"

        st.markdown(f"""
        <div class='card' style='margin-top:0.5rem'>
            <div class='card-title'>Detected localisations</div>
            <div>{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("""
        <div class='card-title' style='font-family:Space Mono,monospace;font-size:0.68rem;
             letter-spacing:0.12em;text-transform:uppercase;color:#6b7280;
             margin-top:1.2rem;margin-bottom:0.6rem'>
            Class probability distribution
        </div>
        """, unsafe_allow_html=True)

        chart_col, detail_col = st.columns([3, 2])
        with chart_col:
            fig_bar = prob_bar_chart(probs, preds, threshold)
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        with detail_col:
            # ── Top-5 rendered as a matplotlib figure (avoids Streamlit HTML sanitisation) ──
            top5_idx = np.argsort(probs)[::-1][:5]

            fig_top5, ax_t = plt.subplots(figsize=(4.2, 4.8))
            fig_top5.patch.set_facecolor("#111318")
            ax_t.set_facecolor("#111318")
            ax_t.set_xlim(0, 1)
            ax_t.set_ylim(-0.5, 4.5)
            ax_t.axis("off")

            # Title
            ax_t.text(0, 4.72, "TOP 5 PREDICTIONS",
                      color="#6b7280", fontsize=7.5,
                      fontweight="bold", ha="left",
                      fontfamily="monospace")

            for row_i, idx in enumerate(top5_idx):
                y   = 3.8 - row_i * 0.95
                prob_val = probs[idx]
                active   = preds[idx] == 1
                txt_col  = "#00e5ff" if active else "#6b7280"
                bar_col  = "#00e5ff" if active else "#2a3040"

                # Label
                label_text = CLASS_LABELS[idx]
                if len(label_text) > 22:
                    label_text = label_text[:21] + "…"
                ax_t.text(0, y + 0.22, label_text,
                          color=txt_col, fontsize=8.2,
                          fontweight="600" if active else "400", ha="left")

                # Prob value
                ax_t.text(1.0, y + 0.22, f"{prob_val:.3f}",
                          color="#a0aab4", fontsize=7.8,
                          fontfamily="monospace", ha="right")

                # Track bar background
                ax_t.barh(y - 0.12, 1.0, height=0.13,
                          left=0, color="#1e2530", align="center")
                # Filled bar
                ax_t.barh(y - 0.12, max(prob_val, 0.008), height=0.13,
                          left=0, color=bar_col, align="center")

                # Separator line
                if row_i < 4:
                    ax_t.axhline(y - 0.42, color="#1e2530",
                                 linewidth=0.7, xmin=0, xmax=1)

            plt.tight_layout(pad=0.5)
            st.pyplot(fig_top5, use_container_width=True)
            plt.close(fig_top5)

            # ── All 28 classes expander (native Streamlit — safe) ──────────────
            with st.expander(f"Show all {NUM_CLASSES} classes"):
                sorted_idx = np.argsort(probs)[::-1]
                for idx in sorted_idx:
                    active     = preds[idx] == 1
                    marker     = "✦" if active else "·"
                    name_col   = "#00e5ff" if active else "#6b7280"
                    prob_str   = f"{probs[idx]:.3f}"
                    # Use native columns so Streamlit renders it safely
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(
                            f"<span style='color:{name_col};font-size:0.78rem'>"
                            f"{marker} {CLASS_LABELS[idx]}</span>",
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"<span style='color:#a0aab4;font-size:0.75rem;"
                            f"font-family:monospace'>{prob_str}</span>",
                            unsafe_allow_html=True,
                        )

elif not all_uploaded:
    st.markdown("""
    <div style='text-align:center;padding:3rem;color:#2a3040;
                border:1px dashed #1e2530;border-radius:12px;margin-top:1.5rem'>
        <div style='font-size:2.5rem;margin-bottom:0.8rem'>🔬</div>
        <div style='font-family:Space Mono,monospace;font-size:0.85rem;color:#6b7280'>
            Upload all four channel images to begin
        </div>
    </div>
    """, unsafe_allow_html=True)
