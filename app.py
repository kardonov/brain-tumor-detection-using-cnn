import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
import os
import random
import warnings

warnings.filterwarnings('ignore')

# Optional Kaggle dataset download.
# This app can also work if you provide a local dataset path in the sidebar.



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
#  PAGE CONFIG & SPOTIFY-GREEN THEME
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection – CNN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPOTIFY_GREEN   = "#1DB954"
SPOTIFY_DARK    = "#191414"
SPOTIFY_GRAY    = "#282828"
SPOTIFY_LGRAY   = "#3E3E3E"
SPOTIFY_WHITE   = "#FFFFFF"
SPOTIFY_OFFWHITE= "#B3B3B3"

st.markdown(f"""
<style>
/* ── Base ── */
html, body, [class*="css"] {{
    background-color: {SPOTIFY_DARK};
    color: {SPOTIFY_WHITE};
    font-family: 'Circular', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}}
.main .block-container {{
    background-color: {SPOTIFY_DARK};
    padding-top: 1.5rem;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {SPOTIFY_GRAY};
    border-right: 1px solid {SPOTIFY_LGRAY};
}}
[data-testid="stSidebar"] .css-1d391kg {{
    color: {SPOTIFY_WHITE};
}}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {{
    color: {SPOTIFY_WHITE};
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background-color: {SPOTIFY_GRAY};
    border: 1px solid {SPOTIFY_LGRAY};
    border-radius: 12px;
    padding: 1rem;
}}
[data-testid="stMetricValue"] {{
    color: {SPOTIFY_GREEN} !important;
    font-size: 2rem !important;
    font-weight: 700;
}}
[data-testid="stMetricLabel"] {{
    color: {SPOTIFY_OFFWHITE} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {SPOTIFY_GREEN};
    color: {SPOTIFY_DARK};
    border: none;
    border-radius: 500px;
    font-weight: 700;
    font-size: 0.875rem;
    letter-spacing: 0.08em;
    padding: 0.6rem 2rem;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background-color: #1ed760;
    transform: scale(1.03);
    box-shadow: 0 4px 15px rgba(29,185,84,0.4);
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {SPOTIFY_GRAY};
    border-radius: 8px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {SPOTIFY_OFFWHITE};
    border-radius: 6px;
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{
    background-color: {SPOTIFY_GREEN} !important;
    color: {SPOTIFY_DARK} !important;
}}

/* ── Progress bar ── */
.stProgress > div > div > div > div {{
    background-color: {SPOTIFY_GREEN};
}}

/* ── Info boxes ── */
.stInfo {{
    background-color: {SPOTIFY_GRAY};
    border-left: 4px solid {SPOTIFY_GREEN};
    border-radius: 4px;
}}
.stSuccess {{
    background-color: rgba(29,185,84,0.15);
    border-left: 4px solid {SPOTIFY_GREEN};
}}

/* ── Selectbox / Slider ── */
.stSelectbox > div > div {{
    background-color: {SPOTIFY_LGRAY};
    color: {SPOTIFY_WHITE};
    border: 1px solid {SPOTIFY_LGRAY};
    border-radius: 8px;
}}
.stSlider .css-1aumxhk {{
    color: {SPOTIFY_GREEN};
}}

/* ── DataFrames ── */
.dataframe thead {{
    background-color: {SPOTIFY_GREEN};
    color: {SPOTIFY_DARK};
}}

/* ── Card ── */
.spotify-card {{
    background: {SPOTIFY_GRAY};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid {SPOTIFY_LGRAY};
    transition: box-shadow 0.2s;
}}
.spotify-card:hover {{
    box-shadow: 0 4px 20px rgba(29,185,84,0.2);
}}
.metric-green {{
    color: {SPOTIFY_GREEN};
    font-size: 2rem;
    font-weight: 700;
}}
.section-header {{
    color: {SPOTIFY_GREEN};
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}}
.badge {{
    display:inline-block;
    background:{SPOTIFY_GREEN};
    color:{SPOTIFY_DARK};
    border-radius:500px;
    padding:2px 12px;
    font-size:0.78rem;
    font-weight:700;
    margin-right:4px;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME  (dark + green accents)
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : SPOTIFY_DARK,
    "axes.facecolor"    : SPOTIFY_GRAY,
    "axes.edgecolor"    : SPOTIFY_LGRAY,
    "axes.labelcolor"   : SPOTIFY_WHITE,
    "xtick.color"       : SPOTIFY_OFFWHITE,
    "ytick.color"       : SPOTIFY_OFFWHITE,
    "text.color"        : SPOTIFY_WHITE,
    "grid.color"        : SPOTIFY_LGRAY,
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "legend.facecolor"  : SPOTIFY_GRAY,
    "legend.edgecolor"  : SPOTIFY_LGRAY,
    "font.family"       : "sans-serif",
})

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CATEGORIES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 15
DATASET_URL = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
COLOR_MAP   = {
    'glioma'     : '#1DB954',
    'meningioma' : '#1ed760',
    'notumor'    : '#17a845',
    'pituitary'  : '#148f3a',
}

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, val in {
    "model"      : None,
    "x_train"    : None, "x_test": None,
    "y_train"    : None, "y_test": None,
    "y_pred"     : None,
    "history"    : None,
    "categories" : CATEGORIES,
    "data_loaded": False,
    "trained"    : False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset_from_path(data_path: str, img_size: int = IMG_SIZE):
    categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
    X, y = [], []
    counts = {}
    for idx, cat in enumerate(categories):
        folder = os.path.join(data_path, cat)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")
        imgs = [f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[cat] = len(imgs)
        for img_name in imgs:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(idx)
    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)
    y_cat = to_categorical(y, num_classes=len(categories))
    return X, y_cat, counts


def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape),
        Activation("relu"),
        Conv2D(32, (5, 5), strides=(2, 2)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        # Use GlobalAveragePooling to make the CNN independent of image size.
        # This avoids Dense input shape mismatch when IMG_SIZE changes.
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    opt = RMSprop(learning_rate=0.0005)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )
    return model



def spotify_fig(fig):
    """Apply Spotify dark background to a fig and return it."""
    fig.patch.set_facecolor(SPOTIFY_DARK)
    for ax in fig.axes:
        ax.set_facecolor(SPOTIFY_GRAY)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPOTIFY_LGRAY)
    return fig


def get_model_summary_str(model):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1rem 0 0.5rem;">
        <span style="font-size:2.5rem;">🧠</span>
        <h2 style="color:{SPOTIFY_GREEN};margin:0.3rem 0 0;">Brain Tumor</h2>
        <p style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;margin:0;">Detection via CNN</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<p class="section-header">📂 Dataset</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="spotify-card" style="font-size:0.82rem;">
        <b style="color:{SPOTIFY_GREEN};">Sumber:</b><br>
        <a href="{DATASET_URL}" target="_blank" style="color:{SPOTIFY_GREEN};">
            Brain Tumor MRI Dataset<br>(Kaggle)
        </a><br><br>
        <b>Kelas:</b><br>
        {'  '.join([f'<span class="badge">{c}</span>' for c in CATEGORIES])}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<p class="section-header">⚙️ Konfigurasi Model</p>', unsafe_allow_html=True)
    img_size_opt = st.selectbox("Image Size", [64, 128, 224], index=1)
    epochs_opt   = st.slider("Epochs", 5, 30, EPOCHS)
    batch_opt    = st.select_slider("Batch Size", [16, 32, 64, 128], value=32)
    test_split   = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)

    st.markdown("---")
    st.markdown(f'<p class="section-header">📁 Path Dataset Lokal</p>', unsafe_allow_html=True)
    data_path = st.text_input(
        "Training folder path",
        placeholder="/path/to/Training",
        help="Folder berisi sub-folder: glioma, meningioma, notumor, pituitary",
    )

    load_btn  = st.button("📥 Muat Dataset", use_container_width=True)
    train_btn = st.button("🚀 Latih Model", use_container_width=True,
                          disabled=not st.session_state.data_loaded)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;color:{SPOTIFY_OFFWHITE};font-size:0.75rem;">
        Made with ❤️ &amp; Streamlit<br>
        <span style="color:{SPOTIFY_GREEN};">TensorFlow / Keras CNN</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATASET LOGIC
# ─────────────────────────────────────────────
if load_btn:
    if not data_path or not os.path.isdir(data_path):
        st.sidebar.error("❌ Path tidak valid atau folder tidak ditemukan.")
    else:
        with st.spinner("⏳ Memuat dan memproses dataset…"):
            try:
                X, y_cat, counts = load_dataset_from_path(data_path, img_size_opt)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y_cat,
                    test_size=test_split,
                    random_state=42,
                    stratify=np.argmax(y_cat, axis=1),
                )
                st.session_state.x_train     = X_tr
                st.session_state.x_test      = X_te
                st.session_state.y_train     = y_tr
                st.session_state.y_test      = y_te
                st.session_state.data_loaded = True
                st.session_state.counts      = counts
                st.session_state.data_path   = data_path
                st.sidebar.success(f"✅ Dataset berhasil dimuat — {len(X)} gambar")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────
#  TRAIN LOGIC
# ─────────────────────────────────────────────
if train_btn and st.session_state.data_loaded:
    model = build_cnn(
        input_shape=(img_size_opt, img_size_opt, 3),
        num_classes=len(CATEGORIES),
    )

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs):
            self.epochs = epochs
            self.prog   = st.sidebar.progress(0)
            self.status = st.sidebar.empty()

        def on_epoch_end(self, epoch, logs=None):
            pct = int((epoch + 1) / self.epochs * 100)
            self.prog.progress(pct)
            self.status.markdown(
                f"<span style='color:{SPOTIFY_GREEN}'>Epoch {epoch+1}/{self.epochs} "
                f"— acc: {logs.get('accuracy',0):.4f} "
                f"| val_acc: {logs.get('val_accuracy',0):.4f}</span>",
                unsafe_allow_html=True,
            )

    cb = StreamlitCallback(epochs_opt)
    with st.spinner("🧠 Training CNN…"):
        history = model.fit(
            st.session_state.x_train,
            st.session_state.y_train,
            batch_size=batch_opt,
            epochs=epochs_opt,
            validation_data=(st.session_state.x_test, st.session_state.y_test),
            shuffle=True,
            callbacks=[cb],
            verbose=0,
        )

    y_pred = model.predict(st.session_state.x_test, verbose=0)
    st.session_state.model   = model
    st.session_state.history = history.history
    st.session_state.y_pred  = y_pred
    st.session_state.trained = True
    st.sidebar.success("✅ Training selesai!")

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{SPOTIFY_GRAY} 0%,{SPOTIFY_DARK} 100%);
            border-left:5px solid {SPOTIFY_GREEN};
            border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
    <h1 style="margin:0;color:{SPOTIFY_WHITE};">
        🧠 Brain Tumor Detection
    </h1>
    <p style="color:{SPOTIFY_OFFWHITE};margin:0.3rem 0 0;font-size:1rem;">
        Deteksi Tumor Otak menggunakan <span style="color:{SPOTIFY_GREEN};font-weight:700;">
        Convolutional Neural Network (CNN)</span> &mdash;
        MRI Classification: Glioma · Meningioma · No Tumor · Pituitary
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Dataset Explorer",
    "🏗️ Arsitektur Model",
    "📈 Training & Evaluasi",
    "🔬 Prediksi",
    "ℹ️ Tentang",
])

# ══════════════════════════════════════════════
#  TAB 1 — DATASET EXPLORER
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown(f'<h2 style="color:{SPOTIFY_GREEN};">📊 Eksplorasi Dataset</h2>',
                unsafe_allow_html=True)

    # ── Dataset info cards (always shown)
    dist_data = {
        'glioma': 1321, 'meningioma': 1339,
        'notumor': 1595, 'pituitary': 1457,
    }
    if st.session_state.data_loaded and hasattr(st.session_state, "counts"):
        dist_data = st.session_state.counts

    total = sum(dist_data.values())

    cols = st.columns(5)
    for i, (cat, cnt) in enumerate(dist_data.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="spotify-card" style="text-align:center;">
                <div style="font-size:1.8rem;">
                    {'🧬' if cat=='glioma' else '🔵' if cat=='meningioma'
                      else '✅' if cat=='notumor' else '🟣'}
                </div>
                <div class="metric-green">{cnt:,}</div>
                <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;
                            text-transform:capitalize;">{cat}</div>
                <div style="color:{SPOTIFY_OFFWHITE};font-size:0.72rem;">
                    {cnt/total*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    with cols[4]:
        st.markdown(f"""
        <div class="spotify-card" style="text-align:center;">
            <div style="font-size:1.8rem;">📁</div>
            <div class="metric-green">{total:,}</div>
            <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;">Total Images</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown(f'<p class="section-header">Distribusi Kelas</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            list(dist_data.keys()), list(dist_data.values()),
            color=[COLOR_MAP[c] for c in dist_data],
            edgecolor=SPOTIFY_DARK, linewidth=0.8, width=0.6,
        )
        ax.set_title("Jumlah Gambar per Kelas Tumor", color=SPOTIFY_WHITE, fontsize=12, pad=12)
        ax.set_xlabel("Tumor Type", color=SPOTIFY_OFFWHITE)
        ax.set_ylabel("Number of Images", color=SPOTIFY_OFFWHITE)
        ax.set_ylim(0, max(dist_data.values()) * 1.2)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                    f'{int(b.get_height()):,}', ha='center', va='bottom',
                    color=SPOTIFY_WHITE, fontsize=10, fontweight='bold')
        ax.tick_params(colors=SPOTIFY_OFFWHITE)
        ax.grid(axis='y', color=SPOTIFY_LGRAY, linestyle='--', alpha=0.5)
        ax.set_facecolor(SPOTIFY_GRAY)
        fig = spotify_fig(fig)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown(f'<p class="section-header">Proporsi Kelas (Pie)</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        wedge_props = dict(linewidth=1.5, edgecolor=SPOTIFY_DARK)
        wedges, texts, autotexts = ax2.pie(
            list(dist_data.values()),
            labels=list(dist_data.keys()),
            autopct='%1.1f%%',
            colors=[COLOR_MAP[c] for c in dist_data],
            wedgeprops=wedge_props,
            startangle=90,
            pctdistance=0.75,
        )
        for t in texts:
            t.set_color(SPOTIFY_WHITE)
            t.set_fontsize(10)
        for at in autotexts:
            at.set_color(SPOTIFY_DARK)
            at.set_fontsize(9)
            at.set_fontweight('bold')
        ax2.set_title("Proporsi Dataset Training", color=SPOTIFY_WHITE, fontsize=12, pad=12)
        fig2 = spotify_fig(fig2)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Dataset link info
    st.markdown("---")
    st.markdown(f"""
    <div class="spotify-card">
        <p class="section-header">🔗 Sumber Dataset</p>
        <p style="color:{SPOTIFY_OFFWHITE};">
            Dataset yang digunakan dalam proyek ini berasal dari Kaggle:
        </p>
        <a href="{DATASET_URL}" target="_blank"
           style="color:{SPOTIFY_GREEN};font-size:1rem;font-weight:700;">
            📦 Brain Tumor MRI Dataset — Kaggle
        </a>
        <p style="color:{SPOTIFY_OFFWHITE};margin-top:0.8rem;font-size:0.87rem;">
            Dataset terdiri dari gambar MRI otak yang diklasifikasikan ke dalam 4 kategori:
            <b style="color:{SPOTIFY_WHITE};">Glioma</b>,
            <b style="color:{SPOTIFY_WHITE};">Meningioma</b>,
            <b style="color:{SPOTIFY_WHITE};">No Tumor</b>, dan
            <b style="color:{SPOTIFY_WHITE};">Pituitary</b>.
            Total ~5.712 gambar training.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample images preview
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown(f'<p class="section-header">🖼️ Contoh Gambar per Kelas</p>',
                    unsafe_allow_html=True)
        data_path_loaded = st.session_state.data_path
        samples = 5
        fig3, axes = plt.subplots(4, samples, figsize=(14, 10))
        fig3.patch.set_facecolor(SPOTIFY_DARK)

        for row_i, cat in enumerate(CATEGORIES):
            folder = os.path.join(data_path_loaded, cat)
            imgs_list = [f for f in os.listdir(folder)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            sample_imgs = random.sample(imgs_list, min(samples, len(imgs_list)))
            for col_j, img_name in enumerate(sample_imgs):
                img = cv2.imread(os.path.join(folder, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax = axes[row_i][col_j]
                ax.imshow(img)
                ax.axis("off")
                if col_j == 0:
                    ax.set_ylabel(cat.capitalize(), color=SPOTIFY_GREEN,
                                  fontsize=11, fontweight='bold', rotation=0,
                                  labelpad=60, va='center')
        fig3.suptitle("Sample MRI Images per Tumor Class",
                      color=SPOTIFY_WHITE, fontsize=14, y=1.02)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    else:
        st.info("📁 Muat dataset terlebih dahulu melalui sidebar untuk melihat contoh gambar.")

    # ── Train/Test split info
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown(f'<p class="section-header">✂️ Train / Test Split</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="spotify-card">
                <div class="metric-green">{st.session_state.x_train.shape[0]:,}</div>
                <div style="color:{SPOTIFY_OFFWHITE};">Training Samples</div>
                <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;">
                    Shape: {st.session_state.x_train.shape}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="spotify-card">
                <div class="metric-green">{st.session_state.x_test.shape[0]:,}</div>
                <div style="color:{SPOTIFY_OFFWHITE};">Testing Samples</div>
                <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;">
                    Shape: {st.session_state.x_test.shape}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 — ARSITEKTUR MODEL
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown(f'<h2 style="color:{SPOTIFY_GREEN};">🏗️ Arsitektur CNN</h2>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="spotify-card">
        <p class="section-header">Desain Model</p>
        <p style="color:{SPOTIFY_OFFWHITE};">
            Model CNN dirancang dengan dua blok konvolusi, diikuti MaxPooling dan Dropout
            untuk regularisasi, kemudian dua fully-connected layer sebagai classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<p class="section-header">Layer Stack</p>', unsafe_allow_html=True)
        layers_info = [
            ("Conv2D (32 filters, 5×5, stride 2)",  "Input→64×64×32", "2,432"),
            ("Activation (ReLU)",                    "64×64×32",       "0"),
            ("Conv2D (32 filters, 5×5, stride 2)",  "30×30×32",       "25,632"),
            ("Activation (ReLU)",                    "30×30×32",       "0"),
            ("MaxPooling2D (2×2)",                   "15×15×32",       "0"),
            ("Dropout (0.25)",                       "15×15×32",       "0"),
            ("Flatten",                              "7,200",          "0"),
            ("Dense (512, ReLU)",                    "512",            "3,686,912"),
            ("Dropout (0.50)",                       "512",            "0"),
            ("Dense (4, Softmax)",                   "4",              "2,052"),
        ]
        df_arch = pd.DataFrame(layers_info, columns=["Layer", "Output Shape", "Params"])
        df_arch.index = range(1, len(df_arch)+1)
        st.dataframe(
            df_arch.style.set_properties(**{
                'background-color': SPOTIFY_GRAY,
                'color': SPOTIFY_WHITE,
                'border': f'1px solid {SPOTIFY_LGRAY}',
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', SPOTIFY_GREEN),
                    ('color', SPOTIFY_DARK),
                    ('font-weight', 'bold'),
                ]}
            ]),
            use_container_width=True,
        )

        st.markdown(f"""
        <div class="spotify-card" style="margin-top:1rem;">
            <div class="section-header">Parameter Summary</div>
            <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:0.5rem;">
                <div>
                    <div class="metric-green">3,717,028</div>
                    <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;">Total Params</div>
                </div>
                <div>
                    <div class="metric-green">14.18 MB</div>
                    <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;">Model Size</div>
                </div>
                <div>
                    <div class="metric-green">0.0005</div>
                    <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;">Learning Rate</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f'<p class="section-header">Visualisasi Arsitektur</p>',
                    unsafe_allow_html=True)

        fig_arch, ax_arch = plt.subplots(figsize=(5, 8))
        ax_arch.set_xlim(0, 10)
        ax_arch.set_ylim(0, 22)
        ax_arch.axis('off')
        fig_arch.patch.set_facecolor(SPOTIFY_DARK)
        ax_arch.set_facecolor(SPOTIFY_DARK)

        blocks = [
            (1,  "Input (128×128×3)",   "#3E3E3E",  SPOTIFY_WHITE),
            (3,  "Conv2D 32×5×5 s2",   SPOTIFY_GREEN, SPOTIFY_DARK),
            (5,  "ReLU Activation",     "#17a845",  SPOTIFY_WHITE),
            (7,  "Conv2D 32×5×5 s2",   SPOTIFY_GREEN, SPOTIFY_DARK),
            (9,  "ReLU Activation",     "#17a845",  SPOTIFY_WHITE),
            (11, "MaxPool2D 2×2",       "#1565C0",  SPOTIFY_WHITE),
            (13, "Dropout 0.25",        "#7B1FA2",  SPOTIFY_WHITE),
            (15, "Flatten → 7200",      "#E65100",  SPOTIFY_WHITE),
            (17, "Dense 512 ReLU",      SPOTIFY_GREEN, SPOTIFY_DARK),
            (19, "Dropout 0.50",        "#7B1FA2",  SPOTIFY_WHITE),
            (21, "Dense 4 Softmax",     "#1DB954",  SPOTIFY_DARK),
        ]

        for y_pos, label, bg, fg in blocks:
            rect = mpatches.FancyBboxPatch(
                (1, y_pos - 0.7), 8, 1.3,
                boxstyle="round,pad=0.1",
                facecolor=bg, edgecolor=SPOTIFY_LGRAY, linewidth=1,
            )
            ax_arch.add_patch(rect)
            ax_arch.text(5, y_pos, label, ha='center', va='center',
                         color=fg, fontsize=8.5, fontweight='bold')

        for y_pos in [2.3, 4.3, 6.3, 8.3, 10.3, 12.3, 14.3, 16.3, 18.3, 20.3]:
            ax_arch.annotate("", xy=(5, y_pos), xytext=(5, y_pos - 0.5),
                             arrowprops=dict(arrowstyle="->",
                                             color=SPOTIFY_GREEN, lw=1.5))

        ax_arch.set_title("CNN Architecture Flow",
                          color=SPOTIFY_WHITE, fontsize=11, pad=8)
        st.pyplot(fig_arch, use_container_width=True)
        plt.close()

    # Hyperparameters
    st.markdown("---")
    st.markdown(f'<p class="section-header">🔧 Hyperparameters</p>', unsafe_allow_html=True)
    hc1, hc2, hc3, hc4 = st.columns(4)
    for col, label, val in zip(
        [hc1, hc2, hc3, hc4],
        ["Optimizer", "Loss Function", "Epochs", "Image Size"],
        ["RMSprop", "Categorical\nCrossEntropy", f"{epochs_opt}", f"{img_size_opt}×{img_size_opt}"]
    ):
        with col:
            st.markdown(f"""
            <div class="spotify-card" style="text-align:center;">
                <div class="metric-green" style="font-size:1.3rem;">{val}</div>
                <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.trained and st.session_state.model:
        st.markdown("---")
        st.markdown(f'<p class="section-header">📋 Model Summary (Live)</p>',
                    unsafe_allow_html=True)
        summary_str = get_model_summary_str(st.session_state.model)
        st.code(summary_str, language="text")


# ══════════════════════════════════════════════
#  TAB 3 — TRAINING & EVALUASI
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown(f'<h2 style="color:{SPOTIFY_GREEN};">📈 Training & Evaluasi</h2>',
                unsafe_allow_html=True)

    if not st.session_state.trained:
        # Show reference results from the notebook
        st.info("🏋️ Model belum dilatih. Menampilkan referensi hasil dari notebook asli…")


    # ── Metric cards
    best_val_acc = max(use_hist['val_accuracy'])
    best_val_loss= min(use_hist['val_loss'])
    best_train_acc=max(use_hist['accuracy'])

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, fmt in zip(
        [m1, m2, m3, m4],
        ["Best Val Accuracy", "Best Train Accuracy", "Best Val Loss", "Test Accuracy"],
        [best_val_acc, best_train_acc, best_val_loss, 0.9405],
        ["{:.2%}", "{:.2%}", "{:.4f}", "{:.2%}"],
    ):
        with col:
            st.markdown(f"""
            <div class="spotify-card" style="text-align:center;">
                <div class="metric-green">{fmt.format(val)}</div>
                <div style="color:{SPOTIFY_OFFWHITE};font-size:0.8rem;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training curves
    st.markdown(f'<p class="section-header">📉 Learning Curves{ref_tag}</p>',
                unsafe_allow_html=True)

    fig_lc, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep_x = list(range(1, len(use_hist['accuracy'])+1))

    for ax, tr_key, vl_key, title, ylabel in [
        (ax1, 'accuracy',  'val_accuracy', 'Model Accuracy',  'Accuracy'),
        (ax2, 'loss',      'val_loss',     'Model Loss',      'Loss'),
    ]:
        ax.plot(ep_x, use_hist[tr_key], color=SPOTIFY_GREEN, lw=2.5,
                marker='o', markersize=4, label='Train')
        ax.plot(ep_x, use_hist[vl_key], color='#1ed760', lw=2.5,
                linestyle='--', marker='s', markersize=4, label='Validation')
        ax.set_title(title, color=SPOTIFY_WHITE, fontsize=12, pad=10)
        ax.set_xlabel("Epoch", color=SPOTIFY_OFFWHITE)
        ax.set_ylabel(ylabel, color=SPOTIFY_OFFWHITE)
        ax.legend(facecolor=SPOTIFY_GRAY, edgecolor=SPOTIFY_LGRAY, labelcolor=SPOTIFY_WHITE)
        ax.set_xticks(ep_x)
        ax.tick_params(colors=SPOTIFY_OFFWHITE)
        ax.grid(True, color=SPOTIFY_LGRAY, linestyle='--', alpha=0.5)
        ax.set_facecolor(SPOTIFY_GRAY)

    fig_lc = spotify_fig(fig_lc)
    plt.tight_layout()
    st.pyplot(fig_lc, use_container_width=True)
    plt.close()

    # ── Epoch table
    st.markdown("---")
    st.markdown(f'<p class="section-header">📋 Tabel Epoch-by-Epoch</p>',
                unsafe_allow_html=True)
    ep_df = pd.DataFrame({
        "Epoch"          : ep_x,
        "Train Accuracy" : [f"{v:.4f}" for v in use_hist['accuracy']],
        "Val Accuracy"   : [f"{v:.4f}" for v in use_hist['val_accuracy']],
        "Train Loss"     : [f"{v:.4f}" for v in use_hist['loss']],
        "Val Loss"       : [f"{v:.4f}" for v in use_hist['val_loss']],
    })
    st.dataframe(ep_df, use_container_width=True, hide_index=True)

    # ── Confusion matrix & classification report
    st.markdown("---")
    st.markdown(f'<p class="section-header">🔢 Confusion Matrix & Classification Report</p>',
                unsafe_allow_html=True)

    if st.session_state.trained and st.session_state.y_pred is not None:
        y_true_live = np.argmax(st.session_state.y_test, axis=1)
        y_pred_live = np.argmax(st.session_state.y_pred, axis=1)
        cm_live     = confusion_matrix(y_true_live, y_pred_live)
        cr_live     = classification_report(y_true_live, y_pred_live,
                                             target_names=CATEGORIES, output_dict=True)
        use_cm      = cm_live
        use_cr      = cr_live
    else:
        # Reference confusion matrix from notebook
        use_cm = np.array([
            [289, 0,  2,  1],
            [  1, 309, 7,  2],
            [  9, 9,  242, 8],
            [  2, 0,  27, 235],
        ])
        use_cr = {
            'pituitary'   : {'precision':0.96,'recall':0.99,'f1-score':0.97,'support':292},
            'notumor'     : {'precision':0.97,'recall':0.97,'f1-score':0.97,'support':319},
            'meningioma'  : {'precision':0.87,'recall':0.90,'f1-score':0.89,'support':268},
            'glioma'      : {'precision':0.96,'recall':0.89,'f1-score':0.92,'support':264},
            'accuracy'    : 0.94,
            'macro avg'   : {'precision':0.94,'recall':0.94,'f1-score':0.94,'support':1143},
            'weighted avg': {'precision':0.94,'recall':0.94,'f1-score':0.94,'support':1143},
        }

    # Order the CM to match pituitary, notumor, meningioma, glioma
    cf1, cf2 = st.columns([1, 1])
    with cf1:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        mask_colors = np.zeros_like(use_cm, dtype=float)
        for i in range(use_cm.shape[0]):
            row_sum = use_cm[i].sum()
            for j in range(use_cm.shape[1]):
                mask_colors[i, j] = use_cm[i, j] / row_sum if row_sum > 0 else 0

        im = ax_cm.imshow(mask_colors, cmap='Greens', vmin=0, vmax=1)
        for i in range(use_cm.shape[0]):
            for j in range(use_cm.shape[1]):
                txt_color = SPOTIFY_DARK if mask_colors[i, j] > 0.5 else SPOTIFY_GRAY
                ax_cm.text(j, i, str(use_cm[i, j]),
                           ha='center', va='center',
                           color=txt_color, fontsize=12, fontweight='bold')

        ax_cm.set_xticks(range(len(CATEGORIES)))
        ax_cm.set_yticks(range(len(CATEGORIES)))
        ax_cm.set_xticklabels(CATEGORIES, rotation=30, ha='right', color=SPOTIFY_DARK)
        ax_cm.set_yticklabels(CATEGORIES, color=SPOTIFY_DARK)
        ax_cm.set_xlabel("Predicted", color=SPOTIFY_DARK, fontsize=11)
        ax_cm.set_ylabel("True", color=SPOTIFY_DARK, fontsize=11)
        ax_cm.set_title("Confusion Matrix", color=SPOTIFY_DARK, fontsize=12, pad=10)
        plt.colorbar(im, ax=ax_cm)
        fig_cm = spotify_fig(fig_cm)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

    with cf2:
        st.markdown(f'<p class="section-header">Classification Report</p>',
                    unsafe_allow_html=True)
        cr_rows = []
        for cat in CATEGORIES:
            if cat in use_cr:
                row = use_cr[cat]
                cr_rows.append({
                    "Class"    : cat.capitalize(),
                    "Precision": f"{row['precision']:.4f}",
                    "Recall"   : f"{row['recall']:.4f}",
                    "F1-Score" : f"{row['f1-score']:.4f}",
                    "Support"  : int(row['support']),
                })
        cr_df = pd.DataFrame(cr_rows)
        st.dataframe(cr_df, use_container_width=True, hide_index=True)

        # Per-class bar chart
        fig_cr, ax_cr = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(CATEGORIES))
        w = 0.25
        prec  = [use_cr[c]['precision'] for c in CATEGORIES]
        rec   = [use_cr[c]['recall']    for c in CATEGORIES]
        f1    = [use_cr[c]['f1-score']  for c in CATEGORIES]
        ax_cr.bar(x - w,   prec, w, label='Precision', color=SPOTIFY_GREEN,  alpha=0.9)
        ax_cr.bar(x,       rec,  w, label='Recall',    color='#1ed760',      alpha=0.9)
        ax_cr.bar(x + w,   f1,   w, label='F1-Score',  color='#17a845',      alpha=0.9)
        ax_cr.set_xticks(x)
        ax_cr.set_xticklabels([c.capitalize() for c in CATEGORIES],
                               color=SPOTIFY_WHITE, fontsize=9)
        ax_cr.set_ylim(0.7, 1.05)
        ax_cr.set_title("Precision / Recall / F1", color=SPOTIFY_WHITE, fontsize=11, pad=8)
        ax_cr.legend(facecolor=SPOTIFY_GRAY, edgecolor=SPOTIFY_LGRAY,
                     labelcolor=SPOTIFY_WHITE, fontsize=8)
        ax_cr.tick_params(colors=SPOTIFY_OFFWHITE)
        ax_cr.set_facecolor(SPOTIFY_GRAY)
        ax_cr.grid(axis='y', color=SPOTIFY_LGRAY, linestyle='--', alpha=0.5)
        fig_cr = spotify_fig(fig_cr)
        plt.tight_layout()
        st.pyplot(fig_cr, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════
#  TAB 4 — PREDIKSI
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown(f'<h2 style="color:{SPOTIFY_GREEN};">🔬 Prediksi Gambar MRI</h2>',
                unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("⚠️ Latih model terlebih dahulu untuk mengaktifkan fitur prediksi.")
    else:
        st.markdown(f"""
        <div class="spotify-card">
            <p style="color:{SPOTIFY_OFFWHITE};">
                Upload gambar MRI otak (JPG/PNG) untuk memprediksi jenis tumor menggunakan
                model CNN yang telah dilatih.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload MRI Image", type=["jpg", "jpeg", "png"],
            help="Upload gambar MRI otak untuk prediksi"
        )

        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            pc1, pc2 = st.columns([1, 2])
            with pc1:
                st.image(img_rgb, caption="Input MRI Image", use_column_width=True)

            with pc2:
                # Preprocess
                img_resized = cv2.resize(img_rgb, (img_size_opt, img_size_opt))
                img_input   = img_resized.astype("float32") / 255.0
                img_input   = np.expand_dims(img_input, axis=0)

                with st.spinner("🧠 Menganalisis gambar…"):
                    predictions = st.session_state.model.predict(img_input, verbose=0)

                pred_idx    = np.argmax(predictions[0])
                pred_class  = CATEGORIES[pred_idx]
                confidence  = predictions[0][pred_idx]
                color_class = SPOTIFY_GREEN

                st.markdown(f"""
                <div class="spotify-card" style="text-align:center;margin-bottom:1rem;">
                    <div style="font-size:2.5rem;margin-bottom:0.3rem;">
                        {'🧬' if pred_class=='glioma' else '🔵' if pred_class=='meningioma'
                          else '✅' if pred_class=='notumor' else '🟣'}
                    </div>
                    <div style="color:{color_class};font-size:1.6rem;font-weight:700;
                                text-transform:capitalize;">
                        {pred_class}
                    </div>
                    <div style="color:{SPOTIFY_OFFWHITE};font-size:1rem;">
                        Confidence: <b style="color:{SPOTIFY_GREEN};">{confidence:.2%}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar chart
                fig_pred, ax_pred = plt.subplots(figsize=(5, 3))
                probs = predictions[0] * 100
                bar_colors = [SPOTIFY_GREEN if i == pred_idx else SPOTIFY_LGRAY
                              for i in range(len(CATEGORIES))]
                bars_pred = ax_pred.barh(CATEGORIES, probs, color=bar_colors,
                                         edgecolor=SPOTIFY_DARK, linewidth=0.6)
                for bar, p in zip(bars_pred, probs):
                    ax_pred.text(min(p + 1, 97), bar.get_y() + bar.get_height()/2,
                                 f"{p:.1f}%", va='center', color=SPOTIFY_WHITE,
                                 fontsize=9, fontweight='bold')
                ax_pred.set_xlim(0, 105)
                ax_pred.set_xlabel("Probability (%)", color=SPOTIFY_OFFWHITE)
                ax_pred.set_title("Class Probabilities", color=SPOTIFY_WHITE, fontsize=11)
                ax_pred.tick_params(colors=SPOTIFY_WHITE)
                ax_pred.set_facecolor(SPOTIFY_GRAY)
                fig_pred = spotify_fig(fig_pred)
                plt.tight_layout()
                st.pyplot(fig_pred, use_container_width=True)
                plt.close()

        # ── Random test samples
        st.markdown("---")
        st.markdown(f'<p class="section-header">🎲 Prediksi Sampel Acak dari Test Set</p>',
                    unsafe_allow_html=True)

        n_samples = st.slider("Jumlah Sampel", 4, 16, 9, 1)
        if st.button("🔀 Prediksi Sampel Acak", use_container_width=False):
            model     = st.session_state.model
            x_test    = st.session_state.x_test
            y_test    = st.session_state.y_test
            y_pred_all= st.session_state.y_pred
            y_true_all= np.argmax(y_test,    axis=1)
            y_pred_cls= np.argmax(y_pred_all, axis=1)

            idxs = np.random.choice(len(x_test), n_samples, replace=False)
            ncols = 3
            nrows = int(np.ceil(n_samples / ncols))
            fig_s, axes_s = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
            axes_s = np.array(axes_s).flatten()
            fig_s.patch.set_facecolor(SPOTIFY_DARK)

            for ax_i, idx in enumerate(idxs):
                img_show  = x_test[idx]
                true_lbl  = CATEGORIES[y_true_all[idx]]
                pred_lbl  = CATEGORIES[y_pred_cls[idx]]
                correct   = true_lbl == pred_lbl
                border_c  = SPOTIFY_GREEN if correct else '#E53935'

                axes_s[ax_i].imshow(img_show)
                axes_s[ax_i].axis("off")
                axes_s[ax_i].set_title(
                    f"True: {true_lbl}\nPred: {pred_lbl}",
                    color=border_c, fontsize=9, fontweight='bold',
                )
                for spine in axes_s[ax_i].spines.values():
                    spine.set_edgecolor(border_c)
                    spine.set_linewidth(3)

            for ax_i in range(len(idxs), len(axes_s)):
                axes_s[ax_i].axis('off')

            plt.tight_layout()
            st.pyplot(fig_s, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════
#  TAB 5 — TENTANG
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown(f'<h2 style="color:{SPOTIFY_GREEN};">ℹ️ Tentang Proyek</h2>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"""
        <div class="spotify-card">
            <p class="section-header">🧠 Deskripsi Proyek</p>
            <p style="color:{SPOTIFY_OFFWHITE};">
                Proyek ini mengimplementasikan sistem deteksi tumor otak berbasis
                <b style="color:{SPOTIFY_WHITE};">Convolutional Neural Network (CNN)</b>
                menggunakan gambar MRI. Model dilatih untuk mengklasifikasikan 4 jenis kondisi:
            </p>
            <ul style="color:{SPOTIFY_OFFWHITE};">
                <li><b style="color:{SPOTIFY_GREEN};">Glioma</b> — Tumor yang berasal dari sel glial</li>
                <li><b style="color:{SPOTIFY_GREEN};">Meningioma</b> — Tumor pada selaput otak (meninges)</li>
                <li><b style="color:{SPOTIFY_GREEN};">No Tumor</b> — Tidak ada tumor</li>
                <li><b style="color:{SPOTIFY_GREEN};">Pituitary</b> — Tumor pada kelenjar pituitari</li>
            </ul>
        </div>

        <div class="spotify-card" style="margin-top:1rem;">
            <p class="section-header">📦 Stack Teknologi</p>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.5rem;">
                {''.join([f'<span class="badge">{t}</span>' for t in
                  ['Python 3.10', 'TensorFlow 2.x', 'Keras', 'OpenCV',
                   'Streamlit', 'NumPy', 'Pandas', 'Matplotlib',
                   'Seaborn', 'Scikit-learn']])}
            </div>
        </div>

        <div class="spotify-card" style="margin-top:1rem;">
            <p class="section-header">📊 Hasil Referensi (Notebook Asli)</p>
            <table style="color:{SPOTIFY_OFFWHITE};width:100%;border-collapse:collapse;">
                <tr style="border-bottom:1px solid {SPOTIFY_LGRAY};">
                    <th style="color:{SPOTIFY_GREEN};text-align:left;padding:6px;">Metrik</th>
                    <th style="color:{SPOTIFY_GREEN};text-align:right;padding:6px;">Nilai</th>
                </tr>
                <tr>
                    <td style="padding:6px;">Test Accuracy</td>
                    <td style="text-align:right;color:{SPOTIFY_GREEN};font-weight:700;padding:6px;">94.05%</td>
                </tr>
                <tr>
                    <td style="padding:6px;">Best Val Accuracy</td>
                    <td style="text-align:right;color:{SPOTIFY_GREEN};font-weight:700;padding:6px;">94.58%</td>
                </tr>
                <tr>
                    <td style="padding:6px;">Macro F1-Score</td>
                    <td style="text-align:right;color:{SPOTIFY_GREEN};font-weight:700;padding:6px;">0.94</td>
                </tr>
                <tr>
                    <td style="padding:6px;">Total Parameters</td>
                    <td style="text-align:right;color:{SPOTIFY_GREEN};font-weight:700;padding:6px;">3,717,028</td>
                </tr>
                <tr>
                    <td style="padding:6px;">Epochs</td>
                    <td style="text-align:right;padding:6px;">15</td>
                </tr>
                <tr>
                    <td style="padding:6px;">Optimizer</td>
                    <td style="text-align:right;padding:6px;">RMSprop (lr=0.0005)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="spotify-card">
            <p class="section-header">🔗 Dataset</p>
            <a href="{DATASET_URL}" target="_blank"
               style="color:{SPOTIFY_GREEN};font-weight:700;font-size:0.95rem;">
                Brain Tumor MRI Dataset
            </a>
            <p style="color:{SPOTIFY_OFFWHITE};font-size:0.82rem;margin-top:0.5rem;">
                Kaggle Dataset oleh Masoud Nickparvar
            </p>
            <hr style="border-color:{SPOTIFY_LGRAY};">
            <div style="font-size:0.85rem;color:{SPOTIFY_OFFWHITE};">
                <b style="color:{SPOTIFY_WHITE};">Training Set:</b><br>
                • Glioma: 1,321 images<br>
                • Meningioma: 1,339 images<br>
                • No Tumor: 1,595 images<br>
                • Pituitary: 1,457 images<br>
                <br>
                <b style="color:{SPOTIFY_WHITE};">Total: 5,712 images</b>
            </div>
        </div>

        <div class="spotify-card" style="margin-top:1rem;">
            <p class="section-header">🚀 Cara Penggunaan</p>
            <ol style="color:{SPOTIFY_OFFWHITE};font-size:0.83rem;padding-left:1.2rem;">
                <li>Download dataset dari Kaggle</li>
                <li>Masukkan path folder Training di sidebar</li>
                <li>Klik <b style="color:{SPOTIFY_GREEN};">Muat Dataset</b></li>
                <li>Atur hyperparameter (opsional)</li>
                <li>Klik <b style="color:{SPOTIFY_GREEN};">Latih Model</b></li>
                <li>Lihat hasil di tab Evaluasi & Prediksi</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;color:{SPOTIFY_OFFWHITE};padding:1rem 0;">
        <span style="color:{SPOTIFY_GREEN};font-weight:700;font-size:1.1rem;">
            🧠 Brain Tumor Detection CNN
        </span><br>
        <span style="font-size:0.8rem;">
            Dibangun dengan TensorFlow · Keras · Streamlit
        </span>
    </div>
    """, unsafe_allow_html=True)