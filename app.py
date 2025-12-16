"""
===============================================================================
STREAMLIT APP - KLASIFIKASI SINYAL EEG
Deep Learning Model Deployment
===============================================================================
Aplikasi web untuk demo model deep learning klasifikasi sinyal EEG
Model: CNN1D, LSTM, CNN-LSTM Hybrid, EEGNet
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
import io
import os
from pathlib import Path

# Import TensorFlow dengan error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("⚠️ TensorFlow tidak terinstal. Install dengan: pip install tensorflow")

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="EEG Signal Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konstanta Preprocessing (HARUS SAMA dengan training)
SAMPLING_RATE = 200  # Hz
LOWCUT = 0.5
HIGHCUT = 45.0
EPOCH_LENGTH = 4  # detik
MODEL_DIR = "models"

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNGSI PREPROCESSING ====================

@st.cache_data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Band-pass filter"""
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        st.error(f"Error in bandpass filter: {e}")
        return data


@st.cache_data
def notch_filter(data, fs, freq=50.0, quality=30):
    """Notch filter untuk powerline noise"""
    try:
        b, a = signal.iirnotch(freq, quality, fs)
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        return filtered_data
    except Exception as e:
        st.error(f"Error in notch filter: {e}")
        return data


def preprocess_eeg_data(data):
    """
    Preprocessing pipeline untuk data EEG
    
    Args:
        data: numpy array shape (n_samples, n_channels)
    
    Returns:
        epochs: numpy array shape (n_epochs, samples_per_epoch, n_channels)
    """
    with st.spinner("🔄 Applying filters..."):
        # 1. Band-pass filter
        filtered_data = butter_bandpass_filter(data, LOWCUT, HIGHCUT, SAMPLING_RATE)
        
        # 2. Notch filter
        filtered_data = notch_filter(filtered_data, SAMPLING_RATE, freq=50.0)
    
    with st.spinner("✂️ Segmenting signal..."):
        # 3. Epoching
        samples_per_epoch = int(SAMPLING_RATE * EPOCH_LENGTH)
        n_samples = filtered_data.shape[0]
        n_epochs = n_samples // samples_per_epoch
        
        # Trim data
        trimmed_data = filtered_data[:n_epochs * samples_per_epoch]
        
        # Reshape ke epochs
        epochs = trimmed_data.reshape(n_epochs, samples_per_epoch, -1)
    
    with st.spinner("📊 Normalizing..."):
        # 4. Baseline correction
        epochs = epochs - np.mean(epochs, axis=1, keepdims=True)
        
        # 5. Z-score normalization
        mean = np.mean(epochs, axis=(0, 1), keepdims=True)
        std = np.std(epochs, axis=(0, 1), keepdims=True)
        normalized_epochs = (epochs - mean) / (std + 1e-8)
    
    return normalized_epochs


# ==================== FUNGSI LOAD MODEL ====================

def load_model_safe(model_path):
    """
    Load model dengan multiple fallback strategies
    """
    if not TF_AVAILABLE:
        return None, "TensorFlow not available"
    
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    
    try:
        # Strategy 1: Load without compile
        model = keras.models.load_model(model_path, compile=False)
        
        # Re-compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model, "success"
    
    except Exception as e1:
        try:
            # Strategy 2: Normal load
            model = keras.models.load_model(model_path)
            return model, "success"
        except Exception as e2:
            error_msg = f"Failed to load model: {str(e2)[:200]}"
            return None, error_msg


@st.cache_resource
def load_model(model_name):
    """Load trained model with caching"""
    model_path = os.path.join(MODEL_DIR, f"{model_name}_final.h5")
    
    with st.spinner(f"⏳ Loading {model_name}..."):
        model, status = load_model_safe(model_path)
    
    if model is None:
        st.error(f"❌ Failed to load {model_name}")
        st.error(f"Error: {status}")
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Kemungkinan penyebab:**
            1. File model tidak ada di folder `models/`
            2. Format file model corrupt
            3. TensorFlow version mismatch
            
            **Solusi:**
            1. Pastikan file ada: `models/{model_name}_final.h5`
            2. Re-download model dari training
            3. Check requirements.txt untuk versi TensorFlow yang sesuai
            """)
    else:
        st.success(f"✅ Model {model_name} loaded successfully!")
    
    return model


def get_available_models():
    """Get list of available models"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        return []
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_final.h5')]
    model_names = [f.replace('_final.h5', '') for f in model_files]
    return sorted(model_names)


# ==================== FUNGSI VISUALISASI ====================

def plot_eeg_signal(data, title="EEG Signal", max_channels=5):
    """Plot time series EEG signal"""
    n_channels = min(max_channels, data.shape[1])
    time = np.arange(data.shape[0]) / SAMPLING_RATE
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i in range(n_channels):
        fig.add_trace(go.Scatter(
            x=time,
            y=data[:, i] + i * 3,
            mode='lines',
            name=f'Channel {i+1}',
            line=dict(width=1.5, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (Normalized + Offset)",
        height=400,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_power_spectrum(data, channel=0):
    """Plot power spectral density"""
    try:
        f, psd = signal.welch(data[:, channel], fs=SAMPLING_RATE, nperseg=256)
    except Exception as e:
        st.error(f"Error calculating PSD: {e}")
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=f,
        y=psd,
        mode='lines',
        fill='tozeroy',
        line=dict(color='steelblue', width=2)
    ))
    
    # Add frequency band regions
    bands = {
        'Delta': (0.5, 4, 'rgba(255, 0, 0, 0.1)'),
        'Theta': (4, 8, 'rgba(255, 165, 0, 0.1)'),
        'Alpha': (8, 13, 'rgba(255, 255, 0, 0.1)'),
        'Beta': (13, 30, 'rgba(0, 255, 0, 0.1)'),
        'Gamma': (30, 45, 'rgba(0, 0, 255, 0.1)')
    }
    
    for band_name, (low, high, color) in bands.items():
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=band_name,
            annotation_position="top left",
            annotation_font_size=10
        )
    
    fig.update_layout(
        title=f"Power Spectral Density - Channel {channel+1}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (V²/Hz)",
        yaxis_type="log",
        height=400,
        template='plotly_white',
        xaxis_range=[0, 50]
    )
    
    return fig


def plot_prediction_confidence(avg_predictions):
    """Plot prediction confidence"""
    classes = ['Training', 'Online']
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=avg_predictions,
            marker_color=['#3498db', '#e74c3c'],
            text=[f'{p:.2%}' for p in avg_predictions],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis_range=[0, 1.1],
        height=350,
        template='plotly_white'
    )
    
    return fig


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown('<div class="main-header">🧠 EEG Signal Classification</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧠 EEG Classifier")
        st.markdown("---")
        
        page = st.radio(
            "📍 Navigation:",
            ["🏠 Home", "📊 Model Demo", "📈 Model Comparison", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.info(f"""
        **Preprocessing:**
        - SR: {SAMPLING_RATE} Hz
        - Filter: {LOWCUT}-{HIGHCUT} Hz
        - Epoch: {EPOCH_LENGTH}s
        """)
        
        st.markdown("---")
        st.markdown("### 📦 Models")
        models = get_available_models()
        if models:
            st.success(f"✅ {len(models)} model(s) available")
            for m in models:
                st.text(f"• {m}")
        else:
            st.warning("⚠️ No models found")
    
    # Main Content
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Model Demo":
        show_demo_page()
    elif page == "📈 Model Comparison":
        show_comparison_page()
    elif page == "ℹ️ About":
        show_about_page()


# ==================== PAGE: HOME ====================

def show_home_page():
    st.markdown('<div class="sub-header">Welcome to EEG Signal Classification App</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models", len(get_available_models()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Channels", "20")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Classes", "2")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Project Overview")
    st.markdown("""
    Aplikasi ini mendemonstrasikan model deep learning untuk klasifikasi sinyal EEG.
    Model dilatih untuk mengklasifikasikan sinyal EEG ke dalam dua kategori:
    
    - **Training Data**: Sinyal dari sesi training
    - **Online Data**: Sinyal dari sesi online/testing
    """)
    
    st.markdown("### 🤖 Available Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **CNN1D**
        - 1D Convolutional Neural Network
        - Ekstraksi fitur temporal lokal
        - Cocok untuk pattern recognition
        
        **LSTM**
        - Long Short-Term Memory
        - Menangkap dependensi temporal jangka panjang
        - Bidirectional architecture
        """)
    
    with col2:
        st.markdown("""
        **CNN-LSTM Hybrid**
        - Kombinasi CNN dan LSTM
        - Fitur spatial-temporal
        - Best of both worlds
        
        **EEGNet**
        - State-of-the-art untuk EEG
        - Depthwise separable convolutions
        - Efficient dan akurat
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    
    st.markdown("""
    1. 📊 Klik **Model Demo** di sidebar
    2. 📁 Upload file CSV berisi sinyal EEG Anda
    3. 🤖 Pilih model yang ingin digunakan
    4. 🎯 Klik **Preprocess & Predict**
    5. 📈 Lihat hasil prediksi dan visualisasi!
    """)
    
    st.info("💡 **Tip**: Gunakan file `sample_eeg.csv` di folder `data/` untuk testing")


# ==================== PAGE: DEMO ====================

def show_demo_page():
    st.markdown('<div class="sub-header">Model Demo & Prediction</div>', unsafe_allow_html=True)
    
    # Check TensorFlow
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow tidak terinstal!")
        st.info("Install dengan: `pip install tensorflow`")
        return
    
    # Check available models
    available_models = get_available_models()
    
    if not available_models:
        st.warning("⚠️ Tidak ada model yang ditemukan!")
        st.info("""
        **Cara menambahkan model:**
        1. Pastikan folder `models/` ada
        2. Upload file model dengan format: `NamaModel_final.h5`
        3. Contoh: `CNN1D_final.h5`, `LSTM_final.h5`
        """)
        return
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "🤖 Select Model:",
            available_models,
            help="Pilih model untuk prediksi"
        )
    
    with col2:
        st.metric("Model Status", "✅ Ready" if selected_model else "❌ None")
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📁 Upload EEG Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file (samples × channels)",
        type=['csv'],
        help="File CSV dengan shape (n_samples, n_channels). Expected: 20 channels"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = load_eeg_data(uploaded_file)
            
            if data is None:
                return
            
            # Display info
            display_data_info(data)
            
            # Visualize raw signal
            st.markdown("### 📊 Raw Signal Preview")
            fig_signal = plot_eeg_signal(data[:1000], "Raw EEG Signal (First 5 seconds)")
            st.plotly_chart(fig_signal, use_container_width=True)
            
            # Predict button
            st.markdown("---")
            if st.button("🚀 Preprocess & Predict", type="primary", use_container_width=True):
                predict_and_display(data, selected_model)
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)


def load_eeg_data(uploaded_file):
    """Load and validate EEG data from uploaded file"""
    try:
        # Try loading without header
        data = pd.read_csv(uploaded_file, header=None).values
    except:
        try:
            # Try with header
            df = pd.read_csv(uploaded_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data = df[numeric_cols].values
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            return None
    
    # Validate
    if len(data.shape) != 2:
        st.error(f"❌ Data harus 2D (samples × channels), got: {data.shape}")
        return None
    
    if data.shape[1] != 20:
        st.warning(f"⚠️ Expected 20 channels, got {data.shape[1]}")
    
    if data.shape[0] < 800:
        st.error(f"❌ Data terlalu pendek! Min 800 samples, got {data.shape[0]}")
        st.info(f"Duration: {data.shape[0]/SAMPLING_RATE:.2f}s (need ≥4s)")
        return None
    
    # Check for NaN/Inf
    if np.isnan(data).any() or np.isinf(data).any():
        st.warning("⚠️ Data contains NaN/Inf values. Replacing with zeros...")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    st.success(f"✅ Data loaded: {data.shape[0]:,} samples × {data.shape[1]} channels")
    
    return data


def display_data_info(data):
    """Display data information"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", f"{data.shape[0]:,}")
    with col2:
        st.metric("Channels", data.shape[1])
    with col3:
        st.metric("Duration", f"{data.shape[0]/SAMPLING_RATE:.1f}s")
    with col4:
        epochs = data.shape[0] // (SAMPLING_RATE * EPOCH_LENGTH)
        st.metric("Epochs", epochs)
    
    with st.expander("📊 Data Statistics"):
        stats = pd.DataFrame({
            'Mean': [f"{data.mean():.4f}"],
            'Std': [f"{data.std():.4f}"],
            'Min': [f"{data.min():.4f}"],
            'Max': [f"{data.max():.4f}"],
            'NaN': [np.isnan(data).sum()]
        })
        st.dataframe(stats, use_container_width=True)


def predict_and_display(data, model_name):
    """Preprocess data and make predictions"""
    
    # Preprocess
    epochs = preprocess_eeg_data(data)
    st.success(f"✅ Created {epochs.shape[0]} epochs")
    
    # Load model
    model = load_model(model_name)
    
    if model is None:
        st.error("❌ Model gagal di-load. Tidak bisa melakukan prediksi.")
        return
    
    # Predict
    with st.spinner("🔮 Predicting..."):
        predictions = model.predict(epochs, verbose=0)
    
    # Calculate average
    avg_pred = np.mean(predictions, axis=0)
    pred_class = np.argmax(avg_pred)
    confidence = avg_pred[pred_class]
    
    # Display results
    st.markdown("---")
    st.markdown("### 🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        class_name = "Training" if pred_class == 0 else "Online"
        st.metric("Predicted Class", class_name)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    
    with col3:
        st.metric("Epochs", epochs.shape[0])
    
    # Confidence plot
    fig_conf = plot_prediction_confidence(avg_pred)
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Per-epoch predictions
    with st.expander("📋 Detailed Per-Epoch Predictions"):
        pred_df = pd.DataFrame({
            'Epoch': range(1, len(predictions) + 1),
            'Training Prob': [f"{p[0]:.4f}" for p in predictions],
            'Online Prob': [f"{p[1]:.4f}" for p in predictions],
            'Predicted': ['Training' if p[0] > p[1] else 'Online' for p in predictions]
        })
        st.dataframe(pred_df, use_container_width=True, height=300)
    
    # Visualizations
    st.markdown("### 📊 Signal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_processed = plot_eeg_signal(epochs[0], "Preprocessed Epoch 1", max_channels=5)
        st.plotly_chart(fig_processed, use_container_width=True)
    
    with col2:
        fig_psd = plot_power_spectrum(epochs[0], channel=0)
        if fig_psd:
            st.plotly_chart(fig_psd, use_container_width=True)


# ==================== PAGE: COMPARISON ====================

def show_comparison_page():
    st.markdown('<div class="sub-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    
    # Try multiple possible locations
    possible_paths = [
        "assets/model_comparison_detailed.csv",
        "model_comparison_detailed.csv",
        "results/model_comparison_detailed.csv"
    ]
    
    results_file = None
    for path in possible_paths:
        if os.path.exists(path):
            results_file = path
            break
    
    if results_file and os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            
            st.markdown("### 📊 Performance Metrics")
            st.dataframe(df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']), 
                        use_container_width=True)
            
            # Visualization
            create_comparison_chart(df)
            
            # Best model
            if 'F1-Score' in df.columns:
                best_idx = pd.to_numeric(df['F1-Score'], errors='coerce').idxmax()
                best_model = df.loc[best_idx, 'Model']
                best_f1 = df.loc[best_idx, 'F1-Score']
                st.success(f"🏆 **Best Model:** {best_model} (F1-Score: {best_f1})")
            
            # Confusion matrices
            show_confusion_matrices()
        
        except Exception as e:
            st.error(f"Error loading comparison data: {e}")
    else:
        show_comparison_placeholder()


def create_comparison_chart(df):
    """Create comparison bar chart"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    fig = go.Figure()
    
    for metric, color in zip(metrics, colors):
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors='coerce')
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Model'],
                y=values,
                marker_color=color,
                text=[f'{v:.3f}' for v in values],
                textposition='outside'
            ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1.1],
        barmode='group',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_confusion_matrices():
    """Show confusion matrix images"""
    st.markdown("### 🔢 Confusion Matrices")
    
    cm_paths = [
        "assets/confusion_matrices.png",
        "confusion_matrices.png",
        "results/confusion_matrices.png"
    ]
    
    cm_image = None
    for path in cm_paths:
        if os.path.exists(path):
            cm_image = path
            break
    
    if cm_image:
        st.image(cm_image, caption="Confusion Matrices", use_column_width=True)
    else:
        st.info("📊 Confusion matrix visualization not available")


def show_comparison_placeholder():
    """Show placeholder when no comparison data"""
    st.warning("⚠️ Model comparison data not found!")
    
    st.info("""
    **Untuk generate comparison results:**
    
    1. Train semua model (CNN1D, LSTM, CNN-LSTM, EEGNet)
    2. Jalankan evaluation script
    3. Download `model_comparison_detailed.csv`
    4. Upload ke folder `assets/` atau root directory
    """)
    
    # Show example structure
    st.markdown("### 📋 Expected Data Format")
    example_df = pd.DataFrame({
        'Model': ['CNN1D', 'LSTM', 'CNN-LSTM', 'EEGNet'],
        'Accuracy': [0.85, 0.88, 0.90, 0.92],
        'Precision': [0.84, 0.87, 0.89, 0.91],
        'Recall': [0.86, 0.89, 0.91, 0.93],
        'F1-Score': [0.85, 0.88, 0.90, 0.92]
    })
    st.dataframe(example_df, use_container_width=True)


# ==================== PAGE: ABOUT ====================

def show_about_page():
    st.markdown('<div class="sub-header">About This Project</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📖 Overview", "🔬 Methodology", "👥 Team"])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_methodology_tab()
    
    with tab3:
        show_team_tab()


def show_overview_tab():
    st.markdown("""
    ### 📋 Project Information
    
    **Title:** Deep Learning for EEG Signal Classification
    
    **Objective:** Mengembangkan dan membandingkan berbagai arsitektur deep learning 
    untuk klasifikasi sinyal EEG ke dalam kategori training dan online.
                """)