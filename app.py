import os
import random
import cv2
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# 1. Page Configuration
st.set_page_config(
    page_title="PotholeEye - Deteksi Lubang Jalan",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Injection (Rich Glassmorphism & Cyberpunk-inspired Dark Theme)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.2);
    }
    .sub-title {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 30px;
    }
    
    /* Card Glassmorphism layout */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 25px;
        margin-bottom: 25px;
    }
    
    /* Glowing Metrics Card */
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #38bdf8;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Severity Status Styling */
    .status-safe {
        border-left: 5px solid #10b981;
        background: rgba(16, 185, 129, 0.1);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
    }
    .status-warning {
        border-left: 5px solid #f59e0b;
        background: rgba(245, 158, 11, 0.1);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
    }
    .status-danger {
        border-left: 5px solid #ef4444;
        background: rgba(239, 68, 68, 0.1);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    /* Footer styles */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        color: #64748b;
        font-size: 0.85rem;
    }
    
    /* Adjust sidebar background */
    .css-1d391tw, [data-testid="stSidebar"] {
        background-color: #070a12;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Navigation & Parameters Setup
with st.sidebar:
    st.markdown("<h2 style='color: #38bdf8;'>Pengaturan Model</h2>", unsafe_allow_html=True)
    
    # Model Weights Selection
    weights_dir = Path("runs/detect/pothole_detection/yolov8n_ep30_tuned/weights")
    default_model = weights_dir / "best.pt"

    # Fallback to local weights if not existing
    if not default_model.exists():
        default_model = Path("yolov8n.pt")

    st.info(f"Model aktif: {default_model.name}")
    
    # Confidence and IoU Threshold sliders
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
                               help="Batas minimum skor keyakinan untuk mendeteksi lubang")
    iou_threshold = st.slider("IoU Threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05,
                              help="Batas tumpang tindih bounding box (NMS)")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #94a3b8;'>Navigasi Menu</h3>", unsafe_allow_html=True)
    menu = st.radio("Pilih Halaman:", [
        "Beranda (Overview)",
        "Deteksi Lubang Jalan",
        # "Hasil Analisis Deteksi",
        # "Kinerja Model (Evaluasi)"
    ])

# 4. Helper function to load model
@st.cache_resource
def load_yolo_model(path):
    return YOLO(str(path))

try:
    model = load_yolo_model(default_model)
except Exception as e:
    st.error(f"Gagal memuat model YOLO: {e}")
    st.stop()

# Cache detection results between pages to display on the Analysis page
if 'last_detection' not in st.session_state:
    st.session_state['last_detection'] = {
        'has_run': False,
        'image_name': None,
        'pothole_count': 0,
        'boxes': [],
        'orig_img': None,
        'annotated_img': None
    }

def render_detection_analysis(detect_data):
    if not detect_data['has_run']:
        st.markdown("""
            <div class='glass-card' style='text-align: center; padding: 50px 20px;'>
                <h3 style='color: #64748b;'>Belum Ada Hasil Deteksi</h3>
                <p style='color: #475569;'>Silakan lakukan deteksi terlebih dahulu.</p>
            </div>
        """, unsafe_allow_html=True)
        return

    pothole_count = detect_data['pothole_count']
    image_name = detect_data['image_name']
    boxes = detect_data['boxes']

    if pothole_count == 0:
        status_class = "status-safe"
        status_label = "AMAN"
        status_desc = "Jalan terdeteksi mulus dan aman dilewati. Tidak ditemukan lubang jalan pada area pantauan."
    elif pothole_count <= 2:
        status_class = "status-warning"
        status_label = "HATI-HATI"
        status_desc = "Terdeteksi sedikit kerusakan jalan (1-2 lubang). Kemudikan kendaraan dengan kecepatan sedang."
    else:
        status_class = "status-danger"
        status_label = "BAHAYA"
        status_desc = "Terdeteksi banyak lubang jalan (3+ lubang). Sangat rawan kecelakaan. Kurangi kecepatan dan hindari lubang."

    st.markdown(f"""
        <div class='glass-card {status_class}'>
            <div class='metric-label'>Status Kerusakan Jalan ({image_name})</div>
            <div style='font-size: 2.2rem; font-weight: 800; margin: 5px 0;'>STATUS: {status_label}</div>
            <p style='margin: 0; color: #cbd5e1;'>{status_desc}</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        # st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### Daftar Deteksi")

        if pothole_count > 0:
            df_data = []
            for b in boxes:
                x1, y1, x2, y2 = b['bbox']
                df_data.append({
                    "ID": b['id'],
                    "Akurasi": f"{b['confidence']:.1%}",
                    "Posisi Bbox": f"({x1}, {y1}) s/d ({x2}, {y2})",
                    "Ukuran (px)": f"{b['width']}x{b['height']}",
                    "Luas (px²)": b['area']
                })
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada lubang jalan yang dianalisis.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("### Statistik Sebaran Akurasi")

        if pothole_count > 0:
            confidences = [b['confidence'] for b in boxes]
            fig = px.histogram(
                x=confidences,
                nbins=10,
                labels={'x': 'Skor Keyakinan', 'y': 'Jumlah Deteksi'},
                title="Distribusi Confidence Score",
                color_discrete_sequence=['#38bdf8']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#cbd5e1',
                margin=dict(l=20, r=20, t=40, b=20),
                height=280
            )
            fig.update_xaxes(range=[0.1, 1.0])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Grafik tidak tersedia karena 0 lubang terdeteksi.")
        st.markdown("</div>", unsafe_allow_html=True)

    if pothole_count > 0:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Analisis Ukuran Lubang Jalan")

        ids = [f"Lubang #{b['id']}" for b in boxes]
        areas = [b['area'] for b in boxes]

        fig_bar = px.bar(
            x=ids,
            y=areas,
            labels={'x': 'ID Lubang', 'y': 'Luas'},
            title="Luas Relatif Lubang Jalan yang Terdeteksi",
            color=areas,
            color_continuous_scale='blues'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 1: BERANDA -----------------
if menu == "Beranda (Overview)":
    st.markdown("<h1 class='main-title'>PotholeEye Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Sistem Deteksi Lubang Jalan Berbasis AI Menggunakan YOLOv8</p>", unsafe_allow_html=True)
    
    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
            <div class='glass-card' style='text-align: center;'>
                <div class='metric-value'>1.581</div>
                <div class='metric-label'>Gambar Training</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
            <div class='glass-card' style='text-align: center;'>
                <div class='metric-value'>396</div>
                <div class='metric-label'>Gambar Validasi</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
            <div class='glass-card' style='text-align: center;'>
                <div class='metric-value'>1</div>
                <div class='metric-label'>Kelas Target (Pothole)</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
            <div class='glass-card' style='text-align: center;'>
                <div class='metric-value'>YOLOv8n</div>
                <div class='metric-label'>Arsitektur Model</div>
            </div>
        """, unsafe_allow_html=True)

    # Main Intro and description
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
            <div class='glass-card'>
                <h3 style='color: #38bdf8; margin-top: 0;'>Tentang Proyek</h3>
                <p style='line-height: 1.6; color: #cbd5e1;'>
                    Kerusakan jalan raya, khususnya <b>lubang jalan (potholes)</b>, merupakan salah satu penyebab utama kecelakaan lalu lintas dan kerusakan kendaraan di Indonesia. Proyek <b>PotholeEye</b> dikembangkan untuk mengotomatiskan identifikasi lubang jalan secara real-time menggunakan kecerdasan buatan (deep learning).
                </p>
                <p style='line-height: 1.6; color: #cbd5e1;'>
                    Menggunakan arsitektur model <b>YOLOv8 (You Only Look Once v8)</b> yang ringan namun akurat, model ini dapat mendeteksi kerusakan jalan secara otomatis, memberikan laporan posisi lubang, dan meminimalkan risiko kecelakaan.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='glass-card'>
                <h3 style='color: #10b981; margin-top: 0;'>Fitur Utama Dashboard</h3>
                <ul style='line-height: 1.8; color: #cbd5e1; padding-left: 20px;'>
                    <li><b>Deteksi Presisi Tinggi:</b> Unggah foto jalan apa pun, model akan menggambar bounding box di sekitar lubang.</li>
                    <li><b>Skoring Tingkat Bahaya:</b> Mengkategorikan tingkat bahaya jalan (Aman, Hati-hati, Bahaya) berdasarkan jumlah lubang jalan yang terdeteksi.</li>
                    <li><b>Analisis Spasial & Dimensi:</b> Menyajikan analisis tabel deteksi berisi lokasi piksel dan luas lubang di gambar.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Check if we can display results sample image
        st.markdown("""
            <div class='glass-card'>
                <h4 style='color: #E62626; margin-top: 0;'>Anggota Kelompok B</h4>
                <ul style='line-height: 1.7; color: #cbd5e1; padding-left: 15px;'>
                    <li>Komang Krisna Jaya Nova Antara (2308561029)</li>
                    <li>Anak Agung Gde Agung Pranandita	(2308561106)</li>
                    <li>I Putu Chandra Ananda Putra.S (2308561126)</li>
                    <li>BENEDIKTUS SILABAN (2308561139)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        sample_img_path = Path("runs/detect/pothole_detection/yolov8n_ep30_tuned/train_batch0.jpg")
        if sample_img_path.exists():
            # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.image(str(sample_img_path), caption="Sampel Batch Data Pelatihan", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='glass-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;'>
                    <h4 style='color: #64748b;'>Model Trained Output</h4>
                    <p style='color: #475569;'>Visualisasi pelatihan akan tampil jika file batch training tersedia.</p>
                </div>
            """, unsafe_allow_html=True)

# ----------------- PAGE 2: DETEKSI LUBANG -----------------
elif menu == "Deteksi Lubang Jalan":
    st.markdown("<h1 class='main-title'>Deteksi Lubang Jalan</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Unggah gambar atau gunakan sampel untuk memulai deteksi lubang jalan secara otomatis.</p>", unsafe_allow_html=True)
    
    # Options for Image Input
    # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])
    with c2:
        st.markdown("<p style='margin-bottom: 8px; color:#94a3b8;'>Atau coba sampel instan:</p>", unsafe_allow_html=True)
        use_demo = st.button("Gunakan Gambar Demo")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle Demo Image logic
    if use_demo:
        demo_dir = Path("data/valid/images")
        if demo_dir.exists():
            demo_images = list(demo_dir.glob("*.jpg"))
            if demo_images:
                chosen_demo = random.choice(demo_images)
                st.session_state["demo_image_path"] = str(chosen_demo)
                st.session_state["demo_image_name"] = chosen_demo.name
            else:
                st.warning("Folder 'data/valid/images' kosong.")
        else:
            st.warning("Folder data validasi tidak ditemukan secara lokal.")

    # Selected image to process
    img_to_process = None
    img_name = ""

    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file)
        img_name = uploaded_file.name
        st.session_state["demo_image_path"] = None
        st.session_state["demo_image_name"] = ""
    elif st.session_state.get("demo_image_path"):
        demo_path = Path(st.session_state["demo_image_path"])
        if demo_path.exists():
            img_to_process = Image.open(demo_path)
            img_name = st.session_state["demo_image_name"]
            st.success(f"Menggunakan gambar demo: `{img_name}`")
        else:
            st.session_state["demo_image_path"] = None
            st.session_state["demo_image_name"] = ""
        
    if img_to_process is not None:
        # Layout for Detection Comparison
        # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        # Convert PIL image to numpy array for inference
        img_np = np.array(img_to_process)
        
        # Run prediction
        with st.spinner("Model sedang memproses gambar..."):
            results = model.predict(img_np, conf=conf_threshold, iou=iou_threshold)
            
        result = results[0]
        
        # Get annotated image
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        # Extract detected boxes details
        boxes_list = []
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            width = x2 - x1
            height = y2 - y1
            area = width * height
            boxes_list.append({
                "id": i + 1,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "width": int(width),
                "height": int(height),
                "area": int(area)
            })
            
        # Store in session state for Analysis page
        st.session_state['last_detection'] = {
            'has_run': True,
            'image_name': img_name,
            'pothole_count': len(boxes_list),
            'boxes': boxes_list,
            'orig_img': img_np,
            'annotated_img': annotated_rgb
        }
        if "demo_image_path" not in st.session_state:
            st.session_state["demo_image_path"] = None
        if "demo_image_name" not in st.session_state:
            st.session_state["demo_image_name"] = ""
        
        st.markdown("---")
        st.markdown("<h2 class='main-title' style='font-size: 2rem;'>Hasil Analisis Deteksi</h2>", unsafe_allow_html=True)
        render_detection_analysis(st.session_state['last_detection'])
        
        with col1:
            st.markdown("<h4 style='color: #94a3b8; text-align: center;'>Gambar Asli</h4>", unsafe_allow_html=True)
            st.image(img_np, use_container_width=True)
            
        with col2:
            st.markdown("<h4 style='color: #38bdf8; text-align: center;'>Hasil Deteksi YOLOv8</h4>", unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Post-detection quick info
        # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        c_info, c_action = st.columns([3, 1])
        with c_info:
            p_count = len(boxes_list)
            st.markdown(f"### Ditemukan **{p_count}** lubang jalan pada gambar ini.")
            st.markdown("Buka tab **Hasil Analisis Deteksi** di sidebar untuk melihat laporan kondisi jalan secara mendalam.")
        with c_action:
            # Download button for annotated image
            result_img = Image.fromarray(annotated_rgb)
            temp_path = "temp_result.jpg"
            result_img.save(temp_path)
            with open(temp_path, "rb") as file:
                btn = st.download_button(
                    label="Unduh Hasil Deteksi",
                    data=file,
                    file_name=f"detected_{img_name}",
                    mime="image/jpeg"
                )
            if os.path.exists(temp_path):
                os.remove(temp_path)
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        st.markdown("""
            <div class='glass-card' style='text-align: center; padding: 50px 20px;'>
                <h3 style='color: #64748b;'>Belum Ada Gambar yang Dipilih</h3>
                <p style='color: #475569;'>Silakan unggah gambar jalan raya di atas atau klik tombol <b>Gunakan Gambar Demo</b> untuk mencoba.</p>
            </div>
        """, unsafe_allow_html=True)

# ----------------- PAGE 3: HASIL ANALISIS -----------------
# elif menu == "Hasil Analisis Deteksi":
#     st.markdown("<h1 class='main-title'>Hasil Analisis Deteksi</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='sub-title'>Analisis kuantitatif, sebaran tingkat keyakinan, dan penaksiran tingkat bahaya jalan.</p>", unsafe_allow_html=True)
    
#     detect_data = st.session_state['last_detection']
    
#     if detect_data['has_run']:
#         # Upper dashboard metrics
#         pothole_count = detect_data['pothole_count']
#         image_name = detect_data['image_name']
#         boxes = detect_data['boxes']
        
#         # Hazard evaluation
#         if pothole_count == 0:
#             status_class = "status-safe"
#             status_label = "AMAN"
#             status_desc = "Jalan terdeteksi mulus dan aman dilewati. Tidak ditemukan lubang jalan pada area pantauan."
#         elif pothole_count <= 2:
#             status_class = "status-warning"
#             status_label = "HATI-HATI"
#             status_desc = "Terdeteksi sedikit kerusakan jalan (1-2 lubang). Kemudikan kendaraan dengan kecepatan sedang."
#         else:
#             status_class = "status-danger"
#             status_label = "BAHAYA"
#             status_desc = "Terdeteksi banyak lubang jalan (3+ lubang)! Sangat rawan kecelakaan. Kurangi kecepatan dan hindari lubang."
            
#         st.markdown(f"""
#             <div class='glass-card {status_class}'>
#                 <div class='metric-label'>Status Kerusakan Jalan (`{image_name}`)</div>
#                 <div style='font-size: 2.2rem; font-weight: 800; margin: 5px 0;'>STATUS: {status_label}</div>
#                 <p style='margin: 0; color: #cbd5e1;'>{status_desc}</p>
#             </div>
#         """, unsafe_allow_html=True)
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
#             st.markdown("### Daftar Deteksi")
            
#             if pothole_count > 0:
#                 # Build dataframe
#                 df_data = []
#                 for b in boxes:
#                     x1, y1, x2, y2 = b['bbox']
#                     df_data.append({
#                         "ID": b['id'],
#                         "Akurasi": f"{b['confidence']:.1%}",
#                         "Posisi Bbox": f"({x1}, {y1}) s/d ({x2}, {y2})",
#                         "Ukuran (px)": f"{b['width']}x{b['height']}",
#                         "Luas (px²)": b['area']
#                     })
#                 df = pd.DataFrame(df_data)
#                 st.dataframe(df, use_container_width=True, hide_index=True)
#             else:
#                 st.info("Tidak ada lubang jalan yang dianalisis.")
#             st.markdown("</div>", unsafe_allow_html=True)
            
#         with col2:
#             st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
#             st.markdown("### Statistik Sebaran Akurasi")
            
#             if pothole_count > 0:
#                 confidences = [b['confidence'] for b in boxes]
#                 # Plotly histogram
#                 fig = px.histogram(
#                     x=confidences,
#                     nbins=10,
#                     labels={'x': 'Skor Keyakinan (Confidence Score)', 'y': 'Jumlah Deteksi'},
#                     title="Distribusi Confidence Score",
#                     color_discrete_sequence=['#38bdf8']
#                 )
#                 fig.update_layout(
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     plot_bgcolor='rgba(0,0,0,0)',
#                     font_color='#cbd5e1',
#                     margin=dict(l=20, r=20, t=40, b=20),
#                     height=280
#                 )
#                 fig.update_xaxes(range=[0.1, 1.0])
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("Grafik tidak tersedia karena 0 lubang terdeteksi.")
#             st.markdown("</div>", unsafe_allow_html=True)
            
#         # Pothole sizes graph
#         if pothole_count > 0:
#             st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#             st.markdown("### Analisis Ukuran Lubang Jalan (Luas Area Bounding Box)")
            
#             ids = [f"Lubang #{b['id']}" for b in boxes]
#             areas = [b['area'] for b in boxes]
            
#             fig_bar = px.bar(
#                 x=ids,
#                 y=areas,
#                 labels={'x': 'ID Lubang', 'y': 'Luas (Pixel Persegi)'},
#                 title="Luas Relatif Lubang Jalan yang Terdeteksi",
#                 color=areas,
#                 color_continuous_scale='blues'
#             )
#             fig_bar.update_layout(
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 font_color='#cbd5e1',
#                 margin=dict(l=20, r=20, t=40, b=20),
#                 height=300
#             )
#             st.plotly_chart(fig_bar, use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
            
#     else:
#         st.markdown("""
#             <div class='glass-card' style='text-align: center; padding: 50px 20px;'>
#                 <h3 style='color: #64748b;'>Belum Ada Hasil Deteksi</h3>
#                 <p style='color: #475569;'>Buka tab <b> Deteksi Lubang Jalan</b> terlebih dahulu untuk mengunggah gambar dan mendeteksi lubang.</p>
#             </div>
#         """, unsafe_allow_html=True)

# # ----------------- PAGE 4: KINERJA MODEL -----------------
# elif menu == "Kinerja Model (Evaluasi)":
#     st.markdown("<h1 class='main-title'>Kinerja Model Pelatihan</h1>", unsafe_allow_html=True)
#     st.markdown("<p class='sub-title'>Metrik evaluasi pelatihan YOLOv8n yang dijalankan pada dataset lokal.</p>", unsafe_allow_html=True)
    
#     # Check if results.csv exists
#     results_csv_path = Path("runs/detect/pothole_detection/yolov8n_ep30_tuned/results.csv")
    
#     if results_csv_path.exists():
#         df_results = pd.read_csv(results_csv_path)
        
#         # Clean column names (strip spaces)
#         df_results.columns = [c.strip() for c in df_results.columns]
        
#         # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#         st.markdown("### Kurva Pelatihan (Loss Progression)")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             # Box Loss plotting
#             fig_loss = go.Figure()
#             fig_loss.add_trace(go.Scatter(x=df_results['epoch'], y=df_results['train/box_loss'], name='Train Box Loss', line=dict(color='#38bdf8', width=2)))
#             fig_loss.add_trace(go.Scatter(x=df_results['epoch'], y=df_results['val/box_loss'], name='Val Box Loss', line=dict(color='#ef4444', width=2, dash='dash')))
#             fig_loss.update_layout(
#                 title="Progres Box Loss",
#                 xaxis_title="Epoch",
#                 yaxis_title="Loss Value",
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 font_color='#cbd5e1',
#                 height=300
#             )
#             st.plotly_chart(fig_loss, use_container_width=True)
            
#         with col2:
#             # Class Loss plotting
#             fig_cls = go.Figure()
#             fig_cls.add_trace(go.Scatter(x=df_results['epoch'], y=df_results['train/cls_loss'], name='Train Class Loss', line=dict(color='#10b981', width=2)))
#             fig_cls.add_trace(go.Scatter(x=df_results['epoch'], y=df_results['val/cls_loss'], name='Val Class Loss', line=dict(color='#f59e0b', width=2, dash='dash')))
#             fig_cls.update_layout(
#                 title="Progres Class Loss",
#                 xaxis_title="Epoch",
#                 yaxis_title="Loss Value",
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 font_color='#cbd5e1',
#                 height=300
#             )
#             st.plotly_chart(fig_cls, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#         st.markdown("### Kurva Akurasi (mAP50 & Precision/Recall)")
        
#         col3, col4 = st.columns(2)
#         with col3:
#             fig_map = go.Figure()
#             fig_map.add_trace(go.Scatter(x=df_results['epoch'], y=df_results['metrics/mAP50(B)'], name='mAP50', line=dict(color='#a855f7', width=3)))
#             fig_map.add_trace(go.Scatter(x=df_results['epoch'], y=df_results['metrics/mAP50-95(B)'], name='mAP50-95', line=dict(color='#ec4899', width=2, dash='dot')))
#             fig_map.update_layout(
#                 title="mAP Accuracy Metrics",
#                 xaxis_title="Epoch",
#                 yaxis_title="Score",
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 font_color='#cbd5e1',
#                 height=300
#             )
#             st.plotly_chart(fig_map, use_container_width=True)
            
#         with col4:
#             fig_pr = go.Figure()
#             fig_pr.add_trace(go.Scatter(x=df_results['epoch'], y=df_results['metrics/precision(B)'], name='Precision', line=dict(color='#14b8a6', width=2)))
#             fig_pr.add_trace(go.Scatter(x=df_results['epoch'], y=df_results['metrics/recall(B)'], name='Recall', line=dict(color='#f97316', width=2)))
#             fig_pr.update_layout(
#                 title="Precision & Recall Metrics",
#                 xaxis_title="Epoch",
#                 yaxis_title="Score",
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 font_color='#cbd5e1',
#                 height=300
#             )
#             st.plotly_chart(fig_pr, use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
        
#     else:
#         st.warning(f"File log training tidak ditemukan di {results_csv_path}. Jalankan training terlebih dahulu untuk melihat visualisasi kurva.")

#     # Show training visualization images
#     # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#     st.markdown("### Visualisasi Data Training & Label")
    
#     col_batch1, col_batch2 = st.columns(2)
#     labels_jpg = Path("runs/detect/pothole_detection/yolov8n_ep30_tuned/labels.jpg")
#     batch0_jpg = Path("runs/detect/pothole_detection/yolov8n_ep30_tuned/train_batch0.jpg")
    
#     with col_batch1:
#         if labels_jpg.exists():
#             st.image(str(labels_jpg), caption="Distribusi & Korelasi Label", use_container_width=True)
#         else:
#             st.info("Visualisasi labels.jpg tidak ditemukan.")
#     with col_batch2:
#         if batch0_jpg.exists():
#             st.image(str(batch0_jpg), caption="Visualisasi Batch 0 Pelatihan", use_container_width=True)
#         else:
#             st.info("Visualisasi train_batch0.jpg tidak ditemukan.")
            
#     st.markdown("</div>", unsafe_allow_html=True)

# 5. Footer HTML
st.markdown("""
    <div class='footer'>
        <p>PotholeEye Dashboard © 2026 - Siklus Hidup Kecerdasan Buatan Proyek Akhir</p>
    </div>
""", unsafe_allow_html=True)
