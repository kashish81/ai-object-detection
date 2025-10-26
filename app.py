import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
from collections import Counter
import io
import time

# Page configuration - MOBILE OPTIMIZED
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto",  # Auto-collapse on mobile
    menu_items={
        'Get Help': 'https://github.com/kashish81/ai-object-detection',
        'Report a bug': "https://github.com/kashish81/ai-object-detection/issues",
        'About': "AI Object Detection System - Built with YOLOv8 & Streamlit"
    }
)

# Enhanced Custom CSS - Modern Design
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4facfe;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling - BIGGER & BOLDER */
    .main-header {
        font-size: 5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(102, 126, 234, 0.3);
        animation: fadeInDown 1s ease-in;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.5rem;
        margin-bottom: 3rem;
        font-weight: 600;
        animation: fadeInUp 1s ease-in;
        letter-spacing: 0.02em;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Glassmorphism effect for containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    /* Button styling - BIGGER */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 1rem 2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        font-size: 1.2rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="metric-container"] label {
        font-weight: 700 !important;
        color: #475569 !important;
        font-size: 1.1rem !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea05 0%, #764ba205 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        font-weight: 700;
        color: #ffff;
        font-size: 1.1rem;
    }
    
    section[data-testid="stSidebar"] h2 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea05 0%, #764ba205 100%);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 4px solid #667eea;
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Images */
    img {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.02);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Download button special styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-radius: 12px;
        font-weight: 600;
        padding: 1rem;
    }
    
    /* Success message */
    .success-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
        animation: slideInRight 0.5s ease;
        margin: 1rem 0;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-radius: 15px;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* General text improvements */
    body {
        font-size: 16px;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
    }
    
    p, li, span {
        font-size: 1.05rem;
        line-height: 1.7;
    }
    
    .stMarkdown {
        font-size: 1.05rem;
    }
    
    /* Section headers - bigger */
    h3 {
        font-size: 2rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h4 {
        font-size: 1.5rem !important;
        margin-top: 1rem !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5568d3 0%, #65408b 100%);
    }
    
    /* ========================================
       MOBILE RESPONSIVE DESIGN
       ======================================== */
    
    /* Tablets and smaller (max-width: 768px) */
    @media screen and (max-width: 768px) {
        /* Header adjustments */
        .main-header {
            font-size: 3rem !important;
            padding: 0 1rem;
        }
        
        .sub-header {
            font-size: 1.2rem !important;
            padding: 0 1rem;
        }
        
        /* Reduce padding on mobile */
        .block-container {
            padding: 1rem !important;
        }
        
        /* Stack columns vertically */
        div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
        }
        
        /* Buttons bigger on mobile */
        .stButton>button {
            padding: 1.2rem 1rem !important;
            font-size: 1.1rem !important;
        }
        
        /* Metrics stack better */
        div[data-testid="metric-container"] {
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }
        
        /* Sidebar full width when expanded */
        section[data-testid="stSidebar"] {
            width: 100% !important;
        }
        
        /* File uploader easier on mobile */
        .stFileUploader {
            padding: 1.5rem 1rem !important;
        }
        
        /* Images take full width */
        img {
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Better touch targets */
        .stSelectbox, .stSlider {
            font-size: 1.1rem !important;
        }
        
        /* Progress bars thicker */
        .stProgress > div > div {
            height: 12px !important;
        }
    }
    
    /* Mobile phones (max-width: 480px) */
    @media screen and (max-width: 480px) {
        /* Even smaller header */
        .main-header {
            font-size: 2.5rem !important;
            line-height: 1.2 !important;
        }
        
        .sub-header {
            font-size: 1rem !important;
        }
        
        /* Compact metrics */
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
        }
        
        div[data-testid="metric-container"] label {
            font-size: 0.9rem !important;
        }
        
        /* Smaller buttons */
        .stButton>button {
            padding: 1rem 0.8rem !important;
            font-size: 1rem !important;
        }
        
        /* Compact sidebar */
        section[data-testid="stSidebar"] h2 {
            font-size: 1.4rem !important;
        }
        
        /* Reduce spacing */
        .block-container {
            padding: 0.5rem !important;
        }
        
        /* Camera input bigger on mobile */
        video {
            max-width: 100% !important;
            height: auto !important;
        }
    }
    
    /* Landscape mode on phones */
    @media screen and (max-height: 500px) and (orientation: landscape) {
        .main-header {
            font-size: 2rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        .sub-header {
            font-size: 0.9rem !important;
            margin-bottom: 1rem !important;
        }
        
        section[data-testid="stSidebar"] {
            max-height: 100vh !important;
            overflow-y: auto !important;
        }
    }
    
    /* Large screens optimization */
    @media screen and (min-width: 1920px) {
        .block-container {
            max-width: 1600px !important;
            margin: 0 auto !important;
        }
    }
    
    /* Touch device optimizations */
    @media (hover: none) and (pointer: coarse) {
        /* Bigger touch targets */
        .stButton>button {
            min-height: 50px !important;
        }
        
        .stFileUploader {
            min-height: 120px !important;
        }
        
        /* Remove hover effects on touch devices */
        .stButton>button:hover {
            transform: none !important;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: none !important;
        }
        
        img:hover {
            transform: none !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching"""
    model = YOLO('yolov8n.pt')
    return model

def detect_objects(model, image, conf_threshold):
    """Detect objects in image"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    results = model(image, conf=conf_threshold)
    result = results[0]
    annotated_img = result.plot()
    
    detected_objects = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        
        detected_objects.append({
            'class': class_name,
            'confidence': confidence
        })
    
    return result, detected_objects, annotated_img

def main():
    # Animated Header - MOBILE RESPONSIVE
    st.markdown('''
    <div style="text-align: center; padding: 2rem 1rem;">
        <h1 style="
            font-size: clamp(2.5rem, 8vw, 5.5rem);
            font-weight: 900;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            padding: 0;
            letter-spacing: -0.03em;
            text-shadow: 0 0 40px rgba(102, 126, 234, 0.3);
            animation: glow 2s ease-in-out infinite alternate;
            line-height: 1.1;
        ">
            AI Object Detection
        </h1>
        <div style="
            font-size: clamp(0.9rem, 3vw, 1.2rem);
            color: #94a3b8;
            margin-top: 0.5rem;
            padding: 0 1rem;
        ">
            Real-time Object Recognition • 80+ Classes • High Accuracy
        </div>
    </div>
    <style>
    @keyframes glow {
        from {
            text-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }
        to {
            text-shadow: 0 0 40px rgba(102, 126, 234, 0.6), 0 0 60px rgba(240, 147, 251, 0.4);
        }
    }
    </style>
    ''', unsafe_allow_html=True)
    
    # Load model with spinner
    with st.spinner('🚀 Initializing AI Model...'):
        model = load_model()
    
    # Sidebar with modern styling
    with st.sidebar:
        st.markdown("## ⚙️ Configuration Panel")
        st.markdown("---")
        
        # Detection mode with emoji icons
        detection_mode = st.selectbox(
            "🎬 Select Detection Mode",
            ["📸 Image Detection", "🎥 Video Detection", "📹 Webcam Detection"],
            help="Choose your preferred detection method"
        )
        
        st.markdown("---")
        
        # Confidence threshold with dynamic emoji
        confidence = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher = More accurate but fewer detections"
        )
        
        # Visual indicator for confidence
        if confidence < 0.3:
            st.info("🔵 Low - More detections, some false positives")
        elif confidence < 0.7:
            st.success("🟢 Balanced - Recommended setting")
        else:
            st.warning("🟡 High - Very accurate, may miss some objects")
        
        st.markdown("---")
        
        # Model stats in expandable section
        with st.expander("📊 Model Statistics", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Classes", len(model.names), help="Total object types")
                st.metric("Speed", "Fast ⚡", help="Real-time capable")
            with col2:
                st.metric("Accuracy", "High 🎯", help="85-95% accurate")
                st.metric("Model", "YOLOv8n", help="Nano version")
        
        st.markdown("---")
        
        # About section with better formatting
        with st.expander("ℹ️ About This App", expanded=False):
            st.markdown("""
            ### 🧠 Deep Learning Object Detection
            
            This application uses **YOLOv8** (You Only Look Once) 
            neural network to detect and identify objects in real-time.
            
            **✨ Key Features:**
            - 🎯 80+ object classes
            - ⚡ Real-time processing
            - 🖼️ Multiple input formats
            - 📊 Detailed analytics
            - 💾 Export results
            
            **🎓 Technology Stack:**
            - YOLOv8 CNN
            - Transfer Learning
            - Python + Streamlit
            - OpenCV
            """)
        
        # Quick tips
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - 📸 Use clear, well-lit images
        - 🎯 Adjust threshold for best results
        - 💾 Download your results
        - 🔄 Try different modes
        """)
    
    # Main content with tabs
    st.markdown("---")
    
    # IMAGE DETECTION MODE
    if detection_mode == "📸 Image Detection":
        st.markdown("### 📸 Image Object Detection")
        st.markdown("Upload an image to detect objects with AI-powered analysis")
        st.markdown("")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### 📤 Upload Your Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Supported formats: JPG, JPEG, PNG",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="✨ Original Image", use_container_width=True)
                
                # Image info
                st.caption(f"📐 Size: {image.size[0]}×{image.size[1]} | 📁 Format: {image.format}")
                
                st.markdown("")
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner('🤖 AI is analyzing your image...'):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        result, detected_objects, annotated_img = detect_objects(
                            model, image, confidence
                        )
                        
                        st.session_state['detected_img'] = annotated_img
                        st.session_state['detected_objects'] = detected_objects
                        st.session_state['detection_done'] = True
                        
                        st.balloons()
        
        with col2:
            st.markdown("#### 🎯 Detection Results")
            
            if st.session_state.get('detection_done', False):
                st.image(
                    st.session_state['detected_img'],
                    caption="🎯 Detected Objects",
                    use_container_width=True
                )
                
                detected_objects = st.session_state['detected_objects']
                
                if detected_objects:
                    st.markdown('<div class="success-banner">✅ Detection Completed Successfully!</div>', 
                              unsafe_allow_html=True)
                    
                    # Metrics in beautiful cards
                    st.markdown("#### 📊 Quick Statistics")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("🎯 Total Objects", len(detected_objects))
                    with col_b:
                        avg_conf = np.mean([obj['confidence'] for obj in detected_objects])
                        st.metric("📈 Avg Confidence", f"{avg_conf*100:.1f}%")
                    with col_c:
                        unique = len(set([obj['class'] for obj in detected_objects]))
                        st.metric("🏷️ Unique Types", unique)
                    
                    st.markdown("")
                    
                    # Detailed breakdown
                    st.markdown("#### 📋 Detailed Breakdown")
                    object_counts = Counter([obj['class'] for obj in detected_objects])
                    
                    for obj_class, count in object_counts.most_common():
                        avg_conf = np.mean([obj['confidence'] for obj in detected_objects 
                                           if obj['class'] == obj_class])
                        
                        # Progress bar for confidence
                        st.markdown(f"**{obj_class.title()}** × {count}")
                        st.progress(avg_conf, text=f"Confidence: {avg_conf*100:.1f}%")
                        st.markdown("")
                    
                    # Download section
                    st.markdown("---")
                    result_pil = Image.fromarray(st.session_state['detected_img'])
                    buf = io.BytesIO()
                    result_pil.save(buf, format='JPEG')
                    
                    st.download_button(
                        label="💾 Download Result",
                        data=buf.getvalue(),
                        file_name="detected_image.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ No objects detected. Try lowering the confidence threshold!")
            else:
                st.info("👈 Upload an image and click 'Analyze' to begin detection")
                st.markdown("")
                st.image("https://via.placeholder.com/500x300/667eea/ffffff?text=Awaiting+Detection", 
                        use_container_width=True)
    
    # VIDEO DETECTION MODE
    elif detection_mode == "🎥 Video Detection":
        st.markdown("### 🎥 Video Object Detection")
        st.markdown("Process videos frame-by-frame with AI detection")
        st.markdown("")
        
        uploaded_video = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_video is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📹 Original Video")
                st.video(uploaded_video)
            
            with col2:
                st.markdown("#### ⚙️ Processing Options")
                st.info("Video processing analyzes each frame individually")
                
                if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    
                    st.markdown("---")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    cap = cv2.VideoCapture(tfile.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    frame_count = 0
                    start_time = time.time()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        results = model(frame, conf=confidence, verbose=False)
                        annotated_frame = results[0].plot()
                        out.write(annotated_frame)
                        
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        
                        elapsed = time.time() - start_time
                        eta = (elapsed / frame_count) * (total_frames - frame_count)
                        status_text.markdown(f"**Processing:** Frame {frame_count}/{total_frames} | "
                                           f"⏱️ ETA: {eta:.1f}s")
                    
                    cap.release()
                    out.release()
                    
                    status_text.markdown('<div class="success-banner">✅ Video Processing Complete!</div>', 
                                       unsafe_allow_html=True)
                    st.balloons()
                    
                    st.markdown("#### 🎬 Processed Video")
                    st.video(output_path)
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="💾 Download Processed Video",
                            data=f,
                            file_name="detected_video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
    
    # WEBCAM DETECTION MODE
    else:  # Webcam Detection
        st.markdown("### 📹 Live Webcam Detection")
        st.markdown("Capture snapshots from your camera and detect objects instantly")
        st.markdown("")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### 📷 Camera Feed")
            camera_image = st.camera_input("Enable your camera")
            
            if camera_image:
                image = Image.open(camera_image)
                st.caption(f"📐 Captured: {image.size[0]}×{image.size[1]} pixels")
        
        with col2:
            st.markdown("#### 🎯 Live Detection")
            
            if camera_image is not None:
                if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
                    with st.spinner('🤖 Analyzing snapshot...'):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        result, detected_objects, annotated_img = detect_objects(
                            model, image, confidence
                        )
                        
                        st.session_state['webcam_detected'] = True
                        st.session_state['webcam_objects'] = detected_objects
                        st.session_state['webcam_image'] = annotated_img
                        
                        st.balloons()
            else:
                st.info("👈 Enable camera and capture an image")
        
        # Show results
        if st.session_state.get('webcam_detected', False):
            st.markdown("---")
            st.markdown("### 🎯 Detection Results")
            
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.image(st.session_state['webcam_image'], 
                        caption="Detected Objects", use_container_width=True)
            
            with col_b:
                detected_objects = st.session_state['webcam_objects']
                
                if detected_objects:
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Total", len(detected_objects))
                    with col2:
                        avg_conf = np.mean([obj['confidence'] for obj in detected_objects])
                        st.metric("📈 Avg Conf", f"{avg_conf*100:.0f}%")
                    with col3:
                        unique = len(set([obj['class'] for obj in detected_objects]))
                        st.metric("🏷️ Types", unique)
                    with col4:
                        max_conf = max([obj['confidence'] for obj in detected_objects])
                        st.metric("🔝 Max", f"{max_conf*100:.0f}%")
                    
                    st.markdown("")
                    
                    # Object list
                    object_counts = Counter([obj['class'] for obj in detected_objects])
                    for obj_class, count in object_counts.most_common():
                        avg_conf = np.mean([obj['confidence'] for obj in detected_objects 
                                           if obj['class'] == obj_class])
                        st.markdown(f"**{obj_class.title()}** × {count}")
                        st.progress(avg_conf, text=f"{avg_conf*100:.1f}%")
                    
                    st.markdown("")
                    
                    # Download
                    result_pil = Image.fromarray(st.session_state['webcam_image'])
                    buf = io.BytesIO()
                    result_pil.save(buf, format='JPEG')
                    
                    st.download_button(
                        label="💾 Download Result",
                        data=buf.getvalue(),
                        file_name="webcam_detection.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                        key="webcam_download"
                    )
                else:
                    st.warning("⚠️ No objects detected")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;'>
                   Deep Learning Project
        </h3>
        <p style='color: #64748b; margin-top: 0.5rem;'>
            Built with ❤️ by <b>Kashish Rajan </b> & <b>Divyanshi Verma</b>
        </p>
        <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 1rem;'>
            Powered by Deep Learning | Transfer Learning | Computer Vision
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()