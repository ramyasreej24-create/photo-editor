"""
Photo Editor using OpenCV and Streamlit
-----------------------------------------
A simple, interactive photo-editing web app built with Streamlit and OpenCV.

Run with:
    streamlit run app.py
"""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(page_title="Photo Editor - OpenCV & Streamlit", layout="wide")

st.title("📸 Photo Editor using OpenCV and Streamlit")
st.write(
    "Upload an image, tweak it with the controls in the sidebar, "
    "and download your edited result."
)


# --------------------------------------------------------------------------
# Helper functions (all operate on OpenCV BGR numpy arrays)
# --------------------------------------------------------------------------
def pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert a PIL image (RGB) to an OpenCV image (BGR)."""
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV image (BGR or grayscale) to a PIL image."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_image(image: np.ndarray, width_pct: int) -> np.ndarray:
    """Resize image by a percentage of its original size."""
    if width_pct == 100:
        return image
    h, w = image.shape[:2]
    new_w = max(1, int(w * width_pct / 100))
    new_h = max(1, int(h * width_pct / 100))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def adjust_brightness_contrast(image: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    """
    brightness: -100 to 100 (added to pixel values)
    contrast:   -100 to 100 (scales pixel values around 127)
    """
    brightness = float(brightness)
    contrast = float(contrast)

    # Contrast factor: maps -100..100 -> 0..2 (approx), 0 => no change
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness

    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def to_grayscale(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # keep 3 channels for consistency


def apply_blur(image: np.ndarray, strength: int) -> np.ndarray:
    """strength: 0 (no blur) - 50"""
    if strength <= 0:
        return image
    k = strength * 2 + 1  # ensure odd kernel size
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_warm_filter(image: np.ndarray, intensity: int) -> np.ndarray:
    """
    Warm filter: boosts red/yellow channels and slightly reduces blue,
    simulating a warm, golden-hour look.
    intensity: 0-100
    """
    if intensity <= 0:
        return image

    img = image.astype(np.float32)
    b, g, r = cv2.split(img)

    factor = intensity / 100.0
    r = np.clip(r + 40 * factor, 0, 255)
    g = np.clip(g + 15 * factor, 0, 255)
    b = np.clip(b - 30 * factor, 0, 255)

    warm = cv2.merge([b, g, r]).astype(np.uint8)
    return warm


def apply_sharpen(image: np.ndarray, amount: int) -> np.ndarray:
    """amount: 0-100, controls sharpening strength via unsharp masking."""
    if amount <= 0:
        return image

    strength = amount / 100.0
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened


@st.cache_resource
def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def portrait_blur(image: np.ndarray, blur_strength: int) -> np.ndarray:
    """
    Simple portrait-mode background blur:
    - Detect faces using a Haar cascade.
    - Build a soft elliptical mask around the detected face(s) (or the
      image center if no face is found) to represent the "subject".
    - Blend a sharp foreground with a blurred background using that mask.
    """
    h, w = image.shape[:2]
    k = blur_strength * 2 + 1
    blurred_bg = cv2.GaussianBlur(image, (k, k), 0)

    mask = np.zeros((h, w), dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = load_face_cascade()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        for (x, y, fw, fh) in faces:
            cx, cy = x + fw // 2, y + fh // 2
            # Expand ellipse beyond face box to cover head + shoulders roughly
            axes = (int(fw * 1.3), int(fh * 2.2))
            cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    else:
        # Fallback: soft ellipse centered on the image
        cx, cy = w // 2, h // 2
        axes = (int(w * 0.28), int(h * 0.4))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)

    # Feather the mask edges for a natural blend
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    mask_3c = cv2.merge([mask, mask, mask])

    result = (image.astype(np.float32) * mask_3c) + (blurred_bg.astype(np.float32) * (1 - mask_3c))
    return result.astype(np.uint8)


def apply_edge_detection(image: np.ndarray, t1: int, t2: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_sketch_effect(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = 255 - blurred
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_cartoon_effect(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(image, d=9, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, matrix, (new_w, new_h))


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("1. Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is None:
    st.info("👈 Upload an image from the sidebar to get started.")
    st.stop()

original_pil = Image.open(uploaded_file)
original_cv = pil_to_cv(original_pil)

st.sidebar.header("2. Resize")
resize_pct = st.sidebar.slider("Resize (% of original)", 10, 200, 100, step=5)

st.sidebar.header("3. Brightness & Contrast")
brightness = st.sidebar.slider("Brightness", -100, 100, 0)
contrast = st.sidebar.slider("Contrast", -100, 100, 0)

st.sidebar.header("4. Basic Effects")
grayscale_on = st.sidebar.checkbox("Convert to Grayscale")
blur_strength = st.sidebar.slider("Blur strength", 0, 25, 0)
warm_intensity = st.sidebar.slider("Warm filter intensity", 0, 100, 0)
sharpen_amount = st.sidebar.slider("Sharpen amount", 0, 100, 0)

st.sidebar.header("5. Portrait Mode")
portrait_on = st.sidebar.checkbox("Enable portrait-style background blur")
portrait_strength = st.sidebar.slider("Background blur strength", 5, 40, 15, disabled=not portrait_on)

st.sidebar.header("6. Extra Effects")
extra_effect = st.sidebar.selectbox(
    "Choose an extra effect (optional)",
    ["None", "Edge Detection", "Sketch", "Cartoon"],
)
edge_t1, edge_t2 = 100, 200
if extra_effect == "Edge Detection":
    edge_t1 = st.sidebar.slider("Canny threshold 1", 0, 300, 100)
    edge_t2 = st.sidebar.slider("Canny threshold 2", 0, 300, 200)

rotation_angle = st.sidebar.slider("Rotate (degrees)", -180, 180, 0, step=5)


# --------------------------------------------------------------------------
# Processing pipeline
# --------------------------------------------------------------------------
processed = original_cv.copy()

processed = resize_image(processed, resize_pct)
processed = adjust_brightness_contrast(processed, brightness, contrast)

if portrait_on:
    processed = portrait_blur(processed, portrait_strength)

if blur_strength > 0:
    processed = apply_blur(processed, blur_strength)

if warm_intensity > 0:
    processed = apply_warm_filter(processed, warm_intensity)

if sharpen_amount > 0:
    processed = apply_sharpen(processed, sharpen_amount)

if grayscale_on:
    processed = to_grayscale(processed)

if extra_effect == "Edge Detection":
    processed = apply_edge_detection(processed, edge_t1, edge_t2)
elif extra_effect == "Sketch":
    processed = apply_sketch_effect(processed)
elif extra_effect == "Cartoon":
    processed = apply_cartoon_effect(processed)

if rotation_angle != 0:
    processed = rotate_image(processed, rotation_angle)

result_pil = cv_to_pil(processed)


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original")
    st.image(original_pil, use_container_width=True)

with col2:
    st.subheader("Edited")
    st.image(result_pil, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Download button
# --------------------------------------------------------------------------
buf = io.BytesIO()
result_pil.save(buf, format="PNG")
byte_data = buf.getvalue()

st.download_button(
    label="⬇️ Download Edited Image",
    data=byte_data,
    file_name="edited_image.png",
    mime="image/png",
)

if st.sidebar.button("Reset all settings"):
    st.rerun()
