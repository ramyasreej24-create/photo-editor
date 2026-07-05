# 📸 Photo Editor using OpenCV and Streamlit

## Project Description
A lightweight, browser-based photo editor built with **Streamlit** and **OpenCV**.
Users can upload an image and interactively apply a range of edits — resizing,
brightness/contrast adjustment, filters, blur effects, sharpening, and a simple
portrait-style background blur — all through sliders and checkboxes in a live
UI, then download the final result as a PNG.

The app follows the flow:

```
Upload Image → Adjust brightness/contrast → Apply filters → View edited image → Download edited image
```

## Tools Used
- **[Python](https://www.python.org/)** — core language
- **[Streamlit](https://streamlit.io/)** — web UI framework
- **[OpenCV](https://opencv.org/)** (`opencv-python-headless`) — image processing
- **[Pillow (PIL)](https://python-pillow.org/)** — image I/O and format conversion
- **[NumPy](https://numpy.org/)** — array/matrix operations

## Features

### Core Features
| Feature | Description |
|---|---|
| **Upload Image** | Upload `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` images |
| **Resize** | Scale the image from 10% to 200% of its original size |
| **Brightness** | Adjust image brightness from -100 to +100 |
| **Contrast** | Adjust image contrast from -100 to +100 |
| **Grayscale** | Convert the image to black & white |
| **Blur** | Apply adjustable Gaussian blur |
| **Warm Filter** | Boosts red/yellow tones and reduces blue for a golden, warm look |
| **Portrait-Style Background Blur** | Detects faces (Haar cascade) and blurs the background while keeping the subject sharp — falls back to a center-focused blur if no face is detected |
| **Sharpen** | Unsharp-mask-based sharpening with adjustable strength |
| **Download Edited Image** | Download the final result as a PNG file |

### Extra Features (Go the Extra Mile)
| Feature | Description |
|---|---|
| **Edge Detection** | Canny edge detection with adjustable thresholds |
| **Sketch Effect** | Converts the photo into a pencil-sketch style drawing |
| **Cartoon Effect** | Combines edge masks and bilateral filtering for a cartoon look |
| **Image Rotation** | Rotate the image freely from -180° to 180° |
| **Reset Settings** | One-click reset button in the sidebar |

All adjustments are applied live and can be combined together (e.g., resize +
warm filter + sharpen + rotate, all at once).

## Steps to Run the Project

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open the app**
   Streamlit will print a local URL (typically `http://localhost:8501`) —
   open it in your browser.

6. **Use the app**
   - Upload an image using the sidebar uploader.
   - Adjust sliders for resize, brightness, contrast, blur, warm filter,
     sharpen, and portrait blur.
   - Optionally select an extra effect (Edge Detection, Sketch, or Cartoon).
   - Optionally rotate the image.
   - Click **Download Edited Image** to save your result.

## Project Structure
```
.
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Notes
- The portrait blur feature uses OpenCV's built-in Haar cascade
  (`haarcascade_frontalface_default.xml`) for face detection. If no face is
  found, the effect defaults to blurring everything outside a soft ellipse in
  the center of the image.
- All processing is done in-memory; no images are stored on a server.
