# CyberShield - Integrated Video Analytics

An advanced, AI-powered video analytics platform designed for smart city infrastructure, transportation networks, and public surveillance systems. This unified system consolidates real-time object detection, facial recognition, demographic analysis, and automatic number plate recognition (ANPR) into a single, professional dashboard.

## 🚀 Key Features

*   **Vehicle Counting & Classification**: Utilizes YOLOv5s to detect, count, and classify vehicles (Cars, Motorcycles, Buses, Trucks) in real-time, displaying live distributions on dynamic charts.
*   **Automatic Number Plate Recognition (ANPR)**: Employs EasyOCR with specialized image preprocessing (targeted cropping, CLAHE contrast enhancement, and deduplication) to accurately extract alphanumeric license plates and store them with precise timestamps.
*   **Facial Recognition System & Demographics**: Integrates DeepFace to analyze human faces, logging gender and estimated age statistics to generate live crowd demographic insights.
*   **Searchable Intelligent Logs**: A professional, dark-themed UI sidebar that aggregates time-stamped ANPR and FRS events, allowing operators to instantly search for specific plates or demographics across the video feed.
*   **Live Dashboard**: Built with FastAPI, TailwindCSS, and Chart.js, serving a responsive, low-latency MJPEG video stream directly to the browser alongside real-time analytical metrics.

## 🛠️ Technology Stack

*   **Backend**: Python 3.11, FastAPI, Uvicorn
*   **AI Models**:
    *   Ultralytics YOLOv5 (Object Detection)
    *   DeepFace (Facial Analysis)
    *   EasyOCR (Number Plate Recognition)
    *   OpenCV (Haar Cascades & Image Manipulation)
*   **Frontend**: HTML5, Vanilla JS, TailwindCSS, Chart.js

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Aravind2674/CyberShield.git
    cd CyberShield/integrated-video-analytics
    ```

2.  **Install Dependencies**
    Ensure you have Python 3.9+ installed, then install the required AI libraries:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first time you run the application, PyTorch Hub and EasyOCR will automatically download their required pre-trained weights (yolov5s.pt, craft_mlt_25k.pth, english_g2.pth).*

3.  **Run the Server**
    Start the FastAPI application on port 8080:
    ```bash
    python main.py
    ```

4.  **Access the Dashboard**
    Open your web browser and navigate to:
    ```
    http://localhost:8080
    ```

## 🎥 Usage

1.  Click **"Upload Source Video"** in the top right corner of the dashboard.
2.  Select an MP4 video file from your local machine.
3.  The system will automatically begin processing the video, rendering bounding boxes on the live feed.
4.  Monitor the live Metric Cards, Vehicle/Demographic Charts, and search through the Intelligent Event Log sidebar for specific data points.

## 📝 License
Proprietary / Internal Use.
