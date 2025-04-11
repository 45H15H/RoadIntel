# RoadIntel

RoadIntel is an advanced vehicle analytics pipeline that leverages AI to detect, analyze, and extract insights from vehicle videos. It processes vehicle data in real-time, providing details such as vehicle type, color, make, and license plate information. This project is designed to be a robust solution for traffic monitoring and vehicle tracking.

## Features

- Real-time vehicle detection and tracking.
- On-the-fly image analysis using Gemini AI.
- Automatic saving of analysis results to a CSV file.
- Supports multiple vehicle types (e.g., cars, trucks).
- Easy integration with video feeds for highway monitoring.

## Setup Instructions

### Prerequisites

1. **Python**: Ensure Python 3.8 or later is installed.
2. **Pip**: Make sure `pip` is installed for managing Python packages.
3. **Virtual Environment (Optional)**: It is recommended to use a virtual environment to manage dependencies.

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/45H15H/RoadIntel.git
   cd RoadIntel
   ```

2. **Set Up Virtual Environment (Optional)**

   ```bash
   python -m venv .vehicle_analytics
   source .vehicle_analytics/bin/activate  # On Windows: .vehicle_analytics\Scripts\activate
   ```

3. **Install Dependencies**
   Before installing the dependencies using `requirements.txt`, ensure you have PyTorch installed as per your system configuration. You can find the installation instructions at [PyTorch's official site](https://pytorch.org/get-started/locally/).

   Install rest of the required Python packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models**
   Ensure the YOLO models are present in the `models/` directory. If not, download the required models and place them in the folder.

5. **Prepare Input Data**
   - Place your video files in the `videos/` directory.
   - Ensure the `cropped_images/` folder exists for saving processed images.

6. **Update the Script**
   Modify the `vehicle_analytics_pipeline.py` script to set the correct paths for your video files and models if necessary.

   ```python
   # Example path settings
   video_path = 'videos/your_video.mp4'
   model_path = 'models/yolo_model.pt'
   ```

   Update the detection polygon and filter polygon as per your requirements.
   Upload the video frame to polygonzone.roboflow.com and get the polygon array.

7. **Configure .env File**
   Create a `.env` file in the root directory and set the following environment variables:

   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   ```

8. **Run the Pipeline**
   Execute the main script to start the vehicle analytics pipeline:

   ```bash
   python vehicle_analytics_pipeline.py
   ```

### Output

- Processed images are saved in the `cropped_images/` folder.
- Analysis results are stored in `vehicle_analytics_result.csv`.

## Folder Structure

```
Vehicle_Analytics/
├── cropped_images/          # Folder for saving cropped vehicle images
├── models/                  # Pre-trained YOLO models
├── videos/                  # Input video files
├── vehicle_analytics_pipeline.py  # Main pipeline script
├── vehicle_analytics_result.csv   # CSV file for analysis results
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## Acknowledgments

- YOLO for object detection.
- Gemini AI for advanced image analysis.
- Supervision (SV) for detection and tracking utilities.
