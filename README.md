# RoadIntel

RoadIntel is an advanced vehicle analytics pipeline that leverages AI to detect, analyze, and extract insights from vehicle images and videos. It processes vehicle data in real-time, providing details such as vehicle type, color, make, and license plate information. This project is designed to be a robust solution for traffic monitoring, vehicle tracking, and intelligent transportation systems.

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
   git clone <repository-url>
   cd Vehicle_Analytics
   ```

2. **Set Up Virtual Environment (Optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required Python packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models**
   Ensure the YOLO models are present in the `models/` directory. If not, download the required models and place them in the folder.

5. **Prepare Input Data**
   - Place your video files in the `videos/` directory.
   - Ensure the `cropped_images/` folder exists for saving processed images.

6. **Run the Pipeline**
   Execute the main script to start the vehicle analytics pipeline:
   ```bash
   python vehicle_analytics_pipeline.py
   ```

### Output
- Processed images are saved in the `cropped_images/` folder.
- Analysis results are stored in `vehicle_analytics_result.csv`.
- Annotated videos are saved in the `annotated_videos/` folder (if configured).

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

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- YOLO for object detection.
- Gemini AI for advanced image analysis.
- Supervision (SV) for detection and tracking utilities.
