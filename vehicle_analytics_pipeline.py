import os
import csv
import json
import threading
import time
from PIL import Image
from google import genai
from google.genai import types
import PIL.Image
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from typing import Any, List
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Generative AI client
gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)

prompt_template = """Return analysis of the car. Include the color, make, type, and license plate. Return result in the following format like a python dictionary: {"color": "red", "make": "Toyota", "type": "car", "license plate": "ABC123"}. return the response in raw string format."""

BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)
tracker = sv.ByteTrack()

detection_polygon = np.array([[64, 1063], [3680, 1707], [3716, 2099], [84, 2075]])  # clear highway
detection_zone = sv.PolygonZone(polygon=detection_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT))  # highway

filter_polygon = np.array([[44, 987], [3740, 1647], [3752, 2135], [32, 2119]])  # clear highway
filter_zone = sv.PolygonZone(polygon=filter_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT))

unique_cars = set()

my_video = r"./videos/highway_2.mp4"

# To get the frame size of the video
cap = cv2.VideoCapture(my_video)
ret, first_frame = cap.read()
frame_height, frame_width = first_frame.shape[:2]
# get fps of the video
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

def analyze_image(image_path):
    img = PIL.Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt_template, img]
    )
    response_text = response.text.replace("```json", "").replace("```", "").strip()
    try:
        dictionary = json.loads(response_text)
        return dictionary
    except json.JSONDecodeError:
        print(f"Error decoding JSON for image {image_path}")
        return None

def process_images_in_folder(folder_path, output_csv):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {image_path}")
            result = analyze_image(image_path)
            if result:
                result['filename'] = filename
                results.append(result)
            time.sleep(1)  # Avoid overloading the API

    # Save results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'color', 'make', 'type', 'license plate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def get_crops(detections, image):
    result = {}
    for bbox, class_name in zip(detections.xyxy, detections.data["class_name"]):
        x1, y1, x2, y2 = bbox
        crop = image[int(y1):int(y2), int(x1):int(x2)]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if class_name in result:
            result[class_name].append(crop)
        else:
            result[class_name] = [crop]

    return result

def background_task(results, frame):
    detections = sv.Detections.from_ultralytics(results)
    car_features = detections[filter_zone.trigger(detections)]
    crops = get_crops(car_features, frame.image)

    for vehicle_type, vehicle_crops in crops.items():
        for i, crop in enumerate(vehicle_crops):
            vehicle_img = Image.fromarray(crop)

            total_width = vehicle_img.width
            max_height = vehicle_img.height

            combined_img = Image.new('RGB', (total_width, max_height))
            combined_img.paste(vehicle_img)

            # Save the image with a unique name
            unique_filename = f"./cropped_images/{vehicle_type}_{uuid.uuid1()}_{i}.png"
            if not os.path.exists("./cropped_images"):
                os.makedirs("./cropped_images")
            try:
                combined_img.save(unique_filename)
                print(f"Image saved: {unique_filename}")

                # Process the image immediately after saving
                result = analyze_image(unique_filename)
                if result:
                    print(f"Analysis result for {unique_filename}: {result}")

                    # Save the result to the CSV file
                    csv_file = "vehicle_analytics_result.csv"
                    file_exists = os.path.isfile(csv_file)
                    with open(csv_file, mode='a', newline='') as csvfile:
                        fieldnames = ['filename', 'color', 'make', 'type', 'license plate']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Write the header only if the file is new
                        if not file_exists:
                            writer.writeheader()

                        # Add the filename to the result and write to the CSV
                        result['filename'] = unique_filename
                        writer.writerow(result)

            except Exception as e:
                print("Error saving or processing image: ", e)

class MyModel:

    def __init__(self, weights_path: str):
        self._model = YOLO(weights_path)

    def infer(self, video_frames: List[VideoFrame]) -> List[Any]:
        return self._model([v.image for v in video_frames])

def save_prediction(prediction: dict, video_frame: VideoFrame) -> None:
    desired_classes = [2]  # Example class IDs for vehicles
    detections = sv.Detections.from_ultralytics(prediction)
    mask = np.isin(detections.class_id, desired_classes)
    detections = detections[mask]

    annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(scene=video_frame.image.copy(), detections=detections)
    annotated_frame = LABEL_ANNOTATOR.annotate(scene=annotated_frame, detections=detections)

    cv2.namedWindow("Vehicle Analytics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Analytics", frame_width // 3, frame_height // 3)
    cv2.imshow("Vehicle Analytics", annotated_frame)
    video_writer.write(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        pipeline.terminate()

    tracked_detections = tracker.update_with_detections(detections)
    detected_cars = tracked_detections[tracked_detections.class_id == 2]
    cars_in_zone = detected_cars[detection_zone.trigger(detected_cars)]

    global unique_cars

    for car in cars_in_zone.tracker_id:
        if car not in unique_cars:
            unique_cars.add(car)
            background_thread = threading.Thread(target=background_task, args=(prediction, video_frame))
            background_thread.start()

my_model = MyModel(r"./models/yolo11m.pt")

pipeline = InferencePipeline.init_with_custom_logic(
    video_reference=my_video,
    on_video_frame=my_model.infer,
    on_prediction=save_prediction,
)

# Start the pipeline
pipeline.start()
# Wait for the pipeline to finish
pipeline.join()