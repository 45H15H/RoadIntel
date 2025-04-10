
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from typing import Any, List

from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import uuid
import threading
import os

BOUNDING_BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)

tracker = sv.ByteTrack()

detection_polygon =np.array([[64, 1063],[3680, 1707],[3716, 2099],[84, 2075]]) # clear highway
detection_zone = sv.PolygonZone(polygon=detection_polygon, triggering_anchors=(sv.Position.BOTTOM_RIGHT, sv.Position.BOTTOM_LEFT)) # highway

filter_polygon = np.array([[44, 987],[3740, 1647],[3752, 2135],[32, 2119]]) # clear highway 
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

# Initialize the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
output_video_path = r".\annotated_videos\annotated_highway_2.mp4"  # Path for the output video
frame_rate = fps  # Adjust according to the input video's frame rate
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

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
    vehicle_img = Image.fromarray(crops["car"][0])

    total_width = vehicle_img.width
    max_height = vehicle_img.height

    combined_img = Image.new('RGB', (total_width, max_height))
    combined_img.paste(vehicle_img)
    combined_img.show() # comment this line if you don't want to see the image

    # save the image with different names for each car
    unique_filename = f"./cropped_images/vehicle_{uuid.uuid1()}.png"
    # first just check if the directory exists
    if not os.path.exists("./cropped_images"):
        os.makedirs("./cropped_images")
    try:
        combined_img.save(unique_filename)
        print("Image saved")
    except Exception as e:
        print("Error saving image: ", e)


class MyModel:

    def __init__(self, weights_path: str):
        self._model = YOLO(weights_path)


  # after v0.9.18  
    def infer(self, video_frames: List[VideoFrame]) -> List[Any]: 
        # result must be returned as list of elements representing model prediction for single frame
        # with order unchanged.
        return self._model([v.image for v in video_frames])

def save_prediction(prediction: dict, video_frame: VideoFrame) -> None:
    desired_classes = [2]
    detections = sv.Detections.from_ultralytics(prediction)
    # Use numpy.isin to filter detections by class IDs
    mask = np.isin(detections.class_id, desired_classes)
    detections = detections[mask]  # Apply the mask to filter detections    annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(scene=video_frame.image.copy(), detections=detections)
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
    detected_cars = tracked_detections[tracked_detections.class_id == 2] # also change this according to class index
    cars_in_zone = detected_cars[detection_zone.trigger(detected_cars)]

    global unique_cars

    for car in cars_in_zone.tracker_id:
        if not car in unique_cars:
            unique_cars.add(car)
            background_thread = threading.Thread(target=background_task, args=(prediction, video_frame))
            background_thread.start()

my_model = MyModel(r"./models/yolo11m.pt")

pipeline = InferencePipeline.init_with_custom_logic(
    video_reference=my_video,
    on_video_frame=my_model.infer,
    on_prediction=save_prediction,
)

# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()