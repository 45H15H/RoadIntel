import os
import csv
import json
from PIL import Image
from google import genai
import time
from google.genai import types
import PIL.Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Generative AI client
gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key)

prompt_template = """Return analysis of the car. Include the color, make, type, and license plate. Return result in the following format like a python dictionary: {"color": "red", "make": "Toyota", "type": "car", "license plate": "ABC123"}.
Don't use any formatting, just return the raw string."""

def analyze_image(image_path):
    img = PIL.Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt_template, img]
    )
    # remove ```json from the response
    response_text = response.text.replace("```json", "").replace("```", "").strip()
    print(response_text)
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
            time.sleep(1)

    # Save results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'color', 'make', 'type', 'license plate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

# Specify the folder containing images and the output CSV file
folder_path = r'.\cropped_images'
output_csv = 'vehicle_analytics_result.csv'

# Process images and save results to CSV
process_images_in_folder(folder_path, output_csv)
print(f"Results saved to {output_csv}")