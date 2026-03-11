import argparse
import os
import cv2
from ultralytics import YOLO

def main():
    # Set up argument parser to match the command in the lab handout
    parser = argparse.ArgumentParser(description="Run YOLO object detection on a folder of images.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images to process")
    args = parser.parse_args()

    input_dir = args.dir
    output_dir = "yolo_results"

    # Friendly error handling if the student types the wrong directory
    if not os.path.exists(input_dir):
        print(f"Error: The directory '{input_dir}' does not exist.")
        print("Please make sure you created the folder and spelled it exactly as shown in the handout.")
        return

    # Create the output directory automatically if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the pre-trained YOLOv8 nano model
    print("Loading YOLO model... (This might take a few seconds)")
    try:
        # This will automatically download the lightweight yolov8n.pt file on the first run
        model = YOLO("yolov8n.pt") 
    except Exception as e:
        print("Error loading the YOLO model. Please ask an instructor for help.")
        print(f"Technical details: {e}")
        return

    # Look for standard image files
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No images found in '{input_dir}'. Did you forget to move your photos into this folder?")
        return

    print(f"\nFound {len(image_files)} images. Starting detection...\n")

    # Process each image one by one
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"detected_{filename}")
        
        print(f"Processing: {filename}...")
        
        # Run inference
        # conf=0.25 means the model will only draw boxes if it is at least 25% confident
        # verbose=False hides the complex terminal outputs that might confuse beginners
        results = model(img_path, conf=0.25, verbose=False)
        
        # results[0].plot() automatically draws the bounding boxes, labels, and confidence scores
        annotated_img = results[0].plot()
        
        # Save the newly drawn image to the yolo_results folder
        cv2.imwrite(output_path, annotated_img)
        
    print(f"\nSuccess! All images have been processed.")
    print(f"Please check the '{output_dir}' folder to see your results.")

if __name__ == "__main__":
    main()