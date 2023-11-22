import os
import dlib
import cv2
import shutil
import argparse
import json
import logging

# Configuração do logging
logging.basicConfig(filename='execute.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def main(image_folder_path, output_folder_path, face_ratio_threshold):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Load the face detector
    face_detector = dlib.get_frontal_face_detector()

    # Iterate over the images in the folder
    for image_name in os.listdir(image_folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder_path, image_name)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_detector(gray)
            for face in faces:
                # Calculate the face size ratio
                face_area = (face.right() - face.left()) * (face.bottom() - face.top())
                image_area = image.shape[0] * image.shape[1]
                face_ratio = face_area / image_area

                # Check if the face occupies a significant part of the image
                if face_ratio > face_ratio_threshold:
                    message = f'Predominant face detected in: {image_name}'
                    print(message)
                    logging.info(message)
                    # Copy the image to the output folder
                    shutil.copy(image_path, os.path.join(output_folder_path, image_name))
                    break  # Stop processing further faces in the same image
                else:
                    message = f'Low face ratio in: {image_name}'
                    print(message)
                    logging.info(message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images to detect predominant faces.')
    parser.add_argument('--image_folder_path', type=str, help='Path to the folder containing images')
    parser.add_argument('--output_folder_path', type=str, help='Path to the output folder')
    parser.add_argument('--face_ratio', type=float, default=0.2, help='Face to image area ratio threshold')

    args = parser.parse_args()

    # Check if any argument was provided
    if args.image_folder_path and args.output_folder_path:
        main(args.image_folder_path, args.output_folder_path, args.face_ratio)
    else:
        # Load parameters from JSON file
        with open('params.json', 'r') as json_file:
            params = json.load(json_file)
            main(params['image_folder_path'], params['output_folder_path'], params['face_ratio'])
