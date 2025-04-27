import os
import cv2
import csv

def convert_images_to_csv(dataset_path, output_csv):
    """
    Converts images in a dataset directory into a CSV file with labels and pixel data.

    Parameters:
    - dataset_path: Path to the dataset directory containing subdirectories for each emotion.
    - output_csv: Path to the output CSV file.
    """
    emotions = sorted(os.listdir(dataset_path))
    print(f"Detected emotion categories: {emotions}")

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['emotion', 'pixels'])
        total_images = 0

        for emotion in emotions:
            emotion_path = os.path.join(dataset_path, emotion)
            if not os.path.isdir(emotion_path):
                print(f"Skipping non-directory item: {emotion_path}")
                continue

            image_files = os.listdir(emotion_path)
            print(f"\nProcessing '{emotion}' category with {len(image_files)} images.")

            for img_name in image_files:
                img_path = os.path.join(emotion_path, img_name)
                print(f"Reading image: {img_path}")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Warning: Unable to read image '{img_path}'. Skipping.")
                    continue

                try:
                    img_resized = cv2.resize(img, (48, 48))
                except Exception as e:
                    print(f"Error resizing image '{img_path}': {e}. Skipping.")
                    continue

                pixels = ' '.join(str(pixel) for pixel in img_resized.flatten())
                writer.writerow([emotion, pixels])
                total_images += 1

        print(f"\nConversion complete. Total images processed: {total_images}")
        print(f"CSV file saved at: {output_csv}")

# Example usage:
# convert_images_to_csv('path_to_dataset', 'emotions.csv')


# Example usage:
convert_images_to_csv('C:\\Users\\vishn\\Desktop\\MCE_AIML_Workshop\\Day 3\\archive\\train', 'emotions.csv')
