import os
import csv

def find_images_and_labels(root_dir):
    image_data = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(root, file)
                parts = os.path.normpath(image_path).split(os.sep)
                if len(parts) >= 2:
                    label = parts[-2]  # Penultimate folder name
                    if label in ['NORMAL', 'PNEUMONIA']:
                        image_data.append([label, image_path])
    
    return image_data

def save_to_csv(image_data, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'image path'])
        writer.writerows(image_data)

if __name__ == "__main__":
    root_directory = './'  
    output_csv = 'labels.csv'

    image_data = find_images_and_labels(root_directory)
    save_to_csv(image_data, output_csv)

    print(f"Saved {len(image_data)} image paths and labels to {output_csv}")
