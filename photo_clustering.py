import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import time
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract the dominant color of an image
def extract_dominant_color(image, k=5):
    """
    Extract the dominant color from an image using K-Means clustering.
    :param image: Input image as a NumPy array.
    :param k: Number of clusters for K-Means.
    :return: Dominant color as a tuple (R, G, B).
    """
    img_data = image.reshape((-1, 3))  # Flatten the image into 2D array
    img_data = np.float32(img_data)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(img_data)
    
    # Get the dominant color (largest cluster)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return tuple(map(int, dominant_color))

# Function to group images by dominant color
def group_images_by_color(image_folder, output_folder, k=5):
    """
    Group images in a folder by their dominant colors.
    :param image_folder: Path to the folder containing input images.
    :param output_folder: Path to the folder where grouped images will be saved.
    :param k: Number of clusters for K-Means.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if file_path.lower().endswith(('png', 'jpg', 'jpeg')):
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Warning: Could not read file {file_name}. Skipping.")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Extract the dominant color
            dominant_color = extract_dominant_color(image, k)
            color_key = f"{dominant_color[0]}{dominant_color[1]}{dominant_color[2]}"
            
            # Create a folder for this color
            color_folder = os.path.join(output_folder, color_key)
            if not os.path.exists(color_folder):
                os.makedirs(color_folder)
            
            # Save the image to the corresponding folder
            new_file_path = os.path.join(color_folder, file_name)
            cv2.imwrite(new_file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Processed {file_name}: Saved to {color_folder}")

# Function to measure runtime
def analyze_runtime(image_folder, k=5):
    """
    Analyze runtime for the iterative algorithm.
    :param image_folder: Folder containing input images.
    :param k: Number of clusters for K-Means.
    :return: Lists of input sizes and corresponding runtimes.
    """
    runtimes = []
    input_sizes = []

    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if file_path.lower().endswith(('png', 'jpg', 'jpeg')):
            # Read the image
            image = cv2.imread(file_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Flatten the image data
            img_data = image.reshape((-1, 3)).astype(np.float32)

            # Measure runtime for dominant color extraction
            start_time = time.time()
            extract_dominant_color(image, k)
            end_time = time.time()

            # Record runtime and input size
            runtimes.append(end_time - start_time)
            input_sizes.append(len(img_data))
    
    return input_sizes, runtimes

# Function to visualize runtime results
def visualize_runtime(input_sizes, runtimes):
    """
    Visualize the runtime analysis results.
    :param input_sizes: List of input sizes.
    :param runtimes: Corresponding list of runtimes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, runtimes, marker='o', label='Iterative Algorithm')
    plt.title("Runtime Analysis of Dominant Color Extraction")
    plt.xlabel("Input Size (Number of Pixels)")
    plt.ylabel("Runtime (Seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the program
if _name_ == "_main_":
    # Input and output folder paths
    input_folder = "images"  # Change this to the folder where your images are stored
    output_folder = "output_groups"  # Output folder where grouped images will be saved
    
    # Number of clusters for K-Means
    clusters = 5  # Adjust this value as needed
    
    # Run the grouping function
    print("Starting to group images by dominant color...")
    group_images_by_color(input_folder, output_folder, k=clusters)
    print("Grouping complete! Check the output folder for results.")
    
    # Analyze runtime
    print("Analyzing runtime...")
    input_sizes, runtimes = analyze_runtime(input_folder, k=clusters)
    
    # Visualize runtime results
    print("Visualizing runtime results...")
    visualize_runtime(input_sizes, runtimes)
    
    print("Analysis complete!")\