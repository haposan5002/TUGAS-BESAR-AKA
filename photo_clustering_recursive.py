import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Recursive implementation of extracting the dominant color
def extract_dominant_color_recursive(data, k=5, iteration=0, max_iterations=10, centers=None):
    """
    Recursive implementation of dominant color extraction using K-Means.
    :param data: Flattened image data.
    :param k: Number of clusters.
    :param iteration: Current iteration count.
    :param max_iterations: Maximum number of iterations allowed.
    :param centers: Current cluster centers.
    :return: Dominant color as a tuple (R, G, B).
    """
    if iteration == 0:  # Initialize centers randomly
        centers = data[np.random.choice(data.shape[0], k, replace=False)]
    
    # Assign clusters in smaller batches to avoid memory errors
    labels = np.zeros(data.shape[0], dtype=np.int32)
    batch_size = 100000  # Process in chunks of 100,000 pixels
    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i+batch_size]
        distances = np.linalg.norm(batch[:, None] - centers, axis=2)
        labels[i:i+batch_size] = np.argmin(distances, axis=1)
    
    # Update cluster centers
    new_centers = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
    
    # Check for convergence or max iterations
    if np.allclose(centers, new_centers) or iteration >= max_iterations:
        largest_cluster = np.argmax(np.bincount(labels))
        return tuple(map(int, new_centers[largest_cluster]))
    
    # Recurse with updated centers
    return extract_dominant_color_recursive(data, k, iteration + 1, max_iterations, new_centers)

# Function to group images by dominant color using the recursive method
def group_images_by_color_recursive(image_folder, output_folder, k=5):
    """
    Group images in a folder by their dominant colors using the recursive method.
    :param image_folder: Path to the folder containing input images.
    :param output_folder: Path to the folder where grouped images will be saved.
    :param k: Number of clusters.
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
            img_data = image.reshape((-1, 3)).astype(np.float32)

            # Extract the dominant color
            dominant_color = extract_dominant_color_recursive(img_data, k)
            color_key = f"{dominant_color[0]}{dominant_color[1]}{dominant_color[2]}"
            
            # Create a folder for this color
            color_folder = os.path.join(output_folder, color_key)
            if not os.path.exists(color_folder):
                os.makedirs(color_folder)
            
            # Save the image to the corresponding folder
            new_file_path = os.path.join(color_folder, file_name)
            cv2.imwrite(new_file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Processed {file_name}: Saved to {color_folder}")

# Function to measure runtime for the recursive algorithm
def analyze_runtime_recursive(image_folder, k=5):
    """
    Analyze runtime for the recursive algorithm.
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
            extract_dominant_color_recursive(img_data, k)
            end_time = time.time()

            # Record runtime and input size
            runtimes.append(end_time - start_time)
            input_sizes.append(len(img_data))
    
    return input_sizes, runtimes

# Function to visualize runtime results
def visualize_runtime_recursive(input_sizes, runtimes):
    """
    Visualize the runtime analysis results for the recursive method.
    :param input_sizes: List of input sizes.
    :param runtimes: Corresponding list of runtimes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, runtimes, marker='x', label='Recursive Algorithm', linestyle='--')
    plt.title("Runtime Analysis of Recursive Dominant Color Extraction")
    plt.xlabel("Input Size (Number of Pixels)")
    plt.ylabel("Runtime (Seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the program
if __name__ == "__main__":
    # Input and output folder paths
    input_folder = "images"  # Change this to the folder where your images are stored
    output_folder = "output_groups_recursive"  # Output folder for grouped images
    
    # Number of clusters for K-Means
    clusters = 5  # Adjust this value as needed
    
    # Run the grouping function
    print("Starting to group images by dominant color using recursive method...")
    group_images_by_color_recursive(input_folder, output_folder, k=clusters)
    print("Grouping complete! Check the output folder for results.")
    
    # Analyze runtime
    print("Analyzing runtime for recursive method...")
    input_sizes, runtimes = analyze_runtime_recursive(input_folder, k=clusters)
    
    # Visualize runtime results
    print("Visualizing runtime results for recursive method...")
    visualize_runtime_recursive(input_sizes, runtimes)
    
    print("Analysis complete!")
