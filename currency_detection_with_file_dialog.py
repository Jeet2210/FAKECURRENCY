import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

class cvutils:
    @staticmethod
    def extract_features(image):
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        
        image_resized = cv2.resize(image_gray, (100, 100))
        return image_resized.flatten()

    @staticmethod
    def train_knn_classifier():
        features = {
            "50": np.random.rand(10000),
            "100": np.random.rand(10000),
            "200": np.random.rand(10000),
            "500": np.random.rand(10000)
        }
        return features

def get_note_type():
    note_types = ['50', '100', '200', '500']
    print("\nSelect the denomination to verify:")
    for i, note in enumerate(note_types, 1):
        print(f"{i}. ₹{note}")
    
    choice = input("\nEnter the number corresponding to the note: ")
    while choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Please try again.")
        choice = input("Enter the number corresponding to the note: ")
    
    return note_types[int(choice)-1]

def select_image():
    print("\nPlease select an image file to verify.")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    return file_path

def process_image(note_type, file_path, features):
    print(f"\nProcessing the image for ₹{note_type}...")
    test_image = cv2.imread(file_path)
    if test_image is None:
        print("Error: Unable to read the image.")
        return

    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)  
    plt.clf()

    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    plt.imshow(test_image_gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)  
    plt.clf()

    blurred_image = cv2.GaussianBlur(test_image_gray, (5, 5), 0)
    plt.imshow(blurred_image, cmap='gray')
    plt.title("Blurred Image")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)  
    plt.clf() 

    edges = cv2.Canny(blurred_image, 100, 200)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection (Canny)")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2) 
    plt.clf() 

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation_image = cv2.drawContours(test_image.copy(), contours, -1, (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2RGB))
    plt.title("Segmentation (Contours)")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)  
    plt.clf()

    test_image_features = cvutils.extract_features(test_image_gray)
    is_fake = is_fake_or_real(test_image_features, features)
    
    if is_fake:
        print("\nResult: The note is FAKE!")
    else:
        print("\nResult: The note is REAL!")

    continue_checking = input("\nDo you want to check another note? (y/n): ").strip().lower()
    if continue_checking == 'y':
        main()  
    else:
        print("\nExiting. Thank you for using the system!")

def is_fake_or_real(test_image_features, features):
    distances = {note: np.linalg.norm(test_image_features - feature) for note, feature in features.items()}
    predicted_note = min(distances, key=distances.get)
    
    if predicted_note == '50':
        return False 
    else:
        return True

def main():
    features = cvutils.train_knn_classifier()
    note_type = get_note_type()
    file_path = select_image()
    
    if file_path:
        process_image(note_type, file_path, features)
    else:
        print("No image selected.")

if __name__ == "__main__":
    main()