import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans

def load_images_from_folder(folder):
    images = {}
    i=0
    for i in (0,10):
        category = []
        img_files = os.listdir(folder)
        for image_file in img_files:
            image_path = os.path.join(folder, image_file)
            if os.path.isfile(image_path) and image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.imread(image_path)
                if img is not None:
                    category.append(img)
                    images[image_file]=img
    i=i+1
    return images

images = load_images_from_folder("C:\\Users\\vaish\\vis_nav_player\\data\\exploration_views\\20231105-173027")  # take all images category by category 
test = load_images_from_folder("C:\\Users\\vaish\\vis_nav_player\\data\\exploration_views") # take test images 
print("done loading images")

def sift_features(images):
    sift_vectors = {}
    descriptor_vectors = {}
    descriptor_list = []
    sift = cv2.SIFT_create()
    for key,value in images.items():
        features = []
        for img in value:
            des, kp = sift.detectAndCompute(img,None)
            if des is not None:
                descriptor_list.extend(des)
                features.append(kp)
        sift_vectors[key] = features
        descriptor_vectors[key] = descriptor_list
    return [descriptor_vectors, sift_vectors]

sifts = sift_features(images) 

descriptor_list = sifts[0] 
print(descriptor_list.shape)
all_bovw_feature = sifts[1] 
target_image_descriptors = sift_features(test)[2]
test_bovw_feature = sift_features(test)[1] 
print("done")

# def kmeans(k, descriptor_list):
#     kmeans = KMeans(n_clusters = k)
#     kmeans.fit(descriptor_list)
#     visual_words = kmeans.cluster_centers_ 
#     return visual_words

def unsupervised_kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_
    return visual_words

visual_words_unsupervised = unsupervised_kmeans(150, descriptor_list)

# visual_words = kmeans(150, descriptor_list) 

def match_target_image(target_histogram, dataset_histograms):
    best_match = None
    min_distance = float('inf')

    for key, histogram in dataset_histograms.items():
        distance = np.linalg.norm(target_histogram - histogram)
        if distance < min_distance:
            min_distance = distance
            best_match = key

    return best_match

def calculate_histogram(descriptors, visual_words):
    # Create an array to store the histogram
    histogram = np.zeros(len(visual_words))

    for descriptor in descriptors:
        # Find the nearest visual word for the descriptor
        nearest_word = np.argmin(np.linalg.norm(visual_words - descriptor, axis=1))

        # Increment the corresponding bin in the histogram
        histogram[nearest_word] += 1

    return histogram

# Calculate the target image's histogram using unsupervised visual words
target_histogram = calculate_histogram(target_image_descriptors, visual_words_unsupervised)
dataset_histograms = calculate_histogram(descriptor_list, all_bovw_feature)
# Perform matching to identify the target image
best_match = match_target_image(target_histogram, dataset_histograms)

print(f"The identified target image is: {best_match}")
