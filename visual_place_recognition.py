import numpy as np
import cv2
import os
from sklearn.cluster import KMeans

class BovwPlaceRecognition:
    def __init__(self):
        self.num_clusters = 150 # TODO tune this parameter
        self.sift = cv2.SIFT_create()
        self.training_img_histograms = {}
        self.visual_words = []


    def build_database(self, image_folder):
        training_images = self._load_images_from_folder(image_folder)

        descriptor_list, image_to_descriptors = self._compute_sift_features(training_images)
        self.visual_words = self._build_visual_words(descriptor_list)

        for key, desc in image_to_descriptors.items():
            hist = self._calculate_histogram(desc, self.visual_words)
            self.training_img_histograms[key] = hist


    def query_by_image(self, query_img):
        _kp, des = self.sift.detectAndCompute(query_img, None)
        query_hist = self._calculate_histogram(des, self.visual_words)
        match = self._get_tentative_match(query_hist, self.training_img_histograms)
        # TODO geometric verification
        return match #TODO this is just filename, get the actual image


    def _load_images_from_folder(self, folder):
        images = {}
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            images[image_name] = cv2.imread(image_path)
        return images


    def _compute_sift_features(self, images):
        descriptor_list = []
        image_to_descriptors = {}

        for key, img in images.items():
            _kp, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                descriptor_list.extend(des)
                image_to_descriptors[key] = des
        return descriptor_list, image_to_descriptors


    def _build_visual_words(self, descriptor_list):
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(descriptor_list)
        visual_words = kmeans.cluster_centers_
        return visual_words


    def _calculate_histogram(self, descriptors, visual_words):
        # Create an array to store the histogram
        histogram = np.zeros(len(visual_words))

        for descriptor in descriptors:
            # Find the nearest visual word for the descriptor
            nearest_word = np.argmin(np.linalg.norm(visual_words - descriptor, axis=1))
            # Increment the corresponding bin in the histogram
            histogram[nearest_word] += 1

        return histogram


    def _get_tentative_match(self, target_histogram, dataset_histograms):
        best_match = None
        min_distance = float('inf')

        for key, histogram in dataset_histograms.items():
            distance = np.linalg.norm(target_histogram - histogram)
            if distance < min_distance:
                min_distance = distance
                best_match = key

        return best_match
