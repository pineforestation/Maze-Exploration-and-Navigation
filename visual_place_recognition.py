import numpy as np
import cv2
import os
import faiss
from time import time

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu=True)
        self.kmeans.train(X).astype(np.float32)
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]

class BovwPlaceRecognition:
    def __init__(self):
        self.num_clusters = 150 # TODO tune this parameter
        self.images = {}
        self.training_img_histograms = {}
        self.visual_words = []


    def build_database(self, image_folder):
        start_t = time()
        print("Starting _load_images_from_folder")
        self.images = self._load_images_from_folder(image_folder)
        end_t = time()
        print(f"Finished _load_images_from_folder in {end_t - start_t} seconds")

        start_t = time()
        print("Starting _compute_sift_features")
        descriptor_list, image_to_descriptors = self._compute_sift_features(self.images)
        end_t = time()
        print(f"Finished _compute_sift_features in {end_t - start_t} seconds")

        start_t = time()
        print("Starting _build_visual_words")
        self.visual_words = self._build_visual_words(descriptor_list)
        end_t = time()
        print(f"Finished _build_visual_words in {end_t - start_t} seconds")

        for key, desc in image_to_descriptors.items():
            hist = self._calculate_histogram(desc, self.visual_words)
            self.training_img_histograms[key] = hist


    def query_by_image(self, query_img, shortlist_count=5):
        sift = cv2.SIFT_create()
        kp_qry, descr_qry = sift.detectAndCompute(query_img, None)
        query_hist = self._calculate_histogram(descr_qry, self.visual_words)
        match_filenames = self._get_tentative_matches(query_hist, self.training_img_histograms, shortlist_count)
        match_imgs = [self.images[filename] for filename in match_filenames]

        # For each shortlisted image, match descriptors with the query image
        # Then, we will return the image with the most good matches
        bf = cv2.BFMatcher()
        best_match_count = 0
        best_filename = None
        best_match_img = None
        for filename, img in zip(match_filenames, match_imgs):
            kp_sl, descr_sl = sift.detectAndCompute(img, None)

            good_matches = []
            for m,n in bf.knnMatch(descr_qry, descr_sl, k=2):
                # Lowe's ratio test
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])

            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_filename = filename
                best_match_img = cv2.drawMatchesKnn(query_img, kp_qry, img, kp_sl, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return best_filename, best_match_img


    def _load_images_from_folder(self, folder):
        images = {}
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            images[image_name] = cv2.imread(image_path)
        print(f"Loaded {len(images)} images")
        return images


    def _compute_sift_features(self, images):
        descriptor_list = []
        image_to_descriptors = {}

        sift = cv2.SIFT_create()
        for key, img in images.items():
            _kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                descriptor_list.extend(des)
                image_to_descriptors[key] = des
        descriptor_array = np.array(descriptor_list, dtype=np.float32)
        return descriptor_array, image_to_descriptors


    def _build_visual_words(self, descriptor_list):
        kmeans = FaissKMeans(n_clusters=self.num_clusters)
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


    def _get_tentative_matches(self, target_histogram, dataset_histograms, n_matches):
        match_to_distance = {}

        for key, histogram in dataset_histograms.items():
            distance = np.linalg.norm(target_histogram - histogram)
            match_to_distance[key] = distance

        sorted_matches = sorted(match_to_distance.items(), key=lambda x: x[1])
        return [filename for filename, _distance in sorted_matches[:n_matches]]
