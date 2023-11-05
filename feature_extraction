import cv2
i=11
while i < 24951:
    color_image = cv2.imread("C:\\Users\\vaish\\OneDrive\\Desktop\\Semester 3\\perception\\SLAM2\\screenshot_"+str(i)+".jpg", cv2.IMREAD_COLOR)
    # Load an image
    image = cv2.imread("C:\\Users\\vaish\\OneDrive\\Desktop\\Semester 3\\perception\\SLAM2\\screenshot_"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        i=i+1
    else:
        block_size = 2  # Neighborhood size for corner detection
        ksize = 3       # Aperture parameter for Sobel operator
        k = 0.02       # Harris detector free parameter
        # Detect corners using the Harris Corner Detector
        corners = cv2.cornerHarris(image, block_size, ksize, k)
        # Threshold for an optimal value (adjust as needed)
        threshold = 0.01 * corners.max()
        # Draw corners on the image
        image_with_corners = color_image.copy()
        image_with_corners[corners > threshold] = 0
        grad_x = cv2.Sobel(image_with_corners, cv2.CV_8U, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        # Gradient-Y
        grad_y = cv2.Sobel(image_with_corners, cv2.CV_8U, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # Display the image with corners
        #cv2.imshow('Harris Corners', image_with_corners)
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        # Detect SIFT features
        kp, _ = sift.detectAndCompute(image,None)
        image_with_keypoints = cv2.drawKeypoints(image,kp,grad)
        cv2.imwrite("C:\\Users\\vaish\\OneDrive\\Desktop\\Semester 3\\perception\\SLAM2\\sobel_grad"+str(i)+".jpg",grad)
        #cv2.imshow("sobel",grad)
        cv2.waitKey(0)
        i=i+1
