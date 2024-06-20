import cv2 as cv
import numpy as np




def preprocess_image(img):
    cv.imshow("Original", cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Decrease noise with blurring
    blurred_gray = cv.GaussianBlur(gray, (7, 7), 11)
    # Thresholding
    thresh_img = cv.adaptiveThreshold(blurred_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
    # Remove salt-pepper noise
    opening = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, (3, 3), iterations=3)
    return opening


def find_contours(img):
    processed_img = preprocess_image(img)
    contours, _ = cv.findContours(processed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled_img = np.zeros_like(img)
    filled_img = cv.drawContours(filled_img, contours, -1, (255, 255, 255), cv.FILLED)
    return contours, filled_img


def find_corner_with_harris(img):
    contours, _ = find_contours(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    harris_corners = cv.cornerHarris(gray, 5, 5, 0.1)
    dst = cv.dilate(harris_corners, None)
    threshold = 0.01 * dst.max()

    keypoints = np.argwhere(dst > threshold)
    keypoints = sorted(keypoints, key=lambda x: dst[x[0], x[1]], reverse=True)

    suppressed = np.zeros(dst.shape, dtype=bool)
    filtered_keypoints = []
    radius = 10  # Specify the radius to consider for NMS

    for kp in keypoints:
        if not suppressed[kp[0], kp[1]]:
            filtered_keypoints.append(kp)
            for x in range(kp[0] - radius, kp[0] + radius + 1):
                for y in range(kp[1] - radius, kp[1] + radius + 1):
                    if 0 <= x < dst.shape[0] and 0 <= y < dst.shape[1]:
                        suppressed[x, y] = True

    print(f"Initial number of corners: {len(keypoints)}")
    print(f"Number of corners after NMS: {len(filtered_keypoints)}")

    return np.array(filtered_keypoints), contours


def find_corner_with_shi_tomasi(img):
    contours, _ = find_contours(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.05, minDistance=50)
    corners = np.intp(corners)  # Use np.intp instead of np.int0

    return corners, contours


def detect_and_draw_diagonals(img, corners, contours):
    shape_detection = {0: "Circle", 1: "Dot", 2: "Dot", 3: "Triangle", 4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon",
                       7: "Heptagon", 8: "Octagon"}

    for contour in contours:
        inside_corners = []
        for corner in corners:
            if len(corner) == 2:  # For Harris corners
                corner_x, corner_y = corner[1], corner[0]
            else:  # For Shi-Tomasi corners
                corner_x, corner_y = corner[0][0], corner[0][1]

            if cv.pointPolygonTest(contour, (int(corner_x), int(corner_y)), False) >= 0:
                inside_corners.append((corner_x, corner_y))

        print(f"Number of corners inside current contour: {len(inside_corners)}")

        # Draw detected corners on the image
        for kp in inside_corners:
            cv.circle(img, (int(kp[0]), int(kp[1])), 10, (0, 255, 0), cv.FILLED)

        # Determine the shape based on the number of inside corners
        num_corners = len(inside_corners)
        shape = shape_detection.get(num_corners, "Unknown Shape")

        # Calculate the center of the contour to place the shape name
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw the shape name at the center of the contour
        cv.putText(img, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        print(f"Detected shape: {shape}")

    cv.imshow("Detected Diagonals", cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img







