import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_gray = cv.GaussianBlur(gray, (3, 3), 15)
    thresh_img = cv.adaptiveThreshold(blurred_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 2)
    opening = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, (3, 3), iterations=1)
    return opening


def find_contours(img):
    processed_img = preprocess_image(img)
    contours, _ = cv.findContours(processed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def draw_convex_hull(image_shape, contours):
    blank_image = np.zeros(image_shape, dtype=np.uint8)
    convex_hulls = [cv.convexHull(contour) for contour in contours]
    for hull in convex_hulls:
        cv.polylines(blank_image, [hull], isClosed=True, color=(255, 255, 255), thickness=2)
    return blank_image, convex_hulls


def classify_polygon_by_corners(approx):
    num_corners = len(approx)
    shape = "Unknown"
    if num_corners == 3:
        shape = "Triangle"
    elif num_corners == 4:
        shape = "Quadrilateral"
    elif num_corners == 5:
        shape = "Pentagon"
    elif num_corners == 6:
        shape = "Hexagon"
    elif num_corners == 7:
        shape = "Heptagon"
    elif num_corners == 8:
        shape = "Octagon"
    else:
        shape = "Circle" if num_corners > 8 else "Unknown"
    return shape


def detect_and_classify_polygons(img):
    original_img = img.copy()
    contours = find_contours(img)

    # Filter out small contours based on area
    contours = [contour for contour in contours if cv.contourArea(contour) > 200]

    blank_shape = (img.shape[0], img.shape[1], 3)
    img_with_hulls, convex_hulls = draw_convex_hull(blank_shape, contours)

    simplified_hulls = []
    for hull in convex_hulls:
        epsilon = 0.01 * cv.arcLength(hull, True)
        simplified_contour = cv.approxPolyDP(hull, epsilon, True)
        simplified_hulls.append(simplified_contour)

    img_with_labels = original_img.copy()

    for hull, simplified_hull in zip(convex_hulls, simplified_hulls):
        for point in simplified_hull:
            cv.circle(img_with_hulls, tuple(point[0]), 5, (0, 0, 255), cv.FILLED)

        shape = classify_polygon_by_corners(simplified_hull)
        M = cv.moments(simplified_hull)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cv.putText(img_with_labels, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    axs[0].imshow(cv.cvtColor(img_with_hulls, cv.COLOR_BGR2RGB))
    axs[0].set_title('Convex Hulls and Corner Points')

    axs[1].imshow(cv.cvtColor(img_with_labels, cv.COLOR_BGR2RGB))
    axs[1].set_title('Original Image with Labels')

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return img_with_labels