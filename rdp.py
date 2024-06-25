import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.spatial.distance import euclidean


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


def rdp_recursive(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    def perpendicular_distance(pt, line_start, line_end):
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        mag = math.hypot(dx, dy)
        if mag > 0.0:
            dx /= mag
            dy /= mag
        pvx = pt[0] - line_start[0]
        pvy = pt[1] - line_start[1]
        pvdot = dx * pvx + dy * pvy
        ax = pvx - pvdot * dx
        ay = pvy - pvdot * dy
        return math.hypot(ax, ay)

    if len(points) < 2:
        raise ValueError("Not enough points to simplify")
    dmax = 0.0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if dmax < d:
            index = i
            dmax = d
    if dmax > epsilon:
        rec_results1 = rdp_recursive(points[:index + 1], epsilon)
        rec_results2 = rdp_recursive(points[index:], epsilon)
        return rec_results1[:-1] + rec_results2
    else:
        return [points[0], points[-1]]


def filter_close_points(points: List[Tuple[float, float]], threshold: float) -> List[Tuple[float, float]]:
    if not points:
        return []

    filtered_points = [points[0]]
    for point in points[1:]:
        dist = math.hypot(point[0] - filtered_points[-1][0], point[1] - filtered_points[-1][1])
        if dist > threshold:
            filtered_points.append(point)
    return filtered_points


def merge_close_points(points: List[Tuple[float, float]], threshold: float) -> List[Tuple[float, float]]:
    if len(points) == 0:
        return []

    merged_points = [points[0]]
    for point in points[1:]:
        if all(math.hypot(point[0] - p[0], point[1] - p[1]) > threshold for p in merged_points):
            merged_points.append(point)
    return merged_points


def detect_and_classify_polygons(img):
    original_img = img.copy()
    contours = find_contours(img)
    contours = [contour for contour in contours if cv.contourArea(contour) > 200]

    blank_shape = (img.shape[0], img.shape[1], 3)
    img_with_hulls, convex_hulls = draw_convex_hull(blank_shape, contours)

    def adaptive_epsilon(contour):
        arc_length = cv.arcLength(contour, True)
        return 0.01 * arc_length

    simplified_hulls = []
    for hull in convex_hulls:
        epsilon_value = adaptive_epsilon(hull)
        point_list = [(point[0][0], point[0][1]) for point in hull]
        simplified_contour = rdp_recursive(point_list, epsilon_value)
        perimeter = cv.arcLength(hull, True)
        filtered_contour = filter_close_points(simplified_contour, perimeter / 20)
        simplified_hulls.append(np.array(filtered_contour, dtype=np.int32))

    img_with_labels = original_img.copy()

    for hull, simplified_hull in zip(convex_hulls, simplified_hulls):
        merged_hull = merge_close_points(simplified_hull, 30)

        for point in merged_hull:
            cv.circle(img_with_hulls, tuple(point), 5, (0, 0, 255), cv.FILLED)

        shape = classify_polygon_by_corners(merged_hull)
        M = cv.moments(np.array(merged_hull, dtype=np.int32))
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


def find_corner_with_harris(img):
    contours = find_contours(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    harris_corners = cv.cornerHarris(gray, 5, 5, 0.1)
    harris_corners = np.intp(np.argwhere(harris_corners > 0.01 * harris_corners.max()))  # Convert to integer points
    corners_list = [(corner[1], corner[0]) for corner in harris_corners]  # Switch x and y
    return corners_list, contours


def find_corner_with_shi_tomasi(img):
    contours = find_contours(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.04, minDistance=100)
    corners = np.intp(corners)
    corners_list = [(corner[0][0], corner[0][1]) for corner in corners]
    return corners_list, contours


def detect_and_draw_diagonals(img, corners, contours):
    shape_detection = {0: "Circle", 1: "Dot", 2: "Dot", 3: "Triangle", 4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon",
                       7: "Heptagon", 8: "Octagon"}
    for contour in contours:
        if len(contour) < 3:
            continue
        inside_corners = []
        for corner in corners:
            if len(corner) == 2:
                corner_x, corner_y = corner
                if cv.pointPolygonTest(contour, (int(corner_x), int(corner_y)), False) >= 0:
                    inside_corners.append((corner_x, corner_y))

        for kp in inside_corners:
            cv.circle(img, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), cv.FILLED)
        num_corners = len(inside_corners)
        shape = shape_detection.get(num_corners, "Unknown Shape")
        M = cv.moments(np.array(contour))
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv.putText(img, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow("Detected Diagonals", cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img


def douglas_peucker(points, epsilon):
    dmax = 0
    index = 0
    end = len(points)
    for i in range(1, end - 1):
        d = euclidean(points[i], (points[0] + points[-1]) / 2)
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        results1 = douglas_peucker(points[:index + 1], epsilon)
        results2 = douglas_peucker(points[index:], epsilon)
        return np.vstack((results1[:-1], results2))
    else:
        return np.vstack((points[0], points[-1]))

