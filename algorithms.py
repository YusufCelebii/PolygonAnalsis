import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk



def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_gray = cv.GaussianBlur(gray, (7, 7), 15)
    thresh_img = cv.adaptiveThreshold(blurred_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
    opening = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, (3, 3), iterations=3)
    return opening


def find_contours(img):
    processed_img = preprocess_image(img)
    contours, _ = cv.findContours(processed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def find_corner_with_harris(img):
    contours = find_contours(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray = cv.GaussianBlur(gray, (15, 15), 15)
    harris_corners = cv.cornerHarris(gray, 5, 5, 0.1)
    harris_corners = np.intp(np.argwhere(harris_corners > 0.01 * harris_corners.max()))
    corners_list = [(corner[1], corner[0]) for corner in harris_corners]
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
        inside_corners = []
        for corner in corners:
            corner_x, corner_y = corner
            if cv.pointPolygonTest(contour, (int(corner_x), int(corner_y)), False) >= 0:
                inside_corners.append((corner_x, corner_y))
        for kp in inside_corners:
            cv.circle(img, (int(kp[0]), int(kp[1])), 10, (0, 0, 255), cv.FILLED)
        num_corners = len(inside_corners)
        shape = shape_detection.get(num_corners, "Unknown Shape")
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv.putText(img, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return img





def show_images(titles, imgs):
    root = tk.Tk()
    root.title("Image Display")

    window_width = 1800
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    for ax, title, img in zip(axs, titles, imgs):
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    plt.tight_layout()
    plt.close('all')

