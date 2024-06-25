import tkinter as tk
from tkinter import filedialog, messagebox
import cv2 as cv
import rdp
import algorithms
import approxPolyDP
image = None
import matplotlib.pyplot as plt

def browse_file():
    global image
    file_path = filedialog.askopenfilename()
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)
        image = cv.imread(file_path)

def harris_button():
    global image
    if image is not None:
        corners, contours = algorithms.find_corner_with_harris(image)
        img_with_diagonals = algorithms.detect_and_draw_diagonals(image.copy(), corners, contours)
        algorithms.show_images(["Original Image", "Final Image"], [image, img_with_diagonals])
    else:
        messagebox.showwarning("Alert!", "Please choose an image!")

def shi_tomasi_button():
    global image
    if image is not None:
        corners, contours = algorithms.find_corner_with_shi_tomasi(image)
        img_with_diagonals = algorithms.detect_and_draw_diagonals(image.copy(), corners, contours)
        algorithms.show_images(["Original Image", "Final Image"], [image, img_with_diagonals])
    else:
        messagebox.showwarning("Alert", "Please choose an image!")

def rdp_button():
    global image
    if image is not None:
        rdp.detect_and_classify_polygons(image)
        plt.show()
    else:
        messagebox.showwarning("Alert!", "Please choose an image!")

def approxPolyDP_button():
    global image
    if image is not None:
        approxPolyDP.detect_and_classify_polygons(image)
    else:
        messagebox.showwarning("Alert!", "Please choose an image!")



# Create Main Window
root = tk.Tk()
root.title("GUI")
root.geometry("400x300")

# Entry Widget and Choose file button
file_frame = tk.Frame(root)
file_frame.pack(pady=10)

file_label = tk.Label(file_frame, text="Choose")
file_label.pack(side=tk.LEFT, padx=5)

file_entry = tk.Entry(file_frame, width=40)
file_entry.pack(side=tk.LEFT, padx=5)

browse_button = tk.Button(file_frame, text="Search", command=browse_file)
browse_button.pack(side=tk.LEFT, padx=5)

# Harris button
harris_btn = tk.Button(root, text="Harris", command=harris_button, width=15, height=2)
harris_btn.pack(pady=10)

# Shi-Tomasi button
shi_tomasi_btn = tk.Button(root, text="Shi-Tomasi", command=shi_tomasi_button, width=15, height=2)
shi_tomasi_btn.pack(pady=10)

# Ramer-Douglas-Peucker button
rdp_btn = tk.Button(root, text="RDP", command=rdp_button, width=15, height=2)
rdp_btn.pack(pady=10)

#approxPolyDP button

approxPolyDP_btn= tk.Button(root, text="approxPolyDP", command=approxPolyDP_button, width=15, height=2)
approxPolyDP_btn.pack(pady=10)


root.mainloop()
