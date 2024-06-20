import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2 as cv
import algorithms

# Initialize the global image variable
image = None

def browse_file():
    global image
    file_path = filedialog.askopenfilename()
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)
        # Load the image
        image = cv.imread(file_path)

def harris_button():
    global image
    if image is not None:
        # Call Harris method
        harris_corners, contours_harris = algorithms.find_corner_with_harris(image)
        # Use Harris corners for detection and drawing
        algorithms.detect_and_draw_diagonals(image.copy(), harris_corners, contours_harris)
    else:
        messagebox.showwarning("Uyarı", "Lütfen önce bir dosya seçin.")

def shi_tomasi_button():
    global image
    if image is not None:
        # Call Shi-Tomasi method
        shi_tomasi_corners, contours_shi_tomasi = algorithms.find_corner_with_shi_tomasi(image)
        # Use Shi-Tomasi corners for detection and drawing
        algorithms.detect_and_draw_diagonals(image.copy(), shi_tomasi_corners, contours_shi_tomasi)
    else:
        messagebox.showwarning("Uyarı", "Lütfen önce bir dosya seçin.")

def test_button():
    messagebox.showinfo("Test Buton", "Test butonuna tıklandı")

# Create Main Window
root = tk.Tk()
root.title("Örnek GUI")
root.geometry("400x200")

# Entry Widget and Choose file button
file_frame = tk.Frame(root)
file_frame.pack(pady=10)

file_label = tk.Label(file_frame, text="Dosya Seç:")
file_label.pack(side=tk.LEFT, padx=5)

file_entry = tk.Entry(file_frame, width=40)
file_entry.pack(side=tk.LEFT, padx=5)

browse_button = tk.Button(file_frame, text="Gözat", command=browse_file)
browse_button.pack(side=tk.LEFT, padx=5)

# Harris button
harris_button = tk.Button(root, text="Harris", command=harris_button, width=8, height=1)
harris_button.pack(pady=10)

# Shi-Tomasi button
shi_tomasi_button = tk.Button(root, text="Shi-Tomasi", command=shi_tomasi_button, width=8, height=1)
shi_tomasi_button.pack(pady=10)

# Test button
test_button = tk.Button(root, text="Test", command=test_button, width=8, height=1)
test_button.pack(pady=10)

root.mainloop()
