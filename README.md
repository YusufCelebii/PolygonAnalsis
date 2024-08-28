# Polygon Detection and Classification

This project is focused on detecting and classifying polygons using various algorithms, including Harris Corner Detection, Shi-Tomasi Corner Detection, Ramer-Douglas-Peucker (RDP) simplification, and approxPolyDP. The project aims to compare these algorithms based on their performance in noise-free and real-world scenarios.

## Used Algorithms

- **Harris Corner Detection:** Detects corner points by analyzing the brightness changes around each pixel.
- **Shi-Tomasi Corner Detection:** Works similarly to Harris but selects the strongest corner points by filtering with specific parameters.
- **Ramer-Douglas-Peucker (RDP) Algorithm:** This algorithm, implemented by me, simplifies the points of a shape by removing unnecessary points. It operates on the same fundamental principles as the approxPolyDP function but is manually implemented rather than optimized by OpenCV. The approxPolyDP function, however, benefits from OpenCV's optimization, making it faster and more efficient in practical use.
- **approxPolyDP:** Reduces the number of corner points of a contour to create a simpler polygon. It operates similarly to the RDP algorithm but is optimized by OpenCV for better performance.

## GUI Overview

The application comes with a simple and user-friendly graphical user interface (GUI) that allows users to easily load images and apply different polygon detection algorithms. The interface includes buttons to execute each algorithm and display the results.

![GUI](https://github.com/user-attachments/assets/ba6bb17d-1d07-409d-b101-de30497c772e)


### How to Use the GUI

1. **Choose an Image:**
   - Click on the "Search" button to load an image file from your computer.

2. **Apply Detection Algorithms:**
   - Use the available buttons (Harris, Shi-Tomasi, RDP, approxPolyDP) to run the respective algorithm on the loaded image.
   - The results will be displayed in a new window with the detected shapes.

## Example Outputs and Performance Metrics

### Harris Corner Detection

This method detects corner points by looking at the brightness changes around each pixel.

- **Level 1:**
  
  ![Harris-Level1](https://github.com/user-attachments/assets/ebaef762-b80b-451f-935e-c4208ad21001)

  
  - **Accuracy:** Detected corners correctly but detected too many points as corners in some places.
  - **Classification Accuracy:** Difficult to classify shapes correctly due to excessive corner detection.

- **Level 2 & Level 3:**

![Harris-Level2](https://github.com/user-attachments/assets/33fa7ea8-a9cd-42c1-88ec-3e38ffd12951)


  
  ![Harris-Level3](https://github.com/user-attachments/assets/bc4a269c-3a1a-4283-8141-b9db811dcd65)

  
  - **Real-world Scenarios:** Increased noise leads to more detected corners, making accurate classification difficult.

### Shi-Tomasi Corner Detection

The Shi-Tomasi algorithm works similarly to the Harris corner detection. It selects the strongest corner points by filtering with parameters finding fewer but more distinct corners.

- **Level 1:**
  
  ![Shi-Tomasi Level1](https://github.com/user-attachments/assets/70ca9f18-2893-494b-b8ad-5760aeacfa02)

  
  - **Accuracy:** The algorithm detects fewer, more distinct corners compared to Harris.
  - **Classification Accuracy:** Achieved 53% accuracy on noise-free images.

- **Level 2 & Level 3:**

  ![Shi-Tomasi Level2](https://github.com/user-attachments/assets/edcc5ff0-c21e-4f71-961b-16b0514fb745)


  ![Shi-Tomasi Level3](https://github.com/user-attachments/assets/6cb440a0-0316-4894-8ef6-f599a2912751)

  
  - **Real-world Scenarios:** Like Harris, the Shi-Tomasi algorithm struggles with increased noise, leading to poor classification accuracy, close to 0%.

### Ramer-Douglas-Peucker and Convex Hulls

The Ramer-Douglas-Peucker (RDP) algorithm simplifies the points of a shape by removing unnecessary points, while Convex Hulls create the smallest convex cover by connecting the outermost points of a shape.

- **Level 1:**
  
  ![RDP Level1](https://github.com/user-attachments/assets/c4a5b10d-6117-4dfb-a7a6-dcc211cfb834)

  
  - **Accuracy:** 93% in noise-free tests, successfully simplified and classified shapes.
  - **Classification Accuracy:** Minor issues detected, such as classifying a small circle as an octagon due to filtering.

- **Level 2:**

 ![RDP Level2](https://github.com/user-attachments/assets/8c8c0709-abf1-466b-89f9-6c7e046f58c0))
  
  - **Accuracy:** 100% accuracy, successfully classified all polygons.
  - **Classification Accuracy:** Noise in the image led to extra classifications, reducing reliability in some scenarios.

- **Level 3:**

  ![RDP Level3](https://github.com/user-attachments/assets/f109de5d-707c-40e0-9942-574e4ea81bfc)

  
  - **Accuracy:** 85% accuracy, correctly classified 18 out of 21 polygons.
  - **Classification Accuracy:** High accuracy but may fail in extremely noisy environments.

### approxPolyDP

The approxPolyDP algorithm reduces the number of corner points of a contour to create a simpler polygon, providing the most accurate and fastest results.

- **Level 1:**
  
 ![ApproxPolyDP Level1](https://github.com/user-attachments/assets/e92e3dff-843d-49e4-b1d6-ceea1b6bd57e)

  
  - **Accuracy:** 100%, successfully simplified and classified all shapes.
  - **Classification Accuracy:** Extremely effective on noise-free images.

- **Level 2:**

  ![ApproxPolyDP Level2](https://github.com/user-attachments/assets/1a7b9766-93cd-4678-887b-81789c4a541b)

  
  - **Accuracy:** 100%, maintained high accuracy despite increased complexity.
  - **Classification Accuracy:** Handles complex and noisy scenarios better than other algorithms.

- **Level 3:**

  ![ApproxPolyDP Level3](https://github.com/user-attachments/assets/8409b2b8-2057-464e-818f-dbed6bb18550)

  
  - **Accuracy:** 95%, correctly classified 20 out of 21 polygons.
  - **Classification Accuracy:** The most robust and reliable in both noise-free and noisy scenarios.

## Conclusion

- **Harris Corner Detection:**
  - Successful in detecting corner points.
  - Difficult to classify due to detecting too many corners.
  
- **Shi Tomasi Corner Detection:**
  - Detects more distinct corners.
  - Performs well on noise-free images but poorly in real-world scenarios.

- **Ramer-Douglas-Peucker (RDP) Algorithm:**
  - Simplifies points with high accuracy.
  - May miss small details.

- **approxPolyDP Algorithm:**
  - Provides the most accurate and fastest results.
  - Overall the most effective method.

## Running the Application

To run the application, ensure you have the necessary dependencies installed:

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt


