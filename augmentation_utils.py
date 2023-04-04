import cv2
import numpy as np


def rotate_point(point, angle, pivot):
    """
    Rotate a point around a pivot point by a given angle in degrees
    :param point: tuple, (x, y) coordinates of the point
    :param angle: float, angle in degrees to rotate the point by
    :param pivot: tuple, (x, y) coordinates of the pivot point
    :return: tuple, (x_new, y_new) coordinates of the rotated point
    """
    x, y = point
    cx, cy = pivot
    radians = np.deg2rad(angle)
    x_new = (x - cx) * np.cos(radians) - (y - cy) * np.sin(radians) + cx
    y_new = (x - cx) * np.sin(radians) + (y - cy) * np.cos(radians) + cy
    return int(x_new), int(y_new)


def add_black_lines(image, lines_num_range=(1, 4), line_thickness_range=(1, 3),
                    line_length=0.4, line_rotation=(-45, 45),
                    line_color=(0, 0, 0), **kwargs):
    """
    Add black lines to the input image with different rotation angles
    :param image: numpy array, input image
    :param lines_num_range: tuple, uniform range of lines to add to the image
    :param line_thickness_range: tuple, uniform range of thickness of the line
    :param line_length: float, length of the line as a percentage of the image diagonal
    :param line_rotation: tuple, rotation range for the line in degrees
    :param line_color: tuple, color of the line in (R, G, B) format
    :return: numpy array, image with added black lines
    """

    lines_num = np.random.randint(*lines_num_range)

    h, w, _ = image.shape
    for i in range(lines_num):
        # Generate random endpoints for the line
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        # Calculate line length based on the image diagonal
        line_len = int(np.sqrt(w**2 + h**2) * line_length)
        # Calculate line angle
        angle = np.random.uniform(line_rotation[0], line_rotation[1])
        # Calculate line endpoint coordinates based on the line length and angle
        dx = line_len * np.cos(np.deg2rad(angle))
        dy = line_len * np.sin(np.deg2rad(angle))
        x1_new, y1_new = int(x1 + dx), int(y1 + dy)
        x2_new, y2_new = int(x2 - dx), int(y2 - dy)
        # Draw the line on the original image
        line_thickness = np.random.randint(*line_thickness_range)
        # line_color = tuple(np.random.randint(0, 255, size=3))
        cv2.line(image, (x1_new, y1_new), (x2_new, y2_new), line_color, line_thickness)

    return image
