import cv2
import numpy as np
from scipy.ndimage import zoom


def draw_hollow_circle_by_red_regions(original_img: np.ndarray,
                                       heatmap: np.ndarray,
                                       alpha: float = 0.4,
                                       min_area: int = 100) -> np.ndarray:
    """
    Draw a circle around red hot regions on a heatmap.
    1) Resize heatmap to match original_img.
    2) Convert to uint8 and apply JET colormap.
    3) Convert to HSV and threshold red ranges.
    4) Find the largest contour and compute minEnclosingCircle.
    5) Draw the circle on original_img (BGR).
    Return original_img with circle overlay.
    """
    # 1) Resize heatmap.
    h, w = heatmap.shape
    heatmap_resized = zoom(heatmap, (original_img.shape[0]/h, original_img.shape[1]/w))
    heat_uint8 = np.uint8(255 * heatmap_resized)

    # 2) Create colored heatmap.
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # 3) Convert to HSV and threshold red ranges.
    hsv = cv2.cvtColor(colored, cv2.COLOR_BGR2HSV)
    # Red at low and high hue ends (wrap-around).
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 4) Find contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No contour found: return original image.
        return original_img.copy()

    # Select the largest contour by area.
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        # Too small: return original image.
        return original_img.copy()

    # 5) Compute enclosing circle.
    (x, y), radius = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius) + 2  # Add a small border.

    circled = original_img.copy()
    # Draw circle (BGR blue).
    cv2.circle(circled, center, radius,
               color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    return circled
