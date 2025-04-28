import cv2
import numpy as np
from scipy.ndimage import zoom

def draw_hollow_circle_by_red_regions(original_img: np.ndarray,
                                       heatmap: np.ndarray,
                                       alpha: float = 0.4,
                                       min_area: int = 100) -> np.ndarray:
    """
    วาดวงกลมรอบโซนสีแดง (hot regions) บน heatmap
    1) ย่อ/ขยาย heatmap ให้เท่าขนาด original_img
    2) แปลงเป็น uint8 แล้ว applyColorMap (JET)
    3) แปลงเป็น HSV แล้ว threshold ช่วงแดง
    4) หา contour ใหญ่สุด แล้วคำนวณ minEnclosingCircle
    5) วาดวงกลมบน original_img (BGR)
    คืนค่า original_img ที่มีวงกลม
    """
    # 1) resize heatmap
    h, w = heatmap.shape
    heatmap_resized = zoom(heatmap, (original_img.shape[0]/h, original_img.shape[1]/w))
    heat_uint8 = np.uint8(255 * heatmap_resized)

    # 2) สร้าง colored heatmap
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # 3) แปลงเป็น HSV แล้ว threshold ช่วงแดง
    hsv = cv2.cvtColor(colored, cv2.COLOR_BGR2HSV)
    # ช่วงแดงตอน hue ต่ำ และสูง (wrap-around)
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 4) หา contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # ถ้าไม่มี contour คืน original เลย
        return original_img.copy()

    # เลือก contour ใหญ่สุด (ตามพื้นที่)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        # ถ้าพื้นที่เล็กเกิน return เลย
        return original_img.copy()

    # 5) คำนวณวงกลมครอบ
    (x, y), radius = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius) + 2  # เติม border เล็กน้อย

    circled = original_img.copy()
    # วาดวงกลม (BGR: here ใช้สีฟ้า)
    cv2.circle(circled, center, radius,
               color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    return circled
