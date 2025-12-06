
import cv2
import json
import numpy as np
import os
import time
import torch
import warnings
from ultralytics import YOLO

# Bỏ qua DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Kiểm tra RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[ERROR] pyrealsense2 not installed. Run: pip install pyrealsense2")

# Kiểm tra requests cho Telegram
try:
    import requests
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("[WARN] requests not installed. Telegram notifications disabled. Run: pip install requests")

# ==================== CONFIG ====================
WINDOW_NAME_RGB = "Safe Zone - RGB View"
WINDOW_NAME_DEPTH = "Safe Zone - Depth View"
SAVE_PATH = "polygon_depth.json"
MODEL_NAME = "yolo11m-pose.pt"
IMG_SIZE = 640
CONF_THRES = 0.25
KEYPOINT_CONF_THRES = 0.2

# Depth settings
DISTANCE_THRESHOLD_MM = 300  # Ngưỡng 300mm (30cm) - người phải GẦN hơn bề mặt polygon
MIN_DEPTH_MM = 300       # Depth tối thiểu hợp lệ
MAX_DEPTH_MM = 8000      # Depth tối đa hợp lệ

# ==================== TELEGRAM CONFIG ====================
# Cách lấy TELEGRAM_BOT_TOKEN và TELEGRAM_CHAT_ID:
# 1. Tạo bot: Tìm @BotFather trên Telegram → /newbot → làm theo hướng dẫn
# 2. Lấy token: BotFather sẽ cho bạn token (VD: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
# 3. Lấy chat_id: 
#    - Gửi tin nhắn cho bot
#    - Truy cập: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
#    - Tìm "chat":{"id": 123456789} trong response
TELEGRAM_BOT_TOKEN = "8137546763:AAEOr8zLInmdnL-lCluCoSxJKBLEOlyj-G0"  # ← ĐIỀN TOKEN VÀO ĐÂY
TELEGRAM_CHAT_ID = "6448037928"    # ← ĐIỀN CHAT_ID VÀO ĐÂY
TELEGRAM_ENABLED = False  # Tự động bật nếu có token và chat_id hợp lệ
TELEGRAM_COOLDOWN = 10   # Cooldown giữa các tin nhắn (giây) để tránh spam

# Polygon editor state
points = []
polygon_closed = False
test_mode = False
show_depth_view = True

# Reference depth của polygon (mm)
polygon_reference_depth = None
polygon_reference_distance = None  # Khoảng cách thực tế 3D trung bình

# Mouse-edit state
selected_idx = None
dragging = False
mouse_down_pos = None  # Lưu vị trí khi bắt đầu drag
RADIUS = 7
DRAG_THRESHOLD = 5  # Ngưỡng để phân biệt click và drag

# Global depth image (để access từ mouse callback)
current_depth_image = None
camera_intrinsics = None  # Thông số camera để tính 3D coordinates

# Telegram notification state
last_telegram_time = 0  # Timestamp của lần gửi telegram cuối

# Device: try GPU first
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("[INFO] CUDA available — using GPU.")
else:
    DEVICE = "cpu"
    print("[WARN] CUDA not available — using CPU.")

# Load YOLO model
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_NAME)
print("[INFO] Model loaded.")

# Kiểm tra và bật Telegram nếu đã config
if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    TELEGRAM_ENABLED = True
    print(f"[INFO] Telegram notifications ENABLED (cooldown: {TELEGRAM_COOLDOWN}s)")
else:
    print("[INFO] Telegram notifications DISABLED (not configured)")

# ==================== DEPTH CAMERA SETUP ====================
class DepthCamera:
    def __init__(self):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("RealSense library not available")
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        print("[INFO] Starting RealSense camera...")
        self.profile = self.pipeline.start(self.config)
        
        # Get depth sensor's depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[INFO] Depth Scale: {self.depth_scale}")
        
        # Get camera intrinsics (để tính 3D coordinates)
        depth_profile = self.profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        print(f"[INFO] Camera Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
        print(f"[INFO] Principal Point: cx={self.intrinsics.ppx:.1f}, cy={self.intrinsics.ppy:.1f}")
        
        # Create align object (align depth to color)
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        print("[INFO] RealSense camera started successfully.")
    
    def get_frames(self):
        """Lấy aligned depth frame và color frame"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())  # uint16, đơn vị: mm
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image
    
    def stop(self):
        self.pipeline.stop()
        print("[INFO] RealSense camera stopped.")

# ==================== POLYGON FUNCTIONS ====================
def save_polygon(path=SAVE_PATH):
    """Lưu polygon + reference depth + distance"""
    data = {
        "points": points,
        "reference_depth_mm": polygon_reference_depth,
        "reference_distance_mm": polygon_reference_distance
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved polygon + depth + distance to {path}")

def load_polygon(path=SAVE_PATH):
    """Load polygon + reference depth + distance"""
    global points, polygon_closed, polygon_reference_depth, polygon_reference_distance
    if not os.path.exists(path):
        print(f"[WARN] {path} not found.")
        return
    
    with open(path, "r") as f:
        data = json.load(f)
    
    pts = data.get("points", [])
    if pts:
        points[:] = [tuple(map(int, p)) for p in pts]
        polygon_closed = len(points) >= 3
        polygon_reference_depth = data.get("reference_depth_mm", None)
        polygon_reference_distance = data.get("reference_distance_mm", None)
        print(f"[INFO] Loaded polygon: {len(points)} points")
        if polygon_reference_depth:
            print(f"[INFO] Reference depth: {polygon_reference_depth:.0f}mm ({polygon_reference_depth/1000:.2f}m)")
        if polygon_reference_distance:
            print(f"[INFO] Reference distance: {polygon_reference_distance:.0f}mm ({polygon_reference_distance/1000:.2f}m)")

def depth_to_3d_distance(x, y, depth_mm, intrinsics):
    """
    Chuyển đổi (x, y, depth) sang khoảng cách 3D thực tế từ camera
    
    Args:
        x, y: Pixel coordinates
        depth_mm: Depth value in mm
        intrinsics: Camera intrinsics
    
    Returns:
        distance_mm: Khoảng cách 3D thực tế (Euclidean distance)
    """
    if depth_mm <= 0 or intrinsics is None:
        return None
    
    # Deproject pixel (x, y, depth) sang 3D point (X, Y, Z)
    # X = (x - cx) * Z / fx
    # Y = (y - cy) * Z / fy
    # Z = depth
    
    Z = depth_mm  # mm
    X = (x - intrinsics.ppx) * Z / intrinsics.fx
    Y = (y - intrinsics.ppy) * Z / intrinsics.fy
    
    # Tính khoảng cách Euclidean từ camera (0,0,0) đến (X,Y,Z)
    distance_mm = np.sqrt(X**2 + Y**2 + Z**2)
    
    return distance_mm

def auto_calculate_polygon_depth(depth_image, show_log=True):
    """
    TỰ ĐỘNG tính depth và DISTANCE của polygon NGAY KHI CÓ ≥3 ĐIỂM
    """
    global polygon_reference_depth, polygon_reference_distance
    
    if len(points) < 3:
        polygon_reference_depth = None
        polygon_reference_distance = None
        return
    
    if depth_image is None or camera_intrinsics is None:
        return
    
    # Tạo mask cho polygon (tạm thời đóng để tính)
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    # Lấy tất cả pixels trong polygon
    ys, xs = np.where(mask == 255)
    
    if len(xs) == 0:
        polygon_reference_depth = None
        polygon_reference_distance = None
        return
    
    # Lấy depth values
    depths = depth_image[ys, xs]
    
    # Filter valid depths
    valid_mask = (depths > MIN_DEPTH_MM) & (depths < MAX_DEPTH_MM)
    valid_xs = xs[valid_mask]
    valid_ys = ys[valid_mask]
    valid_depths = depths[valid_mask]
    
    if len(valid_depths) == 0:
        polygon_reference_depth = None
        polygon_reference_distance = None
        return
    
    # Tính median depth
    new_depth = float(np.median(valid_depths))
    
    # Tính 3D distances cho tất cả pixels hợp lệ
    distances = []
    for i in range(len(valid_xs)):
        dist = depth_to_3d_distance(valid_xs[i], valid_ys[i], valid_depths[i], camera_intrinsics)
        if dist is not None:
            distances.append(dist)
    
    if len(distances) == 0:
        polygon_reference_distance = None
    else:
        # Tính median distance (khoảng cách thực tế trung bình)
        new_distance = float(np.median(distances))
        
        # Chỉ update nếu có thay đổi đáng kể
        if polygon_reference_distance is None or abs(new_distance - polygon_reference_distance) > 10:
            polygon_reference_distance = new_distance
    
    # Update depth
    if polygon_reference_depth is None or abs(new_depth - polygon_reference_depth) > 10:
        polygon_reference_depth = new_depth
        if show_log and polygon_reference_distance:
            print(f"[DEPTH] Depth: {polygon_reference_depth:.0f}mm | Distance: {polygon_reference_distance:.0f}mm ({polygon_reference_distance/1000:.2f}m)")

def find_nearest_point(x, y, thresh=RADIUS+4):
    """Tìm điểm polygon gần nhất với chuột"""
    if not points:
        return None
    pts = np.array(points)
    dist = np.sqrt((pts[:,0]-x)**2 + (pts[:,1]-y)**2)
    min_idx = int(np.argmin(dist))
    if dist[min_idx] <= thresh:
        return min_idx
    return None

def is_drag_motion(start_pos, end_pos):
    """Kiểm tra có phải là drag motion không (dựa vào khoảng cách)"""
    if start_pos is None:
        return False
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    return distance > DRAG_THRESHOLD

# ==================== MOUSE CALLBACK ====================
def mouse_callback(event, x, y, flags, param):
    global points, polygon_closed, test_mode
    global selected_idx, dragging, mouse_down_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down_pos = (x, y)
        
        if test_mode:
            if len(points) >= 3:
                inside = cv2.pointPolygonTest(np.array(points, dtype=np.int32), (x,y), False)
                print(f"[TEST] Point {(x,y)} {'INSIDE' if inside>=0 else 'OUTSIDE'} polygon.")
            return

        # Tìm điểm gần nhất
        idx = find_nearest_point(x, y)
        if idx is not None:
            selected_idx = idx
            dragging = True
            print(f"[SELECT] Point {idx+1} selected for dragging")
        # Nếu không có điểm gần → chờ xem có drag không

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_idx is not None:
            # Đang drag điểm
            points[selected_idx] = (x, y)
            # TỰ ĐỘNG cập nhật depth khi di chuyển
            if len(points) >= 3 and current_depth_image is not None:
                auto_calculate_polygon_depth(current_depth_image, show_log=False)

    elif event == cv2.EVENT_LBUTTONUP:
        # Kiểm tra xem có phải drag không
        if dragging and selected_idx is not None:
            # Kết thúc drag
            if is_drag_motion(mouse_down_pos, (x, y)):
                print(f"[MOVE] Point {selected_idx+1} moved to {(x, y)}")
                # Tính lại depth sau khi drag xong
                if len(points) >= 3 and current_depth_image is not None:
                    auto_calculate_polygon_depth(current_depth_image, show_log=True)
        else:
            # Không phải drag → là click để thêm điểm mới
            if not polygon_closed:
                # Kiểm tra không click vào điểm đã có
                idx = find_nearest_point(x, y)
                if idx is None:
                    points.append((x, y))
                    print(f"[ADD] Point {len(points)}: {(x, y)}")
                    
                    # TỰ ĐỘNG tính depth nếu đã có ≥3 điểm
                    if len(points) >= 3 and current_depth_image is not None:
                        auto_calculate_polygon_depth(current_depth_image, show_log=True)
        
        # Reset state
        dragging = False
        selected_idx = None
        mouse_down_pos = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Xóa điểm gần nhất
        idx = find_nearest_point(x, y)
        if idx is not None:
            removed = points.pop(idx)
            print(f"[REMOVE] Removed point {idx+1}: {removed}")
            
            # Reset polygon nếu < 3 điểm
            if len(points) < 3:
                polygon_closed = False
                polygon_reference_depth = None
                polygon_reference_distance = None
                print("[INFO] Polygon opened (< 3 points)")
            else:
                # Tính lại depth nếu còn ≥3 điểm
                if current_depth_image is not None:
                    auto_calculate_polygon_depth(current_depth_image, show_log=True)

# ==================== DETECTION LOGIC ====================
def check_keypoints_depth_intrusion(keypoints_data, depth_image, polygon_pts, reference_distance):
    """
    Kiểm tra xâm nhập dựa trên:
    1. Keypoint có trong polygon không
    2. KHOẢNG CÁCH 3D của keypoint có GẦN hơn reference distance không
    
    Returns:
        (has_intrusion, intrusion_keypoints, distance_info)
    """
    if len(polygon_pts) < 3 or reference_distance is None or camera_intrinsics is None:
        return False, [], []
    
    intrusion_keypoints = []
    distance_info = []  # [(x, y, kp_distance, ref_distance, distance_diff)]
    
    for person_kps in keypoints_data:
        for kp in person_kps:
            x, y, conf = kp
            if conf < KEYPOINT_CONF_THRES:
                continue
            
            ix, iy = int(x), int(y)
            
            # Check 1: Keypoint trong polygon?
            is_inside = cv2.pointPolygonTest(polygon_pts, (float(x), float(y)), False)
            if is_inside < 0:
                continue  # Ngoài polygon → skip
            
            # Check 2: Lấy depth tại keypoint
            if iy < 0 or iy >= depth_image.shape[0] or ix < 0 or ix >= depth_image.shape[1]:
                continue
            
            kp_depth = depth_image[iy, ix]
            
            # Validate depth
            if kp_depth < MIN_DEPTH_MM or kp_depth > MAX_DEPTH_MM:
                continue
            
            # Check 3: Tính khoảng cách 3D thực tế của keypoint
            kp_distance = depth_to_3d_distance(ix, iy, kp_depth, camera_intrinsics)
            
            if kp_distance is None:
                continue
            
            # Check 4: So sánh khoảng cách 3D với reference
            distance_diff = reference_distance - kp_distance  # Positive = người GẦN hơn
            
            if distance_diff > DISTANCE_THRESHOLD_MM:
                # Người VÀO polygon VÀ đủ GẦN (theo khoảng cách thực) → ALERT!
                intrusion_keypoints.append((ix, iy))
                distance_info.append((ix, iy, kp_distance, reference_distance, distance_diff))
    
    return len(intrusion_keypoints) > 0, intrusion_keypoints, distance_info

# ==================== TELEGRAM NOTIFICATION ====================
def send_telegram_alert(message, image=None):
    """
    Gửi cảnh báo qua Telegram
    
    Args:
        message: Text message
        image: OpenCV image (BGR format) to send as photo (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    global last_telegram_time
    
    if not TELEGRAM_ENABLED or not TELEGRAM_AVAILABLE:
        return False
    
    # Kiểm tra cooldown
    current_time = time.time()
    if current_time - last_telegram_time < TELEGRAM_COOLDOWN:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
        
        # Gửi text message
        if message:
            text_url = f"{url}/sendMessage"
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(text_url, data=data, timeout=5)
            
            if not response.ok:
                print(f"[TELEGRAM ERROR] Failed to send message: {response.status_code}")
                return False
        
        # Gửi ảnh nếu có
        if image is not None:
            photo_url = f"{url}/sendPhoto"
            
            # Encode image to jpg
            _, buffer = cv2.imencode('.jpg', image)
            
            files = {
                'photo': ('alert.jpg', buffer.tobytes(), 'image/jpeg')
            }
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': '🚨 Intrusion Alert - Captured Image'
            }
            
            response = requests.post(photo_url, files=files, data=data, timeout=10)
            
            if not response.ok:
                print(f"[TELEGRAM ERROR] Failed to send photo: {response.status_code}")
                return False
        
        # Update last sent time
        last_telegram_time = current_time
        print(f"[TELEGRAM] Alert sent successfully!")
        return True
        
    except requests.exceptions.Timeout:
        print("[TELEGRAM ERROR] Request timeout")
        return False
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")
        return False


# ==================== VISUALIZATION ====================
def colorize_depth(depth_image):
    """Convert depth image to colorized visualization"""
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), 
        cv2.COLORMAP_JET
    )
    return depth_colormap

def draw_overlay(img, polygon_pts, intrusion_keypoints, alerts, distance_info):
    """Vẽ polygon, keypoints vi phạm, cảnh báo"""
    overlay = img.copy()

    # Draw polygon
    if polygon_pts and len(polygon_pts) >= 2:
        pts = np.array(polygon_pts, dtype=np.int32)
        
        # Vẽ viền polygon
        is_closed = polygon_closed and len(polygon_pts) >= 3
        cv2.polylines(overlay, [pts], isClosed=is_closed, color=(0,255,0), thickness=2)
        
        # Fill polygon nếu đã đóng
        if is_closed:
            cv2.fillPoly(overlay, [pts], color=(0,255,0))
            alpha = 0.2
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        
        # Vẽ đường nối tạm nếu chưa đóng
        elif len(polygon_pts) >= 3:
            # Vẽ fill tạm với alpha thấp hơn
            temp_overlay = img.copy()
            cv2.fillPoly(temp_overlay, [pts], color=(0,255,0))
            alpha = 0.1
            cv2.addWeighted(temp_overlay, alpha, img, 1-alpha, 0, img)

        # Draw polygon points
        for i, p in enumerate(polygon_pts):
            if i == selected_idx and dragging:
                color = (255,0,0)  # Xanh dương khi đang drag
            elif i == selected_idx:
                color = (255,165,0)  # Cam khi được chọn
            else:
                color = (0,0,255)  # Đỏ bình thường
            cv2.circle(img, p, RADIUS, color, -1)
            # Vẽ số thứ tự
            cv2.putText(img, str(i+1), (p[0]+10, p[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw intrusion keypoints
    if intrusion_keypoints:
        for (kx, ky) in intrusion_keypoints:
            cv2.circle(img, (kx, ky), 12, (0, 0, 255), 3)
            cv2.circle(img, (kx, ky), 4, (255, 255, 255), -1)

    # Draw distance info near keypoints
    if distance_info:
        for (kx, ky, kp_dist, ref_dist, diff) in distance_info:
            text = f"-{diff:.0f}mm"
            cv2.putText(img, text, (kx+15, ky-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Draw alerts
    if alerts:
        y = 40
        for a in alerts:
            cv2.putText(img, a, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            y += 50

def draw_ui(img):
    """Vẽ hướng dẫn sử dụng"""
    lines = [
        "Left-click: add point | Drag: move point",
        "Right-click: remove | Space: close polygon",
        "'s': save | 'l': load | 'c': clear",
        "'r': toggle depth view | 't': test mode",
        "'n': toggle Telegram notifications",
        "ESC/q: quit"
    ]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 20 + 22*i), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

# ==================== MAIN LOOP ====================
def main():
    global polygon_closed, test_mode, show_depth_view, current_depth_image, polygon_reference_depth
    global polygon_reference_distance, camera_intrinsics, TELEGRAM_ENABLED

    # Initialize camera
    if not REALSENSE_AVAILABLE:
        print("[ERROR] RealSense not available. Exiting.")
        return
    
    try:
        camera = DepthCamera()
        camera_intrinsics = camera.intrinsics  # Lưu intrinsics vào global
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        return

    cv2.namedWindow(WINDOW_NAME_RGB)
    cv2.setMouseCallback(WINDOW_NAME_RGB, mouse_callback)

    print("\n" + "="*60)
    print("3D DISTANCE-BASED INTRUSION DETECTION:")
    print("1. Click chuột để vẽ polygon")
    print("2. Distance tự động tính ngay khi có ≥3 điểm!")
    print("3. So sánh KHOẢNG CÁCH 3D thực tế (không chỉ depth)")
    print("4. Kéo thả điểm → Distance tự động cập nhật!")
    print("5. Nhấn SPACE để đóng polygon (optional)")
    print("6. Hệ thống tự động detect và cảnh báo")
    print("="*60 + "\n")

    while True:
        # Get frames
        color_image, depth_image = camera.get_frames()
        if color_image is None or depth_image is None:
            print("[WARN] No frames received")
            continue

        # Update global depth image cho mouse callback
        current_depth_image = depth_image

        # YOLO detection
        results = model(color_image, imgsz=IMG_SIZE, conf=CONF_THRES, device=DEVICE)
        annotated_frame = results[0].plot()

        # Process keypoints
        alerts = []
        intrusion_keypoints = []
        distance_info = []

        try:
            res = results[0]
            if res.keypoints is not None and len(res.keypoints) > 0:
                np_kps = res.keypoints.data.cpu().numpy()
                
                # Check intrusion dựa trên DISTANCE (chỉ cần ≥3 điểm)
                if len(points) >= 3 and polygon_reference_distance is not None:
                    has_intrusion, intrusion_kps, d_info = check_keypoints_depth_intrusion(
                        np_kps,
                        depth_image,
                        np.array(points, dtype=np.int32),
                        polygon_reference_distance  # Dùng distance thay vì depth
                    )
                    
                    if has_intrusion:
                        intrusion_keypoints = intrusion_kps
                        distance_info = d_info
                        alerts.append("⚠ DISTANCE INTRUSION ALERT ⚠")
                        
                        # Gửi Telegram notification
                        if TELEGRAM_ENABLED:
                            num_intrusions = len(intrusion_kps)
                            closest_dist = min([info[2] for info in d_info]) if d_info else 0
                            
                            telegram_msg = (
                                f"🚨 <b>INTRUSION ALERT</b> 🚨\n\n"
                                f"⚠️ Detected: {num_intrusions} keypoint(s) violated zone\n"
                                f"📏 Closest distance: {closest_dist:.0f}mm ({closest_dist/1000:.2f}m)\n"
                                f"📍 Reference distance: {polygon_reference_distance:.0f}mm\n"
                                f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                            
                            # Gửi với ảnh
                            send_telegram_alert(telegram_msg, annotated_frame)
        
        except Exception as e:
            print(f"[WARN] Detection error: {e}")

        # Draw overlays
        draw_overlay(annotated_frame, points, intrusion_keypoints, alerts, distance_info)
        draw_ui(annotated_frame)

        # Status info
        status = f"Points: {len(points)}"
        if polygon_closed:
            status += " | Status: CLOSED"
        elif len(points) >= 3:
            status += " | Status: OPEN (detecting active)"
        else:
            status += " | Status: Drawing..."
            
        if polygon_reference_distance:
            status += f" | Dist: {polygon_reference_distance:.0f}mm ({polygon_reference_distance/1000:.2f}m)"
        elif len(points) >= 3:
            status += " | Dist: Calculating..."
        
        status += f" | Device: {DEVICE}"
        
        # Thêm Telegram status
        if TELEGRAM_ENABLED:
            status += " | TG: ON"
        
        cv2.putText(annotated_frame, status, (8, annotated_frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        # Show RGB
        cv2.imshow(WINDOW_NAME_RGB, annotated_frame)

        # Show depth visualization
        if show_depth_view:
            depth_colormap = colorize_depth(depth_image)
            
            # Draw polygon on depth view
            if points and len(points) >= 2:
                pts = np.array(points, dtype=np.int32)
                is_closed = polygon_closed and len(points) >= 3
                cv2.polylines(depth_colormap, [pts], isClosed=is_closed, 
                            color=(255,255,255), thickness=2)
            
            # Hiển thị reference distance trên depth view
            if polygon_reference_distance and len(points) >= 3:
                text = f"Ref Dist: {polygon_reference_distance:.0f}mm"
                cv2.putText(depth_colormap, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow(WINDOW_NAME_DEPTH, depth_colormap)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC or Q
            break
        elif key == ord('c'):
            points.clear()
            polygon_closed = False
            polygon_reference_depth = None
            polygon_reference_distance = None
            print("[ACTION] Cleared polygon.")
        elif key == ord('s'):
            save_polygon()
        elif key == ord('l'):
            load_polygon()
        elif key == ord('t'):
            test_mode = not test_mode
            print(f"[ACTION] Test mode = {test_mode}")
        elif key == ord('r'):
            show_depth_view = not show_depth_view
            if not show_depth_view:
                cv2.destroyWindow(WINDOW_NAME_DEPTH)
            print(f"[ACTION] Depth view = {show_depth_view}")
        elif key == ord('n'):
            if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                TELEGRAM_ENABLED = not TELEGRAM_ENABLED
                print(f"[ACTION] Telegram notifications = {TELEGRAM_ENABLED}")
            else:
                print("[WARN] Telegram not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        elif key == 32:  # Space
            if len(points) >= 3:
                polygon_closed = True
                print(f"[ACTION] Polygon closed with {len(points)} points")
                if polygon_reference_distance:
                    print(f"[INFO] Reference distance: {polygon_reference_distance:.0f}mm ({polygon_reference_distance/1000:.2f}m)")
                    print(f"[INFO] Alert threshold: {DISTANCE_THRESHOLD_MM}mm closer than reference")
            else:
                print("[WARN] Need ≥3 points to close polygon.")

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()