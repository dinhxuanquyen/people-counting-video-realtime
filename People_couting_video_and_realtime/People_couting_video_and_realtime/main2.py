import cv2  # Thư viện OpenCV để xử lý ảnh và video
import pandas as pd  # Thư viện Pandas để làm việc với dữ liệu dưới dạng bảng
import numpy as np  # Thư viện NumPy để xử lý các phép toán với mảng
from ultralytics import YOLO  # Thư viện Ultralytics để sử dụng mô hình YOLO
from tracker import Tracker  # Import lớp Tracker để theo dõi đối tượng
from datetime import datetime  # Thêm datetime để lấy thời gian vào/ra

# Sử dụng mô hình YOLO với trọng số đã huấn luyện sẵn
model = YOLO('People_couting_video_and_realtime/content/Runs/detect/Train/weights/Best.pt')

# Khởi tạo các vùng để theo dõi, với các đỉnh có thể chỉnh sửa
area1 = [(100, 370), (100, 400), (900, 400), (900, 370)]
area2 = [(100, 450), (100, 480), (900, 480), (900, 450)]
# Biến kiểm soát thao tác chuột với vùng được chọn
selected_vertex = None 
selected_area = None  
resize_mode = False 
dragging = False
corner_threshold = 10  # Ngưỡng xác định góc của đa giác để chỉnh sửa

# Hàm kiểm tra xem một điểm có nằm gần góc của một đa giác hay không (để chỉnh sửa kích thước)
def is_near_corner(x, y, area, threshold=corner_threshold):
    for i, point in enumerate(area):
        if abs(point[0] - x) < threshold and abs(point[1] - y) < threshold:
            return i
    return None

# Hàm xử lý sự kiện chuột để di chuyển hoặc thay đổi kích thước của các vùng
def mouse_callback(event, x, y, flags, param):
    global selected_vertex, selected_area, resize_mode, dragging

    if event == cv2.EVENT_LBUTTONDOWN:  # Khi người dùng nhấn chuột trái
        vertex_in_area1 = is_near_corner(x, y, area1)
        vertex_in_area2 = is_near_corner(x, y, area2)

        if vertex_in_area1 is not None:  # Nếu nhấn vào góc của area1
            selected_vertex = vertex_in_area1
            selected_area = 'area1'
            resize_mode = True
        elif vertex_in_area2 is not None:  # Nếu nhấn vào góc của area2
            selected_vertex = vertex_in_area2
            selected_area = 'area2'
            resize_mode = True
        elif cv2.pointPolygonTest(np.array(area1, np.int32), (x, y), False) >= 0:  # Nếu nhấn vào bên trong area1
            selected_area = 'area1'
            dragging = True
        elif cv2.pointPolygonTest(np.array(area2, np.int32), (x, y), False) >= 0:  # Nếu nhấn vào bên trong area2
            selected_area = 'area2'
            dragging = True

    elif event == cv2.EVENT_MOUSEMOVE:  # Khi di chuyển chuột
        if resize_mode and selected_vertex is not None:  # Nếu đang ở chế độ chỉnh sửa kích thước
            if selected_area == 'area1':
                area1[selected_vertex] = (x, y)
            elif selected_area == 'area2':
                area2[selected_vertex] = (x, y)
        elif dragging:  # Nếu đang kéo vùng
            dx = x - param['prev_x']
            dy = y - param['prev_y']
            if selected_area == 'area1':
                area1[:] = [(px + dx, py + dy) for px, py in area1]
            elif selected_area == 'area2':
                area2[:] = [(px + dx, py + dy) for px, py in area2]
        param['prev_x'], param['prev_y'] = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # Khi nhả chuột trái
        selected_vertex = None
        resize_mode = False
        dragging = False

# Tạo cửa sổ hiển thị và gán hàm xử lý chuột cho cửa sổ đó
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', mouse_callback, param={'prev_x': 0, 'prev_y': 0})

# Mở video để xử lý hình ảnh
cap = cv2.VideoCapture('People_couting_video_and_realtime/vid1.mp4')

if not cap.isOpened():  # Kiểm tra xem có mở được video không
    print("Error: Unable to open the video.")
    exit()
    

# Khởi tạo các biến dùng để theo dõi
count = 0
tracker = Tracker()
people_entering = {}
entering = set()
exiting = set()
people_exiting = {}
already_counted_entering = set()
already_counted_exiting = set()

# Mở tệp CSV để ghi thông tin thời gian ra vào
with open('people_log.csv', 'w') as log_file:
    log_file.write('ID,Status,Time\n')  # Ghi tiêu đề các cột

    while True:
        ret, frame = cap.read()
        if not ret:  # Nếu không đọc được khung hình
            print("Error: Unable to read frame from the video.")
            break

        count += 1
        if count % 2 != 0:  # Chỉ xử lý một nửa số khung hình để tiết kiệm tài nguyên
            continue

        frame = cv2.resize(frame, (1020, 500))

        results = model(frame)  # Dự đoán đối tượng trong khung hình bằng YOLO
        list_boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ của bounding box
                class_id = int(box.cls[0])
                if class_id == 0:  # Kiểm tra nếu đối tượng là "person" (thường là class_id 0 trong YOLO)
                    list_boxes.append([x1, y1, x2, y2])

        bbox_id = tracker.update(list_boxes)  # Cập nhật danh sách đối tượng đang theo dõi

        for bbox in bbox_id:  # Duyệt qua các đối tượng đang theo dõi
            x3, y3, x4, y4, id = bbox

            # Kiểm tra xem đối tượng có đi vào vùng area2 hay không
            results = cv2.pointPolygonTest(np.array(area2, np.int32), (int(x4), int(y4)), False)
            if results >= 0:
                people_entering[id] = (x4, y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            if id in people_entering:
                # Kiểm tra xem đối tượng có đi vào vùng area1 từ area2 hay không
                results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (int(x4), int(y4)), False)
                if results1 >= 0 and id not in already_counted_entering:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    entering.add(id)
                    already_counted_entering.add(id)
                    log_file.write(f'{id},Entering,{datetime.now()}\n')  # Ghi vào CSV

            # Kiểm tra xem đối tượng có đi vào vùng area1 hay không
            results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (int(x4), int(y4)), False)
            if results2 >= 0:
                people_exiting[id] = (x4, y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

            if id in people_exiting:
                # Kiểm tra xem đối tượng có đi vào vùng area2 từ area1 hay không
                results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (int(x4), int(y4)), False)
                if results3 >= 0 and id not in already_counted_exiting:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                    cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    exiting.add(id)
                    already_counted_exiting.add(id)
                    log_file.write(f'{id},Exiting,{datetime.now()}\n')  # Ghi vào CSV

        # Vẽ các vùng area1 và area2 lên khung hình
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)

        # Hiển thị số lượng người đã vào và ra
        cv2.putText(frame, f'Nguoi vao: {len(entering)}', (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Nguoi ra: {len(exiting)}', (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

        # Tính và hiển thị tổng số người hiện tại
        total_people = len(entering) - len(exiting)
        cv2.putText(frame, f'Tong so nguoi: {total_people}', (60, 200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

        # Hiển thị khung hình đã xử lý
        cv2.imshow("RGB", frame)
        if cv2.waitKey(100) & 0xFF == 27:  # Nhấn phím Esc để thoát
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
