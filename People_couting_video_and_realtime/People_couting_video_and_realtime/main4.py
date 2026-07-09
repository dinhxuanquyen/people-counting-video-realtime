
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
from datetime import datetime

# Sử dụng mô hình YOLO với trọng số đã huấn luyện sẵn
model = YOLO('People_couting_video_and_realtime/content/Runs/detect/Train/weights/Best.pt')

# Khởi tạo các biến dùng để theo dõi
count = 0
tracker = Tracker()
people_entering = {}
people_exiting = {}
entering = set()
exiting = set()
already_counted_entering = set()
already_counted_exiting = set()

# Đặt kích thước frame mong muốn
frame_width = 640
frame_height = 480

# Tọa độ của đường phân chia
dividing_line_x = frame_width // 2
# # Hàm xử lý sự kiện chuột
# def mouse_callback(event, x, y, flags, param):
#     global dividing_line_x
#     if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
#         dividing_line_x = x

# Thay đổi nguồn video thành camera máy tính
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()

# Đặt kích thước cho camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Mở tệp CSV để ghi thông tin thời gian ra vào
with open('people_log.csv', 'w') as log_file:
    log_file.write('ID,Status,Time\n')  # Ghi tiêu đề các cột

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc khung hình từ camera.")
            break

        # Resize frame nếu kích thước thực tế khác với kích thước mong muốn
        if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
            frame = cv2.resize(frame, (frame_width, frame_height))

        count += 1
        if count % 2 != 0:
            continue

        results = model(frame)
        list_boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if class_id == 0 and conf > 0.5:
                    list_boxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 2)

        bbox_id = tracker.update(list_boxes)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = (x3 + x4) // 2  # Tính toán tâm của bounding box

            cv2.rectangle(frame, (x3, y3), (x4, y4), (0,255, 0), 2)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            # Kiểm tra hướng di chuyển
            if cx < dividing_line_x:
                people_entering[id] = cx
            elif cx > dividing_line_x:
                people_exiting[id] = cx

            if id in people_entering and cx > dividing_line_x and id not in already_counted_entering:
                entering.add(id)
                already_counted_entering.add(id)
                log_file.write(f'{id},Entering,{datetime.now()}\n')

            if id in people_exiting and cx < dividing_line_x and id not in already_counted_exiting:
                exiting.add(id)
                already_counted_exiting.add(id)
                log_file.write(f'{id},Exiting,{datetime.now()}\n')

        # Vẽ đường phân chia
        cv2.line(frame, (dividing_line_x,0), (dividing_line_x, frame_height), (0, 255, 255), 8)

        # Hiển thị số lượng người đã vào và ra
        cv2.putText(frame, f'Nguoi vao: {len(entering)}', (30, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f'Nguoi ra: {len(exiting)}', (30, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

        # Tính và hiển thị tổng số người hiện tại
        total_people = len(entering) - len(exiting)
        cv2.putText(frame, f'Tong so nguoi: {total_people}', (30, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

        # Hiển thị khung hình đã xử lý
        cv2.imshow("Camera", frame)
        # cv2.setMouseCallback("Camera", mouse_callback)
        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn Esc để thoát
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()