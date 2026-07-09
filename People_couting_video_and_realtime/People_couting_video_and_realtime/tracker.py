import math


class Tracker:
    def __init__(self):
        #  Lưu trữ vị trí trung tâm của các đối tượng
        self.center_points = {}
         # Đếm số lượng ID
        # Mỗi khi phát hiện một đối tượng mới, số đếm sẽ tăng lên một
        self.id_count = 0


    def update(self, objects_rect):
         # Danh sách chứa bounding boxes và IDs của các đối tượng
        objects_bbs_ids = []

         # Lấy vị trí trung tâm của đối tượng mới
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Kiểm tra xem đối tượng này đã được phát hiện trước đó chưa
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 50:
                      # Cập nhật vị trí trung tâm của đối tượng đã phát hiện
                    self.center_points[id] = (cx, cy)
#                    # Thêm bounding box và ID vào danh sách
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Nếu phát hiện đối tượng mới, gán ID cho đối tượng đó
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

       # Làm sạch dictionary bằng cách loại bỏ các ID không còn sử dụng
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids