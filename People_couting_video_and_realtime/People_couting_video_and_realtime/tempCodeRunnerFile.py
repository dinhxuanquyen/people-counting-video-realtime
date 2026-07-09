# Hàm xử lý sự kiện chuột
def mouse_callback(event, x, y, flags, param):
    global dividing_line_x
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        dividing_line_x = x