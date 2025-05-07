import numpy as np
import pygame as pg
import cv2


class VirtualCamera:
    def __init__(self, grid_map, epsilon=8):
        self.grid_map = grid_map
        self.epsilon = epsilon
        self.camera_view_distance = 5  # Khoảng cách camera có thể nhìn thấy (đơn vị ô lưới)

    def capture_image(self, robot_pos, direction):
        """Chụp ảnh xung quanh robot trong phạm vi nhìn thấy được"""
        # Tính kích thước ảnh dựa trên khoảng cách nhìn
        view_width = self.camera_view_distance * 2 + 1
        view_height = self.camera_view_distance * 2 + 1

        # Tạo ảnh trống
        image = np.ones((view_height * self.epsilon, view_width * self.epsilon, 3), dtype=np.uint8) * 255

        # Xác định góc trên bên trái của khung nhìn
        top_left_row = robot_pos[0] - self.camera_view_distance
        top_left_col = robot_pos[1] - self.camera_view_distance

        # Vẽ các ô trong tầm nhìn
        for i in range(view_height):
            map_row = top_left_row + i
            for j in range(view_width):
                map_col = top_left_col + j

                # Kiểm tra xem ô có nằm trong bản đồ không
                if 0 <= map_row < len(self.grid_map.map) and 0 <= map_col < len(self.grid_map.map[0]):
                    cell_value = self.grid_map.map[map_row, map_col]

                    # Vẽ ô tương ứng trong ảnh
                    cell_rect = (
                        j * self.epsilon,
                        i * self.epsilon,
                        self.epsilon,
                        self.epsilon
                    )

                    # Xác định màu sắc
                    if cell_value == 1 or cell_value == 'o':  # Vật cản
                        color = (0, 0, 0)  # Đen
                    elif cell_value == 'e':  # Đã khám phá
                        color = (0, 255, 0)  # Xanh lá
                    else:  # Chưa khám phá
                        color = (255, 255, 255)  # Trắng

                    # Vẽ ô vào ảnh
                    cv2.rectangle(
                        image,
                        (cell_rect[0], cell_rect[1]),
                        (cell_rect[0] + cell_rect[2], cell_rect[1] + cell_rect[3]),
                        color,
                        -1  # Filled
                    )

        # Vẽ robot
        robot_center = (
            self.camera_view_distance * self.epsilon + self.epsilon // 2,
            self.camera_view_distance * self.epsilon + self.epsilon // 2
        )
        cv2.circle(image, robot_center, self.epsilon // 3, (255, 0, 0), -1)

        # Vẽ hướng của robot
        direction_end = (
            int(robot_center[0] + direction[1] * self.epsilon),
            int(robot_center[1] + direction[0] * self.epsilon)
        )
        cv2.line(image, robot_center, direction_end, (0, 0, 255), 2)

        return image

    def detect_dynamic_obstacles(self, current_image, previous_image):
        """
        Phát hiện vật cản di chuyển bằng cách so sánh 2 frame liên tiếp
        Trả về danh sách các vật cản di chuyển với vị trí
        """
        if previous_image is None:
            return []

        # Chuyển đổi ảnh sang grayscale
        gray_current = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)

        # Tìm sự khác biệt giữa hai frame
        diff = cv2.absdiff(gray_current, gray_prev)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Áp dụng các phép biến đổi hình thái học để loại bỏ nhiễu
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # Tìm contour của các vật cản di chuyển
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc ra các contour quá nhỏ
        dynamic_obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Ngưỡng kích thước tối thiểu
                continue

            # Lấy bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Tính vị trí tương đối so với robot
            rel_row = y // self.epsilon - self.camera_view_distance
            rel_col = x // self.epsilon - self.camera_view_distance

            dynamic_obstacles.append(((rel_row, rel_col), (w, h)))

        return dynamic_obstacles