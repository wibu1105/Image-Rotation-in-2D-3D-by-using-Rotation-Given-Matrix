import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Cách deploy app: đưa file lên 1 repository Github rồi deploy thông qua Streamlit Cloud bằng tài khoản Github đó
# --------------------- Core Logic ---------------------
class ImageRotation2D:
    def __init__(self, image: np.ndarray):
        """
        Khởi tạo đối tượng xoay ảnh màu 2D.

        :param image: Ảnh đầu vào dạng numpy array 3 chiều (H x W x C).
        :raises ValueError: Nếu ảnh không phải là ảnh màu RGB.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Ảnh đầu vào phải là ảnh màu RGB (3 kênh).")
        self.image = image
        self.height, self.width, self.channels = image.shape
        self.xc = (self.height - 1) / 2
        self.yc = (self.width - 1) / 2

    def rotation_matrix(self, angle_rad: float) -> np.ndarray:
        """
        Trả về ma trận xoay 2D Givens theo góc radian.

        :param angle_rad: Góc xoay tính bằng radian.
        :return: Ma trận xoay 2x2.
        """
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        return np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])

    def rotate(self, angle_degrees: float, background_color=(255, 255, 255)) -> np.ndarray:
        """
        Xoay ảnh màu bằng ánh xạ ngược với góc xoay tùy chọn.

        :param angle_degrees: Góc xoay theo độ (dương là xoay ngược chiều kim đồng hồ).
        :param background_color: Tuple RGB cho màu nền (mặc định: trắng).
        :return: Ảnh đã xoay.
        """
        angle_rad = np.deg2rad(angle_degrees)
        R = self.rotation_matrix(angle_rad)

        # Tính kích thước ảnh mới sau khi xoay
        corners = np.array([
            [ self.xc,  self.yc],
            [-self.xc, -self.yc],
            [-self.xc,  self.yc],
            [ self.xc, -self.yc]
        ])
        rotated_corners = (R @ corners.T).T
        x_bounds, y_bounds = rotated_corners[:, 0], rotated_corners[:, 1]

        new_h = int(np.ceil(x_bounds.max() - x_bounds.min()))
        new_w = int(np.ceil(y_bounds.max() - y_bounds.min()))
        x_c_new = (new_h - 1) / 2
        y_c_new = (new_w - 1) / 2

        # Tạo ảnh kết quả
        rotated = np.full((new_h, new_w, self.channels), background_color, dtype=self.image.dtype)

        # Tạo lưới tọa độ cho ảnh kết quả
        X, Y = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing='ij')
        Xc = X - x_c_new
        Yc = Y - y_c_new

        coords = np.stack([Xc.ravel(), Yc.ravel()], axis=1)
        inv_coords = (self.rotation_matrix(-angle_rad) @ coords.T).T

        src_x = inv_coords[:, 0] + self.xc
        src_y = inv_coords[:, 1] + self.yc

        # Gán giá trị pixel hợp lệ
        valid = (0 <= src_x) & (src_x < self.height) & (0 <= src_y) & (src_y < self.width)
        src_x_valid = src_x[valid].astype(int)
        src_y_valid = src_y[valid].astype(int)

        rotated.reshape(-1, self.channels)[valid] = self.image[src_x_valid, src_y_valid]
        return rotated

class ImageRotation3D:
    def __init__(self, image: np.ndarray):
        """
        Khởi tạo bộ xoay ảnh 3D bằng phép chiếu phối cảnh.

        :param image: Ảnh đầu vào dạng numpy array (H x W x C).
        """
        if image.ndim != 3 or image.shape[2] not in [1, 3]:
            raise ValueError("Ảnh phải có 3 chiều (H x W x C), kênh màu RGB hoặc grayscale mở rộng.")

        self.image = image
        self.height, self.width, self.channels = image.shape

        x, y = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        z = np.zeros_like(x)
        self.pixel_coords = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)

    def givens_rotation_matrix(self, i: int, j: int, theta: float) -> np.ndarray:
        """
        Trả về ma trận xoay Givens 3x3.

        :param i: Trục thứ nhất (0, 1, 2).
        :param j: Trục thứ hai (0, 1, 2), khác với i.
        :param theta: Góc xoay (radian).
        :return: Ma trận xoay 3x3.
        """
        if i == j or not (0 <= i <= 2 and 0 <= j <= 2):
            raise ValueError("Chỉ số trục phải khác nhau và nằm trong khoảng [0, 2].")
        if i > j:
            i, j = j, i

        G = np.eye(3)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = s
        G[j, i] = -s
        return G

    def rotate_pixels(self, pixels, alpha, beta, gamma):
        """
        Xoay ảnh bằng ma trận Givens lần lượt theo các trục x, y, z.

        :param pixels: Tọa độ điểm ảnh gốc (N x 3).
        :param alpha: Góc xoay quanh trục x (radian).
        :param beta: Góc xoay quanh trục y (radian).
        :param gamma: Góc xoay quanh trục z (radian).
        :return: Tọa độ đã xoay (N x 3).
        """
        pixels_centered = pixels - np.mean(pixels, axis=0)
        Rx = self.givens_rotation_matrix(0, 2, alpha)
        Ry = self.givens_rotation_matrix(1, 2, beta)
        Rz = self.givens_rotation_matrix(0, 1, gamma)
        return pixels_centered @ Rx @ Ry @ Rz

    def setup_projection(self, max_angle_deg):
        """
        Khởi tạo tham số camera để chiếu phối cảnh.

        :param max_angle_deg: Góc lớn nhất trong các góc xoay (độ).
        """
        max_dim = max(self.height, self.width)
        angle_scale = 1 + max_angle_deg / 90
        self.focal_length = max_dim * 1.2 * angle_scale
        cx, cy = self.height / 2, self.width / 2
        self.camera_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    def project_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        tvec = np.array([0, 0, self.focal_length * 1.5], dtype=np.float32)
        points_cam = points_3d.T + tvec.reshape(3, 1)

        # Ngăn z gần 0 gây vỡ ảnh
        z = np.clip(points_cam[2], a_min=1e-2, a_max=None)

        x = points_cam[0] / z
        y = points_cam[1] / z
        fx = self.camera_matrix[0, 0]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        u = fx * x + cx
        v = fx * y + cy

        return np.stack([u, v], axis=1)


    def rotate(self, alpha=0, beta=0, gamma=0) -> np.ndarray:
        """
        Xoay ảnh quanh các trục x, y, z và chiếu ảnh xuống mặt phẳng 2D.

        :param alpha: Góc xoay trục x (độ).
        :param beta: Góc xoay trục y (độ).
        :param gamma: Góc xoay trục z (độ).
        :return: Ảnh kết quả sau khi xoay và chiếu.
        """
        # Chuyển sang radian
        a_rad = np.deg2rad(alpha)
        b_rad = np.deg2rad(beta)
        g_rad = np.deg2rad(gamma)

        # Xoay
        rotated_pixels = self.rotate_pixels(self.pixel_coords.copy(), a_rad, b_rad, g_rad)

        # Chiếu
        self.setup_projection(max(abs(alpha), abs(beta), abs(gamma)))
        projected_2d = self.project_to_2d(rotated_pixels).astype(int)
        projected_2d -= np.min(projected_2d, axis=0)

        # Tạo ảnh đầu ra trắng
        h_out, w_out = projected_2d[:, 0].max() + 1, projected_2d[:, 1].max() + 1
        output = np.full((h_out, w_out, self.channels), 255, dtype=self.image.dtype)

        # Gán giá trị pixel
        output = assign_pixels_vectorized(self.pixel_coords, projected_2d, self.image, output)
        return output

def assign_pixels_vectorized(original_pixels, projected_pixels, source_image, output_image):
    h_out, w_out = output_image.shape[:2]

    x0 = original_pixels[:, 0]
    y0 = original_pixels[:, 1]
    x1 = projected_pixels[:, 0]
    y1 = projected_pixels[:, 1]

    # Loại bỏ các điểm ra ngoài khung
    valid = (x1 >= 0) & (x1 < h_out) & (y1 >= 0) & (y1 < w_out)

    x0 = x0[valid]
    y0 = y0[valid]
    x1 = x1[valid]
    y1 = y1[valid]

    if source_image.ndim == 3:
        output_image[x1, y1] = source_image[x0, y0]
    else:
        output_image[x1, y1] = source_image[x0, y0]

    return output_image
    
@st.cache_data(ttl=300)
def plot_image(image, rotated_image):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image) 
    ax[0].set_title("Ảnh gốc")
    ax[0].axis("off")

    ax[1].imshow(rotated_image)
    ax[1].set_title("Ảnh đã xử lý")
    ax[1].axis("off")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------- Giao diện Streamlit ---------------------
st.set_page_config(page_title="Xoay ảnh 2D & 3D", layout="wide", initial_sidebar_state="expanded")
st.title("🎨 Ứng dụng Xoay ảnh")

sidebar = st.sidebar
sidebar.subheader("🔧 Tùy chọn")

if "clear_cache" not in st.session_state:
    st.session_state.clear_cache = False

if sidebar.button("🧹 Xóa cache"):
    st.cache_data.clear()
    st.session_state.clear_cache = True
sidebar.caption("🛈 Nhấn để xóa cache (bộ nhớ tạm) nếu app bị crash hoặc chậm.")

if st.session_state.clear_cache:
    st.session_state.clear_cache = False
    st.rerun()

che_do = sidebar.radio("Chế độ xoay", ["2D", "3D"])

if che_do == "2D":
    goc = sidebar.slider("Góc xoay (độ)", -180, 180, 0)
else:
    alpha = sidebar.slider("Alpha (X - pitch, °)", -180, 180, 0)
    theta = sidebar.slider("Theta (Y - yaw, °)", -180, 180, 0)
    gamma = sidebar.slider("Gamma (Z - roll, °)", -180, 180, 0)

uploaded = sidebar.file_uploader("📁 Tải ảnh lên", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])

col1, col2 = st.columns(2)
if uploaded:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    if img is None:
        st.error("❌ File lỗi hoặc không hỗ trợ.")
    else:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if che_do == "2D":
            image_2d = ImageRotation2D(img)
            out = image_2d.rotate(goc)
        else:
            image_3d = ImageRotation3D(img)
            out = image_3d.rotate(alpha, theta, gamma)

        buf = plot_image(img, out)
        buf.seek(0)
        st.image(buf, use_container_width=True)

        # ======= Tải ảnh đã xử lý =======
        pil_img = Image.fromarray(out.astype(np.uint8))
        img_buf = BytesIO()
        pil_img.save(img_buf, format="PNG")
        byte_im = img_buf.getvalue()

        st.download_button(
            label="📥 Tải ảnh đã xử lý",
            data=byte_im,
            file_name="rotated_image.png",
            mime="image/png"
        )
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")

st.markdown("---")
st.markdown("*Hướng dẫn sử dụng:*")
st.markdown("- *2D*: Xoay ảnh theo góc đơn giản")
st.markdown("- *3D*: Xoay ảnh theo 3 trục (X, Y, Z)")
st.markdown("- *Alpha (X)*: Xoay lên/xuống (pitch)")
st.markdown("- *Theta (Y)*: Xoay trái/phải (yaw)")
st.markdown("- *Gamma (Z)*: Xoay nghiêng (roll)")
