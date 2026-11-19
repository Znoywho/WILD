import numpy as np
import cv2 as cv

# OpenCV đọc ảnh theo thứ tự B, G, R
LUMINANCE_WEIGHTS_BGR = np.array([0.114, 0.587, 0.299])

def convert2grey(img):
    """Ảnh BGR -> xám. Công thức: Y = 0.299 R + 0.587 G + 0.114 B."""
    return np.round(img[..., :3] @ LUMINANCE_WEIGHTS_BGR)

def gaussian_blur(img, sigma):
    from scipy.ndimage import gaussian_filter
    filtered_image = gaussian_filter(img, sigma = sigma)
    # filtered_image = cv.filter2D(img, -1, guassian_kernel)

    return filtered_image

def structure_tensor(img, sigma_i, sigma_d):
    """Ma trận mô-men bậc hai theo công thức (1), đã nhân sigma_d**2.
    sigma_d: độ mờ trước khi lấy đạo hàm.
    sigma_i: bán kính cửa sổ lấy trung bình các tích đạo hàm.
    -> (Mxx, Mxy, Myy), mỗi phần tử cùng kích thước ảnh.
    """

    sobelx = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
    ) * 1 / 8
    
    sobely = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]]
    ) * 1 / 8

    filtered_image = gaussian_blur(img, sigma= sigma_d)
    
    L_x = cv.filter2D(filtered_image, -1, sobelx)
    L_y = cv.filter2D(filtered_image, -1, sobely)
    L_xL_y = L_x * L_y

    # print(L_x)

    Mxx = (sigma_d**2)*gaussian_blur(L_x * L_x, sigma_i)
    Myy = (sigma_d**2)*gaussian_blur(L_y * L_y, sigma_i)
    Mxy = (sigma_d**2)*gaussian_blur(L_xL_y, sigma_i)

    return Mxx, Mxy, Myy


def harris_response(img, sigma_i, sigma_d, k=0.04):
    """Đáp ứng R theo công thức (2). -> mảng float cùng kích thước ảnh."""
    Mxx, Mxy, Myy = structure_tensor(img, sigma_i, sigma_d)
    det = (Mxx * Myy) - Mxy ** 2
    trace = Mxx + Myy
    R = det - k * (trace * trace)
    return R

def nms_2d(response, window=3):
    """Cực đại địa phương trong cửa sổ window x window. -> mảng bool."""
    H, W = response.shape
    r = window // 2
    out = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            patch = response[max(0, i-r):i+r+1, max(0, j-r):j+r+1]
            out_window = out[max(0, i-r):i+r+1, max(0, j-r):j+r+1]
            out[i, j] = response[i, j] == patch.max() and not out_window.any()
    return out

def drop_border_points(mask, border):
    """Bỏ mọi điểm trong dải `border` px sát mép ảnh. -> mảng bool mới.
    Lọc Gauss σ với tới ~3σ; trong dải đó ảnh bị phản chiếu ở mép nên sinh góc giả.
    """
    inside = np.zeros_like(mask)
    inside[border:mask.shape[0] - border, border:mask.shape[1] - border] = True
    return mask & inside

def harris_classic(img, sigma_i=2.0, sigma_d=1.4, k=0.04, thresh_rel=0.01):
    """Harris một tỉ lệ: R -> nms_2d -> ngưỡng tương đối theo max(R).
    -> mảng (N, 3): (row, col, response), sắp xếp giảm dần theo response.
    """

    R = harris_response(img, sigma_i= sigma_i, sigma_d = sigma_d, k = k)
    nms = nms_2d(R)
    nms = nms & (R > thresh_rel * R.max())
    # Đo trên 6 ảnh: điểm cách mép < 6 px giả 28-64%, xa hơn chỉ ~7% -> dải viền = ceil(3σ_I) = 6 px
    nms = drop_border_points(nms, int(np.ceil(3 * sigma_i)))

    rows, cols = np.nonzero(nms)
    result = np.column_stack((rows, cols, R[rows, cols]))
    index_of_order = np.argsort(result[:, 2])[::-1] # Take the third argument and reverse that, this function will return the index

    return result[index_of_order] 

def scale_pyramid(sigma0=1.2, xi=1.2, n_levels=10):
    """Danh sách sigma_n = xi^n * sigma0. -> mảng 1D."""
    pyramid = np.zeros(n_levels + 1)
    for i in range(n_levels + 1):
        pyramid[i] = sigma0 * xi ** i
    return pyramid
def log_normalized(img, sigma):
    """Laplacian chuẩn hoá theo tỉ lệ: sigma^2 * |Lxx + Lyy|."""
    filtered_image = gaussian_blur(img, sigma=sigma)
    sobel_xx = np.array([
        [1, 0, -2, 0, 1],
        [4, 0, -8, 0, 4],
        [6, 0, -12, 0, 6],
        [4, 0, -8, 0, 4],
        [1, 0, -2, 0, 1],
    ]) / 64
    sobel_yy = sobel_xx.T
    L_xx = cv.filter2D(filtered_image, -1, sobel_xx)
    L_yy = cv.filter2D(filtered_image, -1, sobel_yy)
    # Laplacian = Lxx + Lyy, not Lxx*Lxx
    norm = (sigma ** 2) * np.abs(L_xx + L_yy)
    return norm

def harris_laplace(img, sigmas, k=0.04, thresh_rel=0.01, log_thresh=0.0):
    """Harris-Laplace (Mikolajczyk & Schmid 2004).
    Bước 1: mỗi sigma -> harris_response -> nms_2d -> ngưỡng.
    Bước 2: giữ điểm chỉ khi |LoG| là cực đại địa phương theo trục tỉ lệ.
    -> mảng (N, 4): (row, col, sigma, response).
    """
    corner_masks = []
    log_maps = []
    response_maps = []

    for sigma in sigmas:
        sigma_i = sigma
        sigma_d = 0.7 * sigma
        R = harris_response(img, sigma_i, sigma_d, k)
        nms = nms_2d(R)
        mask = nms & (R > thresh_rel * R.max())
        corner_masks.append(mask)
        response_maps.append(R)
        log_maps.append(log_normalized(img, sigma))

    log_stack = np.stack(log_maps, axis=0)

    # Explore entire picture of pixel 
    # That can find maximum of LoG in each position
    # Like when you stack the layers in each image with different sigma in LoG

    result = []
    
    for s in range(1, len(sigmas) - 1):
        ys, xs = np.nonzero(corner_masks[s])
        for y, x in zip(ys, xs):
            log_val = log_stack[s, y, x]
            is_max = log_val > log_stack[s - 1, y, x] and log_val > log_stack[s + 1, y, x]
            if is_max and log_val > log_thresh:
                result.append((y, x, sigmas[s], response_maps[s][y, x]))

    return np.array(result).reshape(-1, 4)

def is_inside_of_image(points, img, border=0):
    """Điểm (x, y) có nằm trong ảnh, cách mép ít nhất `border` px không. -> mảng bool."""
    H, W = img[:2]

    x, y = points[:, 0], points[:, 1]

    return (((x >= border) & (x < W - border)) & ((y >= border) & (y < H - border)))

def rotate_matrix(center, angle_deg, scale=1.0):
    cx, cy = center
    theta = np.deg2rad(angle_deg) 
    alpha, beta = scale * np.cos(theta), scale * np.sin(theta)
    return np.array([
        [ alpha, beta, (1 - alpha) * cx - beta * cy],
        [-beta, alpha, beta * cx + (1 - alpha) * cy],
    ])

def repeatability(pts_a, pts_b, transform, img_shape, eps=3.0, border=6):
    """Tỉ lệ điểm của ảnh A tìm lại được trong ảnh B sau phép biến đổi.
    transform: hàm (row, col) của A -> (row, col) tương ứng ở B.
    Chỉ tính trên các điểm mà ảnh chiếu còn nằm trong B.
    -> số thực trong [0, 1].
    Nhận thêm ma trận affine 2x3 theo (x, y), như cv.getRotationMatrix2D trả về.
    border: bỏ điểm chiếu vào dải mép này của B, vì harris_classic không phát hiện ở đó
            (mặc định 6 = ceil(3σ_I) với σ_I = 2). Đặt 0 để tính đúng như đề gốc.
    """
    if callable(transform):          # đúng mô tả của đề: transform(row, col) -> (row, col)
        moved_rc = np.array([transform(r, c) for r, c in pts_a[:, :2]]).reshape(-1, 2)
        transformed_pts_a = moved_rc[:, [1, 0]]          # (row, col) -> (x, y)
    else:
        P = np.column_stack((pts_a[:, 1], pts_a[:, 0], np.ones(len(pts_a))))
        transformed_pts_a = P @ transform.T

    isInside_a = is_inside_of_image(transformed_pts_a, img_shape, border)
    transformed_pts_a = transformed_pts_a[isInside_a]
    if len(transformed_pts_a) == 0 or len(pts_b) == 0:
        return 0.0 

    pts_b_xy = pts_b[:, [1, 0]]
    dists = np.linalg.norm(transformed_pts_a[:, None, :] - pts_b_xy[None, :, :], axis=2).min(axis=1)
    return np.sum(dists <= eps) / len(transformed_pts_a)