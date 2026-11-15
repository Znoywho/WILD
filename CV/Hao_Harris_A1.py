import cv2 as cv
import numpy as np



def guassian_blur(img, sigma):
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
    )
    
    sobely = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]]
    )

    filtered_image = guassian_blur(img, sigma= sigma_d)
    
    L_x = cv.filter2D(filtered_image, -1, sobelx)
    L_y = cv.filter2D(filtered_image, -1, sobely)
    L_xL_y = L_x * L_y

    # print(L_x)

    Mxx = guassian_blur(L_x * L_x, sigma_i)
    Myy = guassian_blur(L_y * L_y, sigma_i)
    Mxy = guassian_blur(L_xL_y, sigma_i)

    return Mxx, Mxy, Myy

def harris_response(img, sigma_i, sigma_d, k=0.04):
    """Đáp ứng R theo công thức (2). -> mảng float cùng kích thước ảnh."""
    Mxx, Mxy, Myy = structure_tensor(img, sigma_i, sigma_d)
    # print(Mxx, Mxy, Myy)
    # print(Mxx)

    det = (Mxx * Myy) - Mxy ** 2
    trace = Mxx + Myy
    # Không hiểu sao ra cái công thức này cần tìm hieru
    R = det - k * (trace * trace)
    # print(R)
    return R

def nms_2d(response, window=3):
    """Cực đại địa phương trong cửa sổ window x window. -> mảng bool."""
    H, W = response.shape
    r = window // 2
    out = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            patch = response[max(0, i-r):i+r+1, max(0, j-r):j+r+1]
            out[i, j] = response[i, j] == patch.max()
    return out

def harris_classic(img, sigma_i=2.0, sigma_d=1.4, k=0.04, thresh_rel=0.01):
    """Harris một tỉ lệ: R -> nms_2d -> ngưỡng tương đối theo max(R).
    -> mảng (N, 3): (row, col, response), sắp xếp giảm dần theo response.
    """

    R = harris_response(img, sigma_i= sigma_i, sigma_d = sigma_d, k = k)
    nms = nms_2d(R)
    nms = nms & (R > thresh_rel * R.max())
    return nms

def scale_pyramid(sigma0=1.2, xi=1.2, n_levels=10):
    """Danh sách sigma_n = xi^n * sigma0. -> mảng 1D."""
    pyramid = np.zeros(n_levels + 1)
    for i in range(n_levels + 1):
        pyramid[i] = sigma0 * xi ** i
    return pyramid

def log_normalized(img, sigma):
    """Laplacian chuẩn hoá theo tỉ lệ: sigma^2 * |Lxx + Lyy|."""
    filtered_image = guassian_blur(img, sigma= sigma)
    sobel_xx = np.array([
        [1, 0, -2, 0, 1],
        [4, 0, -8, 0, 4],
        [6, 0, -12, 0, 6],
        [4, 0, -8, 0, 4],
        [1, 0, -2, 0, 1],
    ])
    sobel_yy = sobel_xx.T
    L_xx = cv.filter2D(filtered_image, -1, sobel_xx)
    L_yy = cv.filter2D(filtered_image, -1, sobel_yy)

    norm = (sigma ** 2) * np.abs(L_xx * L_xx + L_yy + L_yy)

    return norm
    
def harris_laplace(img, sigmas, k=0.04, thresh_rel=0.01, log_thresh=0.0):
    """Harris-Laplace (Mikolajczyk & Schmid 2004).
    Bước 1: mỗi sigma -> harris_response -> nms_2d -> ngưỡng.
    Bước 2: giữ điểm chỉ khi |LoG| là cực đại địa phương theo trục tỉ lệ.
    -> mảng (N, 4): (row, col, sigma, response).
    """

    classic_log_save = []
    
    for sigma in sigmas:
        sigma_i = sigma
        sigma_d = 0.7 * sigma
        
        classic = harris_classic(img, sigma_d = sigma_d, sigma_i = sigma_i)
        log = log_normalized(img, sigma_d)
        classic_log_save.append((classic, log))
        
    actual_result = []

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            maximum = classic_log_save[0][1][x, y] # Take LoG at initial Scale
            optimal_sigma = sigmas[0]
            is_corner = False
            for i in range(1, len(sigmas)):
                if classic_log_save[i][0][x, y]:
                    is_corner = True
                if maximum <  classic_log_save[i][1][x, y]:
                    maximum = classic_log_save[i][1][x, y]
            if maximum > log_thresh and is_corner:
                actual_result.append((x, y, sigmas[i], classic_log_save[i][0][x, y]))

    return actual_result