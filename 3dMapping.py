import cv2
import matplotlib.pyplot as plt
import numpy as np

SEARCH_PIXEL_SIZE = 24 #정사각형 검색 픽셀 크기 #적합한 픽셀 크기: 1,2,3,4,6,8,12,24,43,86,129,172,258,344,516,1032
SEARCH_PIXEL_LENGTH = 1000 #검색 픽셀 길이
LEFT_IMAGE_SRC = "D:\\sion\\Big_Project\\3d_Mapping\\sample_left_image.jpg" #왼쪽 사진 경로
RIGHT_IMAGE_SRC = "D:\\sion\\Big_Project\\3d_Mapping\\sample_right_image.jpg" #오른쪽 사진 경로

def show_original_images():
    fig = plt.figure()
    rows = 1
    cols = 2

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('left image')
    #ax1.axis("off")
    
    #검색 박스 표시
    for i in range(0, original_image_info[0]+1, SEARCH_PIXEL_SIZE):
        ax1.axhline(y=i, color='r', linewidth=0.1)
    for i in range(0, original_image_info[1]+1, SEARCH_PIXEL_SIZE):
        ax1.axvline(x=i, color='r', linewidth=0.1)

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
    ax2.set_title('right image')
    #ax2.axis("off")

    #검색 박스 표시
    for i in range(0, original_image_info[0]+1, SEARCH_PIXEL_SIZE):
        ax2.axhline(y=i, color='r', linewidth=0.1)
    for i in range(0, original_image_info[1]+1, SEARCH_PIXEL_SIZE):
        ax2.axvline(x=i, color='r', linewidth=0.1)

    plt.show()

def stereo_matching():
    depth_array_info = (original_image_info[0] // SEARCH_PIXEL_SIZE, original_image_info[1] // SEARCH_PIXEL_SIZE, 1)
    depth_array = [[0 for j in range(depth_array_info[1])] for i in range(depth_array_info[0])]
    max_dis_match_val = 3 * 255 * SEARCH_PIXEL_SIZE * SEARCH_PIXEL_SIZE #사진간의 일치율을 계산할때 분모로 사용하는 변수

    for v in range(depth_array_info[0]):
        for h in range(depth_array_info[1]):
            s_h = h * SEARCH_PIXEL_SIZE
            s_v = v * SEARCH_PIXEL_SIZE
            m_h = s_h
            min_dis_match_val = max_dis_match_val
            match_length = 0

            for i in range(SEARCH_PIXEL_LENGTH):
                dis_match_val = np.sum(np.abs(left_image[s_v : s_v+SEARCH_PIXEL_SIZE, s_h : s_h+SEARCH_PIXEL_SIZE] - right_image[s_v : s_v+SEARCH_PIXEL_SIZE, m_h : m_h+SEARCH_PIXEL_SIZE]))
                if dis_match_val < min_dis_match_val:
                    min_dis_match_val = dis_match_val
                    match_length = i
                if m_h-1 < 0:
                    break
                else:
                    m_h -= 1
            
            depth_array[v][h] = match_length
    
    return np.array(depth_array)

def show_depth_image(depth_array):
    cmap = plt.get_cmap('Greys_r')

    plt.matshow(depth_array, cmap=cmap)
    plt.colorbar(shrink=0.8, aspect=10)

    plt.show()

def normalize_depth_array(depth_array, p):
    v, h = len(depth_array), len(depth_array[0])

    for i in range(v-p+1):
        for j in range(h-p+1):
            depth_array[i:i+p, j:j+p] = np.mean(depth_array[i:i+p, j:j+p])

    return depth_array

left_image = cv2.imread(LEFT_IMAGE_SRC, cv2.IMREAD_ANYCOLOR)
right_image = cv2.imread(RIGHT_IMAGE_SRC, cv2.IMREAD_ANYCOLOR)
original_image_info = left_image.shape

print("-----", "original image info", "-----")
print(original_image_info)

show_original_images()
show_depth_image(normalize_depth_array(stereo_matching(), 3))

cv2.waitKey()
cv2.destroyAllWindows()