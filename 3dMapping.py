import cv2
import matplotlib.pyplot as plt

SEARCH_PIXEL_SIZE = 24 #정사각형 검색 픽셀 크기 #적합한 픽셀 크기: 1,2,3,4,6,8,12,24,43,86,129,172,258,344,516,1032
LEFT_IMAGE_SRC = "D:\\sion\\Big_Project\\3d_Mapping\\sample_left_image.jpg" #왼쪽 사진 경로
RIGHT_IMAGE_SRC = "D:\\sion\\Big_Project\\3d_Mapping\\sample_right_image.jpg" #오른쪽 사진 경로

def show_original_images():
    fig = plt.figure()
    rows = 1
    cols = 2

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    ax1.set_title('left image')
    ax1.axis("off")

    for i in range(0, original_image_info[0], SEARCH_PIXEL_SIZE):
        ax1.axhline(y=i, color='r', linewidth=1)

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
    ax2.set_title('right image')
    ax2.axis("off")

    plt.show()

left_image = cv2.imread(LEFT_IMAGE_SRC, cv2.IMREAD_ANYCOLOR)
right_image = cv2.imread(RIGHT_IMAGE_SRC, cv2.IMREAD_ANYCOLOR)
original_image_info = left_image.shape

print("-----", "original image info", "-----")
print(original_image_info)

show_original_images()

cv2.waitKey()
cv2.destroyAllWindows()