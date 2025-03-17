import numpy as np
import cv2

# 加载原始图片
image_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v1\Celeb-real\frames\id0_0008\074.png'
image = cv2.imread(image_path)

# 加载关键点数据
landmarks_path = r'E:\github_code\Unnamed1\dataset\processed\Celeb-DF-v1\Celeb-real\landmarks\id0_0008\074.npy'
landmarks = np.load(landmarks_path)
print(landmarks)
# 遍历关键点并绘制到图片上
for landmark in landmarks:
    x, y = landmark
    cv2.circle(image, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)  # 绿色圆点

# # 保存结果
# output_path = 'path/to/save/output_image.jpg'
# cv2.imwrite(output_path, image)

# 或者直接显示图片
cv2.imshow('Image with Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()