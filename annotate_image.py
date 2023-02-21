import os

import cv2

img_paths = ['scenes/scene_0/a/blue_chair_0005.png',
             'scenes/scene_0/a/blue_chair_0105.png',
             'scenes/scene_0/b/red_chair_0023.png',
             'scenes/scene_0/b/red_chair_0117.png']
pts_lst = [[[367, 259], [429, 270]],
           [[323, 185], [432, 157]],
           [[562, 219], [608, 239]],
           [[579, 146], [664, 130]]]
# img_paths = ['scenes/scene_1/a/rgb_1.png',
#              'scenes/scene_1/a/rgb_2.png',
#              'scenes/scene_1/b/rgb_1.png',
#              'scenes/scene_1/b/rgb_2.png']
# pts_lst = [[[230, 145], [257, 120]],
#            [[433, 247], [460, 280]],
#            [[188, 198], [194, 160]],
#            [[337, 266], [355, 328]]]
for img_path, pts in zip(img_paths, pts_lst):
    img_dir = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    # cv2.imshow('', img)
    # cv2.waitKey(0)
    for pt in pts:
        cv2.circle(img, pt, 1, (0, 255, 0), thickness=-1)
    cv2.imwrite(os.path.join(img_dir, 'annotated_' + img_name), img)
