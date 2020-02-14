
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
import pytesseract
from PIL import Image
from scipy.ndimage import zoom
from scipy import ndimage, misc


def prespective_change(img, src, dst):
    h, w = img.shape[:2]
   
    M = cv2.getPerspectiveTransform(src, dst)
    
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    plt.imshow(cv2.flip(warped, 1))
    print(warped.shape)
    
    return cv2.flip(warped, 1)

def zoom_image(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    if zoom_factor < 1:

        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    elif zoom_factor > 1:
      
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    else:
        out = img
    return out

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


img = cv2.imread("G:/3d_rotate/images/ad.jpg",0)

(thresh, im_bw) = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# image =cv2.subtract(255, im_bw) 

# w, h = im.shape[0], im.shape[1]

src = np.float32([(100,     10),
                  (540,  130),
                  (20,    520),
                  (600,  550)])

dst = np.float32([(700, 0),
                  (10, 0),
                  (500, 500),
                  (10, 500)])



a =prespective_change(im, src, dst)

nwi=rotateImage(img,-2)
zm1 =zoom_image(nwi,1.1)


can = cv2.Canny(zm1,200,300)

plt.figure(figsize=(10,10))
plt.imshow(nwi,cmap="gray")

plt.figure(figsize=(10,10))
plt.imshow(zm1,cmap="gray")


plt.figure(figsize=(10,10))
plt.imshow(can,cmap="gray")

pytesseract.pytesseract.tesseract_cmd = "C:/Users/Sanaullah/Anaconda3/envs/tensorflow/Library/bin/tesseract.exe"
text = pytesseract.image_to_string(zm1,lang='ben')
print (text)


