import cv2
import pytesseract
from gtts import gTTS
import os

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('test.png')

# scale_percent = 40 # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
adaptive_th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
##print(pytesseract.image_to_string(img))

# ## Detecting Characters
hImg, wImg, _ = img.shape
text = pytesseract.image_to_string(adaptive_th)
print(text)
myobj = gTTS(text = text, lang = 'en', slow=False)
myobj.save('test.mp3')
os.system('test.mp3')
# print(text)
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    #print(b)
    b = b.split(' ')
    #print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, hImg-y), (w, hImg-h), (0, 0, 255), 3)
    cv2.putText(img, b[0], (x, hImg-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

## Detecting Words
# hImg, wImg, _ = img.shape
# # cong = r'--oem 3 --psm 6 outputbase digits'
# boxes = pytesseract.image_to_data(img, config=cong)
# for x, b in enumerate(boxes.splitlines()): #add counter
#     if x!=0:
#         b = b.split()
#         print(b)
#         if len(b)==12:
#             x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#             cv2.rectangle(img, (x, y), (w+x, h+y), (0, 0, 255), 3)
#             cv2.putText(img, b[11], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
winname = "Test"
cv2.namedWindow(winname)        # Create a named window
cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
cv2.imshow(winname, adaptive_th)
cv2.waitKey(0)