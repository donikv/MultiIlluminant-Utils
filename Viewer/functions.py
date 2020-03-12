import wx
import cv2
import numpy as np

def TemplateFunction(bitmap):
    image_size = bitmap.GetSize()
    image = bitmap.ConvertToImage()
    buf = image.GetDataBuffer()
    array = np.frombuffer(buf, dtype='uint8')
    reshape = np.reshape(array, (image_size[1], image_size[0], 3))
    
    #Add opencv here, image must be rgb at the end 

    fin = np.reshape(img_rgb, (image_size[0]* image_size[1]* 3))
    image.SetData(fin.tostring())
    return wx.Bitmap(image)


def Histogram(bitmap):
    image_size = bitmap.GetSize()
    image = bitmap.ConvertToImage()
    buf = image.GetDataBuffer()
    array = np.frombuffer(buf, dtype='uint8')
    reshape = np.reshape(array, (image_size[1], image_size[0], 3))
    img_yuv = cv2.cvtColor(reshape, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    fin = np.reshape(img_rgb, (image_size[0]* image_size[1]* 3))
    image.SetData(fin.tostring())
    return wx.Bitmap(image)


def GaussianBlur(bitmap):
    image_size = bitmap.GetSize()
    image = bitmap.ConvertToImage()
    buf = image.GetDataBuffer()
    array = np.frombuffer(buf, dtype='uint8')
    reshape = np.reshape(array, (image_size[1], image_size[0], 3))
    blur = cv2.GaussianBlur(reshape, (5, 5), 0)
    fin = np.reshape(blur, (image_size[0]* image_size[1]* 3))
    image.SetData(fin.tostring())
    return wx.Bitmap(image)


def Canny(bitmap):
    image_size = bitmap.GetSize()
    image = bitmap.ConvertToImage()
    buf = image.GetDataBuffer()
    array = np.frombuffer(buf, dtype='uint8')
    reshape = np.reshape(array, (image_size[1], image_size[0], 3))
    img_gray = cv2.cvtColor(reshape, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(img_gray,100,250)
    img_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    fin = np.reshape(img_rgb, (image_size[0]* image_size[1]* 3))
    image.SetData(fin.tostring())
    return wx.Bitmap(image)