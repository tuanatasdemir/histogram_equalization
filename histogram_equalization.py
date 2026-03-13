import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def HistogramEqual():
    imgPath = 'ornek.jpg'
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    cdf = hist.cumsum()
    if cdf.max() > 0:
        cdfNorm = cdf * float(hist.max()) / cdf.max()
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.title("Orijinal Görsel")
    plt.imshow(img, cmap='gray')

    plt.subplot(223)
    plt.title("Orijinal Histogram")
    plt.plot(hist)
    plt.plot(cdfNorm, color='r', linestyle='--')

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    equImg = cdf_final[img]
    equhist = cv.calcHist([equImg], [0], None, [256], [0,256])
    equcdf = equhist.cumsum()
    if equcdf.max() > 0:
        equcdfNorm = equcdf * float(equhist.max()) / equcdf.max()
    plt.subplot(222)
    plt.title("Düzeltilmiş Görsel")
    plt.imshow(equImg, cmap='gray')

    plt.subplot(224)
    plt.title("Düzeltilmiş Histogram")
    plt.plot(equhist)
    plt.plot(equcdfNorm, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    HistogramEqual()
