from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.io as sio
import math
import cv2
import torch
from scipy.fftpack import fft
from PIL import Image


# Bounding box perturbation coordinates
def F_9anchors(x, y, w, h):
    Anchors9ROI = np.zeros((10, 4))

    Anchors9ROI[1, :] = [x, y, w, h]
    Anchors9ROI[2, :] = [x, y - 7, w, h]
    Anchors9ROI[3, :] = [x + 7, y, w, h]
    Anchors9ROI[4, :] = [x, y + 7, w, h]
    Anchors9ROI[5, :] = [x - 7, y, w, h]
    Anchors9ROI[6, :] = [x + 7, y - 7, w, h]
    Anchors9ROI[7, :] = [x + 7, y + 7, w, h]
    Anchors9ROI[8, :] = [x - 7, y + 7, w, h]
    Anchors9ROI[9, :] = [x - 7, y - 7, w, h]

    return Anchors9ROI

# multisacle ROI coordinates
def F_multisacle(x0, y0, w0, h0, k, ImageW, ImageH):
    k_2 = k ** (-2)
    k_1 = k ** (-1)
    k1 = k ** 1
    k2 = k ** 2

    w_2 = w0 * k_2  # Level - 2
    h_2 = h0 * k_2
    x_2 = x0 + w0 / 2 - w_2 / 2
    y_2 = y0 + h0 / 2 - h_2 / 2

    if x_2 < 1:
        x_2 = 1
    if y_2 < 1:
        y_2 = 1
    if x_2 + w_2 > ImageW:
        w_2 = ImageW - x_2
    if y_2 + h_2 > ImageH:
        h_2 = ImageH - y_2

    x_2 = math.floor(x_2)
    y_2 = math.floor(y_2)
    w_2 = math.floor(w_2)
    h_2 = math.floor(h_2)

    w_1 = w0 * k_1  # Level - 1
    h_1 = h0 * k_1
    x_1 = x0 + w0 / 2 - w_1 / 2
    y_1 = y0 + h0 / 2 - h_1 / 2

    if x_1 < 1:
        x_1 = 1
    if y_1 < 1:
        y_1 = 1
    if x_1 + w_1 > ImageW:
        w_1 = ImageW - x_1
    if y_1 + h_1 > ImageH:
        h_1 = ImageH - y_1

    x_1 = math.floor(x_1)
    y_1 = math.floor(y_1)
    w_1 = math.floor(w_1)
    h_1 = math.floor(h_1)

    x0 = math.floor(x0)
    y0 = math.floor(y0)
    w0 = math.floor(w0)
    h0 = math.floor(h0)

    w1 = w0 * k1  # Level 1
    h1 = h0 * k1
    x1 = x0 + w0 / 2 - w1 / 2
    y1 = y0 + h0 / 2 - h1 / 2
    x1 = math.floor(x1)
    y1 = math.floor(y1)
    w1 = math.floor(w1)
    h1 = math.floor(h1)

    w2 = w0 * k2  # Level 2
    h2 = h0 * k2
    x2 = x0 + w0 / 2 - w2 / 2
    y2 = y0 + h0 / 2 - h2 / 2

    x2 = math.floor(x2)
    y2 = math.floor(y2)
    w2 = math.floor(w2)
    h2 = math.floor(h2)

    ROI_mutiscale = [[x_2, y_2, w_2, h_2],
                     [x_1, y_1, w_1, h_1],
                     [x0, y0, w0, h0],
                     [x1, y1, w1, h1],
                     [x2, y2, w2, h2]]

    return ROI_mutiscale


def get_SNR(p, fps):  # define SNR

    pulseEst = p - np.mean(p)
    N = 512
    Y_pulse = np.fft.fft(pulseEst, n=N, axis=0)
    powsPulse = abs(Y_pulse)
    powsPulse = powsPulse[range(int(N / 2))]
    freqPulse = np.arange(int(N / 2)) * fps / N
    freqRange = (freqPulse >= 0.8) & (freqPulse <= 3.5)  # signal rang e0.8 - 3.5Hz

    freqRangeComp = ((freqPulse > 0) & (freqPulse < 0.8)) | (
            (freqPulse > 3.5) & (freqPulse < 15))  # noise range 0-0.8 Hz, 3.5-15Hz

    snr = 10 * np.log10(sum(np.square(powsPulse[freqRange]))
                        / sum(np.square(powsPulse[freqRangeComp])))  # get defined snr

    return snr

if __name__ == '__main__':

    addpath = 'G:/ZMH/Mul/utils'
    nSub = 10
    nVersion = 6
    PUREfps = 30
    winLength = 150
    stepSize = int(winLength / 2)
    hannW = np.hanning(winLength)

    modelName = 'PhysNet_3DCNNs'
    modelDir = 'G:/ZMH/Mul/weight_150/PhysNet_PURE_150_3_epoch03.pth'  # trained model
    # (dataset_test, winLength, RPN, modelName, kers, dataset_weight, winLength, RPN, i_epoch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    checkpoint = torch.load(modelDir)
    model = checkpoint['model']
    model = model.to(device)

    for iSub in range(1, nSub + 1):
        for iVersion in range(1, nVersion + 1):

            subID = str(iSub).zfill(2) + '-' + str(iVersion).zfill(2)
            vidDir = 'E:\\PURE\\Data\\' + subID
            ResultDir = 'G:\\HWR\\Multi-scale rPPG\\Result\\PURE\\' + subID
            NewResultDir = 'G:\\HWR\\ZMH\\Result\\PURE\\' + subID
            roi_File = ResultDir + '/roi_facedetector.mat'  # Original ROI coordinates tracked by KLT algorithm
            file2Save = NewResultDir + '/newMulRPN_0927_PURE_PhysNet3D.mat'

            if not os.path.exists(vidDir):
                print(subID + 'does not exist ')
                continue

            if not os.path.exists(ResultDir):
                os.makedirs(ResultDir)

            imageList = os.listdir(vidDir)
            roi = sio.loadmat(roi_File)
            nImages = len(imageList)
            Num_k = math.floor(nImages / stepSize)
            nImages = int(Num_k * stepSize)

            PulseEst_GS = np.zeros((1, nImages + 1))
            PulseEst_Uni = np.zeros((1, nImages + 1))
            PulseEst_25Max = np.zeros((1, nImages + 1))
            PulseEst_Stg2_W = np.zeros((1, nImages + 1))

            ROI25 = np.zeros((25 + 1, 4, 150 + 1), dtype=np.int32)
            videoRPN25_1 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_2 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_3 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_4 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_5 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_6 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_7 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_8 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_9 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_10 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_11 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_12 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_13 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_14 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_15 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_16 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_17 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_18 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_19 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_20 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_21 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_22 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_23 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_24 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN25_25 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)

            ROI9 = np.zeros((9 + 1, 4, 150 + 1), dtype=np.int32)
            videoRPN9_1 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN9_2 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN9_3 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN9_4 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN9_5 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN9_6 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN9_7 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN9_8 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN9_9 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)

            ROI5 = np.zeros((5 + 1, 4, 150 + 1), dtype=np.int32)
            videoRPN5_1 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN5_2 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN5_3 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN5_4 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)
            videoRPN5_5 = np.zeros((150 + 1, 178, 178, 3), dtype=np.float32)

            for n in range(winLength, nImages + 1, stepSize):

                videoRPN25_150_n = np.zeros((25 + 1, 150 + 1, 178, 178, 3), dtype=np.float32)
                for iImage in range(n - winLength + 1, n + 1):  # 1:150,  76:225,  151:300 ...
                    imageName = imageList[iImage - 1]
                    imagePath1 = vidDir + '/' + imageName
                    currImage = cv2.imread(imagePath1)

                    imageName_150 = iImage - n + winLength
                    vidHeight = np.size(currImage, 1)

                    x0 = roi['rect_klt'][iImage - 1, 0]
                    y0 = roi['rect_klt'][iImage - 1, 1]
                    w0 = roi['rect_klt'][iImage - 1, 2]
                    h0 = roi['rect_klt'][iImage - 1, 3]

                    # get first 25 ROI coordinates
                    w70 = math.ceil(0.70 * w0)
                    w75 = math.ceil(0.75 * w0)
                    w80 = math.ceil(0.80 * w0)
                    w85 = math.ceil(0.85 * w0)
                    w90 = math.ceil(0.90 * w0)

                    h70_11 = math.ceil(w70 * 1.1)
                    h70_12 = math.ceil(w70 * 1.2)
                    h70_13 = math.ceil(w70 * 1.3)
                    h70_14 = math.ceil(w70 * 1.4)
                    h70_15 = math.ceil(w70 * 1.5)
                    h75_11 = math.ceil(w75 * 1.1)
                    h75_12 = math.ceil(w75 * 1.2)
                    h75_13 = math.ceil(w75 * 1.3)
                    h75_14 = math.ceil(w75 * 1.4)
                    h75_15 = math.ceil(w75 * 1.5)
                    h80_11 = math.ceil(w80 * 1.1)
                    h80_12 = math.ceil(w80 * 1.2)
                    h80_13 = math.ceil(w80 * 1.3)
                    h80_14 = math.ceil(w80 * 1.4)
                    h80_15 = math.ceil(w80 * 1.5)
                    h85_11 = math.ceil(w85 * 1.1)
                    h85_12 = math.ceil(w85 * 1.2)
                    h85_13 = math.ceil(w85 * 1.3)
                    h85_14 = math.ceil(w85 * 1.4)
                    h85_15 = math.ceil(w85 * 1.5)
                    h90_11 = math.ceil(w90 * 1.1)
                    h90_12 = math.ceil(w90 * 1.2)
                    h90_13 = math.ceil(w90 * 1.3)
                    h90_14 = math.ceil(w90 * 1.4)
                    h90_15 = math.ceil(w90 * 1.5)

                    x70 = math.ceil(x0 + w0 / 2 - w70 / 2)
                    x75 = math.ceil(x0 + w0 / 2 - w75 / 2)
                    x80 = math.ceil(x0 + w0 / 2 - w80 / 2)
                    x85 = math.ceil(x0 + w0 / 2 - w85 / 2)
                    x90 = math.ceil(x0 + w0 / 2 - w90 / 2)

                    y70_11 = math.ceil(y0 + h0 / 2 - h70_11 / 2)
                    y70_12 = math.ceil(y0 + h0 / 2 - h70_12 / 2)
                    y70_13 = math.ceil(y0 + h0 / 2 - h70_13 / 2)
                    y70_14 = math.ceil(y0 + h0 / 2 - h70_14 / 2)
                    y70_15 = math.ceil(y0 + h0 / 2 - h70_15 / 2)
                    y75_11 = math.ceil(y0 + h0 / 2 - h75_11 / 2)
                    y75_12 = math.ceil(y0 + h0 / 2 - h75_12 / 2)
                    y75_13 = math.ceil(y0 + h0 / 2 - h75_13 / 2)
                    y75_14 = math.ceil(y0 + h0 / 2 - h75_14 / 2)
                    y75_15 = math.ceil(y0 + h0 / 2 - h75_15 / 2)
                    y80_11 = math.ceil(y0 + h0 / 2 - h80_11 / 2)
                    y80_12 = math.ceil(y0 + h0 / 2 - h80_12 / 2)
                    y80_13 = math.ceil(y0 + h0 / 2 - h80_13 / 2)
                    y80_14 = math.ceil(y0 + h0 / 2 - h80_14 / 2)
                    y80_15 = math.ceil(y0 + h0 / 2 - h80_15 / 2)
                    y85_11 = math.ceil(y0 + h0 / 2 - h85_11 / 2)
                    y85_12 = math.ceil(y0 + h0 / 2 - h85_12 / 2)
                    y85_13 = math.ceil(y0 + h0 / 2 - h85_13 / 2)
                    y85_14 = math.ceil(y0 + h0 / 2 - h85_14 / 2)
                    y85_15 = math.ceil(y0 + h0 / 2 - h85_15 / 2)
                    y90_11 = math.ceil(y0 + h0 / 2 - h90_11 / 2)
                    y90_12 = math.ceil(y0 + h0 / 2 - h90_12 / 2)
                    y90_13 = math.ceil(y0 + h0 / 2 - h90_13 / 2)
                    y90_14 = math.ceil(y0 + h0 / 2 - h90_14 / 2)
                    y90_15 = math.ceil(y0 + h0 / 2 - h90_15 / 2)

                    # Boundary restrictions
                    if y70_11 + h70_11 >= vidHeight:
                        h70_11 = vidHeight - y70_11

                    if y70_12 + h70_12 >= vidHeight:
                        h70_12 = vidHeight - y70_12

                    if y70_13 + h70_13 >= vidHeight:
                        h70_13 = vidHeight - y70_13

                    if y70_14 + h70_14 >= vidHeight:
                        h70_14 = vidHeight - y70_14

                    if y70_15 + h70_15 >= vidHeight:
                        h70_15 = vidHeight - y70_15

                    if y75_11 + h75_11 >= vidHeight:
                        h75_11 = vidHeight - y75_11

                    if y75_12 + h75_12 >= vidHeight:
                        h75_12 = vidHeight - y75_12

                    if y70_13 + h70_13 >= vidHeight:
                        h70_13 = vidHeight - y70_13

                    if y75_14 + h75_14 >= vidHeight:
                        h75_14 = vidHeight - y75_14

                    if y75_15 + h75_15 >= vidHeight:
                        h75_15 = vidHeight - y75_15

                    if y80_11 + h80_11 >= vidHeight:
                        h80_11 = vidHeight - y80_11

                    if y80_12 + h80_12 >= vidHeight:
                        h80_12 = vidHeight - y80_12

                    if y80_13 + h80_13 >= vidHeight:
                        h80_13 = vidHeight - y80_13

                    if y80_14 + h80_14 >= vidHeight:
                        h80_14 = vidHeight - y80_14

                    if y80_15 + h80_15 >= vidHeight:
                        h80_15 = vidHeight - y80_15

                    if y85_11 + h85_11 >= vidHeight:
                        h85_11 = vidHeight - y85_11

                    if y85_12 + h85_12 >= vidHeight:
                        h85_12 = vidHeight - y85_12

                    if y85_13 + h85_13 >= vidHeight:
                        h85_13 = vidHeight - y85_13

                    if y85_14 + h85_14 >= vidHeight:
                        h85_14 = vidHeight - y85_14

                    if y85_15 + h85_15 >= vidHeight:
                        h85_15 = vidHeight - y85_15

                    if y90_11 + h90_11 >= vidHeight:
                        h90_11 = vidHeight - y90_11

                    if y90_12 + h90_12 >= vidHeight:
                        h90_12 = vidHeight - y90_12

                    if y90_13 + h90_13 >= vidHeight:
                        h90_13 = vidHeight - y90_13

                    if y90_14 + h90_14 >= vidHeight:
                        h90_14 = vidHeight - y90_14

                    if y90_15 + h90_15 >= vidHeight:
                        h90_15 = vidHeight - y90_15

                    ROI25[1, :, imageName_150] = [x70, y70_11, w70, h70_11]
                    ROI25[2, :, imageName_150] = [x70, y70_12, w70, h70_12]
                    ROI25[3, :, imageName_150] = [x70, y70_13, w70, h70_13]
                    ROI25[4, :, imageName_150] = [x70, y70_14, w70, h70_14]
                    ROI25[5, :, imageName_150] = [x70, y70_15, w70, h70_15]
                    ROI25[6, :, imageName_150] = [x75, y75_11, w75, h75_11]
                    ROI25[7, :, imageName_150] = [x75, y75_12, w75, h75_12]
                    ROI25[8, :, imageName_150] = [x75, y75_13, w75, h75_13]
                    ROI25[9, :, imageName_150] = [x75, y75_14, w75, h75_14]
                    ROI25[10, :, imageName_150] = [x75, y75_15, w75, h75_15]
                    ROI25[11, :, imageName_150] = [x80, y80_11, w80, h80_11]
                    ROI25[12, :, imageName_150] = [x80, y80_12, w80, h80_12]
                    ROI25[13, :, imageName_150] = [x80, y80_13, w80, h80_13]
                    ROI25[14, :, imageName_150] = [x80, y80_14, w80, h80_14]
                    ROI25[15, :, imageName_150] = [x80, y80_15, w80, h80_15]
                    ROI25[16, :, imageName_150] = [x85, y85_11, w85, h85_11]
                    ROI25[17, :, imageName_150] = [x85, y85_12, w85, h85_12]
                    ROI25[18, :, imageName_150] = [x85, y85_13, w85, h85_13]
                    ROI25[19, :, imageName_150] = [x85, y85_14, w85, h85_14]
                    ROI25[20, :, imageName_150] = [x85, y85_15, w85, h85_15]
                    ROI25[21, :, imageName_150] = [x90, y90_11, w90, h90_11]
                    ROI25[22, :, imageName_150] = [x90, y90_12, w90, h90_12]
                    ROI25[23, :, imageName_150] = [x90, y90_13, w90, h90_13]
                    ROI25[24, :, imageName_150] = [x90, y90_14, w90, h90_14]
                    ROI25[25, :, imageName_150] = [x90, y90_15, w90, h90_15]

                    imgcrop1 = currImage[y70_11:y70_11 + h70_11, x70:x70 + w70]
                    imgcrop2 = currImage[y70_12:y70_12 + h70_12, x70:x70 + w70]
                    imgcrop3 = currImage[y70_13:y70_13 + h70_13, x70:x70 + w70]
                    imgcrop4 = currImage[y70_14:y70_14 + h70_14, x70:x70 + w70]
                    imgcrop5 = currImage[y70_15:y70_15 + h70_15, x70:x70 + w70]
                    imgcrop6 = currImage[y75_11:y75_11 + h75_11, x75:x75 + w75]
                    imgcrop7 = currImage[y75_12:y75_12 + h75_12, x75:x75 + w75]
                    imgcrop8 = currImage[y75_13:y75_13 + h75_13, x75:x75 + w75]
                    imgcrop9 = currImage[y75_14:y75_14 + h75_14, x75:x75 + w75]
                    imgcrop10 = currImage[y75_15:y75_15 + h75_15, x75:x75 + w75]
                    imgcrop11 = currImage[y80_11:y80_11 + h70_11, x80:x80 + w80]
                    imgcrop12 = currImage[y80_12:y80_12 + h70_12, x80:x80 + w80]
                    imgcrop13 = currImage[y80_13:y80_13 + h70_13, x80:x80 + w80]
                    imgcrop14 = currImage[y80_14:y80_14 + h70_14, x80:x80 + w80]
                    imgcrop15 = currImage[y80_15:y80_15 + h70_15, x80:x80 + w80]
                    imgcrop16 = currImage[y85_11:y85_11 + h70_11, x85:x85 + w85]
                    imgcrop17 = currImage[y85_12:y85_12 + h70_12, x85:x85 + w85]
                    imgcrop18 = currImage[y85_13:y85_13 + h70_13, x85:x85 + w85]
                    imgcrop19 = currImage[y85_14:y85_14 + h70_14, x85:x85 + w85]
                    imgcrop20 = currImage[y85_15:y85_15 + h70_15, x85:x85 + w85]
                    imgcrop21 = currImage[y90_11:y90_11 + h70_11, x90:x90 + w90]
                    imgcrop22 = currImage[y90_12:y90_12 + h70_12, x90:x90 + w90]
                    imgcrop23 = currImage[y90_13:y90_13 + h70_13, x90:x90 + w90]
                    imgcrop24 = currImage[y90_14:y90_14 + h70_14, x90:x90 + w90]
                    imgcrop25 = currImage[y90_15:y90_15 + h70_15, x90:x90 + w90]

                    r_imgcrop1 = cv2.resize(imgcrop1, (178, 178))
                    r_imgcrop2 = cv2.resize(imgcrop2, (178, 178))
                    r_imgcrop3 = cv2.resize(imgcrop3, (178, 178))
                    r_imgcrop4 = cv2.resize(imgcrop4, (178, 178))
                    r_imgcrop5 = cv2.resize(imgcrop5, (178, 178))
                    r_imgcrop6 = cv2.resize(imgcrop6, (178, 178))
                    r_imgcrop7 = cv2.resize(imgcrop7, (178, 178))
                    r_imgcrop8 = cv2.resize(imgcrop8, (178, 178))
                    r_imgcrop9 = cv2.resize(imgcrop9, (178, 178))
                    r_imgcrop10 = cv2.resize(imgcrop10, (178, 178))
                    r_imgcrop11 = cv2.resize(imgcrop11, (178, 178))
                    r_imgcrop12 = cv2.resize(imgcrop12, (178, 178))
                    r_imgcrop13 = cv2.resize(imgcrop13, (178, 178))
                    r_imgcrop14 = cv2.resize(imgcrop14, (178, 178))
                    r_imgcrop15 = cv2.resize(imgcrop15, (178, 178))
                    r_imgcrop16 = cv2.resize(imgcrop16, (178, 178))
                    r_imgcrop17 = cv2.resize(imgcrop17, (178, 178))
                    r_imgcrop18 = cv2.resize(imgcrop18, (178, 178))
                    r_imgcrop19 = cv2.resize(imgcrop19, (178, 178))
                    r_imgcrop20 = cv2.resize(imgcrop20, (178, 178))
                    r_imgcrop21 = cv2.resize(imgcrop21, (178, 178))
                    r_imgcrop22 = cv2.resize(imgcrop22, (178, 178))
                    r_imgcrop23 = cv2.resize(imgcrop23, (178, 178))
                    r_imgcrop24 = cv2.resize(imgcrop24, (178, 178))
                    r_imgcrop25 = cv2.resize(imgcrop25, (178, 178))

                    videoRPN25_1[imageName_150, :, :, :] = r_imgcrop1[:, :, :]
                    videoRPN25_2[imageName_150, :, :, :] = r_imgcrop2[:, :, :]
                    videoRPN25_3[imageName_150, :, :, :] = r_imgcrop3[:, :, :]
                    videoRPN25_4[imageName_150, :, :, :] = r_imgcrop4[:, :, :]
                    videoRPN25_5[imageName_150, :, :, :] = r_imgcrop5[:, :, :]
                    videoRPN25_6[imageName_150, :, :, :] = r_imgcrop6[:, :, :]
                    videoRPN25_7[imageName_150, :, :, :] = r_imgcrop7[:, :, :]
                    videoRPN25_8[imageName_150, :, :, :] = r_imgcrop8[:, :, :]
                    videoRPN25_9[imageName_150, :, :, :] = r_imgcrop9[:, :, :]
                    videoRPN25_10[imageName_150, :, :, :] = r_imgcrop10[:, :, :]
                    videoRPN25_11[imageName_150, :, :, :] = r_imgcrop11[:, :, :]
                    videoRPN25_12[imageName_150, :, :, :] = r_imgcrop12[:, :, :]
                    videoRPN25_13[imageName_150, :, :, :] = r_imgcrop13[:, :, :]
                    videoRPN25_14[imageName_150, :, :, :] = r_imgcrop14[:, :, :]
                    videoRPN25_15[imageName_150, :, :, :] = r_imgcrop15[:, :, :]
                    videoRPN25_16[imageName_150, :, :, :] = r_imgcrop16[:, :, :]
                    videoRPN25_17[imageName_150, :, :, :] = r_imgcrop17[:, :, :]
                    videoRPN25_18[imageName_150, :, :, :] = r_imgcrop18[:, :, :]
                    videoRPN25_19[imageName_150, :, :, :] = r_imgcrop19[:, :, :]
                    videoRPN25_20[imageName_150, :, :, :] = r_imgcrop20[:, :, :]
                    videoRPN25_21[imageName_150, :, :, :] = r_imgcrop21[:, :, :]
                    videoRPN25_22[imageName_150, :, :, :] = r_imgcrop22[:, :, :]
                    videoRPN25_23[imageName_150, :, :, :] = r_imgcrop23[:, :, :]
                    videoRPN25_24[imageName_150, :, :, :] = r_imgcrop24[:, :, :]
                    videoRPN25_25[imageName_150, :, :, :] = r_imgcrop25[:, :, :]

                    SNRs_25 = np.zeros((1, 25))
                    Weights_9 = np.zeros((1, 9))

                    ROI25_max = np.zeros((4, nImages + 1), dtype=np.int32)
                    ROI9_Weights_max = np.zeros((4, nImages + 1), dtype=np.int32)

                    rank_25_SNRs = np.zeros((1, 25))
                    rank_9_Weights = np.zeros((1, 9))

                    index_25_SNRs = np.zeros((1, 25))
                    index_9_Weights = np.zeros((1, 9))

                    videoRPN25_150_n[1, :, :, :, :] = videoRPN25_1
                    videoRPN25_150_n[2, :, :, :, :] = videoRPN25_2
                    videoRPN25_150_n[3, :, :, :, :] = videoRPN25_3
                    videoRPN25_150_n[4, :, :, :, :] = videoRPN25_4
                    videoRPN25_150_n[5, :, :, :, :] = videoRPN25_5
                    videoRPN25_150_n[6, :, :, :, :] = videoRPN25_6
                    videoRPN25_150_n[7, :, :, :, :] = videoRPN25_7
                    videoRPN25_150_n[8, :, :, :, :] = videoRPN25_8
                    videoRPN25_150_n[9, :, :, :, :] = videoRPN25_9
                    videoRPN25_150_n[10, :, :, :, :] = videoRPN25_10
                    videoRPN25_150_n[11, :, :, :, :] = videoRPN25_11
                    videoRPN25_150_n[12, :, :, :, :] = videoRPN25_12
                    videoRPN25_150_n[13, :, :, :, :] = videoRPN25_13
                    videoRPN25_150_n[14, :, :, :, :] = videoRPN25_14
                    videoRPN25_150_n[15, :, :, :, :] = videoRPN25_15
                    videoRPN25_150_n[16, :, :, :, :] = videoRPN25_16
                    videoRPN25_150_n[17, :, :, :, :] = videoRPN25_17
                    videoRPN25_150_n[18, :, :, :, :] = videoRPN25_18
                    videoRPN25_150_n[19, :, :, :, :] = videoRPN25_19
                    videoRPN25_150_n[20, :, :, :, :] = videoRPN25_20
                    videoRPN25_150_n[21, :, :, :, :] = videoRPN25_21
                    videoRPN25_150_n[22, :, :, :, :] = videoRPN25_22
                    videoRPN25_150_n[23, :, :, :, :] = videoRPN25_23
                    videoRPN25_150_n[24, :, :, :, :] = videoRPN25_24
                    videoRPN25_150_n[25, :, :, :, :] = videoRPN25_25

                # first selection
                win_pulseEst1 = np.zeros((25 + 1, winLength))
                for iROI1 in range(1, 26):
                    videoRPN25_150_i = np.zeros((150, 178, 178, 3))
                    videoRPN25_150_i[:, :, :, :] = videoRPN25_150_n[iROI1, 1:151, :, :, :]
                    videoRPN25_150_i = videoRPN25_150_i / 255
                    videoRPN25_150_i = np.array([videoRPN25_150_i])
                    videoRPN25_150_i = torch.FloatTensor(videoRPN25_150_i)
                    videoRPN25_150_i = videoRPN25_150_i.permute([0, 4, 1, 2, 3])
                    videoRPN25_150_i = videoRPN25_150_i.to(device)

                    ppg_est_25_i = model(videoRPN25_150_i)
                    ppg_est_25_i = ppg_est_25_i.cpu()
                    ppg_est_25_i = ppg_est_25_i.squeeze(0)
                    ppg_est_25_i = ppg_est_25_i.squeeze(2)
                    ppg_est_25_i = ppg_est_25_i.squeeze(2)
                    ppg_est_25_i = ppg_est_25_i.detach().numpy()

                    win_pulseEst1[iROI1, :] = ppg_est_25_i[0, :]  # 25 signals extracted by physnet
                    SNRs_25[0, iROI1 - 1] = get_SNR(win_pulseEst1[iROI1, :], PUREfps)  # get 25 defined SNR

                SNRs_25 = SNRs_25.squeeze(0)
                SNRs_25 = SNRs_25.tolist()
                Index_25_SNR = sorted(range(len(SNRs_25)), key=lambda k: SNRs_25[k], reverse=True)
                First_index_25_SNR = Index_25_SNR[0]

                win_pulseEst_25Max = win_pulseEst1[First_index_25_SNR + 1,:]  # first selection signals extracted by physnet

                ROI25_max[:, n - winLength + 1: n + 1] = ROI25[First_index_25_SNR + 1, :, 1:151]

                videoRPN9_150_n = np.zeros((9 + 1, 150 + 1, 178, 178, 3), dtype=np.float32)
                for iImage2 in range(n - winLength + 1, n + 1):  # 1:150,  76:225,  151:300 ...
                    imageName2 = imageList[iImage2 - 1]
                    imagePath2 = vidDir + '/' + imageName2
                    currImage2 = cv2.imread(imagePath2)

                    imageName_150_2 = iImage2 - n + winLength

                    x_90 = ROI25_max[0, iImage2]
                    y_90 = ROI25_max[1, iImage2]
                    w_90 = ROI25_max[2, iImage2]
                    h_90 = ROI25_max[3, iImage2]

                    ROI9[:, :, imageName_150_2] = F_9anchors(x_90, y_90, w_90, h_90)  # get second 9 ROI coordinates

                    x_91 = ROI9[1, 0, imageName_150_2]
                    y_91 = ROI9[1, 1, imageName_150_2]
                    w_91 = ROI9[1, 2, imageName_150_2]
                    h_91 = ROI9[1, 3, imageName_150_2]

                    x_92 = ROI9[2, 0, imageName_150_2]
                    y_92 = ROI9[2, 1, imageName_150_2]
                    w_92 = ROI9[2, 2, imageName_150_2]
                    h_92 = ROI9[2, 3, imageName_150_2]

                    x_93 = ROI9[3, 0, imageName_150_2]
                    y_93 = ROI9[3, 1, imageName_150_2]
                    w_93 = ROI9[3, 2, imageName_150_2]
                    h_93 = ROI9[3, 3, imageName_150_2]

                    x_94 = ROI9[4, 0, imageName_150_2]
                    y_94 = ROI9[4, 1, imageName_150_2]
                    w_94 = ROI9[4, 2, imageName_150_2]
                    h_94 = ROI9[4, 3, imageName_150_2]

                    x_95 = ROI9[5, 0, imageName_150_2]
                    y_95 = ROI9[5, 1, imageName_150_2]
                    w_95 = ROI9[5, 2, imageName_150_2]
                    h_95 = ROI9[5, 3, imageName_150_2]

                    x_96 = ROI9[6, 0, imageName_150_2]
                    y_96 = ROI9[6, 1, imageName_150_2]
                    w_96 = ROI9[6, 2, imageName_150_2]
                    h_96 = ROI9[6, 3, imageName_150_2]

                    x_97 = ROI9[7, 0, imageName_150_2]
                    y_97 = ROI9[7, 1, imageName_150_2]
                    w_97 = ROI9[7, 2, imageName_150_2]
                    h_97 = ROI9[7, 3, imageName_150_2]

                    x_98 = ROI9[8, 0, imageName_150_2]
                    y_98 = ROI9[8, 1, imageName_150_2]
                    w_98 = ROI9[8, 2, imageName_150_2]
                    h_98 = ROI9[8, 3, imageName_150_2]

                    x_99 = ROI9[9, 0, imageName_150_2]
                    y_99 = ROI9[9, 1, imageName_150_2]
                    w_99 = ROI9[9, 2, imageName_150_2]
                    h_99 = ROI9[9, 3, imageName_150_2]

                    imgcrop9ROI_1 = currImage2[y_91:y_91 + h_91, x_91:x_91 + y_91]
                    imgcrop9ROI_2 = currImage2[y_92:y_92 + h_92, x_91:x_92 + y_92]
                    imgcrop9ROI_3 = currImage2[y_93:y_93 + h_93, x_91:x_93 + y_93]
                    imgcrop9ROI_4 = currImage2[y_94:y_94 + h_94, x_91:x_94 + y_94]
                    imgcrop9ROI_5 = currImage2[y_95:y_95 + h_95, x_91:x_95 + y_95]
                    imgcrop9ROI_6 = currImage2[y_96:y_96 + h_96, x_91:x_96 + y_96]
                    imgcrop9ROI_7 = currImage2[y_97:y_97 + h_97, x_91:x_97 + y_97]
                    imgcrop9ROI_8 = currImage2[y_98:y_98 + h_98, x_91:x_98 + y_98]
                    imgcrop9ROI_9 = currImage2[y_99:y_99 + h_99, x_91:x_99 + y_99]

                    r_imgcrop9ROI_1 = cv2.resize(imgcrop9ROI_1, (178, 178))
                    r_imgcrop9ROI_2 = cv2.resize(imgcrop9ROI_2, (178, 178))
                    r_imgcrop9ROI_3 = cv2.resize(imgcrop9ROI_3, (178, 178))
                    r_imgcrop9ROI_4 = cv2.resize(imgcrop9ROI_4, (178, 178))
                    r_imgcrop9ROI_5 = cv2.resize(imgcrop9ROI_5, (178, 178))
                    r_imgcrop9ROI_6 = cv2.resize(imgcrop9ROI_6, (178, 178))
                    r_imgcrop9ROI_7 = cv2.resize(imgcrop9ROI_7, (178, 178))
                    r_imgcrop9ROI_8 = cv2.resize(imgcrop9ROI_8, (178, 178))
                    r_imgcrop9ROI_9 = cv2.resize(imgcrop9ROI_9, (178, 178))

                    videoRPN9_1[imageName_150_2, :, :, :] = r_imgcrop9ROI_1[:, :, :]
                    videoRPN9_2[imageName_150_2, :, :, :] = r_imgcrop9ROI_2[:, :, :]
                    videoRPN9_3[imageName_150_2, :, :, :] = r_imgcrop9ROI_3[:, :, :]
                    videoRPN9_4[imageName_150_2, :, :, :] = r_imgcrop9ROI_4[:, :, :]
                    videoRPN9_5[imageName_150_2, :, :, :] = r_imgcrop9ROI_5[:, :, :]
                    videoRPN9_6[imageName_150_2, :, :, :] = r_imgcrop9ROI_6[:, :, :]
                    videoRPN9_7[imageName_150_2, :, :, :] = r_imgcrop9ROI_7[:, :, :]
                    videoRPN9_8[imageName_150_2, :, :, :] = r_imgcrop9ROI_8[:, :, :]
                    videoRPN9_9[imageName_150_2, :, :, :] = r_imgcrop9ROI_9[:, :, :]

                    videoRPN9_150_n[1, :, :, :, :] = videoRPN9_1
                    videoRPN9_150_n[2, :, :, :, :] = videoRPN9_2
                    videoRPN9_150_n[3, :, :, :, :] = videoRPN9_3
                    videoRPN9_150_n[4, :, :, :, :] = videoRPN9_4
                    videoRPN9_150_n[5, :, :, :, :] = videoRPN9_5
                    videoRPN9_150_n[6, :, :, :, :] = videoRPN9_6
                    videoRPN9_150_n[7, :, :, :, :] = videoRPN9_7
                    videoRPN9_150_n[8, :, :, :, :] = videoRPN9_8
                    videoRPN9_150_n[9, :, :, :, :] = videoRPN9_9

                # second selection
                win_pulseEst2 = np.zeros((9 + 1, winLength))
                for iROI2 in range(1, 10):
                    videoRPN9_150_i = np.zeros((150, 178, 178, 3))
                    videoRPN9_150_i[:, :, :, :] = videoRPN9_150_n[iROI2, 1:151, :, :, :]
                    videoRPN9_150_i = videoRPN9_150_i / 255
                    videoRPN9_150_i = np.array([videoRPN9_150_i])
                    videoRPN9_150_i = torch.FloatTensor(videoRPN9_150_i)
                    videoRPN9_150_i = videoRPN9_150_i.permute([0, 4, 1, 2, 3])
                    videoRPN9_150_i = videoRPN9_150_i.to(device)

                    ppg_est_9_i = model(videoRPN9_150_i)

                    ppg_est_9_i = ppg_est_9_i.cpu()
                    ppg_est_9_i = ppg_est_9_i.squeeze(0)
                    ppg_est_9_i = ppg_est_9_i.squeeze(2)
                    ppg_est_9_i = ppg_est_9_i.squeeze(2)
                    ppg_est_9_i = ppg_est_9_i.detach().numpy()

                    win_pulseEst2[iROI2, :] = ppg_est_9_i[0, :]  # 9 signals extracted by physnet

                Weights_9 = getVarWeight(win_pulseEst2, winLength)
                Weights_9 = Weights_9.squeeze(0)
                Weights_9 = Weights_9.tolist()
                Index_9_Weights = sorted(range(len(Weights_9)), key=lambda k: Weights_9[k], reverse=True)
                First_index_9_Weights = Index_9_Weights[0]

                win_pulseEst_9_Weights_Max = win_pulseEst2[First_index_9_Weights + 1,:]  # second selection signals extracted by physnet

                ROI9_Weights_max[:, n - winLength + 1: n + 1] = ROI9[First_index_9_Weights + 1, :, 1:151]

                win_pulseEst3 = np.zeros((6, winLength))

                videoRPN5_150_n = np.zeros((9 + 1, 150 + 1, 178, 178, 3), dtype=np.float32)
                for iImage3 in range(n - winLength + 1, n + 1):  # 1:150,76:225,151:300...
                    imageName3 = imageList[iImage3 - 1]
                    imagePath3 = vidDir + '/' + imageName3
                    currImage3 = cv2.imread(imagePath3)

                    imageName_150_3 = iImage3 - n + winLength
                    vidHeight = np.size(currImage3, 0)

                    ImageW = np.size(currImage3, 1)
                    ImageH = np.size(currImage3, 0)

                    x_Weights_9max = ROI9_Weights_max[0, iImage3]
                    y_Weights_9max = ROI9_Weights_max[1, iImage3]
                    w_Weights_9max = ROI9_Weights_max[2, iImage3]
                    h_Weights_9max = ROI9_Weights_max[3, iImage3]

                    ROI5[1:6, :, imageName_150_3] = F_multisacle(x_Weights_9max, y_Weights_9max, w_Weights_9max,
                                                                 h_Weights_9max, k, ImageW, ImageH)
                    # 5 ROI coordinateS
                    x_51 = ROI5[1, 0, imageName_150_3]
                    y_51 = ROI5[1, 1, imageName_150_3]
                    w_51 = ROI5[1, 2, imageName_150_3]
                    h_51 = ROI5[1, 3, imageName_150_3]

                    x_52 = ROI5[2, 0, imageName_150_3]
                    y_52 = ROI5[2, 1, imageName_150_3]
                    w_52 = ROI5[2, 2, imageName_150_3]
                    h_52 = ROI5[2, 3, imageName_150_3]

                    x_53 = ROI5[3, 0, imageName_150_3]
                    y_53 = ROI5[3, 1, imageName_150_3]
                    w_53 = ROI5[3, 2, imageName_150_3]
                    h_53 = ROI5[3, 3, imageName_150_3]

                    x_54 = ROI5[4, 0, imageName_150_3]
                    y_54 = ROI5[4, 1, imageName_150_3]
                    w_54 = ROI5[4, 2, imageName_150_3]
                    h_54 = ROI5[4, 3, imageName_150_3]

                    x_55 = ROI5[5, 0, imageName_150_3]
                    y_55 = ROI5[5, 1, imageName_150_3]
                    w_55 = ROI5[5, 2, imageName_150_3]
                    h_55 = ROI5[5, 3, imageName_150_3]

                    imgcrop5ROI_1 = currImage3[y_51:y_51 + h_51, x_51:x_51 + w_51]
                    imgcrop5ROI_2 = currImage3[y_52:y_52 + h_52, x_52:x_52 + w_52]
                    imgcrop5ROI_3 = currImage3[y_53:y_53 + h_53, x_53:x_53 + w_53]
                    imgcrop5ROI_4 = currImage3[y_54:y_54 + h_54, x_54:x_54 + w_54]
                    imgcrop5ROI_5 = currImage3[y_55:y_55 + h_55, x_55:x_55 + w_55]

                    r_imgcrop5ROI_1 = cv2.resize(imgcrop5ROI_1, (178, 178))
                    r_imgcrop5ROI_2 = cv2.resize(imgcrop5ROI_2, (178, 178))
                    r_imgcrop5ROI_3 = cv2.resize(imgcrop5ROI_3, (178, 178))
                    r_imgcrop5ROI_4 = cv2.resize(imgcrop5ROI_4, (178, 178))
                    r_imgcrop5ROI_5 = cv2.resize(imgcrop5ROI_5, (178, 178))

                    videoRPN5_1[imageName_150_3, :, :, :] = r_imgcrop5ROI_1[:, :, :]
                    videoRPN5_2[imageName_150_3, :, :, :] = r_imgcrop5ROI_2[:, :, :]
                    videoRPN5_3[imageName_150_3, :, :, :] = r_imgcrop5ROI_3[:, :, :]
                    videoRPN5_4[imageName_150_3, :, :, :] = r_imgcrop5ROI_4[:, :, :]
                    videoRPN5_5[imageName_150_3, :, :, :] = r_imgcrop5ROI_5[:, :, :]

                    videoRPN5_150_n[1, :, :, :, :] = videoRPN5_1
                    videoRPN5_150_n[2, :, :, :, :] = videoRPN5_2
                    videoRPN5_150_n[3, :, :, :, :] = videoRPN5_3
                    videoRPN5_150_n[4, :, :, :, :] = videoRPN5_4
                    videoRPN5_150_n[5, :, :, :, :] = videoRPN5_5

                win_pulseEst3 = np.zeros((5 + 1, winLength))
                for iROI3 in range(1, 6):
                    videoRPN5_150_i = np.zeros((150, 178, 178, 3))
                    videoRPN5_150_i[:, :, :, :] = videoRPN25_150_n[iROI3, 1:151, :, :, :]
                    videoRPN5_150_i = videoRPN5_150_i / 255
                    videoRPN5_150_i = np.array([videoRPN5_150_i])
                    videoRPN5_150_i = torch.FloatTensor(videoRPN5_150_i)
                    videoRPN5_150_i = videoRPN5_150_i.permute([0, 4, 1, 2, 3])
                    videoRPN5_150_i = videoRPN5_150_i.to(device)

                    ppg_est_5_i = model(videoRPN5_150_i)

                    ppg_est_5_i = ppg_est_5_i.cpu()
                    ppg_est_5_i = ppg_est_5_i.squeeze(0)
                    ppg_est_5_i = ppg_est_5_i.squeeze(2)
                    ppg_est_5_i = ppg_est_5_i.squeeze(2)
                    ppg_est_5_i = ppg_est_5_i.detach().numpy()

                    win_pulseEst3[iROI3, :] = ppg_est_5_i[0, :] # 5 multi scale signals extracted by physnet

                SNRs_5 = SNRs_5.squeeze(0)
                SNRs_5 = SNRs_5.tolist()
                Index_5_SNR = sorted(range(len(SNRs_5)), key=lambda k: SNRs_5[k], reverse=True)
                First_index_5_SNR = Index_5_SNR[0]

                T_hannW = np.transpose(hannW)
                win_fusion_pulseEst_GS = 0.05 * win_pulseEst3[1, :] + 0.25 * win_pulseEst3[2, :] + 0.4 * win_pulseEst3[
                                                                                                         3, :] \
                                         + 0.25 * win_pulseEst3[4, :] + 0.05 * win_pulseEst3[5,
                                                                               :]  # Gaussian signal fusion
                win_fusion_pulseEst_Uni = 0.2 * win_pulseEst3[1, :] + 0.2 * win_pulseEst3[2, :] + 0.2 * win_pulseEst3[3,
                                                                                                        :] \
                                          + 0.2 * win_pulseEst3[4, :] + 0.2 * win_pulseEst3[5,
                                                                              :]  # Average signal fusion

                win_fusion_pulseEst_Uni = win_fusion_pulseEst_Uni - np.mean(win_fusion_pulseEst_Uni)
                win_fusion_pulseEst_GS = win_fusion_pulseEst_GS - np.mean(win_fusion_pulseEst_GS)
                win_pulseEst_25Max = win_pulseEst_25Max - np.mean(win_pulseEst_25Max)
                win_pulseEst_9_Weights_Max = win_pulseEst_9_Weights_Max - np.mean(win_pulseEst_9_Weights_Max)

                win_fusion_pulseEst_Uni = win_fusion_pulseEst_Uni * T_hannW
                win_fusion_pulseEst_GS = win_fusion_pulseEst_GS * T_hannW
                win_pulseEst_25Max = win_pulseEst_25Max * T_hannW
                win_pulseEst_9_Weights_Max = win_pulseEst_9_Weights_Max * T_hannW

                # Overlap and add to complete signal
                PulseEst_Var[0, n - winLength + 1: n + 1] = PulseEst_Var[0,
                                                            n - winLength + 1: n + 1] + win_fusion_pulseEst_Var
                PulseEst_GS[0, n - winLength + 1: n + 1] = PulseEst_GS[0,
                                                           n - winLength + 1: n + 1] + win_fusion_pulseEst_GS
                PulseEst_Exp[0, n - winLength + 1: n + 1] = PulseEst_Exp[0,
                                                            n - winLength + 1: n + 1] + win_fusion_pulseEst_exp
                PulseEst_Uni[0, n - winLength + 1: n + 1] = PulseEst_Uni[0,
                                                            n - winLength + 1: n + 1] + win_fusion_pulseEst_Uni
                PulseEst_25Max[0, n - winLength + 1: n + 1] = PulseEst_25Max[0,
                                                              n - winLength + 1: n + 1] + win_pulseEst_25Max
                PulseEst_Stg2_W[0, n - winLength + 1: n + 1] = PulseEst_Stg2_W[0,
                                                               n - winLength + 1: n + 1] + win_pulseEst_9_Weights_Max
                PulseEst_Stg2_GS[0, n - winLength + 1: n + 1] = PulseEst_Stg2_GS[0,
                                                                n - winLength + 1: n + 1] + win_fusion_pulseEst_9_GS
                PulseEst_Stg2_Uni[0, n - winLength + 1: n + 1] = PulseEst_Stg2_Uni[0,
                                                                 n - winLength + 1: n + 1] + win_fusion_pulseEst_9_Uni

            sio.savemat(file2Save,
                        {'PulseEst_GS': PulseEst_GS, 'PulseEst_Uni': PulseEst_Uni, 'PulseEst_25Max': PulseEst_25Max,
                         'PulseEst_Stg2_W': PulseEst_Stg2_W, })

            print([subID + ' PluseEst complete'])
