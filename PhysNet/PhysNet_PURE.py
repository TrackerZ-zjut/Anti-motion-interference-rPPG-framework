from matplotlib import pyplot as plt
import numpy as np
import os
import scipy.io as sio
import math
import cv2
import torch
from scipy.fftpack import fft
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    nSub = 10
    nVersion = 6
    PUREfps = 30
    winLength = 150
    stepSize = int(winLength / 2)
    hannW = np.hanning(winLength)
    T_hannW = np.transpose(hannW)

    modelName = 'PhysNet_3DCNNs'
    modelDir = 'G:/ZMH/Mul/weight_150/PhysNet_PURE_150_3_epoch03.pth'  # trained model
    # (dataset_test, winLength, RPN, modelName, kers, dataset_weight, winLength, RPN, i_epoch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            file2Save = NewResultDir + '/PhysNet3D_PURE_1008_epoch03.mat'

            if not os.path.exists(NewResultDir):
                os.makedirs(NewResultDir)

            if not os.path.exists(vidDir):
                print(subID + 'does not exist')
                continue

            if not os.path.exists(ResultDir):
                os.makedirs(ResultDir)

            imageList = os.listdir(vidDir)
            roi = sio.loadmat(roi_File)  # roi2Save
            nImages = len(imageList)
            Num_k = math.floor(nImages / stepSize)
            nImages = int(Num_k * stepSize)

            PulseEst_PhysNet = np.zeros((1, nImages + 1))

            videoRPN25_1 = np.zeros((150, 178, 178, 3), dtype=np.float32)

            for n in range(winLength, nImages + 1, stepSize):  # 150,225,300,375...
                for iImage in range(n - winLength + 1, n + 1):  # 1:150,  76:225,  151:300 ...
                    imageName = imageList[iImage - 1]
                    imagePath1 = vidDir + '/' + imageName
                    currImage = cv2.imread(imagePath1)

                    vidHeight = np.size(currImage, 1)
                    vidWidth = np.size(currImage, 0)

                    imageName_150 = iImage - n + winLength
                    # Original ROI coordinates
                    x0 = int(roi['rect_klt'][iImage - 1, 0])
                    y0 = int(roi['rect_klt'][iImage - 1, 1])
                    w0 = int(roi['rect_klt'][iImage - 1, 2])
                    h0 = int(roi['rect_klt'][iImage - 1, 3])

                    if y0 + h0 >= vidHeight:
                        h0 = vidHeight - y0

                    if x0 + w0 >= vidWidth:
                        x0 = vidWidth - w0

                    imgcrop1 = currImage[y0:y0 + h0, x0:x0 + w0]
                    r_imgcrop1 = cv2.resize(imgcrop1, (178, 178))
                    # cv2.imshow("r_imgcrop1", r_imgcrop1)
                    # cv2.waitKey(0)
                    videoRPN25_1[imageName_150 - 1, :, :, :] = r_imgcrop1[:, :, :]
                videoRPN25_1 = videoRPN25_1 / 255
                videoRPN25_150_i = np.array([videoRPN25_1])
                videoRPN25_150_i = torch.FloatTensor(videoRPN25_150_i)
                videoRPN25_150_i = videoRPN25_150_i.permute([0, 4, 1, 2, 3])
                videoRPN25_150_i = videoRPN25_150_i.to(device)

                ppg_est_25_i = model(videoRPN25_150_i)
                ppg_est_25_i = ppg_est_25_i.cpu()
                ppg_est_25_i = ppg_est_25_i.squeeze(0)
                ppg_est_25_i = ppg_est_25_i.squeeze(2)
                ppg_est_25_i = ppg_est_25_i.squeeze(2)
                ppg_est_25_i = ppg_est_25_i.detach().numpy()  # signal extracted by physnet

                win_fusion_pulseEst = ppg_est_25_i[0, :]
                win_fusion_pulseEst = win_fusion_pulseEst - np.mean(win_fusion_pulseEst)
                win_fusion_pulseEst = win_fusion_pulseEst * T_hannW
                # Overlap and add to complete signal
                PulseEst_PhysNet[0, n - winLength + 1: n + 1] = PulseEst_PhysNet[0,
                                                                n - winLength + 1: n + 1] + win_fusion_pulseEst

            sio.savemat(file2Save, {'PulseEst_PhysNet': PulseEst_PhysNet})

            print([subID + ' PluseEst complete'])
