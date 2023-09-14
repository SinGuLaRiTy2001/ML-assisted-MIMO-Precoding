import numpy
import torch
# from torch import optim

import scipy.io as scio
import h5py
import time
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on {}'.format(device))

HMean_Dataset_r_nd = h5py.File('data/HMean_input_real_Dataset_10ms.mat')  # 50 8 80 64 9
HMean_Dataset_i_nd = h5py.File('data/HMean_input_imaginary_Dataset_10ms.mat')  # 50 8 80 64 9
Omega_Dataset_nd = h5py.File('data/OmegaPost_input_Dataset_10ms.mat')  # 50 8 256 9
SNR_Dataset_nd = h5py.File('data/SNR_input_Dataset_10ms.mat')  # 1 11
mu_Dataset_nd = h5py.File('data/mu_output_Dataset_10ms.mat')  # 50 8 80 9 11

HMean_Dataset_r = torch.tensor(numpy.transpose(numpy.array(HMean_Dataset_r_nd['HMean_input_Dataset_r'])))
HMean_Dataset_i = torch.tensor(numpy.transpose(numpy.array(HMean_Dataset_i_nd['HMean_input_Dataset_i'])))
Omega_Dataset = torch.tensor(numpy.transpose(numpy.array(Omega_Dataset_nd['omega_input_Dataset'])))
SNR_Dataset = torch.tensor(numpy.transpose(numpy.array(SNR_Dataset_nd['snr_Dataset'])))
mu_Dataset = torch.tensor(numpy.transpose(numpy.array(mu_Dataset_nd['mu_output_Dataset'])))

HMean_Dataset_i = HMean_Dataset_i.to(device)
HMean_Dataset_r = HMean_Dataset_r.to(device)
Omega_Dataset = Omega_Dataset.to(device)
SNR_Dataset = SNR_Dataset.to(device)
mu_Dataset = mu_Dataset.to(device)

''' dataset sizes '''
locNum = 50
rbNum = 8
slotNum = 80 - 3
antNum = 64
kNum = 9
snrNum = 11

sampleNum = locNum * slotNum * snrNum
indexInput = numpy.linspace(0, sampleNum - 1, sampleNum, dtype=int)

iterNum = 5000
batchSize = 1
learning_rate = 0.00001


class Net_Hi(torch.nn.Module):
    def __init__(self):
        super(Net_Hi, self).__init__()  # 9 64 9
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(9, 64, kernel_size=(7, 3), padding=(3, 1)),  # 64 64 9
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(7, 3), padding=(3, 1)),  # 128 64 9
            torch.nn.MaxPool2d(kernel_size=(4, 1)),  # 128 16 9
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=(5, 3), padding=(2, 1)),  # 256 16 9
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=(5, 3), padding=(2, 1)),  # 512 16 9
            torch.nn.MaxPool2d(kernel_size=(4, 1)),  # 512 4 9
            torch.nn.ReLU()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),  # 256 4 9
            torch.nn.ReLU()
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),  # 128 4 9
            torch.nn.MaxPool2d(kernel_size=(2, 1)),  # 128 2 9
            torch.nn.ReLU()
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 32, kernel_size=(3, 3), padding=(1, 1)),  # 32 2 9
            torch.nn.ReLU()
        )

    def forward(self, hi):
        hi = hi.view(batchSize, 9, 64, 9)

        hi = self.conv1(hi)
        hi = self.conv2(hi)
        hi = self.conv3(hi)
        hi = self.conv4(hi)
        hi = self.conv5(hi)
        hi = self.conv6(hi)
        hi = self.conv7(hi)

        hi = hi.view(-1, 32 * 2 * 9)

        return hi


class Net_Hr(torch.nn.Module):
    def __init__(self):
        super(Net_Hr, self).__init__()  # 9 64 9
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(9, 64, kernel_size=(7, 3), padding=(3, 1)),  # 64 64 9
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(7, 3), padding=(3, 1)),  # 128 64 9
            torch.nn.MaxPool2d(kernel_size=(4, 1)),  # 128 16 9
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=(5, 3), padding=(2, 1)),  # 256 16 9
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=(5, 3), padding=(2, 1)),  # 512 16 9
            torch.nn.MaxPool2d(kernel_size=(4, 1)),  # 512 4 9
            torch.nn.ReLU()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),  # 256 4 9
            torch.nn.ReLU()
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),  # 128 4 9
            torch.nn.MaxPool2d(kernel_size=(2, 1)),  # 128 2 9
            torch.nn.ReLU()
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 32, kernel_size=(3, 3), padding=(1, 1)),  # 32 2 9
            torch.nn.ReLU()
        )

    def forward(self, hr):
        hr = hr.view(batchSize, 9, 64, 9)

        hr = self.conv1(hr)
        hr = self.conv2(hr)
        hr = self.conv3(hr)
        hr = self.conv4(hr)
        hr = self.conv5(hr)
        hr = self.conv6(hr)
        hr = self.conv7(hr)

        hr = hr.view(-1, 32 * 2 * 9)

        return hr


class Net_Omega(torch.nn.Module):
    def __init__(self):
        super(Net_Omega, self).__init__()  # 9 256 9
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(9, 64, kernel_size=(7, 3), padding=(3, 1)),  # 64 256 9
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(7, 3), padding=(3, 1)),  # 128 256 9
            torch.nn.MaxPool2d(kernel_size=(4, 1)),  # 128 64 9
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=(5, 3), padding=(2, 1)),  # 256 64 9
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=(5, 3), padding=(2, 1)),  # 512 64 9
            torch.nn.MaxPool2d(kernel_size=(4, 1)),  # 512 16 9
            torch.nn.ReLU()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1)),  # 256 16 9
            torch.nn.ReLU()
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),  # 128 16 9
            torch.nn.MaxPool2d(kernel_size=(4, 1)),  # 128 4 9
            torch.nn.ReLU()
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 32, kernel_size=(3, 3), padding=(1, 1)),  # 32 4 9
            torch.nn.MaxPool2d(kernel_size=(2, 1)),  # 32 2 9
            torch.nn.ReLU()
        )

    def forward(self, omega):
        omega = omega.view(batchSize, 9, 256, 9)

        omega = self.conv1(omega)
        omega = self.conv2(omega)
        omega = self.conv3(omega)
        omega = self.conv4(omega)
        omega = self.conv5(omega)
        omega = self.conv6(omega)
        omega = self.conv7(omega)

        omega = omega.view(-1, 32 * 2 * 9)

        return omega


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.net_Hi = Net_Hi()
        self.net_Hr = Net_Hr()
        self.net_Omega = Net_Omega()

        self.fc1 = torch.nn.Linear(1729, 1024)
        self.fc2 = torch.nn.Linear(1025, 512)
        self.fc3 = torch.nn.Linear(513, 256)
        self.fc4 = torch.nn.Linear(257, 72)
        self.fc5 = torch.nn.Linear(129, 72)

    def forward(self, hr, hi, omega, snr):
        hr = self.net_Hr(hr)  # 1 32*2*9 = 1 576
        hi = self.net_Hi(hi)  # 1 32*2*9 = 1 576
        omega = self.net_Omega(omega)  # 1 32*2*9 = 1 576

        union = torch.cat([hr, hi, omega], dim=1)  # 1 1728

        snr = snr.view(batchSize, 1)

        union = torch.cat([union, snr], dim=1)  # 1 1729
        union = self.fc1(union)

        union = torch.cat([union, snr], dim=1)  # 1 1025
        union = self.fc2(union)

        union = torch.cat([union, snr], dim=1)  # 1 513
        union = self.fc3(union)

        union = torch.cat([union, snr], dim=1)  # 1 257
        union = self.fc4(union)

        # union = torch.cat([union, snr], dim=1)  # 1 129
        # union = self.fc5(union)

        union = union.view(8, 9)

        return union


net = torch.load("MIMO_precoding_0502.pth")
print(net)

net = net.to(device)

criterion = torch.nn.MSELoss()
# optimizer = optim.Adam(net.parameters(), lr=learning_rate)

mu_Output = numpy.zeros(shape=(50, 8, 80, 9, 11))


def output():
    net.eval()

    with torch.no_grad():
        for locId in range(locNum):  # 50
            for initial_slotId in range(slotNum):  # 80-3
                for snrId in range(snrNum):  # 11

                    start = time.time()

                    H_r_1 = torch.stack([HMean_Dataset_r[locId, 0, initial_slotId, :, :],
                                         HMean_Dataset_r[locId, 3, initial_slotId, :, :],
                                         HMean_Dataset_r[locId, 6, initial_slotId, :, :]
                                         ], dim=0)  # 3 64 9
                    H_r_2 = torch.stack([HMean_Dataset_r[locId, 1, initial_slotId + 1, :, :],
                                         HMean_Dataset_r[locId, 4, initial_slotId + 1, :, :],
                                         HMean_Dataset_r[locId, 7, initial_slotId + 1, :, :]
                                         ], dim=0)  # 3 64 9
                    H_r_3 = torch.stack([HMean_Dataset_r[locId, 2, initial_slotId + 2][:, :],
                                         HMean_Dataset_r[locId, 5, initial_slotId + 2][:, :],
                                         HMean_Dataset_r[locId, 7, initial_slotId + 2][:, :]
                                         ], dim=0)  # 3 64 9
                    H_r = torch.cat([H_r_1, H_r_2, H_r_3], dim=0)  # 9 64 9

                    H_i_1 = torch.stack([HMean_Dataset_i[locId, 0, initial_slotId, :, :],
                                         HMean_Dataset_i[locId, 3, initial_slotId, :, :],
                                         HMean_Dataset_i[locId, 6, initial_slotId, :, :]
                                         ], dim=0)  # 3 64 9
                    H_i_2 = torch.stack([HMean_Dataset_i[locId, 1, initial_slotId + 1, :, :],
                                         HMean_Dataset_i[locId, 4, initial_slotId + 1, :, :],
                                         HMean_Dataset_i[locId, 7, initial_slotId + 1, :, :]
                                         ], dim=0)  # 3 64 9
                    H_i_3 = torch.stack([HMean_Dataset_i[locId, 2, initial_slotId + 2, :, :],
                                         HMean_Dataset_i[locId, 5, initial_slotId + 2, :, :],
                                         HMean_Dataset_i[locId, 7, initial_slotId + 2, :, :]
                                         ], dim=0)  # 3 64 9
                    H_i = torch.cat([H_i_1, H_i_2, H_i_3], dim=0)  # 9 64 9

                    Omega_1 = torch.stack([Omega_Dataset[locId, 0, :, :],
                                           Omega_Dataset[locId, 3, :, :],
                                           Omega_Dataset[locId, 6, :, :]
                                           ], dim=0)  # 3 256 9
                    Omega_2 = torch.stack([Omega_Dataset[locId, 1, :, :],
                                           Omega_Dataset[locId, 4, :, :],
                                           Omega_Dataset[locId, 7, :, :]
                                           ], dim=0)  # 3 256 9
                    Omega_3 = torch.stack([Omega_Dataset[locId, 2, :, :],
                                           Omega_Dataset[locId, 5, :, :],
                                           Omega_Dataset[locId, 7, :, :]
                                           ], dim=0)  # 3 256 9
                    Omega = torch.cat([Omega_1, Omega_2, Omega_3], dim=0)  # 9 256 9

                    SNR = SNR_Dataset[:, snrId]  # 1 1

                    mu_Label = mu_Dataset[locId, :, initial_slotId + 3, :, snrId]  # 8 9

                    H_r = H_r.to(torch.float32)
                    H_i = H_i.to(torch.float32)
                    Omega = Omega.to(torch.float32)
                    SNR = SNR.to(torch.float32)
                    mu_Label = mu_Label.to(torch.float32)

                    H_r.to(device)
                    H_i.to(device)
                    Omega.to(device)
                    SNR.to(device)
                    mu_Label.to(device)

                    mu_Predict = net(H_r, H_i, Omega, SNR)
                    loss = criterion(mu_Predict, mu_Label)
                    mu_Predict = mu_Predict.cpu()
                    mu_Output[locId, :, initial_slotId + 3, :, snrId] = mu_Predict

                    end = time.time()

                    cost = end - start

                    print('locId: {}\t slotId: {}\t snrId: {}\t Loss: {:.7f}\t Time: {:.2f}ms'
                          .format(locId, initial_slotId + 3, snrId, loss, cost * 1000))


output()
mu_Output_file = 'mu_predicted_Dataset_10ms.mat'
scio.savemat(mu_Output_file, {'mu_predicted_Dataset': mu_Output})

print('Done.')
