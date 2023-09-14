import numpy
import torch
from torch import optim

# import scipy.io as scio
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Running on {}'.format(device))

HMean_Dataset_r_nd = h5py.File('data/HMean_input_real_Dataset_10ms.mat')
HMean_Dataset_i_nd = h5py.File('data/HMean_input_imaginary_Dataset_10ms.mat')
# 50 8 80 64 9  模拟不同的位置  频域的资源块数  探测周期数（1ms上行导频+9ms下行传输）  发射天线  用户数（每个用户1根天线）

Omega_Dataset_nd = h5py.File('data/OmegaPost_input_Dataset_10ms.mat')  # 50 8 256 9  模拟不同的位置  频域的资源块数  波束数  用户数
SNR_Dataset_nd = h5py.File('data/SNR_input_Dataset_10ms.mat')  # 1 11  遍历取值：[0:5:50]
mu_Dataset_nd = h5py.File('data/mu_output_Dataset_10ms.mat')
# 50 8 80 9 11  模拟不同的位置  频域的资源块数  探测周期数（1ms上行导频+9ms下行传输）  用户数  SNR数

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
slotNum = 80 - 4
antNum = 64
kNum = 9
snrNum = 11

sampleNum = locNum * slotNum * snrNum
trainNum = int(0.9 * sampleNum)
testNum = int(0.1 * sampleNum)
indexTrain = numpy.linspace(0, trainNum - 1, trainNum, dtype=int)
indexTest = numpy.linspace(trainNum, sampleNum - 1, testNum, dtype=int)

iterNum = 5000
batchSize = 16
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

        union = union.view(batchSize, 8, 9)

        return union


net = Net()
print(net)

net = net.to(device)

criterion = torch.nn.MSELoss()  # 均方损失函数
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


def train():
    net.train()
    train_loss = 0

    numpy.random.shuffle(indexTrain)
    dataIndex = indexTrain[0:batchSize]

    locId = dataIndex // (slotNum * snrNum)

    tempId = dataIndex % (slotNum * snrNum)
    initial_slotId = tempId // snrNum

    snrId = tempId % snrNum

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

    optimizer.zero_grad()
    mu_Predict = net(H_r, H_i, Omega, SNR)
    loss = criterion(mu_Predict, mu_Label)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()

    return train_loss


def test():
    net.eval()
    test_losses = 0
    max_loss = -1
    min_loss = 9999

    with torch.no_grad():
        for test_id in range(10):
            numpy.random.shuffle(indexTest)
            dataIndex = indexTest[0:batchSize]

            locId = dataIndex // (slotNum * snrNum)

            tempId = dataIndex % (slotNum * snrNum)
            initial_slotId = tempId // snrNum

            snrId = tempId % snrNum

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
            test_losses += loss.item()
            if loss.item() > max_loss:
                max_loss = loss.item()
            if loss.item() < min_loss:
                min_loss = loss.item()

    return test_losses / 10, max_loss, min_loss


train_losses = 0

print('=================TEST=================')
test_loss, max_loss, min_loss = test()
print('Initial loss: {:.7f}\t Max loss: {:.7f}\t Min loss: {:.7f}'
      .format(test_loss, max_loss, min_loss))

for iterId in range(iterNum):
    train_losses += train()
    if (iterId + 1) % 10 == 0:
        print('Iter: {}\t Train loss: {:.7f}'.format(iterId + 1, train_losses / 10))
        train_losses = 0
    if (iterId + 1) % 200 == 0:
        print('=================TEST=================')
        test_loss, max_loss, min_loss = test()
        print('Iter: {}\t Test loss: {:.7f}\t Max loss: {:.7f}\t Min loss: {:.7f}'
              .format(iterId + 1, test_loss, max_loss, min_loss))
    if (iterId + 1) % 200 == 0:
        learning_rate *= 0.1
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
torch.save(net, "MIMO_precoding_0502.pth")
