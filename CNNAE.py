import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv256 = nn.Conv1d(in_channels=22, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv128 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv64 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.upconv32 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.upconv64 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.upconv128 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.upconv256 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Conv1d(in_channels=256, out_channels=22, kernel_size=1, stride=1)

        self.prelu256 = nn.PReLU(num_parameters=256)
        self.prelu128 = nn.PReLU(num_parameters=128)
        self.prelu64 = nn.PReLU(num_parameters=64)
        self.prelu32 = nn.PReLU(num_parameters=32)
        self.min_loss = 100

    def forward(self, x):
        out_256 = self.conv256(x)
        out = self.prelu256(out_256)

        out_128 = self.conv128(out)
        out = self.prelu128(out_128)

        out_64 = self.conv64(out)
        out = self.prelu64(out_64)

        out_32 = self.conv32(out)
        out = self.prelu32(out_32)

        out = self.conv1(out)
        encoded = out

        out = self.upconv32(out) + out_32
        out = self.prelu32(out)

        out = self.upconv64(out) + out_64
        out = self.prelu64(out)

        out = self.upconv128(out) + out_128
        out = self.prelu128(out)

        out = self.upconv256(out) + out_256

        return encoded, self.downsample(out)

    def train_model(self, model, epochs, train, test):
        model.cuda()
        loss_func1 = nn.MSELoss().cuda()

        optim = torch.optim.SGD(model.parameters(), lr=1e-3)
        for i in range(0, epochs+1):
            val_accuracy = 0
            model.train()
            for (x, y) in train:
                optim.zero_grad()
                x = x.cuda()
                encoded, out = model(x)
                loss = loss_func1(out, x)
                loss.backward()
                optim.step()

            print('---testing---')
            with torch.no_grad():
                model.eval()
                for (x,y) in test:
                    x = x.cuda()
                    y = y.cuda()
                    encoded, out = model(x)
                    loss = loss_func(out, x)
                    val_accuracy += loss.item()
                    if torch.argmax(lstm_out) == y.cuda():
                        val_accuracy+=1
                if val_accuracy < self.min_acc:
                    print('saving')
                    print(val_accuracy)
                    torch.save(model.state_dict(), '/home/atharva/encoder_classifier.pth')
                    self.min_acc = val_accuracy
