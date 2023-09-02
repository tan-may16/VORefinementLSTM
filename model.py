import torch
import torch.nn as nn
import torchvision.models as models

class VORefinementLSTM(nn.Module):
    def __init__(self, input_channels = 4, lstm_hidden_size = 128, num_lstm_layers = 2):
        super(VORefinementLSTM, self).__init__()

        # ResNet feature extractor
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d( in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False )
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=6))

        # Weight initialization
        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, odometry):
        # Extract features from the input image
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        x = self.resnet(x)
        x = self.pool(x)
        x = x.view(b, s, -1)

        # Combine image features and odometry
        x_shifted = x.clone()
        x_shifted[:, 1:, :] = x[:, :-1, :]
        x = torch.cat((x, x_shifted, odometry), dim=2)

        # LSTM layer
        x, _ = self.lstm(x.float())

        # Fully connected layers
        x = x[:, -1, :]  # Use only the last time step's output
        x = self.fc(x)

        return x
