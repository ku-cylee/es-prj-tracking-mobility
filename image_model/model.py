import torch.nn as nn

class IdentityResNet(nn.Module):
    
    def __init__(self, image_size, labels_count, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=2 * image_size, kernel_size=3, padding=1),
            ResNetStage(in_channels=2 * image_size, out_channels=2 * image_size, blocks_count=nblk_stage1),
            ResNetStage(in_channels=2 * image_size, out_channels=4 * image_size, blocks_count=nblk_stage2),
            ResNetStage(in_channels=4 * image_size, out_channels=8 * image_size, blocks_count=nblk_stage3),
            ResNetStage(in_channels=8 * image_size, out_channels=16 * image_size, blocks_count=nblk_stage4),
            ResNetStage(in_channels=16 * image_size, out_channels=32 * image_size, blocks_count=nblk_stage4),
            nn.AvgPool2d(kernel_size=4, stride=4),
        )

        self.fc = nn.Linear(in_features=32 * image_size, out_features=labels_count)
    
    def forward(self, input):
        return self.fc(self.layers(input).view(-1, self.fc.in_features))


class ResNetStage(nn.Module):

    def __init__(self, in_channels, out_channels, blocks_count):
        super(ResNetStage, self).__init__()

        first_layer = ResNetChannelConstantBlock(in_channels) if in_channels == out_channels \
                      else ResNetChannelDoubleBlock(in_channels, out_channels)
        layers = [first_layer] + [ResNetChannelConstantBlock(out_channels) for _ in range(blocks_count - 1)]

        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers(input)


class ResNetChannelDoubleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResNetChannelDoubleBlock, self).__init__()

        self.pre_layers = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
        )

        self.unskip_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        )

        self.skip_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2),
        )


    def forward(self, input):
        pre_result = self.pre_layers(input)
        return self.unskip_layers(pre_result) + self.skip_layers(pre_result)


class ResNetChannelConstantBlock(nn.Module):

    def __init__(self, channels):
        super(ResNetChannelConstantBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, input):
        return self.layers(input) + input
