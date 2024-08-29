import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int,
        downsample: nn.Module = None,
    ) -> None:
        super(Block, self).__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BasicBlock(Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int,
        downsample: nn.Module = None,
    ) -> None:
        super(Block, self).__init__()
        self.downsample = downsample
        self.expansion = expansion
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out += identity
        out = self.relu(out)
        return out


class BottleNeckBlock(Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int,
        downsample: nn.Module = None,
    ) -> None:
        super(Block, self).__init__()
        self.downsample = downsample
        self.expansion = expansion
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.batch_norm0 = nn.BatchNorm2d(out_channels)
        in_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv0(x)
        out = self.batch_norm0(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out += identity
        out = self.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        img_channels: int,
        block: type[Block],
        num_classes: int,
        blocks_count: tuple[int, int, int, int],
        expansion: int,
    ) -> None:
        super(_ResNet, self).__init__()
        self.blocks_count = blocks_count
        self.expansion = expansion
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block=block, out_channels=64, count=self.blocks_count[0], stride=1
        )
        self.layer2 = self._make_layer(
            block=block, out_channels=128, count=self.blocks_count[1], stride=2
        )
        self.layer3 = self._make_layer(
            block=block, out_channels=256, count=self.blocks_count[2], stride=2
        )
        self.layer4 = self._make_layer(
            block=block, out_channels=512, count=self.blocks_count[3], stride=2
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(
        self,
        block: type[Block],
        out_channels: int,
        count: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = [
            (
                block(
                    self.in_channels,
                    out_channels,
                    stride,
                    self.expansion,
                    downsample,
                )
            )
        ]
        self.in_channels = out_channels * self.expansion
        layers += [
            block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=1,
                expansion=self.expansion,
            )
            for _ in range(count - 1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        assert tuple(x.shape)[2:] == (7, 7)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18(_ResNet):
    def __init__(
        self,
        img_channels: int,
        num_classes: int,
    ):
        super().__init__(
            img_channels=img_channels,
            block=BasicBlock,
            num_classes=num_classes,
            blocks_count=(2, 2, 2, 2),
            expansion=1,
        )


class ResNet34(_ResNet):
    def __init__(
        self,
        img_channels: int,
        num_classes: int,
    ):
        super().__init__(
            img_channels=img_channels,
            block=BasicBlock,
            num_classes=num_classes,
            blocks_count=(3, 4, 6, 3),
            expansion=1,
        )


class ResNet50(_ResNet):
    def __init__(
        self,
        img_channels: int,
        num_classes: int,
    ):
        super().__init__(
            img_channels=img_channels,
            block=BottleNeckBlock,
            num_classes=num_classes,
            blocks_count=(3, 4, 6, 3),
            expansion=4,
        )


class ResNet101(_ResNet):
    def __init__(
        self,
        img_channels: int,
        num_classes: int,
    ):
        super().__init__(
            img_channels=img_channels,
            block=BottleNeckBlock,
            num_classes=num_classes,
            blocks_count=(3, 4, 23, 3),
            expansion=4,
        )


class ResNet152(_ResNet):
    def __init__(
        self,
        img_channels: int,
        num_classes: int,
    ):
        super().__init__(
            img_channels=img_channels,
            block=BottleNeckBlock,
            num_classes=num_classes,
            blocks_count=(3, 8, 36, 3),
            expansion=4,
        )


if __name__ == "__main__":
    tensor = torch.rand([1, 3, 224, 224])
    for ModelClass in (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152):
        model = ModelClass(img_channels=3, num_classes=1000)
        print(model)

        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"{total_trainable_params:,} training parameters.")
        output = model(tensor)
