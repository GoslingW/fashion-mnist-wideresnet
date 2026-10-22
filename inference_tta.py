import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image


# ==================== Wide ResNet (Same as before) ====================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 7)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


# ==================== Test Dataset ====================
class FashionMNISTTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = row.values.astype(np.uint8).reshape(28, 28)
        image = Image.fromarray(pixels, mode='L')

        if self.transform:
            image = self.transform(image)

        return image, idx


def predict_with_tta(model, test_loader, device, num_augments=5):
    """使用 Test Time Augmentation 进行预测"""
    model.eval()

    # 定义TTA的数据增强（不包含随机性，使用固定增强）
    mean = [0.2860]
    std = [0.3530]

    # TTA增强策略：原图 + 水平翻转 + 4个不同crop
    tta_transforms = [
        # 1. 原图
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        # 2. 水平翻转
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        # 3-6. 不同位置的crop（通过padding实现）
        transforms.Compose([
            transforms.Pad(2),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Pad(2),
            transforms.RandomCrop(28),  # 左上
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Pad(2),
            transforms.RandomCrop(28),  # 右下
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    ]

    all_predictions = []

    print(f'Running TTA with {len(tta_transforms)} augmentations...')

    with torch.no_grad():
        # 重新加载原始数据并应用每种增强
        for aug_idx, transform in enumerate(tta_transforms):
            print(f'Augmentation {aug_idx + 1}/{len(tta_transforms)}...')

            # 创建新的dataset with当前transform
            test_dataset = FashionMNISTTestDataset('fashion-mnist_test_no_gt.csv', transform=transform)
            test_loader_tta = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

            aug_predictions = []
            for inputs, _ in test_loader_tta:
                inputs = inputs.to(device)
                outputs = model(inputs)
                # 保存概率而不是类别
                probs = F.softmax(outputs, dim=1)
                aug_predictions.append(probs.cpu().numpy())

            aug_predictions = np.concatenate(aug_predictions, axis=0)
            all_predictions.append(aug_predictions)

    # 平均所有增强的概率
    avg_predictions = np.mean(all_predictions, axis=0)
    final_predictions = np.argmax(avg_predictions, axis=1)

    return final_predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transforms (for initial load)
    mean = [0.2860]
    std = [0.3530]

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load test dataset (只用于获取数据大小)
    print('Loading test dataset...')
    test_dataset = FashionMNISTTestDataset('fashion-mnist_test_no_gt.csv', transform=transform_test)
    print(f'Test dataset size: {len(test_dataset)} samples')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Load model
    print('Loading model...')
    model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
    model = model.to(device)

    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.2f}%")

    # Generate predictions with TTA
    print('\n' + '=' * 60)
    print('Generating predictions with Test Time Augmentation...')
    print('=' * 60)
    predictions = predict_with_tta(model, test_loader, device, num_augments=5)

    # Create submission file
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'label': predictions
    })

    submission.to_csv('submission_tta.csv', index=False)
    print('\n' + '=' * 60)
    print('✅ Submission file created: submission_tta.csv')
    print(f'Total predictions: {len(predictions)}')
    print('=' * 60)
    print('\nFirst 10 predictions:')
    print(submission.head(10))
    print('\nTTA typically improves accuracy by 0.3-0.5%!')


if __name__ == '__main__':
    main()