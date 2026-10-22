import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Fashion-MNIST 类别名称
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ==================== 复用你的模型定义 ====================
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


class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row['label'])
        pixels = row.drop('label').values.astype(np.uint8).reshape(28, 28)
        image = Image.fromarray(pixels, mode='L')

        if self.transform:
            image = self.transform(image)

        return image, label


# ==================== 分析函数 ====================

def plot_training_history(history_file='training_history.csv', save_dir='analysis_plots'):
    """绘制训练历史曲线"""
    os.makedirs(save_dir, exist_ok=True)

    # 只绘制真实训练日志，避免将模拟数据当作实验结果。
    if not os.path.exists(history_file):
        print(f"⚠️  {history_file} not found. Skipping training-history plot.")
        return

    history = pd.read_csv(history_file)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss曲线
    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(history['epoch'], history['train_acc'], label='Train Accuracy', linewidth=2, color='#2E86AB')
    axes[1].plot(history['epoch'], history['val_acc'], label='Val Accuracy', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/1_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/1_training_history.png")
    plt.close()


def evaluate_and_plot_confusion_matrix(model, data_loader, device, save_dir='analysis_plots'):
    """生成混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/2_confusion_matrix.png")
    plt.close()

    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
    plt.title('Normalized Confusion Matrix (Recall)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/3_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/3_confusion_matrix_normalized.png")
    plt.close()

    return all_preds, all_labels


def plot_per_class_metrics(all_labels, all_preds, save_dir='analysis_plots'):
    """绘制每个类别的性能指标"""
    os.makedirs(save_dir, exist_ok=True)

    # 计算每个类别的precision, recall, f1-score
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)

    metrics_data = []
    for class_name in CLASS_NAMES:
        metrics_data.append({
            'Class': class_name,
            'Precision': report[class_name]['precision'],
            'Recall': report[class_name]['recall'],
            'F1-Score': report[class_name]['f1-score']
        })

    df_metrics = pd.DataFrame(metrics_data)

    # 绘制条形图
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    bars1 = ax.bar(x - width, df_metrics['Precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, df_metrics['Recall'], width, label='Recall', color='#e74c3c')
    bars3 = ax.bar(x + width, df_metrics['F1-Score'], width, label='F1-Score', color='#2ecc71')

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/4_per_class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/4_per_class_metrics.png")
    plt.close()

    return df_metrics


def plot_prediction_confidence(model, data_loader, device, save_dir='analysis_plots', num_samples=1000):
    """绘制预测置信度分布"""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    correct_confidences = []
    incorrect_confidences = []

    with torch.no_grad():
        count = 0
        for inputs, labels in data_loader:
            if count >= num_samples:
                break

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)

            correct_mask = (preds == labels)
            correct_confidences.extend(confidences[correct_mask].cpu().numpy())
            incorrect_confidences.extend(confidences[~correct_mask].cpu().numpy())

            count += inputs.size(0)

    # 绘制置信度分布
    plt.figure(figsize=(12, 6))

    plt.hist(correct_confidences, bins=50, alpha=0.7, label='Correct Predictions',
             color='#2ecc71', edgecolor='black')
    plt.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect Predictions',
             color='#e74c3c', edgecolor='black')

    plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'Correct: Mean={np.mean(correct_confidences):.3f}, Std={np.std(correct_confidences):.3f}\n'
    stats_text += f'Incorrect: Mean={np.mean(incorrect_confidences):.3f}, Std={np.std(incorrect_confidences):.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/5_confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/5_confidence_distribution.png")
    plt.close()


def plot_sample_predictions(model, data_loader, device, save_dir='analysis_plots', num_samples=16):
    """可视化样本预测结果"""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    # 获取一批数据
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images[:num_samples]
    labels = labels[:num_samples]

    # 预测
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = F.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, 1)

    # 绘制
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()

    for i in range(num_samples):
        img = images[i].cpu().numpy().squeeze()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        confidence = confidences[i].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

        # 颜色：正确为绿色，错误为红色
        color = 'green' if true_label == pred_label else 'red'
        title = f'True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}\nConf: {confidence:.2%}'
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')

    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/6_sample_predictions.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/6_sample_predictions.png")
    plt.close()


def plot_error_analysis(model, data_loader, device, save_dir='analysis_plots', top_k=20):
    """错误案例分析"""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    errors = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)

            # 找出错误预测
            incorrect_mask = (preds != labels)
            if incorrect_mask.any():
                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    errors.append({
                        'image': inputs[idx].cpu(),
                        'true_label': labels[idx].item(),
                        'pred_label': preds[idx].item(),
                        'confidence': confidences[idx].item()
                    })

    # 按置信度排序，找出高置信度的错误
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    top_errors = errors[:min(top_k, len(errors))]

    # 绘制
    cols = 5
    rows = (len(top_errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.ravel() if rows > 1 else [axes] if rows == 1 else []

    for i, error in enumerate(top_errors):
        if i >= len(axes):
            break

        img = error['image'].numpy().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

        title = f"True: {CLASS_NAMES[error['true_label']]}\n"
        title += f"Pred: {CLASS_NAMES[error['pred_label']]}\n"
        title += f"Conf: {error['confidence']:.2%}"
        axes[i].set_title(title, fontsize=9, color='red', fontweight='bold')

    # 隐藏多余的子图
    for i in range(len(top_errors), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Top {len(top_errors)} High-Confidence Errors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/7_error_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/7_error_analysis.png")
    plt.close()


def plot_learning_rate_schedule(save_dir='analysis_plots'):
    """绘制学习率调度"""
    os.makedirs(save_dir, exist_ok=True)

    epochs = 200
    initial_lr = 0.1
    milestones = [60, 120, 160]
    gamma = 0.2

    lrs = []
    current_lr = initial_lr
    for epoch in range(1, epochs + 1):
        lrs.append(current_lr)
        if epoch in milestones:
            current_lr *= gamma

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), lrs, linewidth=2, color='#e74c3c')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Learning Rate', fontsize=12, fontweight='bold')
    plt.title('Learning Rate Schedule (MultiStepLR)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 标注milestone
    for milestone in milestones:
        plt.axvline(x=milestone, color='gray', linestyle='--', alpha=0.7)
        plt.text(milestone, initial_lr / 2, f'Epoch {milestone}',
                 rotation=90, verticalalignment='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/8_learning_rate_schedule.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/8_learning_rate_schedule.png")
    plt.close()


def plot_class_distribution(data_loader, save_dir='analysis_plots'):
    """绘制类别分布"""
    os.makedirs(save_dir, exist_ok=True)

    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())

    # 统计每个类别的数量
    unique, counts = np.unique(all_labels, return_counts=True)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(CLASS_NAMES)), counts, color='#3498db', edgecolor='black')
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Validation Set', fontsize=14, fontweight='bold')
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/9_class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/9_class_distribution.png")
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # 数据转换
    mean = [0.2860]
    std = [0.3530]
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载验证集
    print('Loading validation dataset...')
    full_dataset = FashionMNISTDataset('fashion-mnist_train.csv', transform=transform_test)

    # 使用相同的分割方式
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    print(f'Validation dataset size: {len(val_dataset)} samples\n')

    # 加载模型
    print('Loading model...')
    model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
    model = model.to(device)

    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.2f}%\n")

    print('=' * 70)
    print('Starting visualization generation...')
    print('=' * 70 + '\n')

    # 生成所有图表
    plot_training_history()
    all_preds, all_labels = evaluate_and_plot_confusion_matrix(model, val_loader, device)
    df_metrics = plot_per_class_metrics(all_labels, all_preds)
    plot_prediction_confidence(model, val_loader, device)
    plot_sample_predictions(model, val_loader, device)
    plot_error_analysis(model, val_loader, device)
    plot_learning_rate_schedule()
    plot_class_distribution(val_loader)

    print('\n' + '=' * 70)
    print('✅ All visualizations completed!')
    print('=' * 70)
    print('\nGenerated plots:')
    print('  1. Training History (Loss & Accuracy curves)')
    print('  2. Confusion Matrix (Raw counts)')
    print('  3. Normalized Confusion Matrix (Recall)')
    print('  4. Per-Class Metrics (Precision, Recall, F1)')
    print('  5. Confidence Distribution')
    print('  6. Sample Predictions')
    print('  7. Error Analysis (High-confidence errors)')
    print('  8. Learning Rate Schedule')
    print('  9. Class Distribution')
    print('\nAll plots saved in: analysis_plots/')

    # 打印统计摘要
    print('\n' + '=' * 70)
    print('Performance Summary:')
    print('=' * 70)
    print(f"\nOverall Accuracy: {checkpoint['accuracy']:.2f}%")
    print(f"\nPer-class metrics:")
    print(df_metrics.to_string(index=False))


if __name__ == '__main__':
    main()
