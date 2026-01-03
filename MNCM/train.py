import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# 修改导入，使用带CPCA和自注意力的模型
from model_v3_cpca_mha import mobilenet_v3_large_with_attentions


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    batch_size = 16
    epochs = 50

    # 创建唯一的实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/exp_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    save_dir = os.path.join(experiment_dir, "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(experiment_dir, "training_log.csv")
    create_log_file(log_file)

    print(f"实验目录: {experiment_dir}")
    print(f"模型保存目录: {save_dir}")

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val_test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # 数据集根路径
    data_root = r"D:\深度学习\deep-learning-for-image-processing-master\deep-learning-for-image-processing-master\data_set\论文中的数据小麦\小麦data_set"
    image_path = r"D:\深度学习\deep-learning-for-image-processing-master\deep-learning-for-image-processing-master\data_set\论文中的数据小麦\小麦data_set"
    assert os.path.exists(image_path), f"路径 {image_path} 不存在."

    # 加载已划分的数据集
    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transform["train"]
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val_test"]
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "test"),
        transform=data_transform["val_test"]
    )

    # 保存类别索引
    class_indices = {v: k for k, v in train_dataset.class_to_idx.items()}
    with open(os.path.join(experiment_dir, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=4)

    # 数据加载器
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f"使用 {nw} 个数据加载线程.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )

    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}.")

    # 模型初始化 - 使用带CPCA和自注意力的模型构建函数
    # 指定启用CPCA和自注意力的层
    cpca_layers = [3, 4, 5, 10, 11, 12]  # 在这些层启用CPCA
    self_attn_layers = [10, 11, 12]  # 在这些层启用自注意力（与CPCA部分重叠）

    net = mobilenet_v3_large_with_attentions(
        num_classes=len(class_indices),
        cpca_layers=cpca_layers,
        self_attn_layers=self_attn_layers
    )

    # 打印注意力机制启用情况
    print(f"启用CPCA的层: {cpca_layers}")
    print(f"启用自注意力的层: {self_attn_layers}")

    # 加载预训练权重（基础MobileNetV3权重，不含CPCA和自注意力部分）
    model_weight_path = "D:\\深度学习\\deep-learning-for-image-processing-master\\deep-learning-for-image-processing-master\\pytorch_classification\\Test6_mobilenet\\mobilenet_v3_large-8738ca79.pth"
    assert os.path.exists(model_weight_path), f"权重文件 {model_weight_path} 不存在."

    # 加载预训练权重
    pre_weights = torch.load(model_weight_path, map_location="cpu")

    # 过滤不匹配的层
    pre_dict = {k: v for k, v in pre_weights.items() if k in net.state_dict() and
                net.state_dict()[k].shape == v.shape}

    # 移除不匹配的分类头权重
    if 'classifier.0.weight' in pre_dict:
        del pre_dict['classifier.0.weight']
    if 'classifier.0.bias' in pre_dict:
        del pre_dict['classifier.0.bias']
    if 'classifier.3.weight' in pre_dict:
        del pre_dict['classifier.3.weight']
    if 'classifier.3.bias' in pre_dict:
        del pre_dict['classifier.3.bias']

    # 移除CPCA层和自注意力层可能存在的权重
    for k in list(pre_dict.keys()):
        if 'cpca' in k or 'self_attn' in k:
            del pre_dict[k]

    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    print(f"加载预训练权重，缺失层: {missing_keys}, 多余层: {unexpected_keys}")

    # 冻结特征提取层（只训练分类头、CPCA层和自注意力层）
    for name, param in net.named_parameters():
        if 'cpca' not in name and 'self_attn' not in name and 'classifier' not in name:
            param.requires_grad = False

    # 解冻最后几层特征层、CPCA层和自注意力层以提高性能
    for name, param in net.named_parameters():
        if 'features.12' in name or 'features.13' in name or 'features.14' in name or \
                'cpca' in name or 'self_attn' in name:
            param.requires_grad = True

    # 打印可训练参数
    trainable_params = [name for name, param in net.named_parameters() if param.requires_grad]
    print(f"可训练参数: {trainable_params}")

    net.to(device)

    # 训练配置
    loss_function = nn.CrossEntropyLoss()

    # 为不同模块设置不同的学习率
    params = [
        {'params': [p for n, p in net.named_parameters() if 'classifier' in n], 'lr': 0.0001},
        {'params': [p for n, p in net.named_parameters() if 'cpca' in n or 'self_attn' in n], 'lr': 0.0002},
        {'params': [p for n, p in net.named_parameters() if
                    'features.12' in n or 'features.13' in n or 'features.14' in n], 'lr': 0.00005}
    ]
    optimizer = optim.Adam(params, lr=0.0001)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # 开始训练
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        # 训练阶段
        net.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"训练 epoch {epoch + 1}/{epochs}", file=sys.stdout)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练准确率
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_correct / train_total:.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total

        # 验证和测试阶段
        net.eval()
        val_acc = evaluate(net, val_loader, device)
        test_acc = evaluate(net, test_loader, device)

        # 记录学习率
        lr_classifier = optimizer.param_groups[0]['lr']
        lr_cpca_sa = optimizer.param_groups[1]['lr']
        lr_features = optimizer.param_groups[2]['lr']
        learning_rates = [lr_classifier, lr_cpca_sa, lr_features]

        # 记录训练数据到CSV
        log_training_data(log_file, epoch + 1, train_loss, train_acc, val_acc, test_acc, learning_rates)

        # 打印训练信息
        print(f"Epoch {epoch + 1}/{epochs} | 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
              f"验证准确率: {val_acc:.4f} | 测试准确率: {test_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_model_path = os.path.join(save_dir, f"best_model_epoch_{best_epoch}.pth")
            torch.save(net.state_dict(), best_model_path)
            print(f"保存新最佳模型 (Epoch {best_epoch}) - 验证准确率: {val_acc:.4f} | 测试准确率: {test_acc:.4f}")

            # 尝试创建符号链接（Windows需要管理员权限，失败时给出提示）
            best_model_symlink = os.path.join(save_dir, "best_model.pth")
            try:
                if os.path.exists(best_model_symlink):
                    os.remove(best_model_symlink)
                os.symlink(os.path.basename(best_model_path), best_model_symlink)
                print(f"已创建符号链接到最佳模型: {best_model_symlink}")
            except OSError as e:
                print(f"创建符号链接失败: {e}，将直接使用文件路径")

        # 更新学习率
        scheduler.step(val_acc)
        new_lr_classifier = optimizer.param_groups[0]['lr']
        if new_lr_classifier != lr_classifier:
            print(f"学习率调整: Classifier从 {lr_classifier:.6f} 到 {new_lr_classifier:.6f}")

        # 保存每个epoch的模型
        epoch_model_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
            'test_accuracy': test_acc
        }, epoch_model_path)

    # 训练完成
    print(f"训练完成！最佳模型在 Epoch {best_epoch} | 验证准确率: {best_val_acc:.4f} | 测试准确率: {best_test_acc:.4f}")

    # 保存训练摘要
    with open(os.path.join(experiment_dir, "training_summary.txt"), "w") as f:
        f.write(f"实验目录: {experiment_dir}\n")
        f.write(f"总轮次: {epochs}\n")
        f.write(f"最佳轮次: {best_epoch}\n")
        f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
        f.write(f"对应测试准确率: {best_test_acc:.4f}\n")
        f.write(f"\n训练配置:\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"优化器: Adam\n")
        f.write(f"启用CPCA的层: {cpca_layers}\n")
        f.write(f"启用自注意力的层: {self_attn_layers}\n")

    # 在测试集上评估最佳模型（使用保存的最佳模型）
    best_model = mobilenet_v3_large_with_attentions(
        num_classes=len(class_indices),
        cpca_layers=cpca_layers,
        self_attn_layers=self_attn_layers
    )
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)
    best_model.eval()
    final_test_acc = evaluate(best_model, test_loader, device)
    print(f"最终测试准确率: {final_test_acc:.4f}")


# 评估函数
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 创建日志文件
def create_log_file(log_file):
    columns = ['Epoch', 'Train_Loss', 'Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy',
               'LR_Classifier', 'LR_CPCA_SA', 'LR_Features']
    df = pd.DataFrame(columns=columns)
    df.to_csv(log_file, index=False)


# 记录训练数据
def log_training_data(log_file, epoch, train_loss, train_acc, val_acc, test_acc, learning_rates):
    # 确保有3个学习率值
    lr_values = learning_rates + [None] * (3 - len(learning_rates))

    data = {
        'Epoch': [epoch],
        'Train_Loss': [train_loss],
        'Train_Accuracy': [train_acc],
        'Val_Accuracy': [val_acc],
        'Test_Accuracy': [test_acc],
        'LR_Classifier': [lr_values[0]],
        'LR_CPCA_SA': [lr_values[1]],
        'LR_Features': [lr_values[2]]
    }

    df = pd.DataFrame(data)
    df.to_csv(log_file, mode='a', header=False, index=False)


if __name__ == '__main__':
    main()