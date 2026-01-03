import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model_v3_cpca_mha import mobilenet_v3_large_with_attentions
from tqdm import tqdm
import datetime  # 用于生成日志文件名


def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 读取类别映射
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), f"文件不存在: {json_path}"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # 创建类别到索引的反向映射
    class_to_idx = {v: int(k) for k, v in class_indict.items()}

    # 数据集路径 (请根据实际情况修改)
    dataset_path = r'D:\深度学习\deep-learning-for-image-processing-master\deep-learning-for-image-processing-master\data_set\论文中的数据小麦\小麦data_set\test'

    # 初始化统计变量
    class_correct = {cls: 0 for cls in class_indict.values()}
    class_total = {cls: 0 for cls in class_indict.values()}

    # 创建模型并加载权重
    model = mobilenet_v3_large_with_attentions(num_classes=5).to(device)
    model_weight_path = r"D:\深度学习\deep-learning-for-image-processing-master\deep-learning-for-image-processing-master\pytorch_classification\Test6_mobilenet\experiments\exp_20251012_100158\saved_models\best_model_epoch_29.pth"
    assert os.path.exists(model_weight_path), f"模型权重文件不存在: {model_weight_path}"
    # 加载权重
    checkpoint = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 设置模型为评估模式
    model.eval()

    # 创建日志文件（以当前时间命名，避免重复）
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "prediction_logs"
    os.makedirs(log_dir, exist_ok=True)  # 创建日志目录（如果不存在）
    log_file = os.path.join(log_dir, f"prediction_log_{current_time}.txt")

    # 获取所有类别文件夹
    class_folders = [f for f in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, f))]

    # 写入日志头部信息
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"===== 预测日志 - {current_time} =====\n")
        f.write(f"设备: {device}\n")
        f.write(f"模型路径: {model_weight_path}\n")
        f.write(f"测试集路径: {dataset_path}\n")
        f.write(f"类别映射: {class_indict}\n\n")
        f.write("===== 详细预测结果 =====\n")
        f.write(f"{'图片路径':<80} | {'正确标签':<10} | {'预测标签':<10} | {'是否正确'}\n")
        f.write("-" * 120 + "\n")

    # 遍历每个类别文件夹
    for class_folder in tqdm(class_folders, desc="处理类别"):
        if class_folder not in class_to_idx:
            continue  # 跳过不在类别映射中的文件夹

        class_path = os.path.join(dataset_path, class_folder)
        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 遍历该类别下的所有图片
        for img_file in tqdm(image_files, desc=f"处理 {class_folder}", leave=False):
            img_path = os.path.join(class_path, img_file)

            try:
                # 加载并预处理图片
                img = Image.open(img_path).convert('RGB')
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)

                # 预测
                with torch.no_grad():
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                # 获取真实类别和预测类别
                true_class = class_folder
                pred_class = class_indict[str(predict_cla)]
                is_correct = "正确" if true_class == pred_class else "错误"

                # 更新统计
                class_total[true_class] += 1
                if true_class == pred_class:
                    class_correct[true_class] += 1

                # 写入单张图片的预测结果到日志
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{img_path:<80} | {true_class:<10} | {pred_class:<10} | {is_correct}\n")

            except Exception as e:
                error_msg = f"处理图片 {img_path} 时出错: {str(e)}"
                print(error_msg)
                # 将错误信息写入日志
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{img_path:<80} | {'错误':<10} | {'错误':<10} | {error_msg}\n")
                continue

    # 写入准确率汇总到日志
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "-" * 120 + "\n")
        f.write("===== 各类别准确率汇总 =====\n")
        total_correct = 0
        total_samples = 0

        for cls in class_indict.values():
            if class_total[cls] > 0:
                acc = class_correct[cls] / class_total[cls]
                f.write(f"{cls}: 准确率 {acc:.4f} ({class_correct[cls]}/{class_total[cls]})\n")
                total_correct += class_correct[cls]
                total_samples += class_total[cls]

        # 计算总准确率
        if total_samples > 0:
            total_acc = total_correct / total_samples
            f.write(f"\n总准确率: {total_acc:.4f} ({total_correct}/{total_samples})\n")
        else:
            f.write("\n没有处理任何图片\n")

    print(f"\n预测完成，日志已保存至: {log_file}")


if __name__ == '__main__':
    main()