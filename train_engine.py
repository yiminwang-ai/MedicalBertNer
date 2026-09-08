import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


# 定义绘图函数
def draw_fig(losses, figname:str, ylabel:str, save_pth:str):
    save_pth = os.path.join(save_pth, figname)
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label=ylabel)
    plt.title(figname)
    plt.xlabel('Batch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    plt.savefig(save_pth+'.png', format='png')  # 保存图像文件
    plt.show()
    plt.close()  # 关闭当前图形，以释放内存
    # 将损失保存到CSV文件
    losses_df = pd.DataFrame(losses, columns=[ylabel])
    losses_df.to_csv(save_pth+'.csv', index=False)  # 保存为CSV文件，不包括索引


def train(model: torch.nn.Module, train_loader, val_loader, train_dataset, model_save_pth, epoch_num=100, is_crf=False):
    global new_folder_path, epoch

    # 模型保存路径检查
    if not os.path.exists(model_save_pth):
        os.makedirs(model_save_pth)
        print(f"创建文件夹: {model_save_pth}")

    # 获取子文件夹数量
    def get_subfolder_count(folder_path):
        return len([f.path for f in os.scandir(folder_path) if f.is_dir()])

    # 检查子文件夹数量并创建新文件夹
    subfolder_count = get_subfolder_count(model_save_pth)
    new_folder_path = os.path.join(model_save_pth, f'{subfolder_count}')
    os.makedirs(new_folder_path, exist_ok=True)

    #训练参数设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # lr with warm-up and cosine
    # 计算总步数，用于学习率调度
    total_steps = epoch_num * 100
    # 预热步数
    num_warmup_steps = epoch_num // 10
    print(f"warm-up={num_warmup_steps}")
    def lr_lambda(step):
        # 确保 step 是有效的
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        else:
            if total_steps == num_warmup_steps:
                return 0.0  # 防止除零错误

            # 将计算转换为Tensor
            step_tensor = torch.tensor(step, dtype=torch.float32)  # 转换为Tensor
            total_steps_tensor = torch.tensor(total_steps, dtype=torch.float32)

            cos_value = 0.5 * (1 + torch.cos(
                (step_tensor - num_warmup_steps) / (total_steps_tensor - num_warmup_steps) * 3.141592653589793))
            return cos_value.item()  # 返回Python标量

    scheduler = LambdaLR(optimizer, lr_lambda)
    # 早停参数
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    losses = []
    precisions = []
    recalls = []
    F1s = []
    val_losses = []
    model.train()

    for epoch in range(epoch_num):
        model.train()
        print(f'第{epoch + 1}轮')
        batch_losses = []
        for batch in tqdm(train_loader):
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss if not is_crf else outputs
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()  # 更新学习率
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

        # 计算平均损失
        epoch_loss = np.mean(batch_losses)
        losses.append(epoch_loss)
        print(f'第{epoch + 1}轮, Train Average Loss: {epoch_loss}')

        # 验证集评估
        val_loss, precision, recall, f1 = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        precisions.append(precision)
        recalls.append(recalls)
        F1s.append(f1)

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 保存最佳模型
            torch.save(model.state_dict(), f'{new_folder_path}/best_model.pth')
            print(f'Model successfully saved to {new_folder_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered.')
                break
    # 将超参数写入 TXT 文件
    hyperparameters_file_path = f'{new_folder_path}/hyperparameters.txt'
    with open(hyperparameters_file_path, 'w') as f:
        f.write(f'epoch={epoch},scheduler={str(scheduler)}\n')
        f.write(str(model))
    print(f'Info successfully saved to {hyperparameters_file_path}')
    # 绘制损失曲线
    draw_fig(losses, figname='train_loss', ylabel='loss', save_pth=new_folder_path)
    draw_fig(val_losses, figname='val_loss', ylabel='loss', save_pth=new_folder_path)
    draw_fig(precisions, figname='Precision', ylabel='P', save_pth=new_folder_path)
    draw_fig(recalls, figname='Recall', ylabel='R', save_pth=new_folder_path)
    draw_fig(F1s, figname='F1 Score', ylabel='F1', save_pth=new_folder_path)
    # 保存模型和tokenizer
    train_dataset.tokenizer.save_pretrained(new_folder_path)
    print("tokenizer已成功保存到:", new_folder_path)


from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            # 是labels
            outputs, loss = model(evaluate=True, **batch)
            #print(outputs)
            val_loss += loss.item()

            all_preds.extend(np.asarray(outputs).flatten())
            all_labels.extend(batch['labels'].cpu().numpy().flatten())  # 确保这里的 labels 在 batch 中

    avg_val_loss = val_loss / len(val_loader)

    # 计算 P, R, F1
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Validation Loss: {avg_val_loss}, Precision: {precision}, Recall: {recall}, F1: {f1}')

    return avg_val_loss, precision, recall, f1
