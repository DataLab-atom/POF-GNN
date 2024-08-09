import torch


def fair_metric_gpu(output, labels, sens, idx):
    # 确保所有的张量都在同一个设备上，这里假设是在 CUDA 设备上
    #print("output",output[:20])
    device = output.device
    labels = labels.to(device)
    sens = sens.to(device)
    idx = idx.to(device)
    #print(idx)
    val_y = labels[idx]
    idx_s0 = (sens[idx] == 0)
    idx_s1 = (sens[idx] > 0)
    idx_s0_y1 = torch.bitwise_and(idx_s0, val_y > 0)
    idx_s1_y1 = torch.bitwise_and(idx_s1, val_y > 0)
    #idx_s0_y1 = idx_s0 & (val_y > 0)
    #idx_s1_y1 = idx_s1 & (val_y > 0)

    pred_y = (output[idx].squeeze() > 0.5).type_as(labels)
    #pred_y = torch.argmax(output[idx], dim=1).type_as(labels)
    # 使用 torch.sum 和逻辑索引来计算
    sum_idx_s0 = torch.sum(idx_s0).float()
    sum_idx_s1 = torch.sum(idx_s1).float()
    sum_idx_s0_y1 = torch.sum(idx_s0_y1).float()
    sum_idx_s1_y1 = torch.sum(idx_s1_y1).float()

    # 避免除以零
    if sum_idx_s0 == 0 or sum_idx_s1 == 0 or sum_idx_s0_y1 == 0 or sum_idx_s1_y1 == 0:
        return float('nan'), float('nan')
    #print("a",torch.sum(pred_y[idx_s0]) / sum_idx_s0)
    #print("b",torch.sum(pred_y[idx_s1]) / sum_idx_s1)
    parity = torch.abs(torch.sum(pred_y[idx_s0]) / sum_idx_s0 - torch.sum(pred_y[idx_s1]) / sum_idx_s1)
    equality = torch.abs(torch.sum(pred_y[idx_s0_y1]) / sum_idx_s0_y1 - torch.sum(pred_y[idx_s1_y1]) / sum_idx_s1_y1)
    return parity, equality


if __name__ == '__main__':
    # 示例用法
    output = torch.rand(100, device='cuda')  # 模拟输出
    labels = torch.randint(0, 2, (100,), device='cuda')  # 模拟标签
    sens = torch.randint(0, 2, (100,), device='cuda')  # 模拟敏感特征
    idx = torch.arange(100, device='cuda')  # 模拟索引

    parity, equality = fair_metric_gpu(output, labels, sens, idx)
    print("Demographic Parity:", parity)
    print("Equal Opportunity:", equality)