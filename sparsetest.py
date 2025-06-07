import torch
import torchsparse
from torchsparse import SparseTensor
import torchsparse.nn as spnn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import argparse

# 定义点云数据集
class PointCloudDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.loadtxt(file_path)
        xyz = data[:, :3]
        if data.shape[1] > 3:  # 如果有标签列
            label = data[:, 3].astype(int)
        else:
            label = np.zeros(xyz.shape[0], dtype=int)  # 没有标签时创建虚拟标签
        
        # 添加批量索引列（全为0）
        batch_index = np.zeros((xyz.shape[0], 1), dtype=np.int32)
        coords = np.hstack([batch_index, xyz])
        
        # 将点云数据转换为稀疏张量所需的坐标和特征
        coords = torch.from_numpy(coords).int()
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float32)  # 简单起见，使用全1特征
        labels = torch.from_numpy(label).long()
        
        return (coords, feats), labels

# 定义网络模型
class SparseSegmentationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = spnn.Conv3d(1, 16, kernel_size=3)
        self.conv2 = spnn.Conv3d(16, 32, kernel_size=3)
        self.conv3 = spnn.Conv3d(32, 64, kernel_size=3)
        self.mlp = torch.nn.Linear(64, 3)  # 3个语义类别
    
    def forward(self, inputs):
        coords, feats = inputs
        x = SparseTensor(feats, coords)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 将稀疏张量转换为密集张量以进行全连接层处理
        x = x.dense().squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.mlp(x)
        
        return x

# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for batch_idx, ((coords, feats), labels) in enumerate(dataloader):
        coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
        
        # 前向传播
        outputs = model((coords, feats))
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Train Batch {batch_idx}, Loss: {loss.item()}')

# 预测函数
def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for (coords, feats), _ in dataloader:
            coords, feats = coords.to(device), feats.to(device)
            outputs = model((coords, feats))
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu().numpy())
    return np.concatenate(predictions)

# 主函数
def parse_args():
    parser = argparse.ArgumentParser(description='Sparse Point Cloud Segmentation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], help='Mode: train or predict')
    parser.add_argument('--model_path', type=str, help='Path to the trained model .pth file (required for predict mode)')
    parser.add_argument('--predict_file', type=str, help='Path to the txt file for prediction (only xyz, required for predict mode)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # 数据加载
    if args.mode == 'train':
        dataset = PointCloudDataset('data/train')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # 模型、损失函数和优化器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SparseSegmentationNet().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练
        for epoch in range(5):  # 默认5个epoch
            print(f'Epoch {epoch+1}')
            train(model, dataloader, criterion, optimizer, device)
        
        # 保存模型
        torch.save(model.state_dict(), 'trained_model.pth')
        print("Model saved as 'trained_model.pth'")
    
    elif args.mode == 'predict':
        if not args.model_path or not args.predict_file:
            print("For predict mode, both --model_path and --predict_file are required.")
            exit(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SparseSegmentationNet().to(device)
        model.load_state_dict(torch.load(args.model_path))
        print(f"Model loaded from {args.model_path}")
        
        # 准备预测数据
        data = np.loadtxt(args.predict_file)
        xyz = data[:, :3]
        # 创建虚拟标签（因为数据集中没有）
        labels = np.zeros(xyz.shape[0], dtype=int)
        
        # 添加批量索引列（全为0）
        batch_index = np.zeros((xyz.shape[0], 1), dtype=np.int32)
        coords = np.hstack([batch_index, xyz])
        
        # 将数据转换为张量
        coords = torch.from_numpy(coords).int()
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float32)
        labels = torch.from_numpy(labels).long()
        
        # 创建预测数据集和加载器
        predict_dataset = [( (coords, feats), labels )]
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        
        # 预测
        predictions = predict(model, predict_dataloader, device)
        print(predictions)
        np.savetxt('predictions.txt', predictions, fmt='%d')
        print("Predictions saved as 'predictions.txt'")