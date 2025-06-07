import numpy as np
import os

def generate_point_cloud(num_points, x_limit=50, y_limit=50, z_limit=15):
    # 在指定范围内生成点云数据
    x = np.random.uniform(-x_limit/2, x_limit/2, num_points)
    y = np.random.uniform(-y_limit/2, y_limit/2, num_points)
    z = np.random.uniform(0, z_limit, num_points)
    
    return np.column_stack((x, y, z))

def generate_labels(num_points, label_ratio=(0.4, 0.3, 0.3)):
    # 根据比例生成标签数据
    labels = np.random.choice([0, 1, 2], size=num_points, p=label_ratio)
    return labels

def save_point_cloud(data, file_path):
    # 保存点云数据到txt文件
    np.savetxt(file_path, data, fmt='%0.6f', delimiter=' ')

def generate_point_cloud_dataset(output_dir, num_files=8):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        # 随机生成1000到1200个点
        num_points = np.random.randint(1000, 1201)
        
        # 生成点云数据
        xyz = generate_point_cloud(num_points)
        
        # 生成标签数据（0、1、2三类，使用不同比例）
        labels = generate_labels(num_points)
        
        # 合并为完整的点云数据
        point_cloud = np.column_stack((xyz, labels))
        
        # 保存到文件
        file_path = os.path.join(output_dir, f'cloud_{i+1}.txt')
        save_point_cloud(point_cloud, file_path)
        print(f'已生成文件: {file_path}，包含 {num_points} 个点')

def generate_test_point_cloud(output_dir):
    # 创建测试集
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成1000个点的测试集（只有xyz，没有标签）
    num_points = 1000
    xyz = generate_point_cloud(num_points)
    
    # 保存测试集
    file_path = os.path.join(output_dir, 'test_cloud.txt')
    save_point_cloud(xyz, file_path)
    print(f'已生成测试文件: {file_path}，包含 {num_points} 个点')

def main():
    # 生成训练集
    print("生成训练数据集...")
    generate_point_cloud_dataset('data/train', num_files=8)
    
    # 生成测试集
    print("\n生成测试数据集...")
    generate_test_point_cloud('data/test')
    
    # 测试读取数据
    print("\n验证数据:")
    train_file = 'data/train/cloud_1.txt'
    test_file = 'data/test/test_cloud.txt'
    
    if os.path.exists(train_file):
        data = np.loadtxt(train_file)
        print(f"\n训练数据示例（前5行）：\n{data[:5]}")
        print(f"训练数据维度：{data.shape}")
        print(f"训练数据中各标签的数量：0 { -np.sum(data[:,3]==0)}, 1 - {np.sum(data[:,3]==1)}, 2 - {np.sum(data[:,3]==2)}")
    
    if os.path.exists(test_file):
        data = np.loadtxt(test_file)
        print(f"\n测试数据示例（前5行）：\n{data[:5]}")
        print(f"测试数据维度：{data.shape}")

if __name__ == '__main__':
    main()