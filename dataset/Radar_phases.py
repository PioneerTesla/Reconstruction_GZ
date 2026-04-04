import torch
from random import randint
import os
import shutil

#Search model
S = [[9, 10, 11], #G1G2G3
     [12, 13, 14], #G4G5G6
     [14, 15, 16], #G6G7G8
     [17, 18, 19], #H1H2H3
     [20, 21, 22], #H4H5H6
     [23, 24, 25], #H7H8H9
     [26, 27, 28], #H10H11H12
     [29, 30, 31]] #H13H14H15
     

#Blocking interference detection model
D = [[6, 23, 23], #D1H7H7
     [7, 24, 24], #D2H8H8
     [8, 25, 25]] #D3H9H9

#Tracking model
T = [[9, 0, 0, 0], #G1C1C1C1
     [10, 1, 1, 1], #G2C2C2C2
     [11, 2, 2, 2], #G3C3C3C3
     [12, 3, 3, 3], #G4C4C4C4
     [13, 4, 4, 4], #G5C5C5C5
     [14, 5, 5, 5]] #G6C6C6C6

#Guidance model
G = [[17, 0, 0], #H1C1C1
     [18, 1, 1], #H2C2C2
     [19, 2, 2], #H3C3C3
     [20, 3, 3], #H4C4C4
     [21, 4, 4], #H5C5C5
     [22, 5, 5]] #H6C6C6

S_num, D_num, T_num, G_num = len(S), len(D), len(T), len(G)

num = 0
num_of_dataset = 50000

tmp_list = []
while True:   
    
    #添加搜索模式雷达短语
    num_of_search_model = randint(5, 13)  #搜索模式重复5——13个雷达短语
    for i in range(num_of_search_model):
        idx_of_search_phase = randint(0, S_num-1)
        tmp_list = tmp_list + S[idx_of_search_phase]
    
    #有20%的可能啥也没搜到，接着搜
    random_number = randint(1, 10)
    if random_number <= 2:    
        if len(tmp_list) > 80 and len(tmp_list) <= 100: #判断，如果序列太长就直接保存，不再添加雷达短语
            tmp_tensor = torch.tensor(tmp_list)
            torch.save(tmp_tensor, f'MFR_phase_{num}.pt')  
            tmp_list = []
            num += 1
            if num == num_of_dataset:
                break
        continue    #否则重新搜索
    
    #添加跟踪模式雷达短语
    num_of_tracking_model = randint(2, 8) #跟踪模式重复2-8个雷达短语
    for i in range(num_of_tracking_model):
        idx_of_tracking_phase = randint(0, T_num-1)
        tmp_list = tmp_list + T[idx_of_tracking_phase]
    
    #有20%的概率跟踪丢失目标，需要重新搜索
    random_number = randint(1, 10)
    if random_number <= 2:
        if len(tmp_list) > 60 and len(tmp_list) <= 100: #判断，如果序列太长就直接保存，不再添加雷达短语
            tmp_tensor = torch.tensor(tmp_list)
            torch.save(tmp_tensor, f'MFR_phase_{num}.pt')  
            tmp_list = []
            num += 1
            if num == num_of_dataset:
                break
        continue #丢失目标，此时tmp_list不清空，但是要重新执行搜索模式
    
    #添加探测模式雷达短语
    num_of_detection_model = randint(2, 5) #探测模式重复2-5个雷达短语
    for i in range(num_of_detection_model):
        idx_of_detection_phase = randint(0, D_num-1)
        tmp_list = tmp_list + D[idx_of_detection_phase]
    
    #有10%的概率跟踪丢失目标，需要重新搜索
    random_number = randint(1, 10)
    if random_number == 1:
        if len(tmp_list) > 60 and len(tmp_list) <= 100: #判断，如果序列太长就直接保存，不再添加雷达短语
            tmp_tensor = torch.tensor(tmp_list)
            torch.save(tmp_tensor, f'MFR_phase_{num}.pt')    
            tmp_list = []
            num += 1
            if num == num_of_dataset:
                break
        continue #丢失目标，此时tmp_list不清空，但是要重新执行搜索模式
    
    #添加制导模型雷达短语
    num_of_guidance_model = randint(1, 3) #探测模式重复1-3个雷达短语
    for i in range(num_of_guidance_model):
        idx_of_guidance_phase = randint(0, G_num-1)
        tmp_list = tmp_list + G[idx_of_guidance_phase]
    
    if len(tmp_list) >= 60 and len(tmp_list) <= 100: #判断，如果序列太长就直接保存，不再添加雷达短语
        tmp_tensor = torch.tensor(tmp_list)
        torch.save(tmp_tensor, f'MFR_phase_{num}.pt')  
        tmp_list = []
        num += 1
        if num == num_of_dataset:
            break
    
    elif len(tmp_list) > 100: #判断，如果序列太长超过100，就不要这段序列了，保证数据集的雷达短语长度在60——100之间。
        tmp_list = []

current_dir = os.getcwd()

val_dir = os.path.join(current_dir, 'MFR_phase/val_dataset')
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

for filename in os.listdir(current_dir):
    if filename.endswith(('80.pt', '81.pt', '82.pt', '83.pt', '84.pt', '85.pt', '86.pt', '87.pt', '88.pt', '89.pt')) and os.path.isfile(filename):
        shutil.move(filename, os.path.join(val_dir, filename))
        
current_dir = os.getcwd()

test_dir = os.path.join(current_dir, 'MFR_phase/test_dataset')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

for filename in os.listdir(current_dir):
    if filename.endswith(('90.pt', '91.pt', '92.pt', '93.pt', '94.pt', '95.pt', '96.pt', '97.pt', '98.pt', '99.pt')) and os.path.isfile(filename):
        shutil.move(filename, os.path.join(test_dir, filename))
        

current_dir = os.getcwd()

training_dir = os.path.join(current_dir, 'MFR_phase/training_dataset')
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

for filename in os.listdir(current_dir):
    if filename.endswith(('.pt')) and os.path.isfile(filename):
        shutil.move(filename, os.path.join(training_dir, filename))


# 原始文件夹路径
test_dataset_dir = 'MFR_phase/test_dataset/'

# 目标文件夹路径
show_dataset_dir = 'MFR_phase/show_dataset/'

if not os.path.exists(show_dataset_dir):
    os.makedirs(show_dataset_dir)

# 遍历原始文件夹中的文件
for filename in os.listdir(test_dataset_dir):
    # 检查文件是否以PRI开头并且以.pt结尾
    if filename.endswith('.pt'):
        # 获取文件名的后两位或三位数字
        if (filename[-7: -3].isdigit() == False) and (filename[-6: -3].isdigit() == False):
            # 构建目标文件路径
            destination = os.path.join(show_dataset_dir, filename)
            # 复制文件
            shutil.copy2(os.path.join(test_dataset_dir, filename), destination)
            print(f"Successfully copied {filename} to {destination}.")
