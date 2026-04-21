import os
import torch
import math
import random
import numpy as np
import torch.nn.functional as F
import shutil
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # 优先搜索根目录
from utils import load_defaults_config, create_argparser

args = create_argparser().parse_args()

# ==================== 用户配置 ====================
# 处理模式: 'Miss' / 'Spurious' / 'Mix'
PROCESS_NAME='Miss'
# 数据集根目录: 'dataset'(定长) / 'dataset_random_len'(变长60~80)
DATASET_ROOT='dataset_3W'
# ==================================================

SEQUENCE_LENGTH=args.ground_truth_seq_length
MISS_RATE=args.miss_process_ratio
SPURIOUS_RATE=args.suprious_process_ratio_min
NOISE_STD_RATIO=0.02  # 高斯噪声标准差占PRI均值的比例

def miss_process(pri, miss_rate_list, seq_length=None):
    if seq_length is None:
        seq_length = len(pri)
    
    pri_start=torch.tensor([0])
    restored_pri = torch.cat([pri_start, pri_start + torch.cumsum(pri,  dim=0)])
    
    miss_rate=random.sample(miss_rate_list,k=1)[0]
    
    miss_pulse_num=math.floor(seq_length*(miss_rate+random.uniform(-0.05,0.05)))
    miss_pulse_num=min(miss_pulse_num, seq_length - 1)  # 至少保留1个脉冲
    miss_pulse_num=max(miss_pulse_num, 0)
    
    pos=list(np.arange(0, seq_length))
    miss_pos=random.sample(pos, miss_pulse_num)
    miss_pos.sort(reverse=True)
    
    pri_list=restored_pri.tolist()
    
    for p in miss_pos:
        del pri_list[p]
    
    miss_pri=torch.diff(torch.tensor(pri_list))

    return miss_pri, miss_pos

def spurious_process(pri, spurious_rate_list, seq_length=None):
    if seq_length is None:
        seq_length = len(pri)
    
    pri_start=torch.tensor([0])
    restored_pri = torch.cat([pri_start, pri_start + torch.cumsum(pri,  dim=0)])
    
    spurious_rate=random.sample(spurious_rate_list,k=1)[0]
    
    spurious_pulse_num=math.floor(seq_length*(spurious_rate+random.uniform(-0.05,0.05)))
    spurious_pulse_num = max(spurious_pulse_num, 1)
    
    start_time = restored_pri[0].item()
    end_time = restored_pri[-1].item()
    
    # 均匀分布生成虚假脉冲TOA
    false_toas = np.random.uniform(start_time, end_time, spurious_pulse_num)
    
    # 合并并排序
    new_toa_seq = np.concatenate([restored_pri.numpy(), false_toas])
    new_toa_seq.sort()
    new_pri_seq = np.diff(new_toa_seq)
    # 去掉可能出现的0间隔
    new_pri_seq = new_pri_seq[new_pri_seq > 0]
    
    return torch.tensor(new_pri_seq, dtype=torch.float32)


def gaussian_noise_process(pri, noise_std_ratio=0.02):
    """对PRI序列添加高斯噪声。
    noise_std_ratio: 噪声标准差占PRI均值的比例。
    """
    pri_float = pri.float()
    mean_pri = pri_float.mean()
    noise_std = mean_pri * noise_std_ratio
    noise = torch.randn_like(pri_float) * noise_std
    noisy_pri = pri_float + noise
    # 确保PRI值为正
    noisy_pri = torch.clamp(noisy_pri, min=1.0)
    return noisy_pri


def mix_process(pri, spurious_rate_list, miss_rate_list, noise_std_ratio=0.02, seq_length=None):
    """混合场景：缺失观测 + 虚假脉冲 + 高斯噪声"""
    if seq_length is None:
        seq_length = len(pri)
    # Step 1: 缺失观测
    miss_pri, miss_pos = miss_process(pri, miss_rate_list, seq_length=seq_length)
    # Step 2: 虚假脉冲（用缺失后的实际长度）
    mix_pri = spurious_process(miss_pri, spurious_rate_list, seq_length=len(miss_pri))
    # Step 3: 高斯噪声
    mix_pri = gaussian_noise_process(mix_pri, noise_std_ratio)
    return mix_pri



if __name__ == "__main__":
    dataset_path=os.path.join(os.getcwd(), DATASET_ROOT, 'Ground_Truth')
    processed_pri_dataset_path=os.path.join(os.getcwd(), DATASET_ROOT, PROCESS_NAME)
    
    if not os.path.exists(processed_pri_dataset_path):
        os.makedirs(processed_pri_dataset_path)

    file_list = [f for f in os.listdir(dataset_path) if f.endswith('.pt')]
    print(f'Processing {len(file_list)} files with mode: {PROCESS_NAME}')

    for i, file in enumerate(file_list):
        data = torch.load(os.path.join(dataset_path, file), weights_only=False)
        pri = data['seq']

        seq_len = len(pri)

        if PROCESS_NAME == 'Miss':
            processed_pri, miss_pos = miss_process(pri, miss_rate_list=MISS_RATE, seq_length=seq_len)
            save_dict = {'seq': processed_pri, 'error_pos': miss_pos}

        elif PROCESS_NAME == 'Spurious':
            processed_pri = spurious_process(pri, spurious_rate_list=SPURIOUS_RATE, seq_length=seq_len)
            save_dict = {'seq': processed_pri}

        elif PROCESS_NAME == 'Mix':
            processed_pri = mix_process(pri, spurious_rate_list=SPURIOUS_RATE,
                                        miss_rate_list=MISS_RATE,
                                        noise_std_ratio=NOISE_STD_RATIO,
                                        seq_length=seq_len)
            save_dict = {'seq': processed_pri}

        else:
            raise ValueError(f'Unknown PROCESS_NAME: {PROCESS_NAME}')

        torch.save(save_dict, os.path.join(processed_pri_dataset_path, file))

        if (i + 1) % 500 == 0 or (i + 1) == len(file_list):
            print(f'  [{i+1}/{len(file_list)}] done')

    print(f'All files saved to {processed_pri_dataset_path}')
    print(f'Dataset root: {DATASET_ROOT}, Process: {PROCESS_NAME}')



























