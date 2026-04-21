import torch
from random import randint, uniform, choice, gauss, sample
from Modulation_types import sliding, constant, stagger, dwell_switch, agile, uniform_jitter, Guassian_jitter
import os
import shutil
import numpy as np
import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # 优先搜索根目录
from utils import load_defaults_config, create_argparser

# 每个序列长度为60——80之间，随机生成。

def add_noise(data, type, mean=None, var=None):
    if var == None:
        if type == "pri":
            # var = 4
            var = 0
            mean = 0
        if type == "pf":
            # var = 15
            var = 0
            mean = 0
        if type == "pw":
            # var = 0.15
            var=0
            mean = 0
    data = data + torch.randn_like(data) * var + mean
    return data


def word_1_15(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    pri_seq, pri_seq_mean, pri_seq_var = sliding(seq_len, min_priIni=420, max_priIni=473, min_value=420, max_value=710,
                                                R=1.5)

    pf_seq, pf_seq_mean, pf_seq_var = agile(seq_len, min_value=3100, max_value=3450)

    tmp = randint(1, 2)
    if tmp == 1:
        central_value = 6.0
    else:
        central_value = 10.0
    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=central_value)
    
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")


    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (
                    pri_seq_mean - 420.0) / 970.0, pri_seq_var  # feature_scaling
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var
    
    

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_16(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    pri_seq, pri_seq_mean, pri_seq_var = stagger(seq_len, n=3, SBase=[880, 850, 940])

    pf_seq, pf_seq_mean, pf_seq_var = constant(seq_len, central_value=3423)

    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=15.1)
    
    if if_add_noise:
        
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_17(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    pri_seq, pri_seq_mean, pri_seq_var = stagger(seq_len, n=3, SBase=[846, 783, 723])

    pf_seq, pf_seq_mean, pf_seq_var = constant(seq_len, central_value=3365)

    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=6)
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_18(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    pri_seq, pri_seq_mean, pri_seq_var = stagger(seq_len, n=3, SBase=[1060, 1150, 1100])

    pf_seq, pf_seq_mean, pf_seq_var = constant(seq_len, central_value=3236)

    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=29.4)
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_19(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    pri_seq, pri_seq_mean, pri_seq_var = stagger(seq_len, n=3, SBase=[1200, 1250, 1310])

    pf_seq, pf_seq_mean, pf_seq_var = constant(seq_len, central_value=3248)

    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=55.1)
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_20(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    sbase_len = randint(3, 8)
    SBase = sample([665, 722, 722, 651, 502, 530, 427, 439], sbase_len)
    pri_seq, pri_seq_mean, pri_seq_var = stagger(seq_len, n=sbase_len, SBase=SBase)

    pf_seq, pf_seq_mean, pf_seq_var = constant(seq_len, central_value=3348)

    idx = randint(1, 3)
    if idx == 1:
        central_value = 6.0
    elif idx == 2:
        central_value = 10.0
    elif idx == 3:
        central_value = 15.1
    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=central_value)
    
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_21(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    sbase_len = randint(3, 7)
    SBase = sample([1198, 1090, 1057, 949, 1111, 1111, 1066], sbase_len)
    pri_seq, pri_seq_mean, pri_seq_var = stagger(seq_len, n=sbase_len, SBase=SBase)

    pf_seq, pf_seq_mean, pf_seq_var = constant(seq_len, central_value=3300)

    idx = randint(1, 2)
    if idx == 1:
        central_value = 6
    elif idx == 2:
        central_value = 15.1
    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=central_value)
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_22_28(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    sbase_len = randint(2, 4)
    SBase = sample([[745, 4], [825, 4], [1276, 4], [1342, 4], [625, 8], [575, 8], [525, 8]], sbase_len)
    transposed_SBase = [list(x) for x in zip(*SBase)]  # 转置
    pri_seq, pri_seq_mean, pri_seq_var = dwell_switch(seq_len, cands=transposed_SBase[0],
                                                    repeat_len=transposed_SBase[1])

    rf_list = [3380, 3300, 3221, 3185, 3435, 3465, 3495]
    rf_idx = randint(0, 6)
    pf_seq, pf_seq_mean, pf_seq_var = constant(seq_len, central_value=rf_list[rf_idx])

    pw_list = [10, 15.1, 29.4, 55.1, 6, 6, 6]
    pw_idx = randint(0, 6)
    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=pw_list[pw_idx])
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_29(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)
    pri_seq, pri_seq_mean, pri_seq_var = constant(seq_len, central_value=955)

    pf_seq, pf_seq_mean, pf_seq_var = dwell_switch(seq_len, cands=[3220, 3360, 3320], repeat_len=[12, 12, 12])

    pw_seq, pw_seq_mean, pw_seq_var = dwell_switch(seq_len, cands=[10, 15.1, 29.4], repeat_len=[12, 12, 12])
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_30(if_add_noise=False,seq_len=None, if_normalize=False):
    if seq_len==None:
        seq_len = randint(60, 80)

    pri_seq, pri_seq_mean, pri_seq_var = Guassian_jitter(seq_len, mean=1300, var=30)

    pf_seq, pf_seq_mean, pf_seq_var = agile(seq_len, min_value=3085, max_value=3115)

    idx = randint(1, 2)
    if idx == 1:
        central_value = 6
    elif idx == 2:
        central_value = 10
    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=central_value)
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    if if_normalize:
        pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
        pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
        pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_31(if_add_noise=False,seq_len=None):
    if seq_len==None:
        seq_len = randint(60, 80)

    pri_seq, pri_seq_mean, pri_seq_var = Guassian_jitter(seq_len, mean=580, var=30)

    pf_seq, pf_seq_mean, pf_seq_var = agile(seq_len, min_value=3385, max_value=3415)

    idx = randint(1, 2)
    if idx == 1:
        central_value = 6
    elif idx == 2:
        central_value = 10
    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=central_value)
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
    pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
    pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var


def word_32(if_add_noise=False,seq_len=None):
    if seq_len==None:
        seq_len = randint(60, 80)

    pri_seq, pri_seq_mean, pri_seq_var = Guassian_jitter(seq_len, mean=700, var=30)

    pf_seq, pf_seq_mean, pf_seq_var = agile(seq_len, min_value=3285, max_value=3315)

    idx = randint(1, 2)
    if idx == 1:
        central_value = 10
    elif idx == 2:
        central_value = 6
    pw_seq, pw_seq_mean, pw_seq_var = constant(seq_len, central_value=central_value)
    if if_add_noise:
        pri_seq = add_noise(pri_seq, "pri")
        pf_seq = add_noise(pf_seq, "pf")
        pw_seq = add_noise(pw_seq, "pw")

    pri_seq, pri_seq_mean, pri_seq_var = (pri_seq - 420.0) / 970.0, (pri_seq_mean - 420.0) / 970.0, pri_seq_var
    pf_seq, pf_seq_mean, pf_seq_var = (pf_seq - 3085.0) / 410.0, (pf_seq_mean - 3085.0) / 410.0, pf_seq_var
    pw_seq, pw_seq_mean, pw_seq_var = (pw_seq - 6.0) / 49.1, (pw_seq_mean - 6.0) / 49.1, pw_seq_var

    MFR_word_a = torch.cat((pri_seq.unsqueeze(-1), pf_seq.unsqueeze(-1)), dim=1)
    MFR_word = torch.cat((MFR_word_a, pw_seq.unsqueeze(-1)), dim=-1)

    MFR_word_c = torch.cat((pri_seq_mean.unsqueeze(-1), pf_seq_mean.unsqueeze(-1)), dim=1)
    MFR_word_mean = torch.cat((MFR_word_c, pw_seq_mean.unsqueeze(-1)), dim=-1)

    MFR_word_e = torch.cat((pri_seq_var.unsqueeze(-1), pf_seq_var.unsqueeze(-1)), dim=-1)
    MFR_word_var = torch.cat((MFR_word_e, pw_seq_var.unsqueeze(-1)), dim=-1)

    return MFR_word, MFR_word_mean, MFR_word_var

def de_normalize(seq):
    seq[:,0]=(seq[:,0]*970)+420
    seq[:,1]=(seq[:,1]*410)+3085
    seq[:,2]=(seq[:,2]*49.1)+6
    return seq

def de_normalize_and_standard(seq):
    seq[:,0]=(seq[:,0]*970)+420
    a=torch.mean(seq[:,0])
    b=torch.std(seq[:,0])
    seq[:,0]=(seq[:,0]-torch.mean(seq[:,0]))/torch.std(seq[:,0])
    
    seq[:,1]=(seq[:,1]*410)+3085
    seq[:,2]=(seq[:,2]*49.1)+6
    return seq


def quantize_tensor_by_interval(
    tensor, 
    interval=100, 
    min_val=10, 
    max_val=1000, 
    return_type="index"  # "index"返回区间索引，"value"返回区间代表值
):
    """
    对tensor按指定间隔做量化处理
    :param tensor: 输入的torch.Tensor/ndarray，数值分布在min_val~max_val
    :param interval: 量化间隔，默认100
    :param min_val: 数据最小值，默认10
    :param max_val: 数据最大值，默认1000
    :param return_type: 输出类型，"index"返回区间索引，"value"返回区间代表值
    :return: 量化后的tensor
    """
    # 统一转换为torch.Tensor（兼容ndarray输入）
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError("输入必须是torch.Tensor或numpy.ndarray")
    
    # 步骤1：将数值归一化到以interval为步长的区间（先减去最小值，避免偏移）
    normalized = tensor - min_val  # 把最小值10转为0，最大值1000转为990
    
    # 步骤2：计算所属区间（向下取整）
    quantized_index = torch.floor(normalized / interval).long()
    
    # 边界处理：确保索引不超出范围（避免max_val刚好等于区间上限时越界）
    max_index = (max_val - min_val) // interval  # 990//100=9
    quantized_index = torch.clamp(quantized_index, 0, max_index)
    
    if return_type == "index":
        return quantized_index
    elif return_type == "value":
        # 将索引还原为区间代表值（区间下限）
        quantized_value = quantized_index * interval + min_val
        return quantized_value
    else:
        raise ValueError("return_type只能是'index'或'value'")

def discrete_process(seq, interval=1):
    if interval==1:
        seq=seq.round()
    return seq






# 将雷达字的数据集按照调制类型分类，确保每种调制类型的数据数量尽可能的相同。
# 分为了4类，每类60000个，一共24w数据。
if __name__ == "__main__":

    args = create_argparser().parse_args()


    num_of_all_seqs = args.num_seqs
    if_add_noise=False
    seq_len=args.ground_truth_seq_length

    seqs_list=[]
    
    for i in range(num_of_all_seqs):
        # word 1 15
        # for j in range(15):
        # seq, seq_mean, seq_var = word_1_15()
        # torch.save({'seq': seq, 'seq_mean': seq_mean, 'seq_var': seq_var}, f'word1_15_{i*15+j}.pt')
        word_num=random.randint(12,29)
        if (word_num>15 and word_num<=21) or word_num==29:
            word_function_name='word_'+str(word_num)
        elif word_num<=15:
            word_function_name='word_1_15'
        else:
            word_function_name='word_22_28'
        word_function=globals().get(word_function_name)
        seq, _, _ = word_function(if_add_noise, seq_len=seq_len)
        # seq=de_normalize(seq)
        # seq=discrete_process(seq)
        # seqs_list.append(seq)
        pri = seq[:,0]
        torch.save({'seq': pri}, f'word1_{word_num}_{i}.pt')

    # seqs=torch.concatenate(seqs_list,dim=0)
    # mean,var, max_value, min_value=torch.mean(seqs[:,0]), torch.var(seqs[:,0]), torch.max(seqs[:,0]), torch.min(seqs[:,0])

    # print(mean)
    # print(var)
    # print(max_value)
    # print(min_value)
    
    current_dir = os.getcwd()
    
    basic_path='dataset_3W'
    data_dir = os.path.join(current_dir, os.path.join(basic_path,'Ground_Truth'))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # np.savez(data_dir+'/data_stastics',mean=mean,var=var,max_value=max_value,min_value=min_value)
    
    for filename in os.listdir(current_dir):
        if filename.endswith(('.pt')) and os.path.isfile(filename):
            shutil.move(filename, os.path.join(data_dir, filename))
    
    # val_dir = os.path.join(current_dir, os.path.join(basic_path,'Ground_Truth/val_dataset'))
    # if not os.path.exists(val_dir):
    #     os.makedirs(val_dir)

    # for filename in os.listdir(current_dir):
    #     if filename.endswith(('80.pt', '81.pt', '82.pt', '83.pt', '84.pt', '85.pt', '86.pt', '87.pt', '88.pt',
    #                           '89.pt')) and os.path.isfile(filename):
    #         shutil.move(filename, os.path.join(val_dir, filename))

    # current_dir = os.getcwd()

    # test_dir = os.path.join(current_dir, os.path.join(basic_path,'Ground_Truth/test_dataset'))
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)

    # for filename in os.listdir(current_dir):
    #     if filename.endswith(('90.pt', '91.pt', '92.pt', '93.pt', '94.pt', '95.pt', '96.pt', '97.pt', '98.pt',
    #                           '99.pt')) and os.path.isfile(filename):
    #         shutil.move(filename, os.path.join(test_dir, filename))

    # current_dir = os.getcwd()

    # training_dir = os.path.join(current_dir, os.path.join(basic_path,'Ground_Truth/training_dataset'))
    # if not os.path.exists(training_dir):
    #     os.makedirs(training_dir)

    # for filename in os.listdir(current_dir):
    #     if filename.endswith(('.pt')) and os.path.isfile(filename):
    #         shutil.move(filename, os.path.join(training_dir, filename))

    # # 原始文件夹路径
    # test_dataset_dir = os.path.join(basic_path,'Ground_Truth/test_dataset/')

    # # 目标文件夹路径
    # show_dataset_dir = os.path.join(basic_path,'Ground_Truth/show_dataset/')

    # if not os.path.exists(show_dataset_dir):
    #     os.makedirs(show_dataset_dir)

    # # 遍历原始文件夹中的文件
    # for filename in os.listdir(test_dataset_dir):
    #     # 检查文件是否以PRI开头并且以_1.pt结尾
    #     if filename.endswith('_96.pt'):
    #         # 构建目标文件路径
    #         destination = os.path.join(show_dataset_dir, filename)
    #         # 复制文件
    #         shutil.copy2(os.path.join(test_dataset_dir, filename), destination)
    #         print(f"Successfully copied {filename} to {destination}.")