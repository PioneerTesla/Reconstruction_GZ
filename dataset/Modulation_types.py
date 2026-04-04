import torch
from random import randint, uniform, choice, gauss, sample
from math import ceil


####产生三个部分的序列：观测序列、均值序列和方差序列

def constant(Num, min_value=10, max_value=100, central_value=0, var=1):
    '''
    产生Num个固定值的MFR序列
    '''
    if central_value == 0:
        central_value = randint(min_value, max_value)

    # print('产生了固定PRI的MFR序列, 固定值为:{}, 序列长度为:{}'.format(central_value, Num))

    Seq = torch.full((1, Num), central_value)
    Seq = torch.squeeze(Seq)

    seq_mean = Seq.clone()
    seq_var = torch.full((1, Num), var)
    seq_var = torch.squeeze(seq_var)

    return Seq.float(), seq_mean.float(), seq_var.float()


def sliding(Num, min_priIni, max_priIni, min_value=-1, max_value=-1, R=-1, max_num_per_period=32, var=1):
    '''
    产生Num个线性取值的MFR序列
    '''
    In_or_de = randint(0, 1)

    while True:
        priIni = uniform(min_priIni, max_priIni)
        if R == -1: R = randint(2, 10)
        n = randint(ceil(R), max_num_per_period)
        if max_value == -1 and min_value == -1 and n < Num:
            break
        elif max_value != -1 and min_value == -1 and max_value >= priIni * R and n < Num:
            break
        elif min_value != -1 and max_value == -1 and min_value <= priIni and n < Num:
            break
        elif min_value != -1 and max_value != -1 and max_value >= priIni * R and min_value <= priIni and n < Num:
            break

    step = priIni * (R - 1) / n
    T = ceil(Num / n)
    k = torch.arange(0, n)

    if In_or_de == 0:
        # print("产生了线性增加PRI的MFR序列, priIni={}, R={}, n={}, step={}, T={}".format(priIni, R, n, step, T))
        yk = priIni + k * step

    else:
        # print("产生了线性减小PRI的MFR序列, priIni={}, R={}, n={}, step={}, T={}".format(priIni, R, n, step, T))
        yk = R * priIni - k * step

    seq = yk.repeat(T)[0: Num]

    seq_mean = seq.clone()

    seq_var = torch.full((1, Num), var)
    seq_var = torch.squeeze(seq_var)

    return seq.float(), seq_mean.float(), seq_var.float()


def stagger(Num, min_value=30, max_value=80, n=-1, SBase=[], var=1):
    if n == -1: n = randint(8, 16)

    if SBase == []:
        sBase = torch.tensor([randint(min_value, max_value) for _ in range(n)])
    else:
        sBase = torch.tensor(SBase)

    T = ceil(Num / n)

    # print("产生了PRI为参差变化的MFR序列, n={}, T={}".format(n, T))
    seq = sBase.repeat(T)[0: Num]
    # print(seq)
    seq_mean = seq.clone()

    seq_var = torch.full((1, Num), var)
    seq_var = torch.squeeze(seq_var)

    return seq.float(), seq_mean.float(), seq_var.float()


def uniform_jitter(Num, priUpperBound=-1, priLowerBound=-1):
    priIni = randint(100, 1000)
    priDev = uniform(0.05, 0.5)
    if priUpperBound == -1:
        priUpperBound = priIni + priIni * priDev
    if priLowerBound == -1:
        priLowerBound = priIni - priIni * priDev

    seq = torch.tensor([uniform(priLowerBound, priUpperBound) for _ in range(Num)])
    seq_mean = torch.full((1, Num), priLowerBound)
    seq_mean = torch.squeeze(seq_mean)

    seq_var = torch.full((1, Num), priUpperBound)
    seq_var = torch.squeeze(seq_var)

    return seq.float(), seq_mean.float(), seq_var.float()


def Guassian_jitter(Num, mean=-1, var=-1):
    priIni, maxDev = mean, var

    if mean == -1:
        priIni = randint(30, 60)

    if var == -1:
        priDev = uniform(0.05, 0.2)
        maxDev = priIni * priDev

    # print("产生了PRI为高斯抖动的MFR序列, 均值为{}, 方差为{}".format(priIni, maxDev))

    while True:
        seq = torch.randn(Num) * (maxDev ** 0.5) + priIni
        if min(seq) > 0:
            break
    # seq = torch.tensor([gauss(priIni, maxDev) for _ in range(Num)])

    seq_mean = torch.full((1, Num), priIni)
    seq_var = torch.full((1, Num), maxDev)

    seq_mean, seq_var = torch.squeeze(seq_mean), torch.squeeze(seq_var)

    return seq.float(), seq_mean.float(), seq_var.float()


def agile(Num, min_value=100, max_value=1000, var=0):
    n = randint(4, 8)

    sBase = [randint(min_value, max_value) for _ in range(n)]

    T = ceil(Num / n)

    seq = torch.tensor(sBase)
    for _ in range(T - 1):
        seq_part = torch.tensor(sample(sBase, n))
        seq = torch.cat((seq, seq_part), dim=0)

    seq = seq[:Num]
    seq_mean = torch.tensor(sBase).repeat(T)[:Num]
    seq_var = torch.full((1, Num), var)
    seq_var = torch.squeeze(seq_var)
    return seq.float(), seq_mean.float(), seq_var.float()


def dwell_switch(Num, cands=[], repeat_len=[], var=1):
    if cands == []:
        number_of_cands = randint(2, 16)
        cands = [uniform(100, 1000) for _ in range(number_of_cands)]

    if repeat_len == []:
        while True:
            repeat_len = [randint(8, 30) for _ in range(number_of_cands)]
            if max(repeat_len) <= 3 * min(repeat_len): break

    sBase = torch.tensor([cands[i] for i in range(len(cands)) for _ in range(repeat_len[i])])
    T = ceil(Num / len(sBase))

    seq = sBase.repeat(T)[: Num]
    seq_mean = seq.clone()
    seq_var = torch.full((1, Num), var)
    seq_var = torch.squeeze(seq_var)

    return seq.float(), seq_mean.float(), seq_var.float()