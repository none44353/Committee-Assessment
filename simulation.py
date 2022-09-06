import profile
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing


n = 6 # the number of judgers
m = 2 # the number of players
rnd = 100 # 迭代次数
yita = 5
fstPrice = 100
profileA, profileT, profileS = [], [], []
Bias = (-0.1, 0, 0.1)
cases = []
numT = [0, 0]

def initprofileA():
    global profileA
    for b0 in Bias:
        for b1 in Bias:
            profileA.append((b0, b1))
    return

def initStrategy(profileA):
    global profileS
    for i in range(0, m):
        profileS.append([1/len(profileA)] * len(profileA))
    return

def getCases(n, len):
    cases = ()
    for x in range(0, n+1):
        cases.append([x])
    for i in (1, len):
        newcases = ()
        for x in cases:
            sum = 0
            for e in x:
                sum = sum + e
            for i in (0, n - sum + 1):
                newcases.append(list(x + [i]))
        cases = newcases
    
    newcases = ()
    for x in cases:
        sum = 0
        for i in x:
            sum = sum + i
        if sum == n:
            newcases.append(x)
    return newcases

def setJudgerType():
    global profileT
    profileT = [0] * math.floor(n/2) + [1] * math.ceil(n/2)

    global numT
    for i in profileT:
        numT[i] = numT[i] + 1
    
    global cases, profileA
    cases = [0, 0, 0, 0]
    for i in range(0, 2):
        cases[i] = getCases(numT[i], len(profileA))
        cases[i + 2] = getCases(numT[i] - 1, len(profileA))

    return 

#假设选手的表现是[0, 1]之间的实数
#裁判的策略空间是S(p, t) = p + bias_t 其中bias_t \in[-1, +1]
#这里因为CCE只能处理离散的决策空间的情形，我们让bias_t实际上取到[-.2, -.1, 0, .1, .2]
#裁判的最终给分会被round到[0, 0.2, 0.4, 0.6, 0.8, 1]其中的一个，这里有截断的问题
#考虑到规模问题，我们直接认定偏好相同的裁判会采用一致的策略，但是他们没有一个共同信号
#   也就是说，已知type1的裁判的策略是(p1=0.2,p2=0.3,p3=0.5), type2的裁判策略是(p1=0.1, p2=0.3, p3=0.6)
#   计算的时候，应该先枚举情况: 选择p1的裁判人数为a, 选择p2的裁判人数为b，选择p3的裁判人数为c(a+b+c=n-1)这种情况对应概率
#   conditioning on这种情况，可以计算每个action的punishment
#   计算punishment之后再更新strategic
#
def PlayerPerformance(): 
    profileP = ()
    for i in range(0, 2):
        profileP.append(random.random)
    return profileP
        
def getProb(n, Strategy, cases):
    prob = []
    newcases = []
    for x in cases:
        sum = n
        value = 1
        for a in (0, len(Strategy)):
            p = Strategy[a]
            value = value * (p ** x[a]) * math.comb(sum, x[a])
            sum = sum - x[a]
        if value != 0:
            newcases.append(x)
            prob.append(value) 

    return newcases, prob

def merge(c1, c2, p1, p2):
    cases, prob = [], []
    for i in range(0, len(c1)):
        x, p = c1[i], p1[i]
        for j in range(0, len(c2)):
            y, q = c2[i], p2[i]
            z = (np.array(x) + np.array(y)).tolist()
            cases.append(z)
            prob.append(p * q)
    return cases, prob
      
def calcUtility(type, others): # 返回一个list, 表示如果你的动作是A, 你的收益有多少
    global profileA, fstPrice
    utilities = []
    sum = [0, 0]
    for i in range(0, len(profileA)):
        a = profileA[i]
        for j in range(0, 2):
            sum[j] = sum[j] + a[j] * others[i]
        
    for i in range(0, len(profileA)):
        a = profileA[i]
        nsum = (np.array(sum) + np.array(a)).tolist()
        if (nsum[type] > nsum[1 - type]):
            utilities.append(fstPrice)
        if (nsum[type] < nsum[1 - type]):
            utilities.append(0)
        if (nsum[type] == nsum[1 - type]):
            utilities.append(fstPrice / 2)

    return utilities

def getCost(numT, cases, profileS):
    global profileA
    ntype = len(numT)
    cost = []
    for i in range(0, ntype):
        c1, p1 = getProb(numT[i] - 1, profileS[i], cases[i + ntype])
        c2, p2 = getProb(numT[1 - i], profileS[1 - i], cases[1 - i])
        all, prob = merge(c1, c2, p1, p2)

        price = [0] * len(profileA)
        for i in range(0, len(all)):
            x, p = all[i], prob[i]
            price = (np.array(price) + p * np.array(calcUtility(i, x))).tolist()
        cost.append(price)

    return cost

def update():
    global numT, cases, profileS, profileW
    cost = getCost(numT, cases, profileS)
    for i in range(0, 2):
        profileW[i] = np.array(profileW[i]) * (1 + yita * np.array(cost[i]))
        profileS[i] = preprocessing.normalize(profileW[i])
        profileW[i] = profileW[i].tolist()
        profileS[i] = profileS[i].tolist()
    return

def main():
    initprofileA()
    initprofileS(profileA) 
    setJudgerType()
    profileW = list([1] * n, [1] * n)
    for i in range(0, rnd)：
        profileW = update(profileW)
    print(profileT)

# 设定裁判的偏好，奖项奖励和选手的performance，要求CCE
if __name__ == "__main__":
    main()