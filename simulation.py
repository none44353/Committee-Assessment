import math
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing


n = 4 # the number of judgers
m = 8 # the number of players
rnd = 30 # 迭代次数
yita = 0.5
fstPrice = 100
profileA, profileT, profileS, profileW, profileP = [], [], [], [], []
Bias = (-0.1, 0, 0.1)
cases = []

def initprofileA():
    global profileA
    for b0 in Bias:
        for b1 in Bias:
            profileA.append((b0, b1))
    return

def initprofileS(profileA):
    global profileS, profileW
    for i in range(0, n):
        profileS.append([1/len(profileA)] * len(profileA))
        profileW.append([1] * len(profileA))
    return

def setJudgerType():
    global profileT #profileT是裁判的type列表，其中每个element对应一个裁判的偏好选手列表
    profileT = [0] * n 

    l = math.floor(n/4)
    #for i in range(0, math.floor(n/2)):
    #    profileT[i] = [0,1,2,3]
    #for i in range(math.floor(n/2), n):
    #    profileT[i] = [4,5,6,7]
    for i in range(0, l):
        profileT[i] = [0,1,2,3]
    for i in range(l, 2 * l):
        profileT[i] = [4,5,6,7]
    for i in range(2 * l, 3 * l):
        profileT[i] = [1,3,5,7]
    for i in range(3 * l, n):
        profileT[i] = [0,2,4,6]
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
def setPlayerPerformance(): 
    global profileP
    profileP = [0.05, 0, 0, 0, -0.05, 0, 0, 0]
    #for i in range(0, m):
    #    profileP.append(random.random())
    return 
      
def noPunish(act, pivot, others):
    return 0
    return (act[0]**2 + act[1]**2) * 1000

def sgn(x):
    if x == 0: return 0
    if x < 0: return -1
    return 1

def getPairs(JA, JB, JC, all):
    pairs = []
    for j in range(0, m):
        sA = all[JA][0] if j in profileT[JA] else all[JA][1]
        sB = all[JB][0] if j in profileT[JB] else all[JB][1]
        sC = all[JC][0] if j in profileT[JB] else all[JB][1]
        pairs.append((sgn(sA - sC), sgn(sB - sC)))
    return pairs   

#pairs是一些联合分布的抽样 形如[(x_i, y_i), ...] 其中-1 <= xi,yi <= 1
def getDMI(pairs): 
    DMI2 = []
    random.shuffle(pairs)
    M1, M2 = np.zeros((3, 3)), np.zeros((3, 3))
    for i in range(0, len(pairs) // 2):
        x, y = int(pairs[i][0]), int(pairs[i][1])
        M1[x][y] = M1[x][y] + 1
    M1 /= (len(pairs) // 2)

    for i in range(len(pairs) // 2, len(pairs)):
        x, y = int(pairs[i][0]), int(pairs[i][1])
        M2[x][y] = M2[x][y] + 1
    M2 /= (len(pairs) - len(pairs) // 2)
    
    DMI2.append(np.linalg.det(M1) * np.linalg.det(M2))
    #print(np.mean(np.array(DMI2)))
    return np.mean(np.array(DMI2))

def allPunish(act, pivot, others):
    all = others
    all[pivot] = act

    score, total = 0, 0
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                if i != k and i != j and j < k:
                    score = score + getDMI(getPairs(i, j, k, all))
                    total = total + 1

    return score / total

def personalPunish(act, pivot, others):
    all = others
    all[pivot] = act

    score, total = 0, 0
    for i in range(0, n):
        for j in range(0, n):
            if i < j:
                score = score + getDMI(getPairs(pivot, i, j, all))
                total = total + 1

    return score / total

punishType = "no"
Punishment = {"no": noPunish, "all": allPunish, "personal": personalPunish}

def calcUtility(pivot, others): # 返回一个list, 表示如果你的动作是A, 你的收益有多少
    global m, fstPrice, profileA, profileP, profileT

    utilities = []
    sum = [0] * m
    for i in range(0, n):
        if i != pivot:
            act = others[i]
            addition = [act[0] if j in profileT[pivot] else act[1] for j in range(0, m)]
            sum = (np.array(sum) + np.array(addition)).tolist()
            #for j in range(0, m):
            #    if j in profileT[i]: # player j是Judger i偏好的对象
            #        sum[j] = sum[j] + act[0]
            #    else:
            #        sum[j] = sum[j] + act[1]

    for act in profileA:
        addition = [act[0] if j in profileT[pivot] else act[1] for j in range(0, m)]
        nsum = (np.array(sum) + np.array(addition)).tolist()

        mxValue, mxNum = nsum[0] + profileP[0], 1
        for j in range(1, m):
            if nsum[j] + profileP[j] > mxValue:
                mxValue = nsum[j] + profileP[j]
                mxNum = 1
            elif nsum[j] + profileP[j] == mxValue:
                mxNum = mxNum + 1
        ui = 0
        for j in profileT[pivot]:
            if nsum[j] + profileP[j] == mxValue:
                ui = ui + fstPrice/mxNum
        ui = ui - Punishment[punishType](act, pivot, others)
        utilities.append(ui)
    return utilities

def getProb(profileS, pivot):
    global n, profileA
    cases, prob = [[]], [1]
    for i in range(0, n):
        Strategy = profileS[i]
        newcases, newProb = [], []
        if i == pivot:
            for event, p in zip(cases, prob):
                newcases.append(event + [-1])
                newProb.append(p)
        else:
            for event, p in zip(cases, prob):
                for act, pi in zip(profileA, Strategy):
                    newcases.append(event + [act])
                    newProb.append(p * pi)
        cases = newcases
        prob = newProb
    return cases, prob

def getCost(profileS):
    global n, profileA
    cost = []
    for i in range(0, n):
        all, prob = getProb(profileS, i)
        price = [0] * len(profileA)
        for x, p in zip(all, prob):
            price = (np.array(price) + p * np.array(calcUtility(i, x))).tolist()
        cost.append(price)
    return cost

def update():
    global profileS, profileW
    cost = getCost(profileS)
    for i in range(0, n):
        profileW[i] = np.array(profileW[i]) * (1 + yita * np.array(cost[i]))
        profileS[i] = preprocessing.normalize([profileW[i]], norm="l1").tolist()[0]
        profileW[i] = profileS[i]
        
    return

def printf(profile):
    for i in range(0,n):
        li = [round(x, 2) for x in profile[i]]
        print(li)
    return

def main():
    global punishType
    punishType = input("Input punishment type (i.e. 'no', 'all', 'personal'):")
    setPlayerPerformance()
    initprofileA()
    initprofileS(profileA) 
    setJudgerType()
    print(profileA)
    for i in range(0, rnd):
        print("Update ",i)
        update()
        printf(profileS)

    plt.subplot(1,2,1)
    sns.heatmap(pd.DataFrame(np.array(profileS[0]).reshape(3,3), columns = Bias, index = Bias))
    plt.subplot(1,2,2)
    sns.heatmap(pd.DataFrame(np.array(profileS[n - 1]).reshape(3,3), columns = Bias, index = Bias))
    plt.show()

# 设定裁判的偏好，奖项奖励和选手的performance，要求CCE
if __name__ == "__main__":
    main()