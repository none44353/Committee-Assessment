import profile
import math
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing


n = 8 # the number of judgers
m = 2 # the number of players
rnd = 30 # 迭代次数
yita = 0.05
fstPrice = 100
profileA, profileT, profileS, profileW, profileP = [], [], [], [], []
Bias = (-0.1, 0, 0.1)
cases = []
numT = [0, 0]

def initprofileA():
    global profileA
    for b0 in Bias:
        for b1 in Bias:
            profileA.append((b0, b1))
    return

def initprofileS(profileA):
    global profileS, profileW
    for i in range(0, m):
        profileS.append([1/len(profileA)] * len(profileA))

    profileW = [[1] * len(profileA), [1] * len(profileA)]
    return

def getCases(n, len):
    cases = []
    for x in range(0, n+1):
        cases.append([x])

    for k in range(1, len):
        newcases = []
        for x in cases:
            sum = 0
            for e in x:
                sum = sum + e
            for i in range(0, n - sum + 1):
                newcases.append(x + [i])
        cases = newcases
    
    newcases = []
    for x in cases:
        sum = 0
        for i in x:
            sum = sum + i
        if sum == n:
            newcases.append(x)
            
    return newcases

def setJudgerType():
    global profileT
    #profileT = [0] * math.floor(n/2) + [1] * math.ceil(n/2)
    profileT = [0] * 1 + [1] * (n-1)

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
def setPlayerPerformance(): 
    global profileP
    profileP = [0.0,-0.15]
    #for i in range(0, 2):
    #    profileP.append(random.random())
    return 
        
def getProb(n, Strategy, cases):
    #print(n, Strategy, cases)
    prob = []
    newcases = []
    for x in cases:
        sum = n
        value = 1
        for i in range(0, len(Strategy)):
            p = Strategy[i]
            value = value * (p ** x[i]) * math.comb(sum, x[i])
            sum = sum - x[i]
        if value != 0:
            newcases.append(x)
            prob.append(value) 

    return newcases, prob

def merge(c1, c2, p1, p2):
    cases, prob = [], []
    for i in range(0, len(c1)):
        x, p = c1[i], p1[i]
        for j in range(0, len(c2)):
            y, q = c2[j], p2[j]
            z = (np.array(x) + np.array(y)).tolist()
            cases.append(z)
            prob.append(p * q)
    return cases, prob
      
def calcUtility(usrtype, others): # 返回一个list, 表示如果你的动作是A, 你的收益有多少
    global profileA, fstPrice, profileP
    utilities = []
    sum = [0, 0]
    for i in range(0, len(profileA)):
        a = profileA[i]
        for j in range(0, 2):
            sum[j] = sum[j] + a[j] * others[i]
    for i in range(0, len(profileA)):
        a = profileA[i]
        nsum = (np.array(sum) + np.array(a)).tolist()
        ui = fstPrice
        if (nsum[usrtype] + profileP[usrtype] > nsum[1 - usrtype] + profileP[1 - usrtype] ):
            ui = fstPrice
            #utilities.append(fstPrice)
        if (nsum[usrtype] + profileP[usrtype]  < nsum[1 - usrtype] + profileP[1 - usrtype] ):
            ui = 0
            #utilities.append(0)
        if (nsum[usrtype] + profileP[usrtype]  == nsum[1 - usrtype] + profileP[1 - usrtype] ):
            ui = fstPrice / 2
            #utilities.append(fstPrice / 2)
        ui = ui - (a[0]**2 + a[1]**2) * 1000
        utilities.append(ui)

    return utilities

def calcUtilityPunishAll(usrtype, others): # 返回一个list, 表示如果你的动作是A, 你的收益有多少  在当前框架下没法写
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
        if (nsum[usrtype] > nsum[1 - usrtype]):
            utilities.append(fstPrice)
        if (nsum[usrtype] < nsum[1 - usrtype]):
            utilities.append(0)
        if (nsum[usrtype] == nsum[1 - usrtype]):
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
        for event_i in range(0, len(all)):
            x, p = all[event_i], prob[event_i]
            price = (np.array(price) + p * np.array(calcUtility(i, x))).tolist()
        cost.append(price)

    return cost

def update():
    global numT, cases, profileS, profileW
    cost = getCost(numT, cases, profileS)
    for i in range(0, 2):
        profileW[i] = np.array(profileW[i]) * (1 + yita * np.array(cost[i]))
        profileS[i] = preprocessing.normalize([profileW[i]], norm="l1").tolist()[0]
        profileW[i] = profileS[i]
        #profileW[i] = profileW[i].tolist()
        #profileS[i] = profileS[i].tolist()[0]
        
    return

def printf(profile):
    for i in range(0,2):
        li = [round(x, 2) for x in profile[i]]
        print(li)
    return

def main():
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
    sns.heatmap(pd.DataFrame(np.array(profileS[1]).reshape(3,3), columns = Bias, index = Bias))
    plt.show()

# 设定裁判的偏好，奖项奖励和选手的performance，要求CCE
if __name__ == "__main__":
    main()