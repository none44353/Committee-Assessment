import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

TIMES = 100
data = {
    "Judger A": [],
    "Judger B": [],
    "Judger C": [],
    "NOC Code": [], 
    "Result": [], 
}

'''Nationality = ["","ROC", "JPN", "KOR", "USA", 
               "GEO", "CZE", "GER", "AZE", "BLR", 
               "SUI", "AUT", "CAN", "EST", "NED", 
               "BUL", "POL", "FIN", "SWE", "CHN", 
               "GBR", "AUS", "UKR", "BEL"]
'''
Nationality = [""]

def sgn(x):
    if x == 0: return 0
    if x < 0: return -1
    return 1

#Execution_Score_Table用的
def calcExecutionPairs(inxA, inxB, inxC, dataFrame, NAT):
    pairs = []
    for i in range(0, len(dataFrame)):
        if len(NAT) == 0 or dataFrame["Code"][i] == NAT:
            pairs.append((sgn(dataFrame[inxA][i] - dataFrame[inxC][i]), sgn(dataFrame[inxB][i] - dataFrame[inxC][i])))
    return pairs

def getPairs(JA, JB, JC, dataFrame, NAT, keyword):
    pairs = []
    for i in range(0, len(dataFrame)):
        if dataFrame["Entry"][i] == keyword:
            if len(NAT) == 0 or dataFrame["Code"][i] == NAT:
                pairs.append((sgn(dataFrame[JA][i] - dataFrame[JC][i]), sgn(dataFrame[JB][i] - dataFrame[JC][i])))
    return pairs    

def getMatrix(pairs):
    M = np.zeros((3,3))
    for i in range(0, len(pairs)):
        x, y = int(pairs[i][0] + 1), int(pairs[i][1] + 1)
        M[x][y] = M[x][y] + 1
    return M

def showMatrix(M, X_index, Y_index, title):
    df = pd.DataFrame(M, index=X_index, columns=Y_index)   
    sns.heatmap(data=df, cmap=sns.cubehelix_palette(as_cmap=True))#渐变色盘：sns.cubehelix_palette()使用)            
    plt.title(title)          
    plt.show()


#pairs是一些联合分布的抽样 形如[(x_i, y_i), ...] 其中-1 <= xi,yi <= 1
def getDMI(pairs): 
    DMI2 = []
    for t in range(0, TIMES):
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
    return np.mean(np.array(DMI2))

def main():
    dataFrame = pd.read_csv("Table.csv")
    keyword = "Interpretation of the Music"
    M = getMatrix(getPairs("J1", "J2", "J3", dataFrame, "", keyword))
    print(getDMI(getPairs("J1", "J2", "J3", dataFrame, "", keyword)))
    print(M)
    showMatrix(M, [-1, 0, 1], [-1, 0, 1], "The joint distribution for the ratings difference of "+keyword)

    keywords = ["Execution Score", "Skating Skills", "Transitions", "Performance", "Composition", "Interpretation of the Music"]
    results = []
    for keyword in keywords:
        results.append(getDMI(getPairs("J1", "J2", "J3", dataFrame, "", keyword)))
    print(keywords)
    print(results)
    '''
    pairs = []
    NAT = ""
    for i in range(0, len(dataFrame)):
        if len(NAT) == 0 or dataFrame["Code"][i] == NAT:
            pairs.append((sgn(dataFrame["J1"][i] - dataFrame["J3"][i]), sgn(dataFrame["J2"][i] - dataFrame["J3"][i])))
            #if pairs[i][0] * pairs[i][1] > 0:
            print(pairs[i][0], pairs[i][1], dataFrame["Code"][i])
    M = np.zeros((3,3))
    for i in range(0, len(pairs)):
        x, y = int(pairs[i][0]), int(pairs[i][1])
        M[x][y] = M[x][y] + 1
    print(M)
    '''

    '''
    每3个裁判比较，算DMI
    for NAT in Nationality:
        for A in range(1, 10):
            for B in range(1, 10):
                if A != B:
                    for C in range(1, 10):
                        if A != C and B != C:
                            val = calc(f'J{A}', f'J{B}', f'J{C}', dataFrame, NAT)
                            data["Judger A"].append(f'J{A}')
                            data["Judger B"].append(f'J{B}')
                            data["Judger C"].append(f'J{C}')
                            data["NOC Code"].append(NAT)
                            data["Result"].append(val)
                            print(f'J{A} J{B} J{C} Code = #{NAT}# %.5e\n' % (val))
    pd.DataFrame(data).to_csv("results.csv")
    '''

if __name__ == "__main__":
    main()