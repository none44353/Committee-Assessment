import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

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

def getSample(JA, JB, dataFrame, NATfilter, keyword):
    samples = []
    for i in range(0, len(dataFrame)):
        if keyword == "" or data["Entry"][i] == keyword:
            if (NATfilter(dataFrame["Code"][i])):
                samples.append(dataFrame[JA][i] - dataFrame[JB][i])
    return samples

def getFrequencyFromSample(x, samples):
    y = np.zeros(len(x))
    for k in samples:
        for i in range(0, len(x)):
            if x[i] == k: y[i] = y[i] + 1
        
    y = y / len(samples)
    return y.tolist()
                

def main():
    dataFrame = pd.read_csv("Execution_Score_Table.csv")
    delta1 = getSample("J1", "J2", dataFrame, lambda x: x == "ROC", "")
    delta2 = getSample("J1", "J2", dataFrame, lambda x: x != "ROC", "")
    
    xList = list(set(delta1 + delta2))
    xList.sort()
    y1 = getFrequencyFromSample(xList, delta1)
    y2 = getFrequencyFromSample(xList, delta2)

    bar_width = 0.3
    index1 = np.arange(len(xList))
    index2 = index1 + bar_width
    plt.bar(index1, height=y1, width=bar_width, color='b', label='ROC')
    plt.bar(index2, height=y2, width=bar_width, color='r', label='!ROC')
    plt.legend()
    plt.xticks(index1 + bar_width/2, xList)
    plt.ylabel('frequency')
    plt.show()
    


if __name__ == "__main__":
    main()