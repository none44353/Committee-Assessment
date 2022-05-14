import numpy as np
import pandas as pd
import random

TIMES = 100
data = {
    "Judger A": [],
    "Judger B": [],
    "Judger C": [],
    "NOC Code": [], 
    "Result": [], 
}

Nationality = ["","ROC", "JPN", "KOR", "USA", 
               "GEO", "CZE", "GER", "AZE", "BLR", 
               "SUI", "AUT", "CAN", "EST", "NED", 
               "BUL", "POL", "FIN", "SWE", "CHN", 
               "GBR", "AUS", "UKR", "BEL"]

def sgn(x):
    if x == 0: return 0
    if x < 0: return -1
    return 1

def calc(inxA, inxB, inxC, dataFrame, NAT):
    pairs = []
    for i in range(0, len(dataFrame)):
        if len(NAT) == 0 or dataFrame["Code"][i] == NAT:
            pairs.append((sgn(dataFrame[inxA][i] - dataFrame[inxC][i]), sgn(dataFrame[inxB][i] - dataFrame[inxC][i])))
            
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

if __name__ == "__main__":
    main()