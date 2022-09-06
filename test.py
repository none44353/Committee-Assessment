import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import palettable #python颜色库
from sklearn import datasets 


plt.rcParams['font.sans-serif']=['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于显示中文

iris=datasets.load_iris()
x, y = iris.data, iris.target
pd_iris = pd.DataFrame(np.hstack((x, y.reshape(150, 1))),columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)','class'] )

plt.figure(dpi=200, figsize=(10,6))
data1 = np.array(pd_iris['sepal length(cm)']).reshape(25,6)#Series转np.array
print(data1)

df = pd.DataFrame(data1, 
                index=[chr(i) for i in range(65, 90)],#DataFrame的行标签设置为大写字母
                columns=["a","b","c","d","e","f"])#设置DataFrame的列标签
plt.figure(dpi=120)
sns.heatmap(data=df,#矩阵数据集，数据的index和columns分别为heatmap的y轴方向和x轴方向标签               
)
plt.title('所有参数默认')
plt.show()