import numpy as np
import pandas as pd
import sys
import importlib
importlib.reload(sys)

def pdf2txt(file_path, text_name):
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LTTextBoxHorizontal, LAParams
    from pdfminer.pdfpage import PDFTextExtractionNotAllowed

    Parser = PDFParser(file_path) # claim a parser
    Document = PDFDocument(Parser) # claim a document
    if not Document.is_extractable: # check if it is extractable
        raise PDFTextExtractionNotAllowed
    else:
        ResourceManager = PDFResourceManager()
        Params = LAParams() # set parameters
        Device = PDFPageAggregator(ResourceManager, laparams=Params)
        Interpreter = PDFPageInterpreter(ResourceManager, Device)
        
        text = ""
        for page in PDFPage.create_pages(Document):
            Interpreter.process_page(page)
            layout = Device.get_result()
            for y in layout:
                if(isinstance(y,LTTextBoxHorizontal)):
                    text = text + y.get_text()+"\n"
        with open("%s"%(text_name),'w',encoding="utf-8") as f:
            f.write(text)
        return text

volume = {
    "# Executed Elements": 7, 
    "Value": 7,
    "GOE": 7,
    "J1": 7,
    "J2": 7,
    "J3": 7,
    "J4": 7,
    "J5": 7,
    "J6": 7,
    "J7": 7,
    "J8": 7,
    "J9": 7,
    "Rank Name": 1, 
    "Code": 1,
}

data = {}
def is_number(s):
    try:
        float(s) # for int, long and float
    except ValueError:
        try:
            complex(s) # for complex
        except ValueError:
            return False
    return True

def isNumbericItem(texts, keyword):
    for i in range(0, len(texts) - 1):
        if texts[i].startswith(keyword): return is_number(texts[i + 1])

    return False

def getStrData(texts, keyword, volume):
    data = []
    for i in range(0, len(texts)):
        if texts[i].startswith(keyword):
            items = []
            for j in range(i + 1, len(texts)):
                if not texts[j].isspace():
                    items.append(texts[j])
                    if len(items) == volume: break
            data = data + items
    return data

def getNumbericData(texts, keyword, volume):
    data = []
    for i in range(0, len(texts)):
        text = texts[i]
        if text.startswith(keyword):
            items = []
            for j in range(i + 1, min(i + 1 + volume, len(texts))):
                word = texts[j].split(' ')[0]
                if not is_number(word):
                    if word == "-": # Some items are omitted
                        items.append(0)
                    else:
                        break
                else:
                    items.append(float(word))
            data = data + items 
            
    return data
    return np.array(data).reshape(-1, volume)

def Extractor1(labels):
    for item in volume:
        if isNumbericItem(labels, item):
            data[item] = getNumbericData(labels, item, volume[item])
        else:
            data[item] = getStrData(labels, item, volume[item])
        
        if volume[item] == 1:
            data[item] = [x for x in data[item] for i in range(7)]
    dataFrame = pd.DataFrame(data)
    dataFrame.to_csv('Execution_Score_Table.csv')
    return
    
def getProgramComponents(texts):
    keyword = "Factor"
    volume = 50
    entry = ["Skating Skills", "Transitions", "Performance", "Composition", "Interpretation of the Music"]
    
    dataList = [[], [], [], [], []]
    for i in range(0, len(texts)):
        text = texts[i]
        if text.startswith(keyword):
            List = []
            lenList = 0
            for j in range(i + 1, len(texts)):
                word = texts[j].split(' ')[0]
                if is_number(word):
                    if float(word) > 0:
                        List.append(float(word))
                        lenList = lenList + 1
                        if (lenList == volume): break
            
            for j in range(5, 50):
                dataList[j % 5].append(List[j])
    return entry, dataList

def Extractor2(labels):
    Extractor1(labels)
    ExecuteScore = {}
    
    n = 30
    items = ["Value", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9"]
    for item in items:
        x = np.array(data[item]).reshape(-1, 7)
        ExecuteScore[item] = np.sum(np.array(data[item]).reshape(-1, 7), axis = 1).tolist()
    for i in range(1, 10):
        item = f'J{i}'
        a = ExecuteScore[item]
        b = ExecuteScore["Value"]
        ExecuteScore[item] = [a[i] + b[i] for i in range(0, len(a))]
    #现在执行分的每个组分组都有30个小分，对应30个选手的成绩[Juger给的ExecuteScore(BaseVale+GOE)]

    #对艺术分也要做类似处理，最后列成比较靠谱的表格
    entries, dataList = getProgramComponents(labels)
    PCs, index = [], 0
    for e in entries:
        dict = {}
        for i in range(1, 10): dict[f'J{i}'] = []
        for i in range(0, len(dataList[index])):
            item = f'J{i % 9 + 1}'
            dict[item].append(dataList[index][i])
        index = index + 1
        PCs.append(dict)
    print(PCs[0]["J1"])

    #"Rank Name", "Code", "Entry", "J1", ..., "J9"
    name = getStrData(labels, "Rank Name", volume["Rank Name"])
    code = getStrData(labels, "Code", volume["Code"])
    #tableData Init
    tableData = {}
    tableData["Rank Name"], tableData["Code"], tableData["Entry"] = [], [], []
    for i in range(1, 10): tableData[f'J{i}'] = []
    for i in range(0, len(name)):
        #每个选手有6行，第一行是执行分，2-6行是表现分
        tableData["Rank Name"].append(name[i])
        tableData["Code"].append(code[i])
        tableData["Entry"].append("Execution Score")
        for j in range(1, 10):
            tableData[f"J{j}"].append(ExecuteScore[f"J{j}"][i])

        for k in range(0, 5):        
            tableData["Rank Name"].append(name[i])
            tableData["Code"].append(code[i])
            tableData["Entry"].append(entries[k])
            for j in range(1, 10):
                tableData[f"J{j}"].append(PCs[k][f"J{j}"][i])
    
    dataFrame = pd.DataFrame(tableData)
    dataFrame.to_csv('Table.csv')

    return

def main():
    #path = open("material.pdf",'rb')
    #Text = pdf2txt(path,"material.txt")
    #labels = list(filter(lambda x: len(x) > 0, Text.split('\n')))
    labels = list(filter(lambda x: len(x) > 0, (open('material.txt', encoding="utf-8").read().split('\n'))))
    Extractor2(labels)    


if __name__=="__main__":
    main()