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

def main():
    path = open("material.pdf",'rb')
    Text = pdf2txt(path,"material.txt")
    #labels = list(filter(lambda x: len(x) > 0, (open('material.txt', encoding="utf-8").read().split('\n'))))
    labels = list(filter(lambda x: len(x) > 0, Text.split('\n')))
    for item in volume:
        if isNumbericItem(labels, item):
            data[item] = getNumbericData(labels, item, volume[item])
        else:
            data[item] = getStrData(labels, item, volume[item])
        
        if volume[item] == 1:
            data[item] = [x for x in data[item] for i in range(7)]
    dataFrame = pd.DataFrame(data)
    dataFrame.to_csv('Table.csv')

if __name__=="__main__":
    main()