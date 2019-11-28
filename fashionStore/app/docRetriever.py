import csv
#from tqdm import tqdm
import os

MY_DRIVE = os.getcwd() + '/app' #'/content/drive/My Drive/Data Mining'

STYLE_WITH_DESC_N_TITLE = MY_DRIVE+'/styles_with_description_title.csv'

STYLE_WITH_DESC_N_TITLE_RANDOM = MY_DRIVE+'/styles_with_description_title_RandomSampled.csv'

DOC_ID_COL_IDX = 0

DISPLAY_NAME_COL_IDX = 9

DESCR_COL_IDX = 18


class ProductDoc():
    def __init__(self):
        self.id = None
        self.image = None
        self.displayName = None
        self.description = None

class DocRetriever():
    def __init__(self):
        pass

    def retrieveDocs(self, docIds):
        docs = []

        docIds=[str(x) for x in docIds]

        with open(STYLE_WITH_DESC_N_TITLE_RANDOM, 'r', encoding='latin-1') as csvfile:
            reader = csv.reader(csvfile)

            #print(docIds)

            for rowNo, row in enumerate(reader):
              #  if rowNo == 0:
               #     print(row[0])
                #print('\t ' + str(row[DOC_ID_COL_IDX]))
                if row[DOC_ID_COL_IDX] in docIds:
                    doc = ProductDoc()
                    doc.id = row[DOC_ID_COL_IDX]
                    doc.image = row[DOC_ID_COL_IDX]
                    doc.displayName = row[DISPLAY_NAME_COL_IDX]
                    doc.description = row[DESCR_COL_IDX]
                    doc.displayName = doc.displayName.replace('`', ',')
                    doc.description = doc.description.replace('`', ',')
                    docs.append(doc)
                    #break
        return docs



