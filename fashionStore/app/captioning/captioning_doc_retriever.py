import csv
#from tqdm import tqdm
import os

file_path = os.path.dirname(os.path.realpath(__file__))

MY_DRIVE = file_path #os.getcwd() + '/app/captioning' #'/content/drive/My Drive/Data Mining'

DIBA_DRIVE = file_path #os.getcwd() + '/app/captioning' #'/content/drive/My Drive/Rubel/DM/img_captioning'
MY_DRIVE = DIBA_DRIVE

FLICKR_FILE_GEN = DIBA_DRIVE+'/captions_sampled_generated.csv'

FLICKR_INVERTED_IDX_FILE = DIBA_DRIVE+'/flickr_index_tf_idf_Random.dat'

COL_INDEX_ID = 0
COL_INDEX_DISPLAY_NAME = 9
COL_INDEX_DESC_TITLE = 13

COL_FLICKR_CAPTION = 3

class ProductDoc():
    def __init__(self):
        self.id = None
        self.image = None
        self.caption = None

class CaptionDocRetriever():
    def __init__(self):
        pass

    def retrieveDocs(self, docIds):
        docs = []

        docIds=[str(x)+'.jpg' for x in docIds]

        with open(FLICKR_FILE_GEN, 'r', encoding='latin-1') as csvfile:
            reader = csv.reader(csvfile)

            #print(docIds)

            for rowNo, row in enumerate(reader):
              #  if rowNo == 0:
               #     print(row[0])
                #print('\t ' + str(row[DOC_ID_COL_IDX]))
                if row[COL_INDEX_ID] in docIds:
                    doc = ProductDoc()
                    doc.id = row[COL_INDEX_ID]
                    doc.image = row[COL_INDEX_ID]
                    doc.caption = row[COL_FLICKR_CAPTION]
                    docs.append(doc)
                    #break
        return docs



