# Commented out IPython magic to ensure Python compatibility.
# %cd "drive/My Drive/Colab Notebooks"

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# import these modules
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from array import array
#from tqdm import tqdm
import csv
import math
import re
import numpy as np
import os
import pickle
from nltk.stem import WordNetLemmatizer
# from DMRankedQuery import QueryType, QueryHandler

file_path = os.path.dirname(os.path.realpath(__file__))

MY_DRIVE_DATA_MINING = file_path#os.getcwd() + '/app' #'/content/drive/My Drive/Data Mining'

STYLE_WITH_DESC_N_TITLE_SAMPLED = MY_DRIVE_DATA_MINING+'/styles_with_description_title_RandomSampled.csv'

STYLE_WITH_DESC_N_TITLE = STYLE_WITH_DESC_N_TITLE_SAMPLED#MY_DRIVE_DATA_MINING+'/styles_with_description_title.csv'

INVERTED_IDX_FILE = MY_DRIVE_DATA_MINING+'/store_index_naive_bayes.dat'
# INVERTED_IDX_FILE = MY_DRIVE_DATA_MINING+'/store_index_tf_idf_Random_lemmatized.dat'

COL_INDEX_ID = 0
COL_INDEX_CATEGORY = 2
COL_INDEX_DISPLAY_NAME = 9
COL_INDEX_DESC_TITLE = 13


def defaultDict():
    return 0

def preprocess_string(str_arg):
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case

    return cleaned_str # returning the preprocessed string

class NaiveBayesClassifier:
  def __init__(self, categories):
    self.categories = categories
    self.smoothing = 1
    self.prepareParams()

  def prepareParams(self):
    self.stopwords = set(stopwords.words('english'))
    self.dataFile = STYLE_WITH_DESC_N_TITLE
    # self.indexFile = INVERTED_IDX_FILE
    self.stemmer = PorterStemmer() # SnowballStemmer('english')
    self.lemmatizer = WordNetLemmatizer()

  def getTerms(self, doc):
    #print('Original\n'+doc)
    doc = doc.lower()
    #print('lowered\n\n'+doc)
    doc = re.sub(r'[^a-z0-9 ]',' ',doc) #put spaces instead of non-alphanumeric characters
    terms = doc.split()

    terms = [term for term in terms if term not in self.stopwords]
#     terms = [self.lemmatizer.lemmatize(term) for term in terms]
    terms = [self.stemmer.stem(term) for term in terms]
    #print('Terms:\n\n')
    #print(terms)
    return terms

  def addToIndex(self, products_of_category, category):
    counter = 0
    for product in products_of_category: #for every word in preprocessed example
      token_words = self.getTerms(product)
      for token in token_words:
        counter += 1
        if not token in self.index[category]:
          self.index[category][token] = 0
        self.index[category][token]+=1 #increment in its count
    # print(counter)
    # print(category)
      #   print(self.index[category][token_word])

  def buildInvertedIndex(self):

    # for i, desc in enumerate(self.productDescriptions):
    #   if 'shirt' not in desc.lower():
    #     print(i)
    self.index = np.array([defaultdict(int) for i in range(self.categories.shape[0])])
    # print(len(self.labels))

    for idx, category in enumerate(self.categories):
      all_category_products = [x for x, label in zip(self.productDescriptions, self.labels) if label == category]

      # for x, label in zip(self.productDescriptions, self.labels):
        # if 'care' in label.lower():
          # print('Shirt  ---- '+label)
      cleaned_desc = [category_product for category_product in all_category_products]

      # print(category + ' ' + str(len(cleaned_desc)))

      np.apply_along_axis(self.addToIndex,0,cleaned_desc,idx)

      pass

    # print(self.index)

    pass

  def precalcNBValues(self):
    probability_classes = np.empty(self.categories.shape[0])
    # print(self.categories.shape[0])

    all_words = []
    cat_word_counts = np.empty(self.categories.shape[0])
    # print(cat_word_counts)
    for cat_index,cat in enumerate(self.categories):

        #Calculating prior probability p(c) for each class
        all_category_products = [x for x, label in zip(self.productDescriptions, self.labels) if label == cat]
        # print(len(all_category_products))
        probability_classes[cat_index] = len(all_category_products) / len(self.labels)

        # print(':::->'+(str(list(self.index[cat_index].values()))))
        #Calculating total counts of all the words of each class
        cat_word_counts[cat_index] = np.sum(np.array(list(self.index[cat_index].values())))+self.smoothing # |v| is remaining to be added

        #get all words of this category
        all_words+=self.index[cat_index].keys()


    #combine all words of every category & make them unique to get vocabulary -V- of entire training set

    self.vocab=np.unique(np.array(all_words))
    self.vocab_length=self.vocab.shape[0]
    # print(self.vocab_length)
    #computing denominator value
    denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.categories)])

    # print(denoms)

    self.cats_info=[(self.index[cat_index],probability_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.categories)]
    # print(type(self.index[0]))
    # self.cats_info=[[(self.index[cat_index])] for cat_index,cat in enumerate(self.categories)]
    self.cats_info=np.array(self.cats_info)

    # outfile = open(INVERTED_IDX_FILE,'wb')
    # np.save(outfile, nbc.cats_info)
    # print(self.cats_info)

  def train(self, dataset, labels):
    self.productDescriptions = dataset
    self.labels = labels

    self.buildInvertedIndex()
    self.precalcNBValues()
  #####################################################################
  ## Test ##
  def classify(self,test_example):
      likelihood_prob=np.zeros(self.categories.shape[0]) #to store probability w.r.t each class

      terms = self.getTerms(test_example)
      #finding probability w.r.t each class of the given test example
      for cat_index,cat in enumerate(self.categories):

          for test_token in terms: #split the test example and get p of each test word

              ####################################################################################

              #This loop computes : for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]

              ####################################################################################

              #get total count of this test token from it's respective training dict to get numerator value
              # print(type(self.cats_info[cat_index][0]))
              test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+self.smoothing

              #now get likelihood of this test_token word
              test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])

              #remember why taking log? To prevent underflow!
              likelihood_prob[cat_index]+=np.log(test_token_prob)

      # we have likelihood estimate of the given example against every class but we need posterior probility
      post_prob=np.empty(self.categories.shape[0])

      for cat_index,cat in enumerate(self.categories):
          post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])


      sortedCategory = reversed(sorted(zip(post_prob, self.categories)))
      sortedCategory = [x for x in sortedCategory]
        #sortedCategory = sortedCategory.sort().reverse()
        #result = [str(category + '=' + str(prob)) for prob, category in sortedCategory]
      sortedProb, sortedCategory = zip(*sortedCategory)

      mxCategory = 0;
      mxProb = -float("inf")
      for idx, category in enumerate(self.categories):
        if post_prob[idx] > mxProb:
            mxProb = post_prob[idx]
            mxCategory = idx

      term_freq_max_category = np.empty(len(terms))
      print(len(terms))
      print(len(term_freq_max_category))
      term_freq_max_category_log = np.empty(len(terms))
      logOfTermProbabilities = np.empty(len(terms))

      print(self.categories[mxCategory])
      for idx, term in enumerate(terms):
        term_freq_max_category[idx] = (self.cats_info[mxCategory][0].get(term,0)+self.smoothing)
        logOfTermProbabilities[idx] = np.log((self.cats_info[mxCategory][0].get(term,0)+self.smoothing) / float(self.cats_info[mxCategory][2]))

      return (post_prob, term_freq_max_category, logOfTermProbabilities, np.log(self.cats_info[mxCategory][1]), float(self.cats_info[mxCategory][2]))


  def test(self,test_set):
    predictions=[] #to store prediction of each test example
    for example in test_set:

        #preprocess the test example the same way we did for training set exampels
        cleaned_example=preprocess_string(example)

        #simply get the posterior probability of every example
        post_prob, _, _, _, _ =self.classify(cleaned_example) #get prob of this example for both classes

        #simply pick the max value and map against self.classes!
        predictions.append(self.categories[np.argmax(post_prob)])
        # print(len(predictions))

    return np.array(predictions)

  def loadIndex(self):
    infile = open(INVERTED_IDX_FILE,'rb')
    categoryIndex = pickle.load(infile)
    infile.close()
    self.cats_info = categoryIndex

###################################################################

def loadData():
  with open(STYLE_WITH_DESC_N_TITLE, 'r', encoding='latin-1') as csvfile:
      reader = csv.reader(csvfile)

      descriptions = []
      labels = []

      for rowNo, row in enumerate(reader):
        if rowNo==0 or row[COL_INDEX_CATEGORY] == '':
          continue

        labels.append(row[COL_INDEX_CATEGORY])
        descriptions.append(row[COL_INDEX_DESC_TITLE])
        # if('polor' in row[COL_INDEX_DESC_TITLE].lower() ):
        #   print(rowNo)
  return (descriptions, labels, np.unique(np.array(labels)))


def splitData(data, label):
   X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, random_state=42)
   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
   print(len(X_train))
   print(len(y_train))
   print(len(X_test))
   print(len(y_test))
   print(len(X_val))
   print(len(y_val))

   return (X_train, X_test, X_val, y_train, y_test, y_val)

productCategories = np.array(['Accessories', 'Apparel', 'Footwear', 'Free Items', 'Home', 'Personal Care',
 'Sporting Goods'])


# import
if __name__ == "__main__":
  descriptions, labels, categories = loadData()
  X_train, X_test, y_train, y_test = splitData(descriptions, labels)
  print(categories)

  print(len(descriptions))
  evaluation = []

#   smoothing = 100.0

#   for i in range(15):
#     nbc = NaiveBayesClassifier(productCategories)
#     nbc.smoothing = smoothing

#     nbc.train(X_train,y_train)

#     pclasses=nbc.test(X_val) #get predcitions for test set
#     # print(pclasses)

#     print('For '+str(smoothing)+' : ')
#     # accuracy: (tp + tn) / (p + n)
#     accuracy = accuracy_score(y_val, pclasses)
#     print('Accuracy: %f' % accuracy)
#     # precision tp / (tp + fp)
#     precision = precision_score(y_val, pclasses, average='macro')
#     print('Precision: %f' % precision)
#     # recall: tp / (tp + fn)
#     recall = recall_score(y_val, pclasses, average='macro')
#     print('Recall: %f' % recall)
#     # f1: 2 tp / (2 tp + fp + fn)
#     f1 = f1_score(y_val, pclasses, average='macro')
#     print('F1 score: %f' % f1)


#     evaluation.append((smoothing, accuracy, precision, recall, f1))

#     smoothing /= 10.0

#   print(evaluation)

#   print('##################Validating on validation set.###################')
#   smoothing = 1
#   nbc = NaiveBayesClassifier(productCategories)
#   nbc.smoothing = smoothing
#   nbc.train(X_train,y_train)
#   pclasses=nbc.test(X_test) #get predcitions for test set
#   # print(pclasses)

#   print('For '+str(smoothing)+' : ')
#   # accuracy: (tp + tn) / (p + n)
#   accuracy = accuracy_score(y_test, pclasses)
#   print('Accuracy: %f' % accuracy)
#   # precision tp / (tp + fp)
#   precision = precision_score(y_test, pclasses, average='macro')
#   print('Precision: %f' % precision)
#   # recall: tp / (tp + fn)
#   recall = recall_score(y_test, pclasses, average='macro')
#   print('Recall: %f' % recall)
#   # f1: 2 tp / (2 tp + fp + fn)
#   f1 = f1_score(y_test, pclasses, average='macro')
#   print('F1 score: %f' % f1)




  # INVERTED_IDX_FILE = MY_DRIVE+'/store_index_naive_bayes.dat'
  print("Training done!!")

  # outfile = open(INVERTED_IDX_FILE,'wb')
  # # np.save(outfile, nbc.cats_info)
  # pickle.dump(nbc.cats_info,outfile)
  # outfile.close()

  # nbc.loadIndex()


  print("Saving done!!")
  # print(nbc.test(X_test))

  # y_test = ['Apparel']

  # pclasses=nbc.test(X_test) #get predcitions for test set
  # print(pclasses)
  # test_acc=np.sum(pclasses==y_test)/len(y_test)
  # print(test_acc)