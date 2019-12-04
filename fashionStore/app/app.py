#!/home/anur0nArm/.virtualenvs/fstore/bin/python
import sys
import os

print(sys.version)

file_path = os.path.dirname(os.path.realpath(__file__))

# print(file_path)
sys.path.append(file_path + '/captioning/')

from flask import Flask, render_template, request, jsonify

import time
import re
import numpy as np
from collections import defaultdict
from dmrankedquery import QueryType, QueryHandler
from dm_captioning_ranked_query import ImageQueryHandler
import dmnbclassifier
from dmnbclassifier import NaiveBayesClassifier, productCategories
import dm_caption_evaluator as capgen
#from tqdm import tqdm
from werkzeug import secure_filename
import datetime

import threading

#import tensorflow as tsf

# print(tsf.VERSION)

app = Flask(__name__)

GITHUB_FLICKR_IMG_URL_BASE = 'https://raw.githubusercontent.com/anur0n/flicker-dataset-for-img-captioning/master/flickr10k_images/images/'

queryHandler = QueryHandler()
imgQueryHandler = ImageQueryHandler()
nbClassifier = NaiveBayesClassifier(productCategories)
isIndexLoaded = False
isNBCIndexLoaded = False
isImgIndexLoaded = False

def highlight_terms(queryTerms, doc):

    termdict = []

    # print(queryTerms)
    docTerms = doc.split()
    # print(queryHandler.getTerms(doc))
    for term in docTerms:
        reducedTerm = queryHandler.getTerms(term)
        if reducedTerm in termdict:
            continue
        if len(reducedTerm) > 0 and reducedTerm[0] in queryTerms:
            replaced_text = re.compile(re.escape(term), re.IGNORECASE)
            doc = replaced_text.sub("<mark>{term}</mark>".format(term=term), doc)
            termdict.append(reducedTerm)

    return doc

def classify():
    print('Classifying')
    global isNBCIndexLoaded
    if not isNBCIndexLoaded:
        nbClassifier.loadIndex()
        isNBCIndexLoaded = True
        print('Index ..' + str(len(nbClassifier.cats_info)))
    print(productCategories)
    query = request.form['query']
    if query == '':
        return render_template('search/index.html', opType = 'classification', query='')
    print('searching ..' + query)

    terms = nbClassifier.getTerms(query)

    categoryProb, term_freqs, logOfProb, priorProb, denom = nbClassifier.classify(query)

    sortedCategory = reversed(sorted(zip(categoryProb, productCategories)))
    sortedCategory = [x for x in sortedCategory]
    #sortedCategory = sortedCategory.sort().reverse()
    result = [str(category + '=' + str(prob)) for prob, category in sortedCategory]
    sortedProb, sortedCategory = zip(*sortedCategory)
    percentage = [(prob + abs(min(sortedProb))) for prob in sortedProb]
    sum = np.sum(percentage)
    percentage = [prob/sum for prob in percentage]
    return render_template('search/index.html', opType = 'classification', categoryProb = str(result), \
            detectedCategory = sortedCategory[0], query = query, term_freqs = term_freqs, \
                priorProb = priorProb, terms=terms, denom=denom, logOfProb = logOfProb, totalScore = sortedProb[0],\
                sortedProb = sortedProb, sortedCategory = sortedCategory, percentage = percentage)


def search(request):
    print('Searching')
    global isIndexLoaded
    if not isIndexLoaded:
        queryHandler.prepareParams()
        queryHandler.readIndex()
        isIndexLoaded = True
        print('Index ..' + str(len(queryHandler.index)))
    print('Searching2')
    query = request.form['query']
    if query == '':
        return render_template('search/index.html', opType = 'search', query='')
    print('searching ..' + query)
    docs, tf_idf_scores, tf_scores, idf_scores = queryHandler.performQuery(query)

    titles = [x.displayName for x in docs]
    images = [IMAGES_PATH+x.image+'.jpg' for x in docs]
    descriptions = [x.description for x in docs]
    docLengths = [len(x.split()) for x in descriptions]

    terms = queryHandler.getTerms(query)


    for i, doc in enumerate(titles):
        titles[i] = highlight_terms(terms, doc)

    for i, doc in enumerate(descriptions):
        descriptions[i] = highlight_terms(terms, doc)



    outStr = ''
    return render_template('search/index.html', opType = 'search', titles = titles, images = images, descriptions = descriptions, \
                tf_idf_scores=tf_idf_scores, tf_scores = tf_scores, idf_scores = idf_scores, query = query, \
                docLengths = docLengths, terms = terms)

def image_search():
    print('Image Searching')
    global isImgIndexLoaded
    if not isImgIndexLoaded:
        print('Loading indices')
        imgQueryHandler.prepareParams()
        imgQueryHandler.readIndex()
        capgen.prepare_params()
        isImgIndexLoaded = True
        print('Index ..' + str(len(imgQueryHandler.index)))
    print('Searching2')
    query = request.form['query']
    img_file = request.files['img_file'] if request.files.get('img_file') else None
    if query == '' and img_file == None:
        return render_template('search/index.html', opType = 'image_search', query='')
    query_img = ''
    if img_file != None:
        print('Saving file to: ' + TMP_IMAGE_PATH)
        img_file.save(TMP_IMAGE_PATH)
        print('file saved to ' + TMP_IMAGE_PATH)
        print('Generating caption')
        caption, _ = capgen.evaluate(TMP_IMAGE_PATH)
        query = ' '.join(caption)
        query = query.replace("<end>", "")
        query_img = '/static/tmpfiles/tmp.jpg'+'?'+str(datetime.datetime.now()) #To disable browser cache
        print('Caption: ' + query)

    print('searching ..' + query)
    docs, tf_idf_scores, tf_scores, idf_scores = imgQueryHandler.performQuery(query)

    titles = [x.caption for x in docs]
    image_urls = [GITHUB_FLICKR_IMG_URL_BASE + x.image for x in docs]
    captions = [x.caption.replace("<end>", "") for x in docs]
    docLengths = [len(x.split()) for x in captions]

    terms = imgQueryHandler.getTerms(query)



    outStr = ''
    return render_template('search/index.html', opType = 'image_search', titles = titles, images = image_urls, captions = captions, \
                tf_idf_scores=tf_idf_scores, tf_scores = tf_scores, idf_scores = idf_scores, query = query, \
                docLengths = docLengths, terms = terms, query_img = query_img)

def uploadImage():
    f = request.files['file']
    f.save(IMAGES_PATH+secure_filename(f.filename))
    return render_template('search/index.html', opType = 'img_search', image = IMAGES_PATH+f.filename)

IMAGES_PATH = '/static/images/'
TMP_IMAGE_PATH = file_path +'/static/tmpfiles/tmp.jpg'
@app.route('/', methods=['GET', 'POST'])
def submit():
    # if 'query' in request.args:
    #     q = request.args.get("q")
    #     query = q.split()
    #     # communicate with sub process
    #     return render_template('search/index.html', q=q, query=query)
    print("Submitted")
    print(request.args)
    if request.method == 'POST':
        print('Here')
        #print()
        if request.form['action'] == 'Search':
            print('Search')
            return search(request)
        elif request.form['action'] == 'Classify':
            print('Classify')
            return classify()
        elif request.form['action'] == 'Image Search':
            print('Image search')
            return image_search()
        elif request.form['action'] == 'Image':
            print('Upload image')
            return uploadImage()
        else:
            print('Default')
            return search(request)

    #     for i, doc in enumerate(docs):
    #         outStr += str(i+1)+'<br> <b>'+doc.displayName + '</b><br> <br>' + doc.description + '<br> <br> <br>'
    #     for j, term in enumerate(queryHandler.getTerms(query)):
    #             highlight_term(term, outStr)
    #     print('search complete')
    #     return outStr
    return render_template('search/index.html') #render_template("index.html")

#def index():
 #   return str(queryHandler.performQuery(' toy'))

def preloadIndices():
    print('Loading indices')
    queryHandler.prepareParams()
    queryHandler.prepareParams()
    queryHandler.readIndex()
    isIndexLoaded = True

    nbClassifier.loadIndex()
    isNBCIndexLoaded = True


    imgQueryHandler.prepareParams()
    imgQueryHandler.readIndex()
    capgen.prepare_params()
    isImgIndexLoaded = True
    # queryHandler.readIndex()
    # queryHandler.performQuery('Blue check shirt')
    # print(highlight_terms(queryHandler.getTerms("blue narrow"), 'Blue narrowed check shirt blue'))
    print('Service loaded successfully')
# if __name__ == '__main__':

# preloading_thread = threading.Thread(target=preloadIndices)
# preloading_thread.start()

print('App loaded')

    # app.run()