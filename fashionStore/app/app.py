from flask import Flask, render_template, request, jsonify
import time
import re
import numpy as np
from collections import defaultdict
from dmrankedquery import QueryType, QueryHandler
import dmnbclassifier
from dmnbclassifier import NaiveBayesClassifier, productCategories
#from tqdm import tqdm

app = Flask(__name__)

queryHandler = QueryHandler()
nbClassifier = NaiveBayesClassifier(productCategories)

isIndexLoaded = False
isNBCIndexLoaded = False

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


IMAGES_PATH = 'static/images/'
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

if __name__ == '__main__':
    queryHandler.prepareParams()
    # queryHandler.readIndex()
    # queryHandler.performQuery('Blue check shirt')
    print(highlight_terms(queryHandler.getTerms("blue narrow"), 'Blue narrowed check shirt blue'))
    print('Search complete')

    app.run()