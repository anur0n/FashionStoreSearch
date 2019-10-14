from flask import Flask, render_template, request
import time
import re
from collections import defaultdict
from dmrankedquery import QueryType, QueryHandler
#from tqdm import tqdm

app = Flask(__name__)

queryHandler = QueryHandler()

isIndexLoaded = False

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

IMAGES_PATH = 'static/images/'
@app.route('/', methods=['GET', 'POST'])
def search():
    # if 'query' in request.args:
    #     q = request.args.get("q")
    #     query = q.split()
    #     # communicate with sub process
    #     return render_template('search/index.html', q=q, query=query)

    if request.method == 'POST':
        global isIndexLoaded
        if not isIndexLoaded:
            queryHandler.prepareParams()
            queryHandler.readIndex()
            isIndexLoaded = True
            print('Index ..' + str(len(queryHandler.index)))

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
        return render_template('search/index.html', titles = titles, images = images, descriptions = descriptions, \
                    tf_idf_scores=tf_idf_scores, tf_scores = tf_scores, idf_scores = idf_scores, query = query, \
                    docLengths = docLengths, terms = terms)

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

    #app.run()