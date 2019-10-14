from flask import Flask, render_template, request
import time
from dmrankedquery_cosine import QueryType, QueryHandler
#from tqdm import tqdm

app = Flask(__name__)

queryHandler = QueryHandler()

isIndexLoaded = False

def highlight_term(term, text):
    replaced_text = text.replace(term, "<mark>{term}</mark>".format(term=term))
    return replaced_text

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
        docs, scores = queryHandler.performQuery(query)

        titles = [x.displayName for x in docs]
        images = [IMAGES_PATH+x.image+'.jpg' for x in docs]
        descriptions = [x.description for x in docs]

        for j, term in enumerate(queryHandler.getTerms(query)):
            for i, title in enumerate(titles):
                titles[i] = highlight_term(term, title)
            for i, description in enumerate(descriptions):
                descriptions[i] = highlight_term(term, description)
        outStr = ''
        return render_template('search/index.html', titles = titles, images = images, descriptions = descriptions, scores=scores, query = query)

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
    queryHandler.readIndex()
    queryHandler.performQuery('Blue check shirt')
    print('Search complete')

    #app.run()