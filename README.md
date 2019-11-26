# Fashion Store Search and classification
This project implements a small search engine which searches over thousands of data of dataset containing info about lifestyle products. 
Also provides feature to classify the category of the applying Naive Bayes Classification on product description.

**Required Libraries**

Download following libraries:

```
pip install flask
pip install numpy
pip install nltk
pip install scipy
pip install sklearn
pip install Flask-Images
pip --default-timeout=60 install "tensorflow-gpu>=1.19,<2"
```


**Steps to run and deploy in localhost**
<body>
<pre>
1. Install flask in your environment.
2. Download this repo to a folder
3. Navigate upto the 'app' directory
4. from a console run the app.py file by the command **python app.py**
5. Hit **localhost:5000** in browser.
</pre>
</body>

**Important Files**
* **app.py** Entry point of the front end. Handles the search request and update the UI with results.
* **dmrankedquery.py** Handles the query request and returns the documents based on tf-idf score.
* **dmrankedquery_cosine.py** Handles the query request and returns the documents based on tf-idf and cosine similarity score.
* **docRetriever.py** Utility file that returns the original product data based on document Ids.
* **dmsearchranked.py** This file creates the inverted index and tf idf, and stores in a file.
* **dmnbclassifier.py** This file creates the inverted index, saves and loads from file. And handles classification requests and returns classification results.
