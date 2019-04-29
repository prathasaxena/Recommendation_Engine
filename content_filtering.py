import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle

location = "vectorizer.pickle"
metadata = pd.read_csv('organisations.csv',usecols = [ u'description',u'name'])
#metadata = organisation.head(1000)
tfidf = TfidfVectorizer(stop_words= 'english')

metadata['description'] = metadata['description'].fillna('')
metadata['name'] = metadata['name'].fillna('')
tfidf_matrix = tfidf.fit_transform(metadata['description'])
pickle.dump(tfidf_matrix, open(location, "wb"))


def get_recommendations(name,tfidf_matrix,description):
    if description == '':
        idx = metadata[metadata["name"]== name].index[0]
        tfidf_matrix_target = tfidf_matrix[idx]
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix_target)
    
    metadata["distance"] = cosine_sim
    metadata["distance"].head()
    n = 10 # or however many you want
    n_largest = metadata['distance'].nlargest(n + 1) 
    indices = [i for i in json.loads(n_largest.to_json()).keys()]
    
        # Return the top 10 most similar movies
    print(metadata['name'].iloc[indices])
    
    

load = pickle.load(open(location))
get_recommendations("TMG Health",load,'')
