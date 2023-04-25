import joblib
import pickle


lang = 'en'
cat = 'adt'
dataset = 'old'

# load the preprocessing 'vectorizer' (converts texts to vectors of floats (tf-ids weights of the words))
vectorizer = pickle.load(open(f'{lang}_{cat}_{dataset}-{lang}_vectorizer.pickle', 'rb'))
print(vectorizer)
X = vectorizer.transform(["crime", "drugs", "war", "rainbow"])
print(X)

# load the trained classifier
clf = joblib.load(open(f'{lang}_{cat}_old-en_LogisticRegression_model.joblib', 'rb'))
predictions = clf.predict_proba(X)  # predictions should be a 2D array with 3 columns for low, moderate, and high probabilities, respectively
print(predictions)



