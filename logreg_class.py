import joblib

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words -= {"no", "nor", "not", "never",
    "none", "nothing", "nobody", "neither"}


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

class LogRegTfidfModel:
    def __init__(self):
        self.pred = None
        self.model = joblib.load("tfidf_logreg_model.pkl")

    def predict(self, text):
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]

        tokens = [t for t in filtered_words if t.isalpha()]

        filtered_tokens = [t for t in tokens if t not in stop_words]

        tagged = nltk.pos_tag(filtered_tokens)

        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
        proc_text = [" ".join(lemmatized)]

        pred, prob_arr = self.model.predict(proc_text), self.model.predict_proba(proc_text)[0]

        state = "Negative"
        if pred == 1:
            state = "Neutral"
        elif pred == 2:
            state =  "Positive"

        self.pred = (state, (prob_arr[pred][0]))
        print(f"** LogReg-Tfidf Model **\nPredicted Sentiment: {self.pred[0]}\nConfidence: %{(prob_arr[pred][0] * 100):.2f}")
