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
        self.pred_label = None
        self.pred_conf = None
        self.pred_number = None
        self.up_model = joblib.load("tfidf_logreg_model_unprocessed.pkl")
        self.p_model = joblib.load("tfidf_logreg_model_processed.pkl")

    def processed_predict(self, text):
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]

        tokens = [t for t in filtered_words if t.isalpha()]

        filtered_tokens = [t for t in tokens if t not in stop_words]

        tagged = nltk.pos_tag(filtered_tokens)

        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
        proc_text = [" ".join(lemmatized)]

        pred, prob_arr = self.p_model.predict(proc_text), self.p_model.predict_proba(proc_text)[0]

        state = "Negative"
        if pred == 1:
            state = "Neutral"
        elif pred == 2:
            state =  "Positive"

        self.pred_label = state
        self.pred_conf = (prob_arr[pred][0])
        self.pred_number = self.convert_label_to_numb()

    def unprocessed_predict(self, text):
        pred, prob_arr = self.up_model.predict([text]), self.up_model.predict_proba([text])[0]

        state = "Negative"
        if pred == 1:
            state = "Neutral"
        elif pred == 2:
            state =  "Positive"

        self.pred_label = state
        self.pred_conf = (prob_arr[pred][0])
        self.pred_number = self.convert_label_to_numb()


    def print_prediction(self):
        print(f"** LogReg-Tfidf Model **\nPredicted Sentiment: {self.pred_label}\nConfidence: %{(self.pred_conf * 100):.2f}")

    def convert_to_label(self):
        if self.pred_number == 0:
            return "Negative"
        elif self.pred_number == 1:
            return "Neutral"
        return "Positive"

    def convert_label_to_numb(self):
        if self.pred_label == "Negative":
            return 0
        elif self.pred_label == "Neutral":
            return 1
        return 2