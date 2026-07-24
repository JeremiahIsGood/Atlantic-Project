import joblib

import nltk
import torch

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
        self.positive_or_negative = None
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

        self.pred_label = state
        self.pred_conf = (prob_arr[pred][0])
        self.pred_number = self.convert_label_to_numb()

        pos_neg = torch.argmax(torch.tensor([prob_arr[0], prob_arr[2]]))
        if pos_neg == 1:
            self.positive_or_negative = 2
        else:
            self.positive_or_negative = 0


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