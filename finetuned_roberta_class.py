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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

def convert_to_label(pred_number):
    if pred_number == 0:
        return "Negative"
    elif pred_number == 1:
        return "Neutral"
    return "Positive"

def convert_label_to_numb(pred_label):
    if pred_label == "Negative":
        return 0
    elif pred_label == "Neutral":
        return 1
    return 2

class FTRobertaModel:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta_sentiment_model")
        self.tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis", use_fast = False)
        self.pred_number = None
        self.pred_label = None
        self.probs = None
        self.pred = None
        self.positive_or_negative = None

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
        )

        outputs = self.model(**inputs) #**kwargs is used for unpacking the dict i give it which contains the input ids and attention mask
        #I dont need to do self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]). using ** unpacks it like this.
        #https://www.geeksforgeeks.org/python/packing-and-unpacking-arguments-in-python/
        logits = outputs.logits
        self.probs = torch.softmax(logits, dim=1)[0]
        self.pred_number = torch.argmax(self.probs)
        self.pred_label = convert_to_label(self.pred_number)
        self.pred = {
            "Negative": self.probs[0],
            "Neutral": self.probs[1],
            "Positive": self.probs[2]
        }
        pos_neg = torch.argmax(torch.tensor([self.probs[0].detach(), self.probs[2].detach()]))
        if pos_neg == 1:
            self.positive_or_negative = 2
        else:
            self.positive_or_negative = 0

    def print_prediction(self):
        print(f"\n** Roberta Model **\nPredicted Sentiment: {self.pred_label}\nConfidence: %{(self.probs[self.pred_number] * 100):.2f}")

