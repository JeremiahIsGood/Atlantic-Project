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
        #This is the hugging face pretrained model we used.
        # @inproceedings
        #
        # {barbieri - etal - 2022 - xlm,
        #  title = "{XLM}-{T}: Multilingual Language Models in {T}witter for Sentiment Analysis and Beyond",
        # author = "Barbieri, Francesco  and
        # Espinosa
        # Anke, Luis and
        #       Camacho - Collados, Jose
        # ",
        # booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
        # month = jun,
        # year = "2022",
        # address = "Marseille, France",
        # publisher = "European Language Resources Association",
        # url = "https://aclanthology.org/2022.lrec-1.27",
        # pages = "258--266"
        # }
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta_sentiment_model") #Load in my model and the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis", use_fast = False)
        self.pred_number = None
        self.pred_label = None
        self.probs = None
        self.pred_conf = None
        self.positive_or_negative = None

    def predict(self, text): #I use the tokenizer and return the tensors pytorch. If not there is errors.
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
        self.probs = torch.softmax(logits, dim=1)[0] #I get the probabilities by using softmax
        self.pred_number = torch.argmax(self.probs)
        self.pred_label = convert_to_label(self.pred_number)
        self.pred_conf = self.probs[self.pred_number]

        if self.pred_number == 1 and self.pred_conf < .6:
            pos_neg = torch.argmax(torch.tensor([self.probs[0].detach(), self.probs[2].detach()]))
            if pos_neg == 1:
                self.pred_number = 2
            else:
                self.pred_number = 0

            self.pred_label = self.convert_to_label()
            self.pred_conf = self.probs[self.pred_number]

        self.pred_number = self.pred_number.detach().tolist() #I detach and send it to list instead of tensor.

    def print_prediction(self):
        print(f"\n** Roberta Model **\nPredicted Sentiment: {self.pred_label}\nConfidence: %{(self.probs[self.pred_number] * 100):.2f}")

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
