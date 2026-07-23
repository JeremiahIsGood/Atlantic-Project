import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher, Matcher, DependencyMatcher
nlp = spacy.load("en_core_web_sm")
from itertools import combinations
import gensim

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

def get_pattern(text):
  split_text = text.split()
  length = len(split_text)
  split_text += split_text

  comb_list = set(list(combinations(split_text, length)) +(list(combinations(split_text, length-1))))
  bad_values = set()

  for comb in comb_list:
    tokens = []
    for tok in comb:
      if tok in tokens:
        bad_values.add(comb)
      else:
        tokens.append(tok)
  comb_list = list(comb_list - bad_values)

  patterns = []
  for comb in comb_list:
    if comb:
      pattern = []
      for tok in comb:
        pattern.append({
            "LEMMA": {"FUZZY": tok}
        })
      patterns.append(pattern)

  return patterns

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

def tokenize_text(new_df):
    new_df["text"] = new_df["text"].astype('string')

    processed_list = []
    bad_idx = []
    for idx, sentence in enumerate(new_df['text']):
        if sentence:
            words = word_tokenize(sentence)
            filtered_words = [word for word in words if word.lower() not in stop_words]

            tokens = [t for t in filtered_words if t.isalpha()]

            filtered_tokens = [t for t in tokens if t not in stop_words]

            tagged = nltk.pos_tag(filtered_tokens)

            lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]

            text = " ".join(lemmatized)
            if text.strip() == "":
                bad_idx.append(idx)
            else:
                processed_list.append(text)
        else:
            bad_idx.append(idx)
    new_df = new_df.drop(index=bad_idx).reset_index(drop=True)

    new_df.insert(1,"processed_text", processed_list)
    new_df = new_df.drop(columns=["text"])
    return new_df

class HouseMatcher():
    def __init__(self):
        self.predicted_item = None
        self.df = pd.read_csv("cleaned_house_repair_dataset.csv")
        self.data = [text.split() for text in self.df["item"].values]
        self.matcher = Matcher(nlp.vocab)
        self.word2vec_model = gensim.models.Word2Vec(self.data, min_count=1,
                                                     vector_size=200, window=10)
        for item in self.df["item"].values:
          self.matcher.add(item.upper().replace(" ", "_"), get_pattern(item))

    def lookup_item(self, sentence):
      sentence = tokenize_text(sentence)
      doc = nlp(sentence)
      matches = self.matcher(doc)

      lookup_items = []
      for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        lookup_items.append(string_id.lower().replace("_", " "))
      total_list = []

      for item in lookup_items:
        similarity = 0
        for i in item.split():
          for word in sentence.split():
            try:
              similarity += self.word2vec_model.wv.similarity(i, word)
            except KeyError:
              similarity += 0
        total_list.append(similarity / len(item.split()))

      if total_list:
        predicted_item = lookup_items[np.argmax(total_list)]

        self.predicted_item = self.df.loc[self.df["sentiment"] == predicted_item]

