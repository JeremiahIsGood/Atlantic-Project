import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher, Matcher, DependencyMatcher
nlp = spacy.load("en_core_web_sm")
from itertools import combinations
from gensim.models import Word2Vec

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
from nltk.tokenize import word_tokenize

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
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
stop_words -= {"no", "nor", "not", "never",
    "none", "nothing", "nobody", "neither"}

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def tokenize_text(sentence):
    if sentence:
        words = word_tokenize(sentence)
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

        tokens = [t for t in filtered_words if t.isalpha()]

        filtered_tokens = [t for t in tokens if t not in stop_words]

        tagged = nltk.pos_tag(filtered_tokens)

        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]

        text = " ".join(lemmatized)

        return text
    return None

class HouseMatcher():
    def __init__(self):
        self.predicted_item = None
        self.df = pd.read_csv("cleaned_house_repair_dataset.csv")
        self.matcher = Matcher(nlp.vocab)
        self.word2vec_model = Word2Vec.load("word2vec.model")

        for item in self.df["item"].values:
            self.matcher.add(str(item).upper().replace(" ", "_"), get_pattern(str(item)))

    def lookup_item(self, sentence):
      words = tokenize_text(sentence)
      doc = nlp(words)
      matches = self.matcher(doc)

      lookup_items = []
      for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        lookup_items.append(string_id.lower().replace("_", " "))

      total_list = []
      for item in lookup_items:

        similarity = self.word2vec_model.wv.n_similarity(item.split(), words.split(" "))
        # print(f"Similarity: {similarity} for item: {item} and sentence: {words.split(' ')}")
        total_list.append(similarity)


      if total_list:
        minim = np.max(total_list)-.10
        predicted_items = [idx for idx, sentiment in enumerate(total_list) if sentiment > minim]

        pred = np.argmax(total_list, axis=0)
        predicted_item = lookup_items[pred]


        lookup_items = np.array(lookup_items)
        item_vocab = list(tokenize_text(" ".join(lookup_items[predicted_items])).split(" "))

        common_words = [voc for voc in item_vocab if item_vocab.count(voc) > (len(predicted_items) * .6) and item_vocab.count(voc) < (len(predicted_items) * .8)]
        sentence_words = [word for word in words.split() if word not in common_words]
        removed_common_words = []
        for item in [voc.split() for voc in lookup_items[predicted_items]]:
            item_removed = []
            for word in item:
                if word not in common_words:
                    item_removed.append(tokenize_text(word))
            removed_common_words.append(item_removed)

        print("Common: ",common_words)
        print("Removed: ",removed_common_words)

        if len(lookup_items[predicted_items]) > 1:
            skewed_list = []
            if sentence_words and removed_common_words:
                for word in removed_common_words:
                    similarity = self.word2vec_model.wv.n_similarity(word, sentence_words)
                    print(f"Similarity: {similarity} for word '{word}' and sentence: {sentence_words}")
                    skewed_list.append(similarity)

            if skewed_list:
                pred_idx = np.argmax(skewed_list, axis=0)
                predicted_item = lookup_items[predicted_items[pred_idx]]

        pred_similarity = self.word2vec_model.wv.n_similarity(predicted_item.split(" "), words.split(" "))
        print(f"Final Prediction\nSimilarity: {pred_similarity} for '{predicted_item}' and sentence: {sentence}")

        if pred_similarity < 0.8:
            return None

        item = self.df.loc[self.df["item"] == predicted_item]

        if len(item) > 1:
            self.predicted_item = item.loc[item["avg_cost"] == item["avg_cost"].median()]
        else:
            self.predicted_item = item

        return self.predicted_item

      return None

