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

def get_pattern(text): # I split the item and use combinations to create multiple combinations. I do this to do order and when some of the words are not all included.
  split_text = text.split()
  length = len(split_text)
  split_text += split_text

  comb_list = set(list(combinations(split_text, length)) +(list(combinations(split_text, length-1)))) # I use set to remove duplicates
  bad_values = set()

  for comb in comb_list: #I also remove ones where it may not be duplicated but has words that appear more than once in one thing.
    tokens = []
    for tok in comb:
      if tok in tokens:
        bad_values.add(comb)
      else:
        tokens.append(tok)
  comb_list = list(comb_list - bad_values) #I set the new set

  patterns = [] #I get patterns using this. I get the lemma of each token and do fuzzy to allow for some differences.
  for comb in comb_list:
    if comb:
      pattern = []
      for tok in comb:
        pattern.append({
            "LEMMA": {"FUZZY": tok}
        })
      patterns.append(pattern) #I return the list of patterns.

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
    return wordnet.NOUN

def tokenize_text(sentence):
    if sentence:
        words = word_tokenize(sentence)
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

        tokens = [t for t in filtered_words if t.isalpha()]

        filtered_tokens = [t for t in tokens if t not in stop_words]

        tagged = nltk.pos_tag(filtered_tokens)

        lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged if not tag.startswith('R')] #I don't include tokens that are adverbs.

        text = " ".join(lemmatized)

        return text
    return None

def remove_j_not_common(sentence, common_words):
    tagged = nltk.pos_tag(sentence.split(" "))

    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged if word not in common_words and not tag.startswith('J')] #I remove adjectives if they are not in the common words.
    #I was having an issue where it was removing bed from bed bug, but bed was a common word, so this makes it so it is not removed.

    text = " ".join(lemmatized)

    return text

# https://medium.com/bi3-technologies/advance-text-matching-with-spacy-and-python-40b558c51413 I used this website when deciding what type of matcher to use.
# https://spacy.io/api/matcher/
# https://www.ancisoft.com/blog/using-phrasematcher-in-spacy-to-find-multiple-match-types/#prerequisites this was useful in learning how to create
class HouseMatcher():
    def __init__(self):
        self.predicted_item = None
        self.df = pd.read_csv("cleaned_house_repair_dataset.csv")
        self.matcher = Matcher(nlp.vocab)
        self.word2vec_model = Word2Vec.load("word2vec.model")

        for item in self.df["item"].values: #I add to the matcher with keys like "ROOF_REPAIR" from "roof repair"
            self.matcher.add(str(item).upper().replace(" ", "_"), get_pattern(str(item)))

    def lookup_item(self, sentence):
      words = tokenize_text(sentence) #I tokenize the text. using nlp, for doc in token if doc.lemma_ or doc.text and stuff can be done, but I wanted to use this for more customality
      doc = nlp(words)
      matches = self.matcher(doc) #for each token in the i am matching it.

      lookup_items = [] #I look up the matches using the match_id. I reverse what we did earlier.
      for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        lookup_items.append(string_id.lower().replace("_", " "))

      total_list = []
      for item in lookup_items:

        similarity = self.word2vec_model.wv.n_similarity(item.split(), words.split(" "))
        print(f"Similarity: {similarity} for item: {item} and sentence: {words.split(' ')}")
        total_list.append(similarity)


      if total_list:
        minim = np.max(total_list)-.10
        predicted_items = [idx for idx, sentiment in enumerate(total_list) if sentiment > minim]

        pred = np.argmax(total_list, axis=0)
        predicted_item = lookup_items[pred]


        lookup_items = np.array(lookup_items)
        item_vocab = list(tokenize_text(" ".join(lookup_items[predicted_items])).split(" "))
        #words like need and this are causing the similarities to be different from what it should. I found this by doing "the roof needs replacement". however needs was causing it to choose a different answer
        unneeded_lookup_words = [
            "need", "want", "require",
            "required", "requiring",
            "must", "should", "have",
            "ask", "look",
            "wish", "desire", "prefer",
            "expect", "insist", "demand",
            "entail", "involve", "necessitate",
            "obligatory", "mandatory",
            "essential", "necessary", "requisite"
        ]

        common_words = [voc for voc in item_vocab if item_vocab.count(voc) > (len(predicted_items) * .65)]
        sentence_words = [word for word in remove_j_not_common(words, common_words).split() if (word not in common_words) and (word not in unneeded_lookup_words)]
        removed_common_words = []
        remove_predicted_items = []
        for item in [voc.split() for voc in lookup_items[predicted_items]]:
            count = 0
            for common_word in common_words:
                if common_word in item:
                    count += 1

            if count > 0:
                item_removed = []
                for word in item:
                    if word not in common_words:
                        if word not in unneeded_lookup_words:
                            item_removed.append(tokenize_text(word))

                removed_common_words.append(item_removed)
            else:
                remove_predicted_items.append(item)

        remove_predicted_items = [" ".join(item) for item in remove_predicted_items]
        predicted_items = [pred_idx for pred_idx in predicted_items if lookup_items[pred_idx] not in remove_predicted_items]

        #get the most similiar without removing common words and after that get the the amount in range of .5 and re do the same thing with removed common words.
        if len(lookup_items[predicted_items]) > 1:
            skewed_list = []
            word_list = []
            if sentence_words and removed_common_words:
                for word in removed_common_words:
                    if not word:
                        skewed_list.append(1)
                    else:
                        similarity = self.word2vec_model.wv.n_similarity(word, sentence_words)
                        print(f"Similarity: {similarity} for word '{word}' and sentence: {sentence_words}")
                        word_list.append(word)
                        skewed_list.append(similarity)

            if skewed_list:
                indexes = np.arange(0, len(skewed_list), 1)
                top3_max = sorted(zip(skewed_list, word_list, indexes), reverse=True)[:3] #this sorts by the first item which is similarity list.
                #I get the top 3 similarities
                new_skewed_list = []
                new_indexes_skewed = []
                for simil, words_l, index in top3_max:
                    word_count = 0
                    for word in words_l:
                        if word not in sentence_words:
                            word_count += 1

                    if word_count > 0:
                        simil -= .10 * word_count #I reduce the similarity by .10 based on the amount of words not in the sentence words.
                        new_skewed_list.append(simil)
                        new_indexes_skewed.append(index)

                print(new_skewed_list)
                if new_skewed_list:
                    pred_idx = np.argmax(new_skewed_list, axis=0)
                    predicted_item = lookup_items[predicted_items[new_indexes_skewed[pred_idx]]]
                else:
                    pred_idx = np.argmax(skewed_list, axis=0)
                    predicted_item = lookup_items[predicted_items[pred_idx]]

        pred_similarity = self.word2vec_model.wv.n_similarity(predicted_item.split(" "), words.split(" "))
        print(f"Final Prediction\nSimilarity: {pred_similarity} for '{predicted_item}' and sentence: {sentence}")

        if pred_similarity < 0.8:
            return None #the prediction must be at least .8

        item = self.df.loc[self.df["item"] == predicted_item]

        if len(item) > 1:
            self.predicted_item = item.loc[item["avg_cost"] == item["avg_cost"].median()] #I get the median cost of the items if dataframe has more than one row
        else:
            self.predicted_item = item

        return self.predicted_item

      return None

