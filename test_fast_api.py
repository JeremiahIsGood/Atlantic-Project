from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
import uvicorn
from logreg_class import LogRegTfidfModel
from finetuned_roberta_class import FTRobertaModel
from phraseMatcher import HouseMatcher
from ranforest_class import RandomForestModel
from nltk.tokenize import sent_tokenize

app = FastAPI()
logreg = LogRegTfidfModel()
ft_roberta = FTRobertaModel()
random_forest = RandomForestModel()
matcher = HouseMatcher()


#https://medium.com/@shiksha.verma1/calling-a-machine-learning-model-from-a-java-spring-boot-application-e42ed097e32f?utm_source=copilot.com
#this website helped me learn the basics of fast api.

# https://fastapi.tiangolo.com/tutorial/ Going through these pages were also very helpful.
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/house/predict")
async def house_predict(inputs: List = Body(embed=False)): #I get a List which is the inputs
    # I say embed = false so the parameter is not expected to be in a JSON body as a
    #             key instead of being the JSON body itself. I am just forcing this. I think embed=False is default.
    #You would use Body(embed=True) when you have like multiple parameters where you need to specify which one is what with the key.
    print(inputs)
    price = random_forest.predict(inputs)
    return {"price" : price}

@app.post("/sentiment/logreg/predict")
async def logreg_predict(description: str = Body(embed=False)):
    sentences = sent_tokenize(description)
    sentiment_list = []
    for sent in sentences:
        logreg.predict(sent)
        sentiment_list.append(logreg.pred_number)
    return {"sentiment" : sentiment_list}

@app.post("/sentiment/fine_tuned/predict")
async def finetuned_predict(description: str = Body(embed=False)):
    sentences = sent_tokenize(description)
    sentiment_list = []
    for sent in sentences:
        ft_roberta.predict(sent)
        sentiment_list.append(ft_roberta.pred_number)
        print(ft_roberta.probs)
        #I return a list for the sentiment of each sentence
    return {"sentimentList" : sentiment_list}

@app.post("/house/lookup")
async def house_lookup(description: str = Body(embed=False)):
    sentences = sent_tokenize(description)
    item_list = []
    for sent in sentences:
        item = matcher.lookup_item(sent)
        if item is not None:
            item_list.append(item.values.tolist()[0])
        else:
            item_list.append(None)
    # I return a list for the item found or not found of each sentence
    return {"itemList": item_list}

if __name__ == "__main__": #this code starts the server on port 8000 of localhost when you run it.
    uvicorn.run("test_fast_api:app", host="127.0.0.1", port=8000, reload=True)