from fastapi import FastAPI, Body
import uvicorn
from logreg_class import LogRegTfidfModel
from finetuned_roberta_class import FTRobertaModel

app = FastAPI()
logreg = LogRegTfidfModel()
ft_roberta = FTRobertaModel()

#https://medium.com/@shiksha.verma1/calling-a-machine-learning-model-from-a-java-spring-boot-application-e42ed097e32f?utm_source=copilot.com
#this website helped me learn the basics of fast api.
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/sentiment/logreg/predict")
async def logreg_predict(description: str = Body(embed=False)):
    logreg.predict(description)
    return {"sentiment" : logreg.pred_label}

@app.post("/sentiment/finetuned/predict")
async def finetuned_predict(description: str = Body(embed=False)):
    ft_roberta.predict(description)
    return {"sentiment" : ft_roberta.pred_label}

if __name__ == "__main__":
    uvicorn.run("test_fast_api:app", host="127.0.0.1", port=8000, reload=True)