from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = FastAPI()


class Item(BaseModel):
    text: str


docs = [
    "Win free money now",
    "Meeting tomorrow at office",
    "Claim your prize now",
    "Project discussion today"
]

labels = [1, 0, 1, 0]


v = TfidfVectorizer()
x = v.fit_transform(docs)

m = LogisticRegression()
m.fit(x, labels)


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(item: Item):
    x = v.transform([item.text])
    y = m.predict(x)[0]
    return {"prediction": int(y)}