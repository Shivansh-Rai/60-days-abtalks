from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = FastAPI()


class Input(BaseModel):
    text: str


def train():
    docs = [
        "I love this product",
        "This is terrible",
        "Amazing experience",
        "Worst service ever"
    ]
    y = [1, 0, 1, 0]

    v = TfidfVectorizer()
    x = v.fit_transform(docs)

    m = LogisticRegression()
    m.fit(x, y)

    return v, m


vectorizer, model = train()


@app.get("/")
def home():
    return {"message": "NLP Model API Running"}


@app.post("/predict")
def predict(data: Input):
    x = vectorizer.transform([data.text])
    y = model.predict(x)[0]
    return {"prediction": int(y)}