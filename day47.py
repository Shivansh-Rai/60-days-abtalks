from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)


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


@app.route("/")
def home():
    return jsonify({"message": "Flask NLP API Running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    
    x = vectorizer.transform([text])
    y = model.predict(x)[0]
    
    return jsonify({"prediction": int(y)})