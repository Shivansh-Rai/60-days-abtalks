import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


nltk.download('punkt')
nltk.download('stopwords')


def clean(t):
    s = ""
    for c in t:
        if c.isalnum() or c.isspace():
            s += c
        else:
            s += " "
    return s.lower()


def preprocess(docs):
    st = set(stopwords.words("english"))
    out = []
    for t in docs:
        t = clean(t)
        w = word_tokenize(t)
        w = [i for i in w if i not in st]
        out.append(" ".join(w))
    return out


def bow(docs):
    v = CountVectorizer()
    x = v.fit_transform(docs)
    return v.get_feature_names_out(), x.toarray()


def tfidf(docs):
    v = TfidfVectorizer()
    x = v.fit_transform(docs)
    return v.get_feature_names_out(), x.toarray()


print("Enter number of documents:")
n = int(input())

docs = []
for _ in range(n):
    docs.append(input())

docs = preprocess(docs)

f1, b = bow(docs)
f2, t = tfidf(docs)

print("\nBag of Words Features:")
print(f1)
print(b)

print("\nTF-IDF Features:")
print(f2)
print(t)