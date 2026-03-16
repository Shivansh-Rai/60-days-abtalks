from sklearn.feature_extraction.text import CountVectorizer


def clean(t):
    s = ""
    for c in t:
        if c.isalnum() or c.isspace():
            s += c
        else:
            s += " "
    return s.lower()


def tokenize(t):
    return t.split()


def remove_stop(w):
    st = {"is","and","the","a","an","to","in","of","for","on","at","this","that"}
    return [i for i in w if i not in st]


def stem(w):
    out = []
    for i in w:
        if len(i) > 4:
            out.append(i[:-1])
        else:
            out.append(i)
    return out



def pipeline(t):
    t = clean(t)
    w = tokenize(t)
    w = remove_stop(w)
    w = stem(w)
    return " ".join(w)


def vectorize(docs):
    v = CountVectorizer()
    x = v.fit_transform(docs)
    return v.get_feature_names_out(), x.toarray()


print("Enter number of messages:")
n = int(input())

docs = []
for _ in range(n):
    docs.append(input())

processed = []
for d in docs:
    processed.append(pipeline(d))

f, m = vectorize(processed)

print("\nProcessed Messages:")
print(processed)

print("\nVocabulary:")
print(f)

print("\nFeature Matrix:")
print(m)