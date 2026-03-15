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


def remove_stop(w, st):
    return [i for i in w if i not in st]


def stem(w):
    out = []
    for i in w:
        if len(i) > 4:
            out.append(i[:-1])
        else:
            out.append(i)
    return out


def pipeline(t, use_stop=True, use_stem=True):
    t = clean(t)
    w = tokenize(t)
    if use_stop:
        st = {"is", "and", "the", "a", "an", "of", "to", "in"}
        w = remove_stop(w, st)
    if use_stem:
        w = stem(w)
    return w


print("Enter text:")
t = input()

w = pipeline(t, use_stop=True, use_stem=True)

print("Processed Output:")
print(w)