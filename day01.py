def tokenize(p):
    s = ""
    for c in p:
        if c.isalnum() or c.isspace():
            s += c
        else:
            s += " "
    return s.lower().split()


def remove_stop(w, st):
    return [i for i in w if i not in st]


def freq(w):
    d = {}
    for i in w:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d


if __name__ == "__main__":
    p = "Python is great. Python is simple and powerful. Learning Python is fun and practical."
    st = {"is", "and", "the", "a", "an"}

    w = tokenize(p)
    w2 = remove_stop(w, st)
    d = freq(w2)

    print(w)
    print(w2)
    print(d)