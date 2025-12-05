def preprocess_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = text.lower()
    text = word_tokenize(text)
    stop = stopwords.words('english')
    text = [word for word in text if word not in stop]
    text = " ".join(text)
    return text
