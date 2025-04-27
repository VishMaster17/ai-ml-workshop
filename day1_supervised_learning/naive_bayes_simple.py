from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

dataset = {
    "text": [
        "Congratulations! You have won a lottery",
        "Limited Offer",
        "You are selected for a free gift",
        "You have a new message",
        "Let's meet tomorrow",
    ],
    "label": [
        "spam",
        "spam",
        "spam",
        "ham",
        "ham",
    ]
}

model = MultinomialNB()
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(dataset['text'])
y = dataset['label']

model.fit(x, y)

predicted_label = model.predict(vectorizer.transform(["urgent click the link below"]))
print(predicted_label)