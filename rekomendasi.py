import numpy as np
import pandas as pd

books = pd.read_csv('book/books.csv')
books = books.dropna(subset=['original_title', 'average_rating'])
books = books[['original_title', 'average_rating']]

# add feature
def mergeCol(i):
    return (i['original_title'] + " " + str(i['average_rating']))
books['features'] = books.apply(mergeCol, axis=1)

# Modeling
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    analyzer='word',
    lowercase=True,
    tokenizer=lambda x: x.split(' ')
)
matrixFeature = model.fit_transform(books['features'])
features = model.get_feature_names()
jmlfeatures = len(features)

from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixFeature)

# user
andi = {
   'The Hunger Games' : 5,
   'Catching Fire' : 5,
   'Mockingjay' : 4,
   'The Hobbit or There and Back Again': 4,
   'Animal Farm: A Fairy Story' : 1
}
budi = {
    'Harry Potter and the Philosopher\'s Stone':5,
    'Harry Potter and the Chamber of Secrets':5,
    'Harry Potter and the Prisoner of Azkaban':5
}
chiko = {
    'The Brightest Star in the Sky':2,
    'The Last Seven Months of Anne Frank':1,
    'The Venetian Betrayal':2,
    'Robots and Empire':5
}
dedi = {
    'Hunter × Hunter #1':1,
    'Peter Pan':2
}
ello = {
    'Being Mortal: Medicine and What Matters in the End':2,
    'Doctor Sleep':4,
    'The Story of Doctor Dolittle':5,
}
users = [(andi,'andi'), (budi,'budi'), (chiko,'chiko'), (dedi,'dedi'), (ello,'ello')]
for user,nama in users:
    print(nama)
    likeIndex = []
    for title, rating in user.items():
        if rating >= 3:
            favBook = title
            bookIndex = books[books['original_title'] == favBook].index.values[0]
            likeIndex.append(bookIndex)

    for index in likeIndex:
        sortScoreList = []
        scoreList = list(enumerate(score[index]))
        sortScore = sorted(
            scoreList,
            key = lambda y: y[1],
            reverse = True
        )
        sortScoreList.extend(sortScore)
    
    topbooks = []
    for v in sortScoreList:
        if v[1] > 0.4:
            topbooks.append(v)

    import random
    booksRecSc = random.choices(topbooks, k=5)

    booksRecIndex = []
    for sc in booksRecSc:
        booksRecIndex.append(sc[0])


    print(books.drop('features', axis=1).iloc[booksRecIndex])
    print()