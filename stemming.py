from nltk.stem.snowball import SnowballStemmer

englishStemmer=SnowballStemmer("italian")

x = englishStemmer.stem("avendo")

print(x)




