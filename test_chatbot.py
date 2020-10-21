import pandas as pd
import spacy
from spacy.util import minibatch
import random


data = pd.read_csv('preguntas_chatboot.csv', sep=';')

#print(data['text'])

nlp = spacy.blank("es")  # create blank Language class

'''
nlp = spacy.load("es_core_news_md")

text = "hoy llueve mucho en Madrid"


doc = nlp(text)

for token in doc:
    print(token.text, token.i, token.lemma_, token.tag_, token.pos_, token.dep_, token.head.text,  token.head.pos_)
'''

# Create the TextCategorizer with exclusive classes and "bow" architecture
textcat = nlp.create_pipe(
              "textcat",
              config={
                "exclusive_classes": True,
                "architecture": "bow"})

# Add the TextCategorizer to the empty model
nlp.add_pipe(textcat)

textcat.add_label("goodbye")
textcat.add_label("thanks")
textcat.add_label("greet")
textcat.add_label("confirm")
textcat.add_label("negation")
textcat.add_label("get_full_debt")
textcat.add_label("get_debt")
textcat.add_label("get_plate")
textcat.add_label("payment")
textcat.add_label("deadlines")

train_texts = data['text'].values

train_labels = [{'cats': {'goodbye': label == 'goodbye',
                          'thanks': label == 'thanks',
                          'greet': label == 'greet',
                          'confirm': label == 'confirm',
                          'negation': label == 'negation',
                          'get_full_debt': label == 'get_full_debt',
                          'get_debt': label == 'get_debt',
                          'get_plate': label == 'get_plate',
                          'payment': label == 'payment',
                          'deadlines': label == 'deadlines'}}
                for label in data['label']]

train_data = list(zip(train_texts, train_labels))

random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        # Each batch is a list of (text, label) but we need to
        # send separate lists for texts and labels to update().
        # This is a quick way to split a list of tuples into lists
        texts, labels = zip(*batch)
        nlp.update(texts, labels, sgd=optimizer, losses=losses)
    print(losses)


texts = ["Hola, cuanto debo a la concesión?",
         ""]
docs = [nlp.tokenizer(text) for text in texts]

textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

predicted_labels = scores.argmax(axis=1)

print([textcat.labels[label] for label in predicted_labels])

