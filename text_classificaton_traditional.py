from collections import defaultdict, Counter
import random

def train_ngram_model(text, n=3):
    model = defaultdict(Counter)
    tokens = text.split()
    for i in range(len(tokens)-n+1):
        context = tuple(tokens[i:i+n-1])
        next_word = tokens[i+n-1]
        model[context][next_word] += 1
    return model

def predict_next_word(model, context):
    context = tuple(context)
    if context in model:
        return model[context].most_common(1)[0][0]
    else:
        return "<UNK>"

def evaluate_ngram_model(model, text, n=3):
    tokens = text.split()
    correct_predictions = 0
    total_predictions = 0

    for i in range(len(tokens)-n):
        context = tuple(tokens[i:i+n-1])
        actual_next_word = tokens[i+n-1]
        predicted_next_word = predict_next_word(model, context)
        
        if predicted_next_word == actual_next_word:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

text = "the cat sat on the mat the cat ate the mouse"
model = train_ngram_model(text, n=3)
print("Predicted next word for 'the cat':", predict_next_word(model, ["the", "cat"]))
accuracy = evaluate_ngram_model(model, text, n=3)
print(f"Model accuracy: {accuracy:.2f}")
