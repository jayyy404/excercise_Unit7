import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import brown


nltk.download('brown', quiet=True)
nltk.download('universal_tagset', quiet=True)

def prepare_data(sentences, max_sentences=2000):
    """Prepare word and tag sequences from tagged sentences."""
    sentences = sentences[:max_sentences]
    word2idx = {'<PAD>': 0, '<UNK>': 1}  
    tag2idx = {'<PAD>': 0}
    X, y = [], []

    for sent in sentences:
        x_seq, y_seq = [], []
        for word, tag in sent:
            word = word.lower()
            word2idx.setdefault(word, len(word2idx))  
            tag2idx.setdefault(tag, len(tag2idx))  
            x_seq.append(word2idx[word])
            y_seq.append(tag2idx[tag])
        X.append(x_seq)
        y.append(y_seq)

    # Pad sequences to the same length
    maxlen = max(len(s) for s in X)
    X = pad_sequences(X, padding='post', maxlen=maxlen, value=0)
    y = pad_sequences(y, padding='post', maxlen=maxlen, value=0)
    y = to_categorical(y, num_classes=len(tag2idx))

    return X, y, word2idx, tag2idx, maxlen

def build_model(vocab_size, tag_count, maxlen):
    """Build and compile the LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)), 
        TimeDistributed(Dense(tag_count, activation='softmax'))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    """Train and evaluate the LSTM POS tagger."""
    tagged_sents = brown.tagged_sents(tagset='universal')
    X, y, word2idx, tag2idx, maxlen = prepare_data(tagged_sents)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Build and train model
    model = build_model(len(word2idx), len(tag2idx), maxlen)
    model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=3,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    idx2tag = {v: k for k, v in tag2idx.items()}
    test_sentence = "the dog barks".lower().split()
    x_test = [word2idx.get(word, word2idx['<UNK>']) for word in test_sentence] 
    x_test = pad_sequences([x_test], maxlen=maxlen, padding='post', value=0)
    pred = model.predict(x_test, verbose=0)
    pred_tags = [idx2tag[np.argmax(tag)] for tag in pred[0][:len(test_sentence)]]
    print(f"Sentence: {' '.join(test_sentence)}")
    print(f"Predicted Tags: {' '.join(f'{w}/{t}' for w, t in zip(test_sentence, pred_tags))}")

if __name__ == "__main__":
    main()
