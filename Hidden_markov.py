from collections import defaultdict
import pprint

def hidden_markov(sentences):
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    for sentence in sentences:
        words_tags = sentence.split()
        prev_tag = 'START'

        for wt in words_tags:
            word, tag = wt.rsplit('_', 1)
            emission_counts[tag][word] += 1
            transition_counts[prev_tag][tag] += 1
            tag_counts[tag] += 1
            prev_tag = tag

        transition_counts[prev_tag]['END'] += 1

    transition_probs = defaultdict(dict)
    emission_probs = defaultdict(dict)

    for prev_tag in transition_counts:
        total = sum(transition_counts[prev_tag].values())
        for curr_tag in transition_counts[prev_tag]:
            transition_probs[prev_tag][curr_tag] = transition_counts[prev_tag][curr_tag] / total

    for tag in emission_counts:
        total = sum(emission_counts[tag].values())
        for word in emission_counts[tag]:
            emission_probs[tag][word] = emission_counts[tag][word] / total

    return transition_probs, emission_probs



if __name__ == "__main__":
    sentences = [
        "The_DET cat_NOUN sleeps_VERB",
        "A_DET dog_NOUN barks_VERB",
        "The_DET dog_NOUN sleeps_VERB",
        "My_DET dog_NOUN runs_VERB fast_ADV",
        "A_DET cat_NOUN meows_VERB loudly_ADV",
        "Your_DET cat_NOUN runs_VERB",
        "The_DET bird_NOUN sings_VERB sweetly_ADV",
        "A_DET bird_NOUN chirps_VERB"
    ]

    trans_probs, emit_probs = hidden_markov(sentences)

    print("Transition Probabilities")
    pprint.pprint(dict(trans_probs))

    print("\nEmission Probabilities")
    pprint.pprint(dict(emit_probs))
