import os
import re

VOWEL_RUNS = re.compile("[aeiouy]+", flags=re.I)
EXCEPTIONS = re.compile(
    # fixes trailing e issues:
    # smite, scared
    "[^aeiou]e[sd]?$|"
    # fixes adverbs:
    # nicely
    + "[^e]ely$",
    flags=re.I
)
ADDITIONAL = re.compile(
    # fixes incorrect subtractions from exceptions:
    # smile, scarred, raises, fated
    "[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$|"
    # fixes miscellaneous issues:
    # flying, piano, video, prism, fire, evaluate
    + ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua",
    flags=re.I
)


def _count_syllables(word):
    vowel_runs = len(VOWEL_RUNS.findall(word))
    exceptions = len(EXCEPTIONS.findall(word))
    additional = len(ADDITIONAL.findall(word))
    return max(1, vowel_runs - exceptions + additional)
    
dir_path = os.path.dirname(os.path.realpath(__file__))
words = {}

with open(f'{dir_path}/cmudict', 'r', encoding='latin-1') as in_file:
    for line in in_file:
        if line.startswith(';;;'):
            continue
        word, phones = line.split(" ", 1)
        words[word] = sum([phones.count(i) for i in '012'])

def syllable_count(word):
    word = word.upper().strip()
    if word in words:
        return words[word]
    return _count_syllables(word)
    