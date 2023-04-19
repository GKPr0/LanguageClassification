import re
from collections import Counter, defaultdict
from pathlib import Path
import pickle

from Config import AllowedCharacters


def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def load_train_data(root, pattern="*.txt", languages=None, encoding="utf8"):
    data = defaultdict(lambda: [])
    for file in Path(root).rglob(pattern):

        with open(file, "r", encoding=encoding) as f:
            text = f.read()

        allowed_chars = get_allowed_characters(languages)
        sentences = clear_and_split_text(text, allowed_chars=allowed_chars)

        lng = str(file.parent.name)
        data[lng].extend(sentences)

    return dict(data)


def load_test_data(root, pattern="*.txt", languages=None, encoding="utf8"):
    data = []
    for file in Path(root).rglob(pattern):
        with open(file, "r", encoding=encoding) as file:
            for line in file.readlines():
                lng, text = line.split(";", maxsplit=1)

                allowed_chars = get_allowed_characters(languages)
                sentences = clear_and_split_text(text, allowed_chars=allowed_chars)

                data.append((lng, sentences))

    return data


def get_allowed_characters(languages: [str]):
    if not languages:
        return "\\p{L}"

    chars = ""
    for lng in languages:
        chars += AllowedCharacters[lng]

    return "".join(set(chars)) + " "


def clear_and_split_text(text: str, allowed_chars: str = "\\p{L}", sym_pre: str = "", sym_post: str = "") -> [str]:
    text = text.lower().replace("\n", " ")

    reg_not_allowed = re.compile(fr"[^{allowed_chars}]")
    reg_double_space = re.compile(r"[ ]+")

    sentences = []
    for sent in re.split(r'[.!?]\s', text):
        cleaned_sent = reg_not_allowed.sub(" ", sent)
        cleaned_sent = reg_double_space.sub(" ", cleaned_sent)
        cleaned_sent = f"{sym_pre} {cleaned_sent} {sym_post}"
        cleaned_sent = cleaned_sent.strip()

        if not cleaned_sent:
            continue

        sentences.append(cleaned_sent)

    return sentences


def create_per_line_char_ngram(text: [str, list[str]], ngram_order: int):
    if type(text) is str:
        text = [text]

    create_ngram = lambda chars, start_i: (chars[i] for i in range(start_i, start_i + ngram_order))
    ngrams = []
    for line in text:
        ngrams.extend([tuple(create_ngram(line, i)) for i in range(len(line) - ngram_order + 1)])

    return Counter(ngrams)


def create_smoothed_ngram(text: [str], ngram_order: int):
    if ngram_order < 2:
        raise Exception("Only ngrams of second order and above can be smoothed by this method.")

    allowedChars = get_allowed_characters(["CS", "SK"])
    ngrams = create_per_line_char_ngram(text, ngram_order=ngram_order)

    unique_preceding_combinations = len(allowedChars) ** ngram_order

    ngrams_model = {}
    for preceding_words in cartesian_product(allowedChars, ngram_order - 1):
        N,T = 0,0
        for ngram in ngrams:
            ngram_exists = True
            for nth, nth_gram in zip(preceding_words, ngram):
                if nth != nth_gram:
                    ngram_exists = False
                    break

            if not ngram_exists:
                continue

            N += ngrams[ngram]
            T += 1

        N = max(N, 1)
        T = max(T, 1)
        Z = unique_preceding_combinations - T
        not_found_prob = T/(Z*(N+T))
        for last in allowedChars:
            ngram = (*preceding_words, *last)
            ngrams_model[ngram] = ngrams[ngram]/(N+T) if ngram in ngrams else not_found_prob

    return ngrams_model


def cartesian_product(input_list, n):
    for item in input_list:
        if n == 1:
            yield [*item]
        else:
            for next_item in cartesian_product(input_list, n-1):
                yield [*item] + next_item
