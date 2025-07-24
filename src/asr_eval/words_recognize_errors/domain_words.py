import os
from random import choice


def get_domain_words(topik: str | None = None) -> list[str] | None:
    dir = "src/asr_eval/words_recognize_errors/domain_words"
    files = os.listdir(dir)
    if topik is None:
        file = os.path.join(dir, choice(files))
    else:
        topiks = [
            f[: f.find("-")] for f in files
        ]  # files must be [topik]-domain-cpecific.txt
        if topik not in topiks:
            return None
        file = os.path.join(dir, files[topiks.index(topik)])
    domain_words = None
    with open(file) as words_file:
        domain_words = words_file.read().split()
    return domain_words
