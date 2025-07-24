import os

def get_domain_words(topik:str='chemistry') -> list[str] | None:
    files = os.listdir('domain_words')
    topiks = [f[:f.find('-')] for f in files] # files must be [topik]-domain-cpecific.txt
    if topik not in topiks:
        return None
    domain_words = None
    with open(os.path.join('domain_words', files[topiks.index(topik)])) as words_file:
        domain_words = words_file.read().split()
    return domain_words

        




