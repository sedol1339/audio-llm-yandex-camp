from domain_words import get_domain_words
from generate_texts import generate_texts
from generate_audio import generate_audio
from pprint import pprint
from .models.voxtral_wrapper import VoxtralmWrapper
from numpy.random import choice
from scipy.io import wavfile

if __name__ =='__main__':
    words = choice(get_domain_words(), size=15)
    pprint(words)
    texts = generate_texts(words)
    pprint(texts)
    audios = generate_audio(texts)
    print(audios)
    voxtral_model = VoxtralmWrapper()
    waveforms = [wavfile.read(a)[1] for a in audios]
    voxtral_model(waveforms)
    
    

    
