from speechkit import model_repository, configure_credentials, creds



class YTextToAudioApi:
    def __init__(self, api_key:str, voice:str = 'jane', role:str = 'good'):
        configure_credentials(yandex_credentials=creds.YandexCredentials(api_key=api_key))
        self.model = model_repository.synthesis_model()
        self.model.voice = voice
        self.model.role = role

    
    def synthesize(self, text:str, export_path:str) -> str:
       # return path
       result = self.model.synthesize(text, raw_format=False)
       result.export(export_path, 'wav')

def generate_audio(texts:list[str]) -> list[str]:
    ln = len(texts)
    from dotenv import load_dotenv
    import os
    load_dotenv()
    tta = YTextToAudioApi(os.getenv('API_KEY'))
    paths:list[str] = []

    for i, text in enumerate(texts):
        path = os.path.join('genereted_audio', f'text_synthesize_{i}.wav')
        paths.append(path)
        tta.synthesize(text, path)
    return paths
    
        

if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    load_dotenv()
    tta = YTextToAudioApi(os.getenv('API_KEY'))
    text = 'Я читал, что настройка гиперпараметрических значений в модели машинного обучения может сильно повлиять на её точность.'
    tta.synthesize(text, 'spech.wav')
    
      