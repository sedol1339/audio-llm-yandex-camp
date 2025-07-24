from random import choice

from speechkit import configure_credentials, creds, model_repository


class YTextToAudioApi:
    def __init__(self, api_key: str):
        configure_credentials(
            yandex_credentials=creds.YandexCredentials(api_key=api_key)
        )
        self.model = model_repository.synthesis_model()

    def synthesize(
        self,
        text: str,
        export_path: str,
        voice: str | None = None,  # none => random
        role: str | None = None,  # none => random
    ) -> str:
        # return path
        if voice is None and role is None:
            voice, role = choice(self.get_voices_and_roles())
        self.model.voice = voice
        self.model.role = role
        result = self.model.synthesize(text, raw_format=False)
        result.export(export_path, "wav")
        return export_path

    def get_voices_and_roles(self) -> list[tuple[str, str]]:
        voices = [
            ("alena", "good"),
            ("ermil", "neutral"),
            ("jane", "neutral"),
            ("jane", "good"),
            ("jane", "evil"),
            ("omazh", "neutral"),
            ("omazh", "evil"),
            ("zahar", "neutral"),
            ("zahar", "good"),
            ("dasha", "neutral"),
            ("dasha", "good"),
            ("dasha", "friendly"),
            ("julia", "neutral"),
            ("julia", "strict"),
            ("lera", "neutral"),
            ("lera", "friendly"),
            ("masha", "good"),
            ("masha", "strict"),
            ("masha", "friendly"),
            ("marina", "neutral"),
            ("marina", "whisper"),
            ("marina", "friendly"),
            ("alexander", "neutral"),
            ("alexander", "good"),
            ("kirill", "neutral"),
            ("kirill", "strict"),
            ("kirill", "good"),
            ("anton", "neutral"),
            ("anton", "good"),
        ]
        return voices


def generate_audio(texts: list[str]) -> list[str]:
    ln = len(texts)
    import os

    from dotenv import load_dotenv

    load_dotenv()
    tta = YTextToAudioApi(os.getenv("TTS_API_KEY"))
    paths: list[str] = []

    for i, text in enumerate(texts):
        path = os.path.join(
            "src/asr_eval/words_recognize_errors/genereted_audio",
            f"text_synthesize_{i}.wav",
        )
        paths.append(path)
        tta.synthesize(text, path)
    return paths


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    tta = YTextToAudioApi(os.getenv("TTS_API_KEY"))
    text = "Я читал, что настройка гиперпараметрических значений в модели машинного обучения может сильно повлиять на её точность."
    tta.synthesize(text, "spech.wav")
