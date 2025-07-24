from llm import TextGenerator
from logging import getLogger
logger = getLogger('text generator')
 

def generate_texts(domain_words: list[str]) -> list[str]:
    """
    Main function to generate texts for a list of domain words.
    """
    # Initialize our text generation agent.
    logger.info("Начало генерации предложений")
    generator = TextGenerator()
    sentences:list[str] = []
    for word in domain_words:
        try:
            sentence:str = generator.generate_sentence(word)
            sentences.append(sentence)
            logger.info(f"Сгенерировано предложения. Слово: '{word}' -> Предложение: {sentence[:30]}")
        except Exception as e:
            logger.error(f"Не удалось сгенерировать предложение для слова '{word}': {e}")
    return sentences
