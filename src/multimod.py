# type: ignore

import os
import ssl
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Настройка SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загрузка ресурсов NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class TranscriptionFixerAgent:
    def __init__(self, openai_api_key = os.environ.get("LCH_API_KEY")):
        self.llm = ChatOpenAI(
            model="openai/gpt-4o",
            temperature=0.3,
            base_url="https://api.vsegpt.ru/v1",
            api_key=openai_api_key
        )

        self.search_tool = DuckDuckGoSearchRun(
            name="web_search",
            description="Поиск информации в интернете для уточнения терминов"
        )

        self.tools = [self.search_tool]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
Ты — профессиональный редактор и эксперт по исправлению транскрибаций аудио на русском языке. 
В твои задачи входит:
1. Анализ текста транскрибации, выявление ошибок в терминах, именах и специфических фразах.
2. Проверка правильного написания терминов с помощью интернета.
3. Возврат полностью исправленного текста транскрибации.

Если термин вызывает сомнения — проверь его с помощью поиска. Если термин написан правильно и ты не нашёл ошибки - ничего не меняй.
После твоего выхода будет считаться WER с истинной разметкой, поэтому правильность каждого слова важно."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            max_iterations=10
        )

    def fix_transcription(self, input_transcription):
        try:
            agent_input = f"""
Это транскрибация аудио. В ней могут быть ошибки, особенно в сложных терминах и именах.
Пожалуйста, найди ошибки и исправь их, используя интернет-поиск. Если ты не находишь ошибок - ничего не меняй. Не добавляй никакие знаки препинания.

Транскрибация:

{input_transcription}


Если нашёл ошибки, верни исправленную транскрибацию. Там где нет ошибок, ничего не меняй.
Верни ТОЛЬКО транскрибацию и ничего больше. Не комментируй. Иначе ты умрешь.
"""
            result = self.agent_executor.invoke({"input": agent_input})
            return result['output'] if isinstance(result, dict) else result

        except Exception as e:
            return f"Произошла ошибка: {str(e)}"

# # Пример использования
# def example_usage():
#     fixer = TranscriptionFixerAgent()

#     input_text = """
# Сегодня мы обсуждаем влияние исскуственного интелекта на развитие нейросетивых технологий
# в медецине, таких как компьютерное зрение и обработки естесвенного языка.
# """

#     corrected_text = fixer.fix_transcription(input_text)

#     print("📌 Исправленная транскрибация:\n")
#     print(corrected_text)

# if __name__ == "__main__":
#     example_usage()
