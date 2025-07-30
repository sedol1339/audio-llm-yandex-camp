# type: ignore

import wikipediaapi
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from collections import defaultdict
from transformers import pipeline

def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'perluniprops', 'nonbreaking_prefixes', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

class WikipediaTermRetriever:
    def __init__(self, lang="ru", candidate_topics = [
            "биология", "химия", "физика", "история", 
            "медицина", "география", "искусство", "литература"
        ]):
        self.wiki = wikipediaapi.Wikipedia("MyRAGASR", lang)
        self.classifier = pipeline(
            "zero-shot-classification",
            model="vicgalle/xlm-roberta-large-xnli-anli"
        )
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.stopwords = set(stopwords.words("russian" if lang == 'ru' else 'english'))
        self.candidate_topics = candidate_topics
        self.cache = {}
        self.term_embeddings = {}
        download_nltk_resources()

    def detect_topic(self, text):
        """Определение темы с помощью zero-shot классификации"""
        result = self.classifier(text, self.candidate_topics)
        return result['labels'][0]

    def get_category_articles(self, category_name, max_articles=500):
        """Рекурсивная загрузка статей категории"""
        if category_name in self.cache:
            return self.cache[category_name]
        
        category = self.wiki.page(f"Category:{category_name}")
        if not category.exists():
            return []
            
        articles = []
        for page in category.categorymembers.values():
            if len(articles) > 5000:
                break
            
            if len(articles) >= max_articles:
                break
            if page.ns == wikipediaapi.Namespace.MAIN:
                articles.append({
                    'title': page.title,
                    'text': page.text[:10000],  # Ограничиваем размер
                    'url': page.fullurl
                })
            elif page.ns == wikipediaapi.Namespace.CATEGORY:
                sub_articles = self.get_category_articles(page.title.split(":")[1])
                articles.extend(sub_articles[:max_articles - len(articles)])
        
        self.cache[category_name] = articles
        return articles

    def preprocess_text(self, text):
        """Токенизация и очистка текста"""
        tokens = word_tokenize(text.lower())
        return [
            token for token in tokens 
            if re.fullmatch(r'[а-яё]{3,}', token) and token not in self.stopwords
        ]

    def build_term_index(self, articles):
        """Создание семантического индекса терминов"""
        term_freq = defaultdict(int)
        all_terms = []
        
        for article in articles:
            terms = self.preprocess_text(article['text'])
            for term in terms:
                term_freq[term] += 1
            all_terms.extend(terms)
            
        # Отбираем редкие и значимые термины
        unique_terms = [term for term in set(all_terms) if 1 < term_freq[term] < 20]
        
        # Создаем эмбеддинги для терминов
        term_embeddings = {}
        batch_size = 100
        for i in range(0, len(unique_terms), batch_size):
            batch = unique_terms[i:i+batch_size]
            embeddings = self.embedder.encode(batch)
            for term, emb in zip(batch, embeddings):
                term_embeddings[term] = emb
                
        return term_embeddings

    def find_similar_terms(self, query_terms, term_index, top_k=5, similarity_threshold=0.7):
        """Поиск семантически похожих терминов с учетом возможных ошибок"""
        similar_terms = {}
        
        # Получаем все термины из индекса
        all_index_terms = list(term_index.keys())
        
        for term in query_terms:
            # Если термин есть в индексе - используем обычный поиск
            if term in term_index:
                term_emb = term_index[term]
                similarities = {}
                
                for other_term, other_emb in term_index.items():
                    if term == other_term:
                        continue
                        
                    sim = cosine_similarity([term_emb], [other_emb])[0][0]
                    similarities[other_term] = sim
                    
                sorted_terms = sorted(similarities.items(), key=lambda x: -x[1])[:top_k]
                similar_terms[term] = sorted_terms
                continue
                
            # Если термина нет в индексе - ищем похожие по написанию и смыслу
            term_emb = self.embedder.encode(term)
            candidate_terms = []
            
            # Этап 1: Быстрый поиск по сходству строк (для опечаток)
            for index_term in all_index_terms:
                # Используем комбинацию семантического и строкового сходства
                semantic_sim = cosine_similarity([term_emb], [term_index[index_term]])[0][0]
                fuzzy_sim = fuzz.ratio(term, index_term) / 100
                combined_score = 0.6*semantic_sim + 0.4*fuzzy_sim
                
                if combined_score > similarity_threshold:
                    candidate_terms.append((index_term, combined_score))
            
            # Этап 2: Выбираем лучшие кандидаты
            if candidate_terms:
                candidate_terms.sort(key=lambda x: -x[1])
                similar_terms[term] = candidate_terms[:top_k]
            else:
                similar_terms[term] = []
                
        return similar_terms

    def process_query(self, asr_text, top_terms=10):
        """Полный цикл обработки запроса"""
        # 1. Определяем тему
        topic = self.detect_topic(asr_text)
        print(f"Определена тема: {topic}")
        
        # 2. Загружаем статьи по теме
        articles = self.get_category_articles(topic)
        print(f"Загружено статей: {len(articles)}")
        
        # 3. Строим семантический индекс терминов
        term_index = self.build_term_index(articles)
        print(f"Проиндексировано терминов: {len(term_index)}")
        
        # 4. Извлекаем термины из запроса
        query_terms = self.preprocess_text(asr_text)
        print(f"Термины для коррекции: {query_terms}")
        
        # 5. Находим похожие термины
        similar_terms = self.find_similar_terms(query_terms, term_index)
        print(f"Найденные похожие термины: {similar_terms}")
        
        # 6. Выбираем топ-N терминов
        all_terms = []
        for term, suggestions in similar_terms.items():
            for suggested_term, score in suggestions:
                all_terms.append((suggested_term, score))
                
        top_terms_list = sorted(all_terms, key=lambda x: -x[1])[:top_terms]
        
        return {
            'original_text': asr_text,
            'detected_topic': topic,
            'query_terms': query_terms,
            'suggested_terms': [term[0] for term in top_terms_list],
            'term_scores': dict(top_terms_list)
        }

# # Пример использования
# if __name__ == "__main__":
#     # Инициализация
#     retriever = WikipediaTermRetriever()
    
#     # Пример запроса с ошибками в транскрипции
#     asr_text = "шван предложил клеточную теорию в 19 веке"
    
#     # Обработка запроса
#     result = retriever.process_query(asr_text)
    
#     print("\nРезультаты:")
#     print(f"Оригинальный текст: {result['original_text']}")
#     print(f"Определенная тема: {result['detected_topic']}")
#     print(f"Термины для коррекции: {result['query_terms']}")
#     print("Рекомендуемые термины:")
#     for term, score in result['term_scores'].items():
#         print(f"  {term}: {score:.3f}")