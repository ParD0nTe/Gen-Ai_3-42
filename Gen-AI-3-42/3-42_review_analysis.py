# 3-42_review_analysis.py
from transformers import pipeline
from collections import defaultdict
import pymorphy3
import json
import re

print("Загрузка моделей...")

# === 1. Инициализация моделей ===
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment"
)

entity_extractor = pipeline(
    "ner",
    model="Gherman/bert-base-NER-Russian",
    grouped_entities=True
)

morph = pymorphy3.MorphAnalyzer()

# === 2. Вспомогательные функции ===
NEGATIVE_HINTS = [
    "плохо", "садится", "слаб", "медленно", "дорого",
    "греется", "бликует", "шумно", "ломается", "не работает", "скользкий"
]

ASPECT_FILTER = [
    "экран", "камера", "звук", "корпус", "батарея", "производительность",
    "дизайн", "скорость", "процессор", "память", "дисплей", "клавиатура",
    "вес", "система", "интерфейс", "качество", "зарядка"
]

def clean_ner_entities(entities):
    cleaned = []
    for e in entities:
        if isinstance(e, dict):
            word = e.get("word", "")
        else:
            word = str(e)
        word = word.replace("##", "").strip()
        if word:
            cleaned.append(word.lower())
    return list(dict.fromkeys(cleaned))

def extract_aspects_fallback(text):
    """Выделяем существительные из текста (только потенциальные аспекты)."""
    words = re.findall(r"[А-Яа-яA-Za-zёЁ]+", text)
    nouns = []
    for w in words:
        p = morph.parse(w)[0]
        if "NOUN" in p.tag:
            lemma = p.normal_form.lower()
            if lemma in ASPECT_FILTER:
                nouns.append(lemma)
    return list(dict.fromkeys(nouns))

def normalize_sentiment(label):
    lab = (label or "").upper()
    if "POS" in lab or "GOOD" in lab or "LABEL_2" in lab:
        return "POSITIVE"
    elif "NEG" in lab or "BAD" in lab or "LABEL_0" in lab:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def detect_negative_lexically(text):
    """Если встречаются явно негативные слова — считаем NEGATIVE."""
    t = text.lower()
    return any(word in t for word in NEGATIVE_HINTS)

def split_review(text):
    """Разбиваем отзыв по 'но' и знакам препинания."""
    parts = re.split(r"[.!?]|,?\s+но\s+", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]

# === 3. Синтетические отзывы ===
reviews = [
    {"product": "Phone X", "review": "Экран очень яркий, но батарея быстро садится."},
    {"product": "Phone X", "review": "Отличная камера и звук, но корпус скользкий."},
    {"product": "Tablet Z", "review": "Работает плавно, но экран бликует на солнце."},
    {"product": "Tablet Z", "review": "Хорошая производительность, но слабая батарея."},
    {"product": "Phone X", "review": "Камера просто супер, батарея держит плохо."}
]

# === 4. Анализ отзывов ===
report = defaultdict(lambda: {"positive_aspects": [], "negative_aspects": []})

for r in reviews:
    product = r["product"]
    parts = split_review(r["review"])

    for part in parts:
        # анализ тональности
        sent_res = sentiment_analyzer(part)[0]
        sentiment = normalize_sentiment(sent_res.get("label", ""))

        # лексическая коррекция
        if detect_negative_lexically(part):
            sentiment = "NEGATIVE"

        # извлекаем аспекты
        entities = entity_extractor(part)
        aspects = clean_ner_entities(entities)
        if not aspects:
            aspects = extract_aspects_fallback(part)

        for aspect in aspects:
            if sentiment == "POSITIVE":
                report[product]["positive_aspects"].append(aspect)
            elif sentiment == "NEGATIVE":
                report[product]["negative_aspects"].append(aspect)

# === 5. Очистка отчёта ===
for product, data in report.items():
    for key in ["positive_aspects", "negative_aspects"]:
        # убираем дубликаты и пересечения
        items = list(dict.fromkeys(data[key]))
        # если аспект и в positive, и в negative — оставляем только в negative
        if key == "negative_aspects":
            items = [a for a in items if a not in data["positive_aspects"]]
        data[key] = items

# === 6. Формирование сводки ===
summary_parts = []
for product, data in report.items():
    pos = data["positive_aspects"]
    neg = data["negative_aspects"]

    pos_text = ", ".join(pos) if pos else "нет положительных аспектов"
    neg_text = ", ".join(neg) if neg else "нет отрицательных аспектов"

    summary_parts.append(f"Для {product}: хвалят {pos_text}, но жалуются на {neg_text}.")

summary_text = " ".join(summary_parts)

# === 7. Сохранение ===
final_report = {
    "summary": summary_text,
    "products": {k: v for k, v in report.items()}
}

with open("review_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=4)

# === 8. Вывод ===
print("\n=== СВОДКА ===")
print(summary_text)
print("\nОтчёт сохранён в 'review_report.json'")
