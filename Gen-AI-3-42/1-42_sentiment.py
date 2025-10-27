from transformers import pipeline
import matplotlib.pyplot as plt

def analyze_sentiment_phrases():
    analyzer = pipeline("sentiment-analysis")

    phrases = [
        "Я счастлив",
        "Мне плохо",
        "Сегодня отличный день",
        "Это ужасно",
        "Неплохо, но могло быть лучше"
    ]

    results = analyzer(phrases)

    for text, res in zip(phrases, results):
        print(f"Фраза: {text}\n → {res['label']} (оценка: {res['score']:.3f})\n")

    labels = [res['label'] for res in results]
    scores = [res['score'] for res in results]

    plt.figure(figsize=(8, 4))
    plt.bar(phrases, scores, color=['green' if l == 'POSITIVE' else 'red' for l in labels])
    plt.xticks(rotation=45, ha='right')
    plt.title("Анализ тональности фраз (GenAI-1-42)")
    plt.ylabel("Уверенность модели")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_sentiment_phrases()
