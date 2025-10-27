from transformers import pipeline

def extract_entities_phrases():
    extractor = pipeline("ner", grouped_entities=True)

    texts = [
        "Экран яркий, но батарея садится быстро.",
        "У телефона отличная камера, но плохой звук.",
        "Ноутбук ASUS работает быстро, но сильно греется.",
        "Планшет Samsung лёгкий, но экран бликует.",
        "Apple iPhone радует дизайном, но дорого стоит."
    ]

    for t in texts:
        entities = extractor(t)
        print(f"\nТекст: {t}")
        for e in entities:
            print(f" → {e['word']} ({e['entity_group']}, {e['score']:.3f})")

if __name__ == "__main__":
    extract_entities_phrases()
