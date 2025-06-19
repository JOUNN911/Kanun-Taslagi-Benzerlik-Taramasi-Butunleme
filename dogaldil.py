import os
import re
import glob
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import zeyrek
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime
from snowballstemmer import TurkishStemmer

# Zeyrek ve SnowballStemmer
analyzer = zeyrek.MorphAnalyzer()
stemmer = TurkishStemmer()

# Stop words listesi
STOP_WORDS = set([
    "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç",
    "birşey", "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa",
    "diye", "eğer", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç",
    "için", "ile", "ise", "kez", "ki", "kim", "mı", "mu", "mü", "nasıl",
    "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
])

# Klasör yapısı oluşturma
os.makedirs("word2vec_models/stemmed", exist_ok=True)
os.makedirs("word2vec_models/lemmatized", exist_ok=True)
os.makedirs("zipf_analizi", exist_ok=True)
os.makedirs("temizlenmis_veriler", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)
os.makedirs("tfidf_models", exist_ok=True)
os.makedirs("benzer_metinler", exist_ok=True)

# .txt dosyalarını al
txt_files = sorted(glob.glob("gazeteler/*.txt"))


def clean_text(text):
    """Metin temizleme fonksiyonu"""
    text = text.lower()
    text = re.sub(r'[^\w\s.,ğüşıöçĞÜŞİÖÇ]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_and_tokenize(text):
    """Zeyrek ile lemmatizasyon ve SnowballStemmer ile stemming"""
    words = text.split()
    tokens = []
    lemmas = []
    stems = []

    for word in words:
        if word not in STOP_WORDS and len(word) >= 1:
            analysis = analyzer.lemmatize(word)
            lemma = analysis[0][1][0] if analysis else word
            stem = stemmer.stemWord(word)
            tokens.append(word)
            lemmas.append(lemma)
            stems.append(stem)
    return tokens, lemmas, stems


def process_text(text):
    """Metin işleme pipeline'ı"""
    cleaned = clean_text(text)
    tokens, lemmas, stems = normalize_and_tokenize(cleaned)
    return tokens, lemmas, stems


def debug_text_processing(txt_files):
    """Metin işleme sürecini analiz eder"""
    print("\n--- Veri Seti Analizi ---")
    all_tokens = []
    all_lemmas = []
    all_stems = []

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens, lemmas, stems = process_text(text)
                print(f"\nDosya: {txt_file}")
                print(f"Ham metin (ilk 100 karakter): {text[:100]}...")
                print(f"Token'lar: {tokens[:20]}...")
                print(f"Lemmatized: {lemmas[:20]}...")
                print(f"Stemmed: {stems[:20]}...")
                all_tokens.extend(tokens)
                all_lemmas.extend(lemmas)
                all_stems.extend(stems)
        except Exception as e:
            print(f"Hata: {txt_file} - {str(e)}")

    print(f"\nVeri seti çeşitliliği analizi:")
    print(f"Toplam dosya sayısı: {len(txt_files)}")
    print(f"Toplam token sayısı: {len(all_tokens)}")
    print(f"Toplam lemma sayısı: {len(all_lemmas)}")
    print(f"Toplam stem sayısı: {len(all_stems)}")
    print(f"Eşsiz token'lar: {len(set(all_tokens))} (Oran: {len(set(all_tokens)) / len(all_tokens):.4f})")
    print(f"Eşsiz lemma'lar: {len(set(all_lemmas))} (Oran: {len(set(all_lemmas)) / len(all_lemmas):.4f})")
    print(f"Eşsiz stem'ler: {len(set(all_stems))} (Oran: {len(set(all_stems)) / len(all_stems):.4f})")
    print(f"En sık 10 lemma: {Counter(all_lemmas).most_common(10)}")
    print(f"En sık 10 stem: {Counter(all_stems).most_common(10)}")


def apply_zipfs_law(words, output_prefix):
    """Zipf analizi"""
    word_counts = Counter(words)
    most_common = word_counts.most_common(1000)
    ranks = np.arange(1, len(most_common) + 1)
    freqs = [count for _, count in most_common]

    plt.figure(figsize=(12, 6))
    plt.plot(ranks, freqs, 'b-', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Zipf Kanunu - {output_prefix} (Log-Log)')
    plt.xlabel('Log(Sıralama)')
    plt.ylabel('Log(Frekans)')
    plt.grid(True, which="both", ls="-")
    plt.savefig(f"zipf_analizi/{output_prefix}_loglog_zipf.png", dpi=300)
    plt.close()

    with open(f"temizlenmis_veriler/{output_prefix}_sik_kelimeler.txt", "w", encoding="utf-8") as f:
        for word, count in most_common[:100]:
            f.write(f"{word}: {count}\n")


def create_tfidf_models(documents):
    """TF-IDF modellerini oluşturur"""
    lemma_texts = [' '.join(doc['lemmas']) for doc in documents]
    stem_texts = [' '.join(doc['stems']) for doc in documents]

    if not lemma_texts or all(len(text.strip()) == 0 for text in lemma_texts):
        print("Hata: Lemma metinleri boş.")
        lemma_texts = ["örnek metin tfidf için"]
    if not stem_texts or all(len(text.strip()) == 0 for text in stem_texts):
        print("Hata: Stem metinleri boş.")
        stem_texts = ["örnek metin tfidf için"]

    lemma_vectorizer = TfidfVectorizer(max_features=10000, min_df=1, token_pattern=r'\b[^\d\W]+\b', ngram_range=(1, 2))
    stem_vectorizer = TfidfVectorizer(max_features=10000, min_df=1, token_pattern=r'\b[^\d\W]+\b', ngram_range=(1, 2))
    lemma_tfidf = lemma_vectorizer.fit_transform(lemma_texts)
    stem_tfidf = stem_vectorizer.fit_transform(stem_texts)

    os.makedirs("tfidf_models", exist_ok=True)
    joblib.dump(lemma_vectorizer, 'tfidf_models/tfidf_lemmatized.model')
    joblib.dump(stem_vectorizer, 'tfidf_models/tfidf_stemmed.model')
    pd.DataFrame(lemma_tfidf.toarray(), columns=lemma_vectorizer.get_feature_names_out()).to_csv(
        'tfidf_models/lemma_tfidf_matrix.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(stem_tfidf.toarray(), columns=stem_vectorizer.get_feature_names_out()).to_csv(
        'tfidf_models/stem_tfidf_matrix.csv', index=False, encoding='utf-8-sig')

    return lemma_vectorizer, stem_vectorizer


def split_into_sentences(tokens):
    """Token'ları cümlelere böler"""
    sentences = []
    current_sentence = []
    for word in tokens:
        if word == ".":
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(word)
    if current_sentence:
        sentences.append(current_sentence)
    return sentences


def create_word2vec_models(documents):
    """Word2Vec modellerini oluşturur"""
    w2v_configs = [
        {'vector_size': 100, 'window': 5, 'sg': 0, 'min_count': 1, 'workers': 4},  # CBOW
        {'vector_size': 100, 'window': 5, 'sg': 1, 'min_count': 1, 'workers': 4},  # Skip-Gram
        {'vector_size': 100, 'window': 10, 'sg': 0, 'min_count': 1, 'workers': 4},
        {'vector_size': 100, 'window': 10, 'sg': 1, 'min_count': 1, 'workers': 4},
        {'vector_size': 200, 'window': 5, 'sg': 0, 'min_count': 1, 'workers': 4},
        {'vector_size': 200, 'window': 5, 'sg': 1, 'min_count': 1, 'workers': 4},
        {'vector_size': 200, 'window': 10, 'sg': 0, 'min_count': 1, 'workers': 4},
        {'vector_size': 200, 'window': 10, 'sg': 1, 'min_count': 1, 'workers': 4},
    ]

    w2v_models = []
    for i, config in enumerate(w2v_configs, 1):
        lemma_sentences = [split_into_sentences(doc['lemmas']) for doc in documents]
        stem_sentences = [split_into_sentences(doc['stems']) for doc in documents]
        lemma_sentences = [sent for doc in lemma_sentences for sent in doc]
        stem_sentences = [sent for doc in stem_sentences for sent in doc]

        lemma_model = Word2Vec(lemma_sentences, **config)
        stem_model = Word2Vec(stem_sentences, **config)
        lemma_model.save(f"word2vec_models/lemmatized/lemma_model_{i}.model")
        stem_model.save(f"word2vec_models/stemmed/stem_model_{i}.model")
        w2v_models.append((lemma_model, stem_model, config))
        print(
            f"Word2Vec Model {i} (Lemma): vector_size={config['vector_size']}, window={config['window']}, sg={config['sg']} ({'CBOW' if config['sg'] == 0 else 'Skip-Gram'})")
        print(
            f"Word2Vec Model {i} (Stem): vector_size={config['vector_size']}, window={config['window']}, sg={config['sg']} ({'CBOW' if config['sg'] == 0 else 'Skip-Gram'})")

    return w2v_models


def compute_cosine_similarities(input_text, documents, tfidf_lemma_model, tfidf_stem_model, w2v_models):
    """TF-IDF ve Word2Vec ile benzerlik hesaplar"""
    _, input_lemmas, input_stems = process_text(input_text)
    input_lemma_vec = tfidf_lemma_model.transform([' '.join(input_lemmas)])
    input_stem_vec = tfidf_stem_model.transform([' '.join(input_stems)])

    similarities = []
    # TF-IDF benzerlikleri
    lemma_tfidf_matrix = tfidf_lemma_model.transform([' '.join(doc['lemmas']) for doc in documents])
    stem_tfidf_matrix = tfidf_stem_model.transform([' '.join(doc['stems']) for doc in documents])
    lemma_similarities = cosine_similarity(input_lemma_vec, lemma_tfidf_matrix)[0]
    stem_similarities = cosine_similarity(input_stem_vec, stem_tfidf_matrix)[0]
    similarities.append(
        ('TF-IDF Lemma', [(doc['document_id'], score) for doc, score in zip(documents, lemma_similarities)]))
    similarities.append(
        ('TF-IDF Stem', [(doc['document_id'], score) for doc, score in zip(documents, stem_similarities)]))

    # Word2Vec benzerlikleri
    def avg_vector(words, model):
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

    for i, (lemma_model, stem_model, config) in enumerate(w2v_models, 1):
        input_vec_lemma = avg_vector(input_lemmas, lemma_model)
        input_vec_stem = avg_vector(input_stems, stem_model)
        lemma_sim_w2v = []
        stem_sim_w2v = []
        for doc in documents:
            doc_vec_lemma = avg_vector(doc['lemmas'], lemma_model)
            doc_vec_stem = avg_vector(doc['stems'], stem_model)
            lemma_score = cosine_similarity([input_vec_lemma], [doc_vec_lemma])[0][0] if np.any(doc_vec_lemma) else 0
            stem_score = cosine_similarity([input_vec_stem], [doc_vec_stem])[0][0] if np.any(doc_vec_stem) else 0
            lemma_sim_w2v.append((doc['document_id'], lemma_score))
            stem_sim_w2v.append((doc['document_id'], stem_score))
        similarities.append(
            (f'Word2Vec Lemma Model {i} (vector_size={config["vector_size"]}, window={config["window"]}, {"CBOW" if config["sg"] == 0 else "Skip-Gram"})',
             lemma_sim_w2v))
        similarities.append(
            (f'Word2Vec Stem Model {i} (vector_size={config["vector_size"]}, window={config["window"]}, {"CBOW" if config["sg"] == 0 else "Skip-Gram"})',
             stem_sim_w2v))

    return similarities


def save_top_matches(similarities, method_name, documents, timestamp):
    """En benzer 5 metni kaydeder"""
    out_dir = f"benzer_metinler/{timestamp}_{method_name.replace(' ', '_')}_top5"
    os.makedirs(out_dir, exist_ok=True)

    top5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    for i, (doc_id, score) in enumerate(top5, 1):
        content = next(doc['original_text'] for doc in documents if doc['document_id'] == doc_id)
        filename = f"{i:02d}_{score:.4f}_{doc_id}.txt"
        with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as out_f:
            out_f.write(f"Document: {doc_id}, Benzerlik Skoru: {score:.4f}\n\n{content}")


def create_top_matches_excel(similarities, timestamp):
    """Tüm modellerin en benzer metinlerini tek bir Excel dosyasına yazar"""
    excel_data = []
    for method_name, sims in similarities:
        top5 = sorted(sims, key=lambda x: x[1], reverse=True)[:5]
        row = {'Model': method_name}
        for i, (doc_id, score) in enumerate(top5, 1):
            # Find the corresponding text file
            folder_name = f"{timestamp}_{method_name.replace(' ', '_')}_top5"
            file_path = glob.glob(f"benzer_metinler/{folder_name}/{i:02d}_{score:.4f}_{doc_id}.txt")
            if file_path:
                try:
                    with open(file_path[0], 'r', encoding='utf-8') as f:
                        # Skip the first line (metadata) and read the content
                        content = ''.join(f.readlines()[2:]).strip()
                        # Remove any trailing "..." if present
                        content = content.rstrip('...')
                        row[f'Document {i} (ID: {doc_id}, Score: {score:.4f})'] = content
                except Exception as e:
                    print(f"Hata: {file_path[0]} okunurken sorun oluştu - {str(e)}")
                    row[f'Document {i} (ID: {doc_id}, Score: {score:.4f})'] = "Hata: İçerik okunamadı"
            else:
                row[f'Document {i} (ID: {doc_id}, Score: {score:.4f})'] = "Dosya bulunamadı"
        excel_data.append(row)

    # Excel dosyasına kaydet
    excel_df = pd.DataFrame(excel_data)
    excel_file = f"benzer_metinler/top_matches_{timestamp}.xlsx"
    excel_df.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"\nExcel dosyası oluşturuldu: {excel_file}")


def subjective_evaluation(similarities, method_name, documents, timestamp):
    """Anlamsal değerlendirme"""
    top5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    scores = []
    print(f"\n--- En Benzer 5 Metin ({method_name}) ---")
    for doc_id, score in top5:
        content = next(doc['original_text'] for doc in documents if doc['document_id'] == doc_id)
        normalized_score = min(5, max(1, round(score * 5)))
        scores.append(normalized_score)
        print(f"Document: {doc_id}, Benzerlik: {score:.4f}")
        print(f"Content (ilk 500 karakter): {content[:500]}...")
        print(f"Puan: {normalized_score}\n")
    avg_score = np.mean(scores) if scores else 0
    print(f"{method_name} Ortalama Puan: {avg_score:.2f}")
    return top5, scores, avg_score


def jaccard_similarity(set1, set2):
    """Jaccard benzerliği"""
    set1 = set(set1)
    set2 = set(set2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def ranking_agreement(similarities, timestamp):
    """Sıralama tutarlılığı analizi"""
    top5_docs = {name: [doc_id for doc_id, _ in sorted(sims, key=lambda x: x[1], reverse=True)[:5]] for name, sims in
                 similarities}
    model_names = [name for name, _ in similarities]
    jaccard_matrix = np.zeros((len(model_names), len(model_names)))

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                jaccard_matrix[i, j] = 1.0
            else:
                jaccard_matrix[i, j] = jaccard_similarity(top5_docs[name1], top5_docs[name2])

    jaccard_df = pd.DataFrame(jaccard_matrix, index=model_names, columns=model_names)
    jaccard_df.to_csv(f"benzer_metinler/jaccard_matrix_{timestamp}.csv", encoding='utf-8-sig')
    return jaccard_df


def process_files():
    """Dosyaları işler ve modelleri oluşturur"""
    if not txt_files:
        print("Hata: 'gazeteler' klasöründe .txt dosyası bulunamadı.")
        return None, None, None

    # Veri setini analiz et
    debug_text_processing(txt_files)

    documents = []
    for i, txt_file in enumerate(txt_files, 1):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens, lemmas, stems = process_text(text)
                documents.append({
                    'document_id': f'doc{i}',
                    'original_text': text,
                    'lemmas': lemmas,
                    'stems': stems
                })
        except Exception as e:
            print(f"Hata: {txt_file} - {str(e)}")

    # Zipf analizi
    all_tokens = [token for doc in documents for token in doc['lemmas']]
    apply_zipfs_law(all_tokens, "lemma")
    all_stems = [stem for doc in documents for stem in doc['stems']]
    apply_zipfs_law(all_stems, "stem")

    # TF-IDF modelleri
    tfidf_lemma_model, tfidf_stem_model = create_tfidf_models(documents)

    # Word2Vec modelleri
    w2v_models = create_word2vec_models(documents)

    # Temizlenmiş verileri kaydet
    with open("temizlenmis_veriler/anlam_butunlugu_bozan_kelimeler.txt", "w", encoding="utf-8") as f:
        f.write("STOP WORDS LİSTESİ:\n")
        f.write("\n".join(sorted(STOP_WORDS)))
        word_counts = Counter(all_tokens)
        f.write("\n\nEN SIK KARŞILAŞILAN ANLAMSIZ KELİMELER:\n")
        for word, count in word_counts.most_common(50):
            if word not in STOP_WORDS and len(word) < 4:
                f.write(f"{word}: {count}\n")

    return documents, tfidf_lemma_model, tfidf_stem_model, w2v_models


def main():
    """Ana iş akışı"""
    print("Metin işleme başlıyor...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dosyaları işle
    result = process_files()
    if result is None:
        return
    documents, tfidf_lemma_model, tfidf_stem_model, w2v_models = result

    # Örnek metni veri setinden seç
    sample_text = documents[0]['original_text']
    print(f"\nSeçilen örnek metin: {sample_text[:100]}...")

    # Benzerlikleri hesapla
    similarities = compute_cosine_similarities(sample_text, documents, tfidf_lemma_model, tfidf_stem_model, w2v_models)

    # Sonuçları kaydet ve değerlendir
    results = []
    for method_name, sims in similarities:
        top5, scores, avg_score = subjective_evaluation(sims, method_name, documents, timestamp)
        save_top_matches(sims, method_name, documents, timestamp)
        results.append({
            'Model': method_name,
            'Top 5 Documents': ', '.join([doc_id for doc_id, _ in top5]),
            'Scores': scores,
            'Average Score': avg_score
        })

    # Excel dosyası oluştur
    create_top_matches_excel(similarities, timestamp)

    # Anlamsal değerlendirme tablosu
    scores_df = pd.DataFrame(results)
    scores_df.to_csv(f"benzer_metinler/model_scores_{timestamp}.csv", index=False, encoding='utf-8-sig')
    print("\nAnlamsal Değerlendirme Sonuçları:")
    print(scores_df[['Model', 'Top 5 Documents', 'Average Score']].to_string(index=False))

    # Sıralama tutarlılığı
    print("\n--- Sıralama Tutarlılığı (Jaccard Benzerliği) ---")
    jaccard_df = ranking_agreement(similarities, timestamp)
    print(jaccard_df)


if __name__ == "__main__":
    main()