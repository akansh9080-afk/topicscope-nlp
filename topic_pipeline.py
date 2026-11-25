"""TopicScope - core NLP pipeline (cleaning, vectorization, LDA / NMF topic modelling)."""
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS)
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Newsgroup/email noise words that carry no topical meaning
EXTRA_STOP = {"edu", "com", "org", "net", "subject", "lines", "writes", "wrote",
              "article", "organization", "just", "like", "know", "think", "use",
              "used", "using", "does", "don", "didn", "going", "say", "said",
              "make", "good", "way", "people", "thanks", "mail", "posting"}
STOP = set(ENGLISH_STOP_WORDS) | EXTRA_STOP


def clean_text(text: str) -> str:
    """Lowercase, strip emails/urls/numbers/punctuation, tokenize, drop stopwords."""
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)            # emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # urls
    text = re.sub(r"[^a-z\s]", " ", text)           # numbers & punctuation
    tokens = [w for w in text.split() if len(w) > 2 and w not in STOP]
    return " ".join(tokens)


class TopicModeler:
    def __init__(self, n_topics=10, method="LDA", max_features=5000, random_state=42):
        self.n_topics = n_topics
        self.method = method.upper()
        self.max_features = max_features
        self.random_state = random_state

    # ---------- training ----------
    def fit(self, docs):
        cleaned = [clean_text(d) for d in docs]
        keep = [i for i, c in enumerate(cleaned) if len(c.split()) >= 3]
        if len(keep) < max(self.n_topics, 2):
            raise ValueError("Not enough usable documents after cleaning. "
                             "Add more/longer documents or reduce the topic count.")
        cleaned = [cleaned[i] for i in keep]
        small = len(cleaned) < 50
        min_df, max_df = (1, 1.0) if small else (2, 0.5)

        if self.method == "LDA":
            self.vectorizer = CountVectorizer(max_features=self.max_features,
                                              min_df=min_df, max_df=max_df)
            X = self.vectorizer.fit_transform(cleaned)
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics, learning_method="online",
                max_iter=15, random_state=self.random_state, n_jobs=-1)
        else:  # NMF on TF-IDF
            self.vectorizer = TfidfVectorizer(max_features=self.max_features,
                                              min_df=min_df, max_df=max_df)
            X = self.vectorizer.fit_transform(cleaned)
            self.model = NMF(n_components=self.n_topics, init="nndsvd",
                             random_state=self.random_state, max_iter=400)
        W = self.model.fit_transform(X)
        self.vocab_size = X.shape[1]
        self.kept_idx = keep
        self.doc_topics = self._normalize(W)
        self.perplexity = (float(self.model.perplexity(X))
                           if self.method == "LDA" else None)
        return self

    # ---------- inference ----------
    @staticmethod
    def _normalize(W):
        s = W.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return W / s

    def transform(self, docs):
        X = self.vectorizer.transform([clean_text(d) for d in docs])
        return self._normalize(self.model.transform(X))

    def top_words(self, n=10):
        words = np.array(self.vectorizer.get_feature_names_out())
        out = []
        for comp in self.model.components_:
            idx = comp.argsort()[::-1][:n]
            w = comp[idx] / comp.sum()
            out.append(list(zip(words[idx], w)))
        return out

    def topic_label(self, k, n=3):
        return f"Topic {k + 1}: " + ", ".join(w for w, _ in self.top_words(n)[k])

    # ---------- persistence ----------
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
