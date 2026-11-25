# 🧠 TopicScope – Text Topic Modelling (NLP Project 9)

Discovers hidden topics in a collection of documents using **LDA** and **NMF**, with a Streamlit GUI.

## Pipeline
Dataset (20 Newsgroups) → cleaning → tokenization / stop-word removal → Bag-of-Words (LDA) or TF-IDF (NMF)
→ topic model → top words per topic → document-topic assignment → evaluation → GUI.

## Setup
```bash
pip install -r requirements.txt
python train.py --topics 20 --method LDA    # optional: train + evaluate (Perplexity, NMI, Homogeneity, Purity)
streamlit run app.py
```
(First run downloads 20 Newsgroups automatically; internet needed once.)

## GUI features
Upload multiple .txt/.csv files (or use 20 Newsgroups / paste text) · Run Topic Modelling button ·
topics with important words · topic assigned per document · topic distribution chart · CSV download ·
predict topic of new text.

## Files
`topic_pipeline.py` core NLP code · `train.py` training + evaluation · `app.py` Streamlit GUI
