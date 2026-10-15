# NLP and Text Processing Toolkit

A comprehensive Google Colab notebook containing implementations of fundamental Natural Language Processing (NLP) techniques, regex pattern matching, statistical text analysis, and speech recognition integration using Python, NLTK, spaCy, and scikit-learn.

## 📋 Table of Contents
1. [Prerequisites & Libraries](#prerequisites--libraries)
2. [Tokenization (NLTK)](#1-tokenization-nltk)
3. [spaCy NLP Pipeline (POS & NER)](#2-spacy-nlp-pipeline-pos--ner)
4. [Stop Word Removal](#3-stop-word-removal)
5. [Regex Pattern Extraction (Register Numbers)](#4-regex-pattern-extraction-register-numbers)
6. [N-Grams Generation](#5-n-grams-generation)
7. [Document-Term Matrix (CountVectorizer)](#6-document-term-matrix-countvectorizer)
8. [Rule-Based Sentiment Analysis](#7-rule-based-sentiment-analysis)
9. [Stemming vs. Lemmatization (NLTK)](#8-stemming-vs-lemmatization-nltk)
10. [Text Data Analysis with Pandas](#9-text-data-analysis-with-pandas)
11. [Speech-to-Text & NLP Audio Pipeline](#10-speech-to-text--nlp-audio-pipeline)

---

## 🛠 Prerequisites & Libraries
The notebook utilizes the following Python libraries:
* `nltk` (Natural Language Toolkit)
* `spacy` (Industrial-Strength Natural Language Processing)
* `scikit-learn` (`CountVectorizer` for text vectorization)
* `pandas` (Tabular data manipulation and statistical summaries)
* `SpeechRecognition` & `pyaudio` / `ffmpeg` (Audio handling and transcription)

---

## Notebook Modules Breakdown

### 1. Tokenization (NLTK)
Splits raw text into individual words and sentences using NLTK's `word_tokenize` and `sent_tokenize`.
* **Example Input:** `"there is no one"`
* **Tokens:** `['there', 'is', 'no', 'one']`
* **Sentences:** `['there is no one']`

### 2. spaCy NLP Pipeline (POS & NER)
Loads the pre-trained English model (`en_core_web_sm`) to perform Part-of-Speech (POS) tagging and Named Entity Recognition (NER).
* **POS Tagging:** Identifies grammatical components (e.g., Nouns, Verbs, Adjectives).
* **NER:** Detects real-world entities such as Organizations (`ORG`), Geopolitical Entities (`GPE`), and Monetary values (`MONEY`).

### 3. Stop Word Removal
Filters out common, non-informative words (like "the", "in", "over") from text using spaCy token flags (`token.is_stop` and `token.is_punct`).

### 4. Regex Pattern Extraction (Register Numbers)
Uses regular expressions to extract structured identifiers from text matching custom formats.
* **Regex Pattern:** `\b(\d{2,4})([A-Za-z]{2,5})(\d{2,4})\b`
* **Extracted Attributes:** Year/Batch, Department Code, and Roll Number.

### 5. N-Grams Generation
Generates contiguous sequences of $n$ items from a given sample of text:
* **Unigrams ($n=1$):** Single words.
* **Bigrams ($n=2$):** Two-word pairs.
* **Trigrams / Custom $n$ ($n=5$):** Multi-word phrases.

### 6. Document-Term Matrix (CountVectorizer)
Converts a collection of text documents to a matrix of token counts using `scikit-learn`:
```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "there is always high wind",
    "wind in the sea and landscape area",
]
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(documents)


7. Rule-Based Sentiment AnalysisEvaluates text sentiment by cross-referencing pre-defined positive and negative keyword dictionaries alongside heuristic rules (e.g., uppercase emphasis, exclamation marks, or currency symbols).
8. Stemming vs. Lemmatization (NLTK)Stemming (PorterStemmer):
Cuts words back to their root stems heuristically (e.g., running $\rightarrow$ run).
Lemmatization (WordNetLemmatizer): Maps words to their morphological root based on dictionary definitions and Part-of-Speech context.
9. Text Data Analysis with PandasPerforms statistical analysis on text attribute lengths (e.g., calculating mean, median, and mode character lengths for entries in a dataset).
10. Speech-to-Text & NLP Audio PipelineIntegrates browser-based audio recording widgets (IPython.display.HTML) with SpeechRecognition (Google Speech API) and ffmpeg to transcribe spoken voice clips into text and feed them directly into the spaCy NLP analysis pipeline.
🚀 Getting StartedOpen the project inside Google Colab.
Run the cells sequentially to install required dependencies (nltk, spacy, SpeechRecognition, etc.).
Execute individual code blocks to test tokenization, extraction, and text analytics pipelines.
