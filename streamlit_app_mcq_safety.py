from __future__ import annotations

import io
import json
import os
import random
import re
import string
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import nltk

import numpy as np
import streamlit as st
import sys
import ssl

import torch
import importlib.util

SENTENCEPIECE_AVAILABLE = importlib.util.find_spec("sentencepiece") is not None
if not SENTENCEPIECE_AVAILABLE:
    print("Warning: sentencepiece not available, some features may be limited")
torch.set_num_threads(2)  # Limit CPU threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# Fix for SSL certificate issues with NLTK download
# Fix for SSL certificate issues with NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK data directory setup
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download required NLTK data
import nltk

print("📥 Checking NLTK resources...")

# Download all required resources
for resource in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']:
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif resource == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        elif resource == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif resource == 'averaged_perceptron_tagger':
            nltk.data.find('taggers/averaged_perceptron_tagger')
        print(f"✅ {resource} already available")
    except LookupError:
        print(f"📥 Downloading {resource}...")
        nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
        print(f"✅ {resource} downloaded")

# COMPREHENSIVE FIX: Monkey patch the entire PerceptronTagger class
import nltk.tag.perceptron

# Save the original load_from_json method
original_load_from_json = nltk.tag.perceptron.PerceptronTagger.load_from_json

def patched_load_from_json(self, lang='eng'):
    """Patch to handle both eng and non-eng tagger names"""
    try:
        # Try the original method first
        original_load_from_json(self, lang)
    except LookupError:
        try:
            # If lang is 'eng', try without the _eng suffix
            if lang == 'eng':
                # Try to find the non-eng version
                loc = nltk.data.find('taggers/averaged_perceptron_tagger/')
                # Load the weights and tagdict directly
                import json
                with open(loc + 'averaged_perceptron_tagger.pickle.weights.json', 'r') as fin:
                    self.model.weights = json.load(fin)
                with open(loc + 'averaged_perceptron_tagger.pickle.tagdict.json', 'r') as fin:
                    self.tagdict = json.load(fin)
                with open(loc + 'averaged_perceptron_tagger.pickle.classes.json', 'r') as fin:
                    self.classes = json.load(fin)
                return
        except:
            pass
        
        # If all else fails, try to download the tagger
        try:
            nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=False)
            # Try again with the original method
            original_load_from_json(self, lang)
        except:
            # Ultimate fallback - create a dummy tagger that just returns noun tags
            print("WARNING: Using fallback POS tagger")
            self.model.weights = {}
            self.tagdict = {}
            self.classes = {'NN'}

# Apply the patch
nltk.tag.perceptron.PerceptronTagger.load_from_json = patched_load_from_json

print("📥 NLTK setup complete!")
# ---------------------------------------------------------------------------
# Teacher / Student quiz storage (file-based)
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QUIZZES_FILE = os.path.join(DATA_DIR, "quizzes.json")
ATTEMPTS_FILE = os.path.join(DATA_DIR, "attempts.json")

# Default teacher credentials
TEACHER_USERNAME = os.environ.get("TEACHER_USERNAME", "teacher")
TEACHER_PASSWORD = os.environ.get("TEACHER_PASSWORD", "teacher123")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_json(path: str, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: str, data: Any) -> None:
    _ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_quizzes() -> Dict[str, Any]:
    return _load_json(QUIZZES_FILE, {})


def save_quiz(quiz_id: str, teacher_name: str, mcqs: List[Dict[str, Any]], title: str = "") -> None:
    data = load_quizzes()
    data[quiz_id] = {
        "teacher_name": teacher_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "mcqs": mcqs,
        "title": title or f"Quiz {quiz_id[:8]}",
    }
    _save_json(QUIZZES_FILE, data)


def load_attempts() -> Dict[str, List[Dict[str, Any]]]:
    return _load_json(ATTEMPTS_FILE, {})


def save_attempt(quiz_id: str, student_name: str, score: int, total: int) -> None:
    data = load_attempts()
    if quiz_id not in data:
        data[quiz_id] = []
    data[quiz_id].append({
        "student_name": student_name,
        "score": score,
        "total": total,
        "completed_at": datetime.utcnow().isoformat() + "Z",
    })
    _save_json(ATTEMPTS_FILE, data)


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded .txt, .docx, or .pdf file."""
    if uploaded_file is None:
        return ""
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.read()
    try:
        raw_str = raw.decode("utf-8", errors="replace")
    except Exception:
        raw_str = str(raw)

    if name.endswith(".txt"):
        return raw_str.strip()

    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
        except ImportError:
            return raw_str.strip()
        except Exception:
            return raw_str.strip()

    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            return "\n".join(parts).strip()
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(raw))
                parts = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
                return "\n".join(parts).strip()
            except ImportError:
                return ""
        except Exception:
            return ""

    return raw_str.strip()


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_qg_model():
    """Load the specialized question generation model (old working version).
    
    ALTERNATIVE MODELS (for better quality):
    - "valhalla/t5-base-qg-hl" - Highlight-based QG (better quality)
    - "allenai/t5-small-squad2-question-generation" - AllenAI's QG model
    - "mrm8488/t5-base-finetuned-question-generation-ap" - Fine-tuned for academic
    
    Current: lmqg/t5-base-squad-qg (balanced speed/quality)
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("lmqg/t5-base-squad-qg")
    model = AutoModelForSeq2SeqLM.from_pretrained("lmqg/t5-base-squad-qg")
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_t5_model():
    """Load FLAN-T5 base for fallback and explanations."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    """Optional sentence transformer for semantic similarity."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_toxicity_model():
    from transformers import pipeline
    return pipeline("text-classification", model="unitary/toxic-bert", truncation=True)

@st.cache_resource(show_spinner=False)
def load_bias_model():
    from transformers import pipeline
    return pipeline("text-classification", model="valurank/distilroberta-bias", truncation=True)


# ---------------------------------------------------------------------------
# Lazy imports and NLTK setup
# ---------------------------------------------------------------------------

@st.cache_resource
def _lazy_imports():
    import torch
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        T5ForConditionalGeneration,
        T5Tokenizer,
        pipeline,
    )
    return True


def ensure_nltk() -> None:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SafetyScore:
    label: str
    score: float
    model: str


@dataclass
class BloomResult:
    level: str
    confidence: float
    method: str


# ---------------------------------------------------------------------------
# Safety detectors
# ---------------------------------------------------------------------------

class ToxicityDetector:
    def __init__(self):
        _lazy_imports()
        self.model_name = "unitary/toxic-bert"
        self._pipe = load_toxicity_model()

    def predict(self, text: str) -> SafetyScore:
        if not text.strip():
            return SafetyScore(label="empty", score=0.0, model=self.model_name)
        out = self._pipe(text[:512])[0]
        label = str(out.get("label", "")).lower()
        score = float(out.get("score", 0.0))
        return SafetyScore(label=label, score=score, model=self.model_name)


class BiasDetector:
    def __init__(self):
        _lazy_imports()
        self.model_name = "valurank/distilroberta-bias"
        self._pipe = load_bias_model()

    def predict(self, text: str) -> SafetyScore:
        if not text.strip():
            return SafetyScore(label="empty", score=0.0, model=self.model_name)
        out = self._pipe(text[:512])[0]
        label = str(out.get("label", "")).upper()
        score = float(out.get("score", 0.0))
        return SafetyScore(label=label, score=score, model=self.model_name)


# ---------------------------------------------------------------------------
# Bloom classifier (classification only!)
# ---------------------------------------------------------------------------

class BloomHeuristicClassifier:
    LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

    VERB_MAP: List[Tuple[str, str]] = [
        ("design", "Create"), ("create", "Create"), ("compose", "Create"),
        ("develop", "Create"), ("formulate", "Create"), ("propose", "Create"),
        ("construct", "Create"), ("evaluate", "Evaluate"), ("justify", "Evaluate"),
        ("critique", "Evaluate"), ("assess", "Evaluate"), ("argue", "Evaluate"),
        ("judge", "Evaluate"), ("recommend", "Evaluate"), ("analyze", "Analyze"),
        ("compare", "Analyze"), ("contrast", "Analyze"), ("differentiate", "Analyze"),
        ("examine", "Analyze"), ("categorize", "Analyze"), ("distinguish", "Analyze"),
        ("apply", "Apply"), ("solve", "Apply"), ("use", "Apply"),
        ("demonstrate", "Apply"), ("calculate", "Apply"), ("implement", "Apply"),
        ("explain", "Understand"), ("summarize", "Understand"), ("describe", "Understand"),
        ("interpret", "Understand"), ("classify", "Understand"), ("discuss", "Understand"),
        ("identify", "Understand"), ("define", "Remember"), ("list", "Remember"),
        ("recall", "Remember"), ("what is", "Remember"), ("who is", "Remember"),
        ("when did", "Remember"), ("where is", "Remember"), ("what are", "Remember"),
    ]

    def predict(self, question: str) -> BloomResult:
        q = question.strip().lower()
        if not q:
            return BloomResult(level="Unknown", confidence=0.0, method="heuristic")
        for phrase, level in self.VERB_MAP:
            if q.startswith(phrase) or f" {phrase} " in q:
                return BloomResult(level=level, confidence=0.72, method="heuristic")
        return BloomResult(level="Understand", confidence=0.45, method="heuristic")


# ---------------------------------------------------------------------------
# Improved MCQ Generator (EXACT OLD WORKING CODE for question generation)
# ---------------------------------------------------------------------------

class ImprovedMCQGenerator:
    def __init__(self, show_progress: bool = False):
        _lazy_imports()
        ensure_nltk()

        import nltk
        from nltk.corpus import stopwords

        self._nltk = nltk
        self.stop_words = set(stopwords.words('english'))
        self.max_length = 128

        # Load models with optional progress UI
        try:
            if show_progress:
                progress_bar = st.progress(0, text="Loading AI models...")
                status = st.empty()

                status.text("📚 Loading question generation model...")
                self.qg_tokenizer, self.qg_model = load_qg_model()
                progress_bar.progress(40)

                status.text("🤖 Loading T5 model...")
                self.t5_tokenizer, self.t5_model = load_t5_model()
                progress_bar.progress(70)

                status.text("🧠 Loading sentence transformer...")
                self._sentence_model = load_sentence_transformer()
                progress_bar.progress(100)

                status.text("✅ Models loaded!")
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status.empty()
            else:
                self.qg_tokenizer, self.qg_model = load_qg_model()
                self.t5_tokenizer, self.t5_model = load_t5_model()
                self._sentence_model = load_sentence_transformer()

            self.has_qg_model = True
        except Exception as e:
            print(f"QG model load failed: {e}, falling back to T5 only")
            self.has_qg_model = False
            self.t5_tokenizer, self.t5_model = load_t5_model()
            self._sentence_model = load_sentence_transformer()

        # TF-IDF utilities
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.TfidfVectorizer = TfidfVectorizer
            self.cosine_similarity = cosine_similarity
        except Exception:
            self.TfidfVectorizer = None
            self.cosine_similarity = None

    # -----------------------------------------------------------------------
    # Text processing helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text or "").strip()
        return text

    def _sent_tokenize(self, text: str) -> List[str]:
        return self._nltk.tokenize.sent_tokenize(text)

    def _word_tokenize(self, text: str) -> List[str]:
        return self._nltk.tokenize.word_tokenize(text)

    def _pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if not tokens:
           return []
        try:
            return self._nltk.pos_tag(tokens)
        except Exception as e:
            print(f"POS tagging failed: {e}, using fallback")
            # Simple fallback - assume everything is a noun
            return [(token, 'NN') for token in tokens]

    # -----------------------------------------------------------------------
    # EXACT OLD WORKING CODE: Question generation
    # -----------------------------------------------------------------------
    def generate_question(self, context: str, answer: str, grade: Optional[str] = None, difficulty: Optional[str] = None, cognitive_level: Optional[str] = None) -> str:
        """Generate a question given a context and answer using specialized QG model - EXACT OLD WORKING CODE with cognitive level support"""
        # Find the sentence containing the answer for better context
        sentences = self._sent_tokenize(context)
        relevant_sentences = []

        for sentence in sentences:
            if answer.lower() in sentence.lower():
                relevant_sentences.append(sentence)

        if not relevant_sentences:
            # If answer not found in any sentence, use a random sentence
            if sentences:
                relevant_sentences = [random.choice(sentences)]
            else:
                relevant_sentences = [context]

        # Use up to 3 sentences for context (the sentence with answer + neighbors)
        if len(relevant_sentences) == 1 and len(sentences) > 1:
            # Find the index of the relevant sentence
            idx = sentences.index(relevant_sentences[0])
            if idx > 0:
                relevant_sentences.insert(0, sentences[idx-1])
            if idx < len(sentences) - 1:
                relevant_sentences.append(sentences[idx+1])

        # Join the relevant sentences
        focused_context = ' '.join(relevant_sentences)
        
        # Build cognitive level hint if provided
        cog_hint = ""
        if cognitive_level:
            # Strong cognitive level guidance with question starters
            cog_prompts = {
                "Remember": ". Start with: What is, Define, List, Name, Identify, or Recall",
                "Understand": ". Start with: Explain, Describe, Summarize, or Interpret",
                "Apply": ". Start with: How would you, Calculate, Solve, Demonstrate, or Apply",
                "Analyze": ". Start with: Compare, Contrast, Analyze, Examine, or Differentiate",
                "Evaluate": ". Start with: Evaluate, Assess, Judge, Critique, or Justify",
                "Create": ". Start with: Design, Develop, Create, Formulate, or Construct"
            }
            cog_hint = cog_prompts.get(cognitive_level, "")

        import torch

        if self.has_qg_model:
            # Use specialized QG model
            input_text = f"answer: {answer} context: {focused_context}"
            inputs = self.qg_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

            with torch.no_grad():
                outputs = self.qg_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=5,
                    top_k=120,
                    top_p=0.95,
                    temperature=1.0,
                    do_sample=True,
                    num_return_sequences=3,
                    no_repeat_ngram_size=2
                )

            # Get multiple questions and pick the best one
            questions = [self.qg_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            valid_questions = [q for q in questions if q.endswith('?') and answer.lower() not in q.lower()]

            if valid_questions:
                return self.clean_text(valid_questions[0])

        # Fallback to T5 model if specialized model fails or isn't available
        input_text = f"generate question for answer: {answer} from context: {focused_context}{cog_hint}"
        inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=5,
                top_k=120,
                top_p=0.95,
                temperature=1.0,
                do_sample=True,
                num_return_sequences=3,
                no_repeat_ngram_size=2
            )

        questions = [self.t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Clean and validate questions
        valid_questions = []
        for q in questions:
            # Format the question properly
            q = self.clean_text(q)
            if not q.endswith('?'):
                q += '?'

            # Avoid questions that contain the answer directly
            if answer.lower() not in q.lower():
                valid_questions.append(q)

        if valid_questions:
            return valid_questions[0]

        # If all else fails, create a simple question
        return f"Which of the following best describes {answer}?"

    # -----------------------------------------------------------------------
    # EXACT OLD WORKING CODE: Entity extraction
    # -----------------------------------------------------------------------
    def extract_key_entities(self, text: str, n: int = 8, exclude: List[str] = None) -> List[str]:
        """Extract key entities from text that would make good answers - EXACT OLD WORKING CODE"""
        sentences = self._sent_tokenize(text)
        exclude = [e.lower() for e in (exclude or [])]
        
        # Get noun phrases and named entities
        key_entities = []

        for sentence in sentences:
            words = self._word_tokenize(sentence)
            pos_tags = self._pos_tag(words)

            # Extract noun phrases (consecutive nouns and adjectives)
            i = 0
            while i < len(pos_tags):
                if pos_tags[i][1].startswith('NN') or pos_tags[i][1].startswith('JJ'):
                    phrase = pos_tags[i][0]
                    j = i + 1
                    while j < len(pos_tags) and (pos_tags[j][1].startswith('NN') or pos_tags[j][1] == 'JJ'):
                        phrase += ' ' + pos_tags[j][0]
                        j += 1
                    if len(phrase.split()) >= 1 and not all(w.lower() in self.stop_words for w in phrase.split()):
                        key_entities.append(phrase)
                    i = j
                else:
                    i += 1

        # Extract important terms based on POS tags
        important_terms = []
        for sentence in sentences:
            words = self._word_tokenize(sentence)
            pos_tags = self._pos_tag(words)

            # Get nouns, verbs, and adjectives
            terms = [word for word, pos in pos_tags if
                   (pos.startswith('NN') or pos.startswith('VB') or pos.startswith('JJ'))
                   and word.lower() not in self.stop_words
                   and len(word) > 2]

            important_terms.extend(terms)

        # Combine and remove duplicates
        all_candidates = key_entities + important_terms
        unique_candidates = []

        for candidate in all_candidates:
            # Clean candidate
            candidate = candidate.strip()
            candidate = re.sub(r'[^\w\s]', '', candidate)

            # Skip if empty or just stopwords
            if not candidate or all(w.lower() in self.stop_words for w in candidate.split()):
                continue

            # Skip if in exclude list
            if candidate.lower() in exclude:
                continue

            # Check for duplicates
            if candidate.lower() not in [c.lower() for c in unique_candidates]:
                unique_candidates.append(candidate)

        # Use TF-IDF to rank entities by importance
        if len(unique_candidates) > n and self.TfidfVectorizer and self.cosine_similarity:
            try:
                vectorizer = self.TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([text] + unique_candidates)
                document_vector = tfidf_matrix[0:1]
                entity_vectors = tfidf_matrix[1:]

                # Calculate similarity to document
                similarities = self.cosine_similarity(document_vector, entity_vectors).flatten()

                # Get top n entities
                ranked_entities = [entity for _, entity in sorted(zip(similarities, unique_candidates), reverse=True)]
                return ranked_entities[:n]
            except:
                # Fallback if TF-IDF fails
                return random.sample(unique_candidates, min(n, len(unique_candidates)))

        return unique_candidates[:n]

    # -----------------------------------------------------------------------
    # EXACT OLD WORKING CODE: Distractor generation
    # -----------------------------------------------------------------------
    def generate_distractors(self, answer: str, context: str, all_answers: List[str] = None, n: int = 3) -> List[str]:
        """Generate plausible distractors for a given answer - EXACT OLD WORKING CODE"""
        all_answers = all_answers or []
        
        # Extract potential distractors from context
        exclude = all_answers + [answer]
        potential_distractors = self.extract_key_entities(context, n=15, exclude=exclude)

        # Remove the correct answer and similar options
        filtered_distractors = []
        answer_lower = answer.lower()

        for distractor in potential_distractors:
            distractor_lower = distractor.lower()

            # Skip if it's the answer or too similar to the answer
            if distractor_lower == answer_lower:
                continue
            if answer_lower in distractor_lower or distractor_lower in answer_lower:
                continue
            if len(set(distractor_lower.split()) & set(answer_lower.split())) > len(answer_lower.split()) / 2:
                continue

            filtered_distractors.append(distractor)

        # If we need more distractors, generate them with T5
        if len(filtered_distractors) < n:
            import torch
            input_text = f"generate alternatives for: {answer} context: {context}"
            inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

            with torch.no_grad():
                outputs = self.t5_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=64,
                    num_beams=5,
                    top_k=50,
                    top_p=0.95,
                    temperature=1.2,
                    do_sample=True,
                    num_return_sequences=5
                )

            model_distractors = [self.t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

            # Clean and validate model distractors
            for distractor in model_distractors:
                distractor = self.clean_text(distractor)

                # Skip if it's the answer or too similar
                if distractor.lower() == answer.lower():
                    continue
                if answer.lower() in distractor.lower() or distractor.lower() in answer.lower():
                    continue

                filtered_distractors.append(distractor)

        # Ensure uniqueness
        unique_distractors = []
        for d in filtered_distractors:
            if d.lower() not in [x.lower() for x in unique_distractors]:
                unique_distractors.append(d)

        # If we still don't have enough, create semantic variations
        while len(unique_distractors) < n:
            if not unique_distractors and not potential_distractors:
                # No existing distractors to work with, create something different
                unique_distractors.append(f"None of the above")
                unique_distractors.append(f"All of the above")
                unique_distractors.append(f"Not mentioned in the text")
            else:
                base = answer if not unique_distractors else random.choice(unique_distractors)
                words = base.split()

                if len(words) > 1:
                    # Modify a multi-word distractor
                    modified = words.copy()
                    pos_to_change = random.randint(0, len(words)-1)

                    # Make sure the new distractor is different
                    modification = f"alternative_{modified[pos_to_change]}"
                    while modification in [x.lower() for x in unique_distractors]:
                        modification += "_variant"

                    modified[pos_to_change] = modification
                    unique_distractors.append(" ".join(modified))
                else:
                    # Modify a single word
                    modification = f"alternative_{base}"
                    while modification in [x.lower() for x in unique_distractors]:
                        modification += "_variant"

                    unique_distractors.append(modification)

        # Return the required number of distractors
        return unique_distractors[:n]

    # -----------------------------------------------------------------------
    # Explanation generation (simple)
    # -----------------------------------------------------------------------
    def generate_explanation(self, question: str, answer: str, context: str, grade: Optional[str] = None) -> str:
        """Generate explanation for the answer"""
        sentences = self._sent_tokenize(context)
        for sent in sentences:
            if answer.lower() in sent.lower():
                return sent.strip()
        # Fallback
        return f"The correct answer is {answer}."

    # -----------------------------------------------------------------------
    # EXACT OLD WORKING CODE: MCQ validation
    # -----------------------------------------------------------------------
    def validate_mcq(self, mcq: Dict[str, Any], context: str) -> bool:
        """Validate if an MCQ meets quality standards - EXACT OLD WORKING CODE"""
        # Check if question ends with question mark
        if not mcq['question'].endswith('?'):
            return False

        # Check if the question is too short
        if len(mcq['question'].split()) < 5:
            return False

        # Check if question contains the answer (too obvious)
        if mcq['answer'].lower() in mcq['question'].lower():
            return False

        # Check if options are sufficiently different
        if len(set([o.lower() for o in mcq['options']])) < len(mcq['options']):
            return False

        # Check if answer is in the context
        if mcq['answer'].lower() not in context.lower():
            return False

        return True

    # -----------------------------------------------------------------------
    # EXACT OLD WORKING CODE: Main generation method
    # -----------------------------------------------------------------------
    def generate_mcqs(
        self,
        paragraph: str,
        num_questions: int = 5,
        grade: Optional[str] = None,
        difficulty: Optional[str] = None,
        cognitive_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple-choice questions from a paragraph with cognitive level filtering"""
        paragraph = self.clean_text(paragraph)
        if not paragraph:
            return []

        mcqs = []
        
        # If cognitive level is specified, we need to generate more and filter
        generate_count = num_questions * 3 if cognitive_level else num_questions
        
        # Extract potential answers
        potential_answers = self.extract_key_entities(paragraph, n=generate_count * 3)

        # Shuffle potential answers
        random.shuffle(potential_answers)

        # Try to generate MCQs for each potential answer
        attempts = 0
        max_attempts = generate_count * 3
        all_generated = []

        while len(all_generated) < generate_count and attempts < max_attempts and potential_answers:
            answer = potential_answers.pop(0)
            attempts += 1

            # Generate question with cognitive level
            question = self.generate_question(paragraph, answer, grade, difficulty, cognitive_level)

            # Generate distractors
            distractors = self.generate_distractors(answer, paragraph, all_answers=[m['answer'] for m in all_generated])

            # Create MCQ
            mcq = {
                'question': question,
                'options': [answer] + distractors,
                'answer': answer
            }

            # Validate MCQ
            if self.validate_mcq(mcq, paragraph):
                # Shuffle options
                shuffled_options = mcq['options'].copy()
                random.shuffle(shuffled_options)

                # Find the index of the correct answer
                correct_index = shuffled_options.index(answer)

                # Update MCQ with shuffled options
                mcq['options'] = shuffled_options
                mcq['answer_index'] = correct_index

                all_generated.append(mcq)
        
        # If cognitive level is specified, filter by that level
        if cognitive_level and all_generated:
            # Import BloomHeuristicClassifier for classification
            bloom = BloomHeuristicClassifier()
            
            # Classify all questions
            classified = []
            for mcq in all_generated:
                result = bloom.predict(mcq['question'])
                mcq['_bloom_level'] = result.level
                mcq['_bloom_confidence'] = result.confidence
                classified.append((mcq, result))
            
            # Filter for target level (with some tolerance)
            target_questions = [mcq for mcq, result in classified if result.level == cognitive_level]
            
            # If we don't have enough exact matches, include close levels
            if len(target_questions) < num_questions:
                # Define level proximity
                level_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
                if cognitive_level in level_order:
                    target_idx = level_order.index(cognitive_level)
                    # Include adjacent levels
                    close_levels = []
                    if target_idx > 0:
                        close_levels.append(level_order[target_idx - 1])
                    if target_idx < len(level_order) - 1:
                        close_levels.append(level_order[target_idx + 1])
                    
                    close_questions = [mcq for mcq, result in classified if result.level in close_levels]
                    target_questions.extend(close_questions)
            
            # Clean up temporary fields
            for mcq in target_questions:
                mcq.pop('_bloom_level', None)
                mcq.pop('_bloom_confidence', None)
            
            return target_questions[:num_questions]
        
        return all_generated[:num_questions]


# ---------------------------------------------------------------------------
# Helper functions for UI
# ---------------------------------------------------------------------------

def sentence_split(text: str) -> List[str]:
    _lazy_imports()
    ensure_nltk()
    import nltk
    return [s.strip() for s in nltk.tokenize.sent_tokenize(text or "") if s.strip()]


def analyze_sentences(
    text: str,
    tox: Optional[ToxicityDetector],
    bias: Optional[BiasDetector],
    max_sentences: int = 12,
) -> Dict[str, Any]:
    sents = sentence_split(text)[:max_sentences]
    rows = []
    for s in sents:
        tox_score = tox.predict(s) if tox else None
        bias_score = bias.predict(s) if bias else None
        rows.append({
            "sentence": s,
            "tox_label": tox_score.label if tox_score else None,
            "tox_score": tox_score.score if tox_score else None,
            "bias_label": bias_score.label if bias_score else None,
            "bias_score": bias_score.score if bias_score else None,
        })
    return {"sentences": rows}


def run_mcq_safety_check(
    mcqs: List[Dict[str, Any]],
    tox: Optional[ToxicityDetector],
    bias: Optional[BiasDetector],
    tox_threshold: float = 0.05,
    bias_threshold: float = 0.60,
) -> List[Dict[str, Any]]:
    results = []
    for i, mcq in enumerate(mcqs, 1):
        q = mcq.get("question", "")
        opts = mcq.get("options", [])
        row = {
            "id": i,
            "question": q,
            "question_tox": None,
            "question_bias": None,
            "options_flags": [],
        }
        if tox:
            t = tox.predict(q)
            row["question_tox"] = {"label": t.label, "score": t.score}
        if bias:
            b = bias.predict(q)
            row["question_bias"] = {"label": b.label, "score": b.score}
        for j, opt in enumerate(opts):
            flags = []
            if tox:
                to = tox.predict(opt[:256])
                if to.label == "toxic" and to.score >= tox_threshold:
                    flags.append(f"toxic({to.score:.2%})")
            if bias:
                bi = bias.predict(opt[:256])
                if bi.label == "BIASED" and bi.score >= bias_threshold:
                    flags.append(f"biased({bi.score:.2%})")
            row["options_flags"].append(flags)
        results.append(row)
    return results


def mcqs_to_export_rows(
    mcqs: List[Dict[str, Any]],
    bloom: Optional[BloomHeuristicClassifier] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for i, mcq in enumerate(mcqs, 1):
        opts = mcq.get("options") or []
        row = {
            "id": i,
            "question": mcq.get("question", ""),
            "A": opts[0] if len(opts) > 0 else "",
            "B": opts[1] if len(opts) > 1 else "",
            "C": opts[2] if len(opts) > 2 else "",
            "D": opts[3] if len(opts) > 3 else "",
            "correct_index": mcq.get("answer_index", -1),
            "correct_option": ["A", "B", "C", "D"][mcq.get("answer_index", 0)] if len(opts) >= 4 else "",
            "answer_text": opts[mcq.get("answer_index", 0)] if opts and 0 <= mcq.get("answer_index", 0) < len(opts) else "",
        }
        if bloom:
            b = bloom.predict(row["question"])
            row["bloom_level"] = b.level
            row["bloom_confidence"] = b.confidence
            row["bloom_method"] = b.method
        if mcq.get("explanation"):
            row["explanation"] = mcq["explanation"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Student flow
# ---------------------------------------------------------------------------

def student_flow(quiz_id: str) -> None:
    """Student quiz taking flow - called when accessing a quiz link"""
    quizzes = load_quizzes()
    quiz_data = quizzes.get(quiz_id)
    if not quiz_data:
        st.error("❌ This quiz link is invalid or has been removed.")
        st.info("💡 Please check the link or contact your teacher.")
        return
    mcqs = quiz_data.get("mcqs", [])
    if not mcqs:
        st.warning("⚠️ This quiz has no questions.")
        return

    # Initialize session state for this quiz
    if "student_name" not in st.session_state or st.session_state.get("student_quiz_id") != quiz_id:
        st.session_state["student_quiz_id"] = quiz_id
        st.session_state.pop("student_name", None)
        st.session_state.pop("student_submitted", None)

    # Name entry screen
    if "student_name" not in st.session_state:
        st.markdown("""
            <div style='text-align: center; padding: 3rem 0;'>
                <h1 style='color: #ffffff; font-size: 2.5rem; margin-bottom: 1rem; border: none;'>
                    📝 {title}
                </h1>
                <p style='font-size: 1.2rem; color: #b0b0b0;'>
                    Enter your name to start the quiz
                </p>
            </div>
        """.format(title=quiz_data.get('title', 'Quiz')), unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("student_name_form"):
                name = st.text_input("👤 Your Name", placeholder="Enter your full name").strip()
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("🚀 Start Quiz", use_container_width=True, type="primary"):
                    if name:
                        st.session_state["student_name"] = name
                        st.rerun()
                    else:
                        st.error("❌ Please enter your name to continue.")
        return

    # Already submitted
    if st.session_state.get("student_submitted"):
        st.success("✅ You have already completed this quiz!")
        score = st.session_state.get("student_score", 0)
        total = st.session_state.get("student_total", len(mcqs))
        percentage = (100 * score // max(1, total)) if total > 0 else 0
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Score", f"{score}/{total}")
        with col2:
            st.metric("📊 Percentage", f"{percentage}%")
        with col3:
            status = "🟢 Excellent" if percentage >= 80 else "🟡 Good" if percentage >= 60 else "🔴 Needs Practice"
            st.metric("📈 Status", status)
        
        st.info("💡 Your teacher can see this result in the Reports section.")
        st.balloons()
        return

    # Quiz taking
    student_name = st.session_state["student_name"]
    st.title(f"📝 {quiz_data.get('title', 'Quiz')}")
    st.caption(f"👤 Student: **{student_name}**")
    st.markdown("---")

    key_prefix = f"student_quiz_{quiz_id}_"
    for idx, mcq in enumerate(mcqs, 1):
        st.markdown(f"### Question {idx}")
        st.markdown(f"**{mcq.get('question', '')}**")
        opts = mcq.get("options", [])
        st.radio(
            "Select your answer:",
            options=opts,
            key=f"{key_prefix}q{idx}",
            index=None,
            label_visibility="collapsed",
        )
        st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📋 Submit Quiz", type="primary", use_container_width=True):
            correct = 0
            total = len(mcqs)
            for idx, mcq in enumerate(mcqs, 1):
                opts = mcq.get("options", [])
                ans_idx = int(mcq.get("answer_index", -1))
                correct_opt = opts[ans_idx] if opts and 0 <= ans_idx < len(opts) else ""
                user_opt = st.session_state.get(f"{key_prefix}q{idx}", None)
                if user_opt == correct_opt:
                    correct += 1
            save_attempt(quiz_id, student_name, correct, total)
            st.session_state["student_submitted"] = True
            st.session_state["student_score"] = correct
            st.session_state["student_total"] = total
            st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Smart MCQ Generator", 
        layout="wide",
        page_icon="📚",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for professional, beautiful UI
    st.markdown("""
        <style>
        /* Import modern fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Poppins', sans-serif !important;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main background with gradient */
        .main {
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
        }
        
        /* Main container */
        .main .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        
        /* Beautiful gradient text */
        .gradient-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Headers with glow effect */
        h1 {
            color: #ffffff !important;
            font-weight: 800 !important;
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
            font-size: 3rem !important;
            letter-spacing: -1px;
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        }
        
        h2 {
            color: #e0e0e0 !important;
            font-weight: 700 !important;
            margin-top: 2.5rem;
            font-size: 2rem !important;
            letter-spacing: -0.5px;
        }
        
        h3 {
            color: #d0d0d0 !important;
            font-weight: 600 !important;
            font-size: 1.5rem !important;
            margin-top: 1.5rem;
        }
        
        /* Regular text with better contrast */
        p, div, span, label, li {
            color: #e0e0e0 !important;
            line-height: 1.7;
        }
        
        /* Stylish buttons with gradient and animation */
        .stButton>button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            font-size: 1rem !important;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* Primary button with gradient */
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        
        .stButton>button[kind="primary"]:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        }
        
        /* Secondary button */
        .stButton>button[kind="secondary"] {
            background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%) !important;
            color: #ffffff !important;
            border: 2px solid #667eea !important;
        }
        
        /* Glass morphism cards for expanders */
        .streamlit-expanderHeader {
            background: rgba(45, 45, 45, 0.6) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            font-weight: 600 !important;
            color: #ffffff !important;
            border: 1px solid rgba(102, 126, 234, 0.2) !important;
            padding: 1.2rem !important;
            transition: all 0.3s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(102, 126, 234, 0.15) !important;
            border-color: rgba(102, 126, 234, 0.5) !important;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
        }
        
        .streamlit-expanderContent {
            background: rgba(37, 37, 37, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 0 0 15px 15px;
            border: 1px solid rgba(102, 126, 234, 0.1);
            border-top: none;
            padding: 1.5rem;
        }
        
        /* Modern tabs with gradient */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
            background: transparent;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 14px 28px;
            font-weight: 600;
            background: rgba(45, 45, 45, 0.6);
            backdrop-filter: blur(10px);
            color: #b0b0b0;
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(102, 126, 234, 0.15);
            color: #ffffff;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Alert boxes with gradient borders */
        .stAlert {
            border-radius: 15px !important;
            border: none !important;
            padding: 1.2rem 1.5rem !important;
            backdrop-filter: blur(10px);
        }
        
        [data-baseweb="notification"] {
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        /* Info box */
        .stAlert[data-baseweb="notification"][kind="info"] {
            background: rgba(33, 150, 243, 0.15) !important;
            border-left: 4px solid #2196F3 !important;
        }
        
        /* Success box */
        .stAlert[data-baseweb="notification"][kind="success"] {
            background: rgba(76, 175, 80, 0.15) !important;
            border-left: 4px solid #4CAF50 !important;
        }
        
        /* Warning box */
        .stAlert[data-baseweb="notification"][kind="warning"] {
            background: rgba(255, 152, 0, 0.15) !important;
            border-left: 4px solid #FF9800 !important;
        }
        
        /* Error box */
        .stAlert[data-baseweb="notification"][kind="error"] {
            background: rgba(244, 67, 54, 0.15) !important;
            border-left: 4px solid #F44336 !important;
        }
        
        /* Sidebar with gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            padding: 2rem 1rem;
            border-right: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        [data-testid="stSidebar"] * {
            color: #e0e0e0 !important;
        }
        
        /* Sidebar headers */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Text inputs with glass effect */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea {
            background: rgba(45, 45, 45, 0.6) !important;
            backdrop-filter: blur(10px);
            color: #ffffff !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
            background: rgba(45, 45, 45, 0.8) !important;
        }
        
        /* File uploader with gradient border */
        [data-testid="stFileUploader"] {
            background: rgba(45, 45, 45, 0.4);
            backdrop-filter: blur(10px);
            border: 2px dashed rgba(102, 126, 234, 0.4);
            border-radius: 15px;
            padding: 2.5rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }
        
        /* Radio buttons with custom styling */
        .stRadio>div {
            background: rgba(45, 45, 45, 0.6);
            backdrop-filter: blur(10px);
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        
        /* Metrics with gradient */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        [data-testid="stMetricLabel"] {
            font-weight: 600 !important;
            color: #b0b0b0 !important;
        }
        
        /* Dividers with gradient */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
            margin: 2.5rem 0;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background: rgba(26, 26, 46, 0.8) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 12px;
        }
        
        code {
            background: rgba(102, 126, 234, 0.1) !important;
            padding: 2px 6px;
            border-radius: 4px;
            color: #a8b3ff !important;
        }
        
        /* Download buttons */
        .stDownloadButton>button {
            background: rgba(45, 45, 45, 0.6) !important;
            backdrop-filter: blur(10px);
            color: #ffffff !important;
            border: 2px solid #667eea !important;
            border-radius: 12px;
        }
        
        .stDownloadButton>button:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-color: transparent !important;
        }
        
        /* Select boxes */
        .stSelectbox>div>div {
            background: rgba(45, 45, 45, 0.6) !important;
            backdrop-filter: blur(10px);
            color: #ffffff !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 12px !important;
        }
        
        /* Sliders with gradient */
        .stSlider>div>div>div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        /* Checkbox */
        .stCheckbox {
            color: #e0e0e0 !important;
        }
        
        /* Mode cards with 3D effect */
        .mode-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2.5rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            border: none;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .mode-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, transparent 0%, rgba(255, 255, 255, 0.1) 100%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .mode-card:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5);
        }
        
        .mode-card:hover::before {
            opacity: 1;
        }
        
        .mode-card h2 {
            color: white !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
        }
        
        .mode-card p, .mode-card li {
            color: rgba(255, 255, 255, 0.95) !important;
            font-size: 1.05rem;
        }
        
        /* Student mode card */
        .mode-card-student {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            box-shadow: 0 10px 40px rgba(245, 87, 108, 0.3);
        }
        
        .mode-card-student:hover {
            box-shadow: 0 20px 60px rgba(245, 87, 108, 0.5);
        }
        
        /* Login container */
        .login-container {
            background: rgba(26, 26, 46, 0.8);
            backdrop-filter: blur(20px);
            padding: 3.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
            border: 1px solid rgba(102, 126, 234, 0.3);
        }
        
        /* Question cards */
        .question-card {
            background: rgba(45, 45, 45, 0.6);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            border-left: 4px solid #667eea;
            margin: 1.5rem 0;
            transition: all 0.3s ease;
        }
        
        .question-card:hover {
            transform: translateX(5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        }
        
        /* Correct answer highlight */
        .correct-answer {
            background: rgba(76, 175, 80, 0.15) !important;
            border-left-color: #4caf50 !important;
        }
        
        /* Dataframe styling */
        .dataframe {
            background: rgba(26, 26, 46, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 12px;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(26, 26, 46, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* Tooltip */
        [data-baseweb="tooltip"] {
            background: rgba(26, 26, 46, 0.95) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 8px;
        }
        
        /* Animation for page load */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .main .block-container > div {
            animation: fadeInUp 0.6s ease-out;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Shared quiz route (for students taking teacher's quiz)
    try:
        if hasattr(st, 'query_params'):
            # New Streamlit API
            qp = st.query_params
            quiz_id = qp.get("quiz") or qp.get("quiz_id")
        else:
            # Old Streamlit API (fallback)
            qp = st.experimental_get_query_params()
            quiz_id = (qp.get("quiz") or qp.get("quiz_id") or [None])[0]
        
        if isinstance(quiz_id, list):
            quiz_id = quiz_id[0] if quiz_id else None
    except Exception:
        quiz_id = None
    
    if quiz_id:
        student_flow(quiz_id)
        return

    # Initialize session state
    if "user_mode" not in st.session_state:
        st.session_state["user_mode"] = None  # None, "teacher", or "student"
    if "teacher_logged_in" not in st.session_state:
        st.session_state["teacher_logged_in"] = False
    if "teacher_name" not in st.session_state:
        st.session_state["teacher_name"] = ""

    # Mode selection screen
    if st.session_state["user_mode"] is None:
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='text-align: center; padding: 3rem 0 2rem 0;'>
                <div style='font-size: 4rem; margin-bottom: 1rem; animation: pulse 2s ease-in-out infinite;'>
                    📚
                </div>
                <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                           -webkit-background-clip: text;
                           -webkit-text-fill-color: transparent;
                           font-size: 4rem; 
                           margin-bottom: 1rem; 
                           border: none;
                           font-weight: 900;
                           letter-spacing: -2px;'>
                    Smart MCQ Generator
                </h1>
                <p style='font-size: 1.4rem; color: #b0b0b0; margin-bottom: 4rem; font-weight: 300;'>
                    ✨ Powered by Advanced AI • Create Perfect Quizzes Instantly ✨
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        
        with col2:
            st.markdown("""
                <h2 style='text-align: center; 
                           color: #ffffff; 
                           margin-bottom: 3rem; 
                           font-weight: 700;
                           font-size: 2rem;'>
                    Choose Your Experience
                </h2>
            """, unsafe_allow_html=True)
            
            col_teacher, col_student = st.columns(2, gap="large")
            
            with col_teacher:
                st.markdown("""
                    <div class='mode-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                        <div style='font-size: 4rem; margin-bottom: 1.5rem;'>👨‍🏫</div>
                        <h2 style='color: white; border: none; margin: 0 0 1rem 0; padding: 0; font-size: 2.5rem; font-weight: 800;'>
                            Teacher
                        </h2>
                        <p style='color: rgba(255, 255, 255, 0.9); margin-bottom: 2rem; font-size: 1.15rem; font-weight: 400;'>
                            Full-Featured Professional Dashboard
                        </p>
                        <div style='text-align: left; padding: 0 1.5rem;'>
                            <div style='margin: 1rem 0; padding-left: 0.5rem;'>
                                <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>✨</span>
                                    <span style='font-size: 1.05rem;'>AI-Powered MCQ Generation</span>
                                </div>
                                <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>🔗</span>
                                    <span style='font-size: 1.05rem;'>Share Quiz Links Instantly</span>
                                </div>
                                <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>📊</span>
                                    <span style='font-size: 1.05rem;'>Track Student Performance</span>
                                </div>
                                <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>📈</span>
                                    <span style='font-size: 1.05rem;'>Advanced Analytics & Reports</span>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🎓 Login as Teacher", use_container_width=True, type="primary", key="btn_teacher"):
                    st.session_state["user_mode"] = "teacher"
                    st.rerun()
            
            with col_student:
                st.markdown("""
                    <div class='mode-card mode-card-student' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                        <div style='font-size: 4rem; margin-bottom: 1.5rem;'>👨‍🎓</div>
                        <h2 style='color: white; border: none; margin: 0 0 1rem 0; padding: 0; font-size: 2.5rem; font-weight: 800;'>
                            Student
                        </h2>
                        <p style='color: rgba(255, 255, 255, 0.9); margin-bottom: 2rem; font-size: 1.15rem; font-weight: 400;'>
                            Interactive Self-Learning Platform
                        </p>
                        <div style='text-align: left; padding: 0 1.5rem;'>
                            <div style='margin: 1rem 0; padding-left: 0.5rem;'>
                                <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>✨</span>
                                    <span style='font-size: 1.05rem;'>Generate Practice Questions</span>
                                </div>
                                <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>🎮</span>
                                    <span style='font-size: 1.05rem;'>Interactive Quiz Mode</span>
                                </div>
                                <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>📝</span>
                                    <span style='font-size: 1.05rem;'>Instant Self-Assessment</span>
                                </div>
                                <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>🚀</span>
                                    <span style='font-size: 1.05rem;'>No Login Required</span>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📖 Continue as Student", use_container_width=True, key="btn_student"):
                    st.session_state["user_mode"] = "student"
                    st.rerun()
        
        # Add footer
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; padding: 2rem; color: #666;'>
                <p style='font-size: 0.9rem;'>
                    🔒 Secure • 🚀 Fast • 💡 Intelligent • 🎯 Accurate
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        return

    # Teacher login
    if st.session_state["user_mode"] == "teacher" and not st.session_state["teacher_logged_in"]:
        # Background particles effect
        st.markdown("""
            <style>
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-20px); }
            }
            .particle {
                position: fixed;
                border-radius: 50%;
                background: rgba(102, 126, 234, 0.1);
                animation: float 3s ease-in-out infinite;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Beautiful header
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0 3rem 0;'>
                <div style='display: inline-block; position: relative;'>
                    <div style='font-size: 5rem; margin-bottom: 0;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 30px;
                                padding: 1rem 1.5rem;
                                box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
                                display: inline-block;'>
                        👨‍🏫
                    </div>
                </div>
                <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           -webkit-background-clip: text;
                           -webkit-text-fill-color: transparent;
                           font-size: 3.5rem; 
                           margin: 1.5rem 0 0.5rem 0; 
                           border: none;
                           font-weight: 900;
                           letter-spacing: -1px;'>
                    Teacher Dashboard
                </h1>
                <p style='font-size: 1.2rem; color: #999; font-weight: 300; margin-top: 0.5rem;'>
                    🔐 Secure Access Portal
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2.5, 1])
        with col2:
            st.markdown("""
                <div style='background: rgba(26, 26, 46, 0.95);
                            backdrop-filter: blur(20px);
                            padding: 4rem 3.5rem;
                            border-radius: 25px;
                            box-shadow: 0 20px 60px rgba(0,0,0,0.5),
                                        0 0 0 1px rgba(102, 126, 234, 0.3);
                            position: relative;
                            overflow: hidden;'>
                    <div style='position: absolute; top: 0; left: 0; right: 0; height: 5px;
                                background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);'></div>
            """, unsafe_allow_html=True)
            
            with st.form("teacher_login"):
                st.markdown("""
                    <div style='text-align: center; margin-bottom: 2.5rem;'>
                        <h3 style='color: #ffffff; 
                                   font-size: 1.8rem; 
                                   font-weight: 700; 
                                   margin-bottom: 0.5rem;
                                   display: flex;
                                   align-items: center;
                                   justify-content: center;'>
                            🔐 <span style='margin-left: 0.5rem;'>Enter Your Credentials</span>
                        </h3>
                        <p style='color: #888; font-size: 0.95rem; margin-top: 0.5rem;'>
                            Please sign in to continue to your dashboard
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Username field with icon
                st.markdown("""
                    <div style='margin-bottom: 0.5rem;'>
                        <label style='color: #b0b0b0; 
                                     font-size: 0.9rem; 
                                     font-weight: 600; 
                                     display: flex; 
                                     align-items: center;
                                     margin-bottom: 0.5rem;'>
                            👤 <span style='margin-left: 0.5rem;'>Username</span>
                        </label>
                    </div>
                """, unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed", key="login_username")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Password field with icon
                st.markdown("""
                    <div style='margin-bottom: 0.5rem;'>
                        <label style='color: #b0b0b0; 
                                     font-size: 0.9rem; 
                                     font-weight: 600; 
                                     display: flex; 
                                     align-items: center;
                                     margin-bottom: 0.5rem;'>
                            🔒 <span style='margin-left: 0.5rem;'>Password</span>
                        </label>
                    </div>
                """, unsafe_allow_html=True)
                password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed", key="login_password")
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                col_login, col_back = st.columns(2, gap="medium")
                with col_login:
                    login_btn = st.form_submit_button("🚀 Log In", use_container_width=True, type="primary")
                with col_back:
                    back_btn = st.form_submit_button("← Back", use_container_width=True)
                
                if back_btn:
                    st.session_state["user_mode"] = None
                    st.rerun()
                
                if login_btn:
                    pw = os.environ.get("TEACHER_PASSWORD", TEACHER_PASSWORD)
                    if username.strip() and password == pw:
                        st.session_state["teacher_logged_in"] = True
                        st.session_state["teacher_name"] = username.strip()
                        st.success("✅ Login successful! Redirecting...")
                        import time
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Info box below login
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div style='background: rgba(33, 150, 243, 0.1);
                            border: 1px solid rgba(33, 150, 243, 0.3);
                            border-radius: 15px;
                            padding: 1.5rem;
                            text-align: center;
                            backdrop-filter: blur(10px);'>
                    <p style='color: #64b5f6; margin: 0; font-size: 0.95rem; font-weight: 500;'>
                        💡 <strong>Default Credentials:</strong><br>
                        <span style='font-family: monospace; background: rgba(100, 181, 246, 0.1); 
                                     padding: 0.3rem 0.8rem; border-radius: 8px; margin-top: 0.5rem; display: inline-block;'>
                            teacher / teacher123
                        </span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        return

    # Sidebar configuration
    with st.sidebar:
        if st.session_state["user_mode"] == "teacher":
            st.markdown(f"### 👨‍🏫 Teacher Mode")
            st.caption(f"Logged in as **{st.session_state['teacher_name']}**")
            if st.button("🚪 Log Out", use_container_width=True):
                st.session_state["teacher_logged_in"] = False
                st.session_state["teacher_name"] = ""
                st.session_state["user_mode"] = None
                st.rerun()
        else:
            st.markdown(f"### 👨‍🎓 Student Mode")
            st.caption("Practice and learn")
            if st.button("← Back to Home", use_container_width=True):
                st.session_state["user_mode"] = None
                st.rerun()
        
        st.divider()

    # Get user mode
    is_teacher = st.session_state["user_mode"] == "teacher"
    teacher_name = st.session_state.get("teacher_name", "")

    # Page title
    if is_teacher:
        st.title("👨‍🏫 Teacher Dashboard - MCQ Generator")
    else:
        st.title("👨‍🎓 Student Practice - MCQ Generator")


    # Sidebar options
    with st.sidebar:
        st.subheader("⚙️ Generation Settings")
        
        # Cognitive Level Selection
        st.markdown("##### 🎯 Question Difficulty Level")
        cognitive_level = st.selectbox(
            "Bloom's Taxonomy Level",
            options=["Auto (Mixed)", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
            index=0,
            help="Select cognitive level for questions. 'Auto' generates mixed difficulty levels.",
            label_visibility="collapsed"
        )
        
        if cognitive_level != "Auto (Mixed)":
            st.caption(f"📌 All questions will target **{cognitive_level}** level")
        else:
            st.caption("📌 Questions will vary in difficulty")
        
        st.markdown("---")
        
        num_q = st.slider("📊 Number of questions", 1, 15, 5, help="Recommended: 3-10 questions for best quality")
        
        # Smart validation warning
        if num_q > 10:
            st.warning("⚠️ Generating 10+ questions may reduce quality for short paragraphs")
        
        grade_options = [f"{i}{'st' if i==1 else 'nd' if i==2 else 'rd' if i==3 else 'th'}" for i in range(1,13)]
        grade = st.selectbox("📚 Class / Grade", grade_options, index=5)
        difficulty = st.selectbox("📈 Content Difficulty", ["Easy", "Medium", "Hard"], index=1)

        st.markdown("---")
        st.subheader("🎨 Display Options")
        if is_teacher:
            show_answers = st.checkbox("✅ Show correct answer", value=True, help="Show correct answers in generated questions")
        else:
            show_answers = False  # Always hide for students
            st.info("💡 Answers hidden in student mode")
        enable_explanations = st.checkbox("📝 Show explanations", value=True)

        st.markdown("---")
        st.subheader("🛡️ Safety Checks")
        enable_tox = st.checkbox("🔍 Toxicity check", value=True)
        enable_bias = st.checkbox("⚖️ Bias check", value=True)
        sentence_level = st.checkbox("📄 Sentence-level (slower)", value=False)
        
        with st.expander("⚙️ Advanced Settings"):
            tox_threshold = st.slider("Toxicity threshold", 0.0, 1.0, 0.05, 0.01)
            bias_threshold = st.slider("Bias threshold", 0.0, 1.0, 0.60, 0.01)
            enable_mcq_safety = st.checkbox("🔒 Check MCQs for safety", value=True)

        st.markdown("---")
        st.subheader("📊 Analysis")
        enable_bloom = st.checkbox("🎯 Show Bloom's classification", value=True)
        st.caption("Educational classification of questions")

    # Create tabs based on user mode
    if is_teacher:
        tabs = st.tabs(["📝 Generate", "🎮 Practice Quiz", "🛡️ Safety & Bias", "📊 Bloom Analysis", "💾 Export", "📈 Reports"])
    else:
        tabs = st.tabs(["📝 Generate", "🎮 Practice Quiz", "🛡️ Safety & Bias", "📊 Bloom Analysis"])

    # ---------- Generate tab ----------
    with tabs[0]:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem 0;'>
                <h2 style='color: #ffffff; 
                           font-size: 2.5rem; 
                           font-weight: 800; 
                           margin-bottom: 0.5rem;
                           display: inline-block;'>
                    ✨ <span style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                   -webkit-background-clip: text;
                                   -webkit-text-fill-color: transparent;'>
                        Generate Questions
                    </span>
                </h2>
                <p style='color: #888; font-size: 1.05rem; margin-top: 0.5rem;'>
                    AI-powered MCQ generation from your content
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_upload, col_text = st.columns([1, 2], gap="large")
        with col_upload:
            st.markdown("""
                <div style='background: rgba(45, 45, 45, 0.6);
                            backdrop-filter: blur(10px);
                            padding: 1.5rem;
                            border-radius: 15px;
                            border: 1px solid rgba(102, 126, 234, 0.2);
                            text-align: center;
                            height: 100%;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>📎</div>
                    <h4 style='color: #ffffff; margin-bottom: 1rem;'>Upload File</h4>
                    <p style='color: #888; font-size: 0.9rem;'>
                        Drag & drop or browse<br>
                        <span style='color: #667eea;'>.txt, .docx, .pdf</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            uploaded = st.file_uploader("📎 Upload File", type=["txt", "docx", "pdf"], help="Upload a text file, Word document, or PDF", label_visibility="collapsed")
        
        with col_text:
            st.markdown("""
                <div style='margin-bottom: 1rem;'>
                    <h4 style='color: #ffffff; font-weight: 600; display: flex; align-items: center;'>
                        <span style='font-size: 1.5rem; margin-right: 0.5rem;'>📝</span>
                        Or Paste Your Text
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            file_text = extract_text_from_file(uploaded) if uploaded else ""
            passage = st.text_area(
                "Paste your text here",
                height=250,
                placeholder="📚 Example: Photosynthesis is the process by which plants convert light energy into chemical energy. During this process, plants use sunlight, water, and carbon dioxide to produce glucose and oxygen...",
                value=file_text,
                label_visibility="collapsed"
            )
        
        merged = (passage or "").strip() or file_text

        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            gen = st.button("🚀 Generate MCQs", type="primary", use_container_width=True)
        
        st.markdown("""
            <div style='text-align: center; margin-top: 0.5rem;'>
                <p style='color: #666; font-size: 0.85rem;'>
                    ⏱️ First run downloads AI models (1-2 min) • Subsequent runs are instant
                </p>
            </div>
        """, unsafe_allow_html=True)

        if gen:
            if not merged:
                st.error("❌ Please paste text or upload a file.")
            else:
                # Smart validation
                word_count = len(merged.split())
                sentence_count = len([s for s in merged.split('.') if s.strip()])
                
                # Calculate recommended question count
                recommended_max = max(3, min(word_count // 30, 10))  # Roughly 1 question per 30 words
                
                if word_count < 50:
                    st.error(f"❌ Text too short! Please provide at least 50 words. Current: {word_count} words")
                    st.info("💡 **Tip:** For best results, provide a detailed paragraph with multiple concepts.")
                    return
                
                if num_q > recommended_max:
                    st.warning(f"""
                    ⚠️ **Quality Warning:** 
                    - Your text has **{word_count} words** 
                    - Recommended max questions: **{recommended_max}**
                    - You're requesting: **{num_q}** questions
                    
                    This may result in:
                    - Duplicate or similar questions
                    - Generic or low-quality questions
                    - Repeated answer options
                    
                    💡 **Suggestion:** Reduce to {recommended_max} questions or add more content.
                    """)
                    
                    col_continue, col_cancel = st.columns(2)
                    with col_continue:
                        if not st.button("⚠️ Generate Anyway", type="secondary", use_container_width=True):
                            return
                    with col_cancel:
                        if st.button("← Go Back", use_container_width=True):
                            return
                
                # Convert cognitive level
                cog_level_map = {
                    "Auto (Mixed)": None,
                    "Remember": "Remember",
                    "Understand": "Understand", 
                    "Apply": "Apply",
                    "Analyze": "Analyze",
                    "Evaluate": "Evaluate",
                    "Create": "Create"
                }
                selected_cog_level = cog_level_map.get(cognitive_level)
                
                with st.spinner("🔄 Initializing AI models and generating questions..."):
                    generator = ImprovedMCQGenerator(show_progress=True)
                    
                    if selected_cog_level:
                        st.info(f"🎯 Filtering questions for **{selected_cog_level}** level (Bloom's Taxonomy). Generating extra questions to ensure quality...")
                    
                    mcqs = generator.generate_mcqs(
                        merged,
                        num_questions=num_q,
                        grade=grade,
                        difficulty=difficulty,
                        cognitive_level=selected_cog_level,
                    )
                    if enable_explanations and mcqs:
                        for m in mcqs:
                            m["explanation"] = generator.generate_explanation(
                                m["question"], m["answer"], merged, grade=grade
                            )
                st.session_state["passage"] = merged
                st.session_state["mcqs"] = mcqs
                st.session_state["cognitive_level"] = selected_cog_level  # Store for display
                st.session_state.pop("shareable_quiz_id", None)
                
                if len(mcqs) < num_q:
                    st.warning(f"⚠️ Generated {len(mcqs)} questions (requested {num_q}). Text may be too short or not enough questions match the selected cognitive level.")
                else:
                    if selected_cog_level:
                        st.success(f"✅ Successfully generated {len(mcqs)} **{selected_cog_level}**-level questions!")
                    else:
                        st.success(f"✅ Successfully generated {len(mcqs)} high-quality questions!")

        mcqs = st.session_state.get("mcqs", [])
        if mcqs:
            # Teacher-only feature: Share quiz
            if is_teacher:
                st.divider()
                st.subheader("🔗 Share Quiz with Students")
                col_title, col_btn = st.columns([3, 1])
                with col_title:
                    quiz_title = st.text_input("Quiz title", key="quiz_title_share", placeholder="Enter a descriptive title")
                with col_btn:
                    st.write("")  # Spacing
                    if st.button("🔗 Create Link", use_container_width=True):
                        qid = str(uuid.uuid4())[:8]
                        save_quiz(qid, teacher_name, mcqs, title=quiz_title or f"Quiz {qid}")
                        st.session_state["shareable_quiz_id"] = qid

                share_id = st.session_state.get("shareable_quiz_id")
                if share_id:
                    link = f"?quiz={share_id}"
                    st.success("✅ Quiz created successfully!")
                    st.code(link, language=None)
                    st.caption(f"📎 Share this link with students: `https://yourapp.com{link}`")
            
            st.divider()
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0;'>
                    <h2 style='color: #ffffff; font-size: 2rem; font-weight: 700;'>
                        📋 Generated Questions
                    </h2>
                    <p style='color: #888; margin-top: 0.5rem;'>
                        {count} high-quality questions ready for use
                    </p>
                </div>
            """.format(count=len(mcqs)), unsafe_allow_html=True)
            
            for idx, mcq in enumerate(mcqs, 1):
                with st.expander(f"**Question {idx}**", expanded=(idx == 1)):
                    st.markdown(f"""
                        <div style='background: rgba(102, 126, 234, 0.05);
                                    border-left: 4px solid #667eea;
                                    padding: 1.5rem;
                                    border-radius: 12px;
                                    margin: 1rem 0;'>
                            <h3 style='color: #ffffff; 
                                       font-size: 1.3rem; 
                                       font-weight: 600; 
                                       margin-bottom: 1.5rem;
                                       line-height: 1.6;'>
                                {mcq['question']}
                            </h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    opts = mcq.get("options", [])
                    ans_idx = int(mcq.get("answer_index", -1))
                    
                    st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
                    for j, opt in enumerate(opts):
                        is_correct = (show_answers and j == ans_idx)
                        
                        if is_correct:
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(76, 175, 80, 0.05) 100%);
                                            border: 2px solid #4caf50;
                                            border-radius: 12px;
                                            padding: 1rem 1.5rem;
                                            margin: 0.8rem 0;
                                            display: flex;
                                            align-items: center;
                                            transition: all 0.3s ease;'>
                                    <span style='color: #4caf50; 
                                                 font-size: 1.5rem; 
                                                 margin-right: 1rem;
                                                 font-weight: 800;'>✓</span>
                                    <span style='color: #ffffff; 
                                                 font-weight: 600; 
                                                 font-size: 1.05rem;
                                                 flex: 1;'>
                                        {string.ascii_uppercase[j]}. {opt}
                                    </span>
                                    <span style='background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
                                                 color: white;
                                                 padding: 0.4rem 1rem;
                                                 border-radius: 20px;
                                                 font-size: 0.85rem;
                                                 font-weight: 600;'>
                                        Correct
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style='background: rgba(45, 45, 45, 0.4);
                                            border: 1px solid rgba(102, 126, 234, 0.2);
                                            border-radius: 12px;
                                            padding: 1rem 1.5rem;
                                            margin: 0.8rem 0;
                                            display: flex;
                                            align-items: center;
                                            transition: all 0.3s ease;'>
                                    <span style='color: #667eea; 
                                                 font-size: 1.2rem; 
                                                 margin-right: 1rem;
                                                 font-weight: 600;'>{string.ascii_uppercase[j]}.</span>
                                    <span style='color: #e0e0e0; font-size: 1.05rem;'>{opt}</span>
                                </div>
                            """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if mcq.get("explanation") and enable_explanations:
                        st.markdown(f"""
                            <div style='background: rgba(33, 150, 243, 0.08);
                                        border-left: 3px solid #2196F3;
                                        padding: 1.2rem;
                                        border-radius: 10px;
                                        margin-top: 1.5rem;'>
                                <div style='display: flex; align-items: start;'>
                                    <span style='font-size: 1.5rem; margin-right: 1rem;'>💡</span>
                                    <div>
                                        <strong style='color: #64b5f6; font-size: 0.95rem; display: block; margin-bottom: 0.5rem;'>
                                            EXPLANATION
                                        </strong>
                                        <p style='color: #e0e0e0; margin: 0; line-height: 1.6;'>
                                            {mcq['explanation']}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

    # ---------- Practice Quiz tab ----------
    with tabs[1]:
        st.markdown("### 🎮 Practice Mode")
        qs = st.session_state.get("mcqs", [])
        if not qs:
            st.info("📝 Generate MCQs first in the Generate tab.")
        else:
            if "quiz_submitted" not in st.session_state:
                st.session_state["quiz_submitted"] = False

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔄 Reset Quiz", use_container_width=True):
                    st.session_state["quiz_submitted"] = False
                    st.session_state["_quiz_reset"] = st.session_state.get("_quiz_reset", 0) + 1
                    st.rerun()

            reset = st.session_state.get("_quiz_reset", 0)
            
            if not st.session_state.get("quiz_submitted"):
                st.markdown("---")
                for idx, mcq in enumerate(qs, 1):
                    st.markdown(f"#### Question {idx}")
                    st.markdown(f"**{mcq['question']}**")
                    st.radio(
                        "Select your answer:",
                        options=mcq.get("options", []),
                        key=f"_quiz_q{idx}_{reset}",
                        index=None,
                        label_visibility="collapsed"
                    )
                    st.divider()

                if st.button("📋 Submit Quiz", type="primary", use_container_width=True):
                    st.session_state["quiz_submitted"] = True
                    st.rerun()
            else:
                # Show results
                correct = 0
                for idx, mcq in enumerate(qs, 1):
                    opts = mcq.get("options", [])
                    ans_idx = int(mcq.get("answer_index", -1))
                    correct_opt = opts[ans_idx] if opts else ""
                    user_opt = st.session_state.get(f"_quiz_q{idx}_{reset}", None)
                    if user_opt == correct_opt:
                        correct += 1
                
                total = len(qs)
                percentage = (100 * correct // total) if total > 0 else 0
                
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("✅ Correct", f"{correct}", f"{percentage}%")
                with col2:
                    st.metric("❌ Incorrect", f"{total - correct}")
                with col3:
                    st.metric("📊 Total", f"{total}")
                
                # Show detailed results
                st.markdown("---")
                st.markdown("### 📝 Detailed Results")
                for idx, mcq in enumerate(qs, 1):
                    opts = mcq.get("options", [])
                    ans_idx = int(mcq.get("answer_index", -1))
                    correct_opt = opts[ans_idx] if opts else ""
                    user_opt = st.session_state.get(f"_quiz_q{idx}_{reset}", None)
                    is_correct = user_opt == correct_opt
                    
                    with st.expander(f"{'✅' if is_correct else '❌'} Question {idx}", expanded=not is_correct):
                        st.markdown(f"**{mcq['question']}**")
                        st.markdown(f"**Your answer:** {user_opt if user_opt else 'Not answered'}")
                        st.markdown(f"**Correct answer:** {correct_opt}")
                        if mcq.get("explanation"):
                            st.info(f"💡 {mcq['explanation']}")

    # ---------- Safety & Bias tab ----------
    with tabs[2]:
        st.markdown("### 🛡️ Content Safety Analysis")
        p = st.session_state.get("passage", "")
        if not p:
            st.info("📝 Generate MCQs first to see safety analysis.")
        else:
            tox = ToxicityDetector() if enable_tox else None
            bias = BiasDetector() if enable_bias else None

            col1, col2 = st.columns(2)
            if tox:
                t = tox.predict(p)
                toxic = (t.label == "toxic") and (t.score >= tox_threshold)
                with col1:
                    if toxic:
                        st.error(f"⚠️ **Toxicity Detected**")
                        st.metric("Toxicity Score", f"{t.score:.2%}")
                    else:
                        st.success(f"✅ **Content is Safe**")
                        st.metric("Toxicity Score", f"{t.score:.2%}")
            else:
                col1.info("Toxicity check disabled")

            if bias:
                b = bias.predict(p)
                biased = (b.label == "BIASED") and (b.score >= bias_threshold)
                with col2:
                    if biased:
                        st.warning(f"⚠️ **Bias Detected**")
                        st.metric("Bias Score", f"{b.score:.2%}")
                    else:
                        st.success(f"✅ **Content is Fair**")
                        st.metric("Bias Score", f"{b.score:.2%}")
            else:
                col2.info("Bias check disabled")

            if sentence_level and (tox or bias):
                st.markdown("---")
                st.markdown("### 📄 Sentence-Level Analysis")
                with st.spinner("Analyzing sentences..."):
                    rep = analyze_sentences(p, tox, bias, 12)
                for row in rep["sentences"]:
                    flags = []
                    if tox and row["tox_label"] == "toxic" and (row["tox_score"] or 0) >= tox_threshold:
                        flags.append(f"🔴 Toxicity: {row['tox_score']:.2%}")
                    if bias and row["bias_label"] == "BIASED" and (row["bias_score"] or 0) >= bias_threshold:
                        flags.append(f"🟡 Bias: {row['bias_score']:.2%}")
                    if flags:
                        st.warning(f"**Sentence:** {row['sentence'][:100]}...\n\n**Flags:** {' | '.join(flags)}")

            if enable_mcq_safety and (tox or bias) and st.session_state.get("mcqs"):
                st.markdown("---")
                st.markdown("### 🔒 MCQ Safety Check")
                with st.spinner("Checking MCQ safety..."):
                    safety = run_mcq_safety_check(
                        st.session_state["mcqs"], tox, bias, tox_threshold, bias_threshold
                    )
                
                issues_found = False
                for r in safety:
                    flags = []
                    if r.get("question_tox") and r["question_tox"]["label"] == "toxic" and r["question_tox"]["score"] >= tox_threshold:
                        flags.append(f"🔴 Question toxic: {r['question_tox']['score']:.2%}")
                    if r.get("question_bias") and r["question_bias"]["label"] == "BIASED" and r["question_bias"]["score"] >= bias_threshold:
                        flags.append(f"🟡 Question biased: {r['question_bias']['score']:.2%}")
                    for j, of in enumerate(r["options_flags"]):
                        if of:
                            flags.append(f"Option {string.ascii_uppercase[j]}: {', '.join(of)}")
                    if flags:
                        issues_found = True
                        st.warning(f"**Question {r['id']}:** {' | '.join(flags)}")
                
                if not issues_found:
                    st.success("✅ All MCQs passed safety checks!")

    # ---------- Bloom Analysis tab ----------
    with tabs[3]:
        st.markdown("### 🎯 Bloom's Taxonomy Analysis")
        st.caption("Educational classification of cognitive levels in questions")
        qs = st.session_state.get("mcqs", [])
        if not qs:
            st.info("📝 Generate MCQs first to see Bloom's analysis.")
        elif not enable_bloom:
            st.info("Enable Bloom estimation in the sidebar settings.")
        else:
            bloom = BloomHeuristicClassifier()
            counts = {level: 0 for level in BloomHeuristicClassifier.LEVELS}
            
            st.markdown("#### Question Classification")
            for i, mcq in enumerate(qs, 1):
                r = bloom.predict(mcq.get("question", ""))
                counts[r.level] = counts.get(r.level, 0) + 1
                
                # Color code based on level
                if r.level in ["Remember", "Understand"]:
                    color = "🟢"
                elif r.level in ["Apply", "Analyze"]:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"{color} **Q{i}** — {r.level} (confidence: {r.confidence:.0%})")
            
            st.markdown("---")
            st.markdown("#### Distribution by Cognitive Level")
            st.bar_chart(counts)
            
            st.info("""
            **Bloom's Taxonomy Levels:**
            - 🟢 **Remember/Understand:** Basic recall and comprehension
            - 🟡 **Apply/Analyze:** Application and analysis of concepts  
            - 🔴 **Evaluate/Create:** Higher-order thinking and creation
            """)

    # Teacher-only tabs
    if is_teacher:
        # ---------- Export tab ----------
        with tabs[4]:
            st.markdown("### 💾 Export Questions")
            qs = st.session_state.get("mcqs", [])
            if not qs:
                st.info("📝 Generate MCQs first to export them.")
            else:
                bloom = BloomHeuristicClassifier() if enable_bloom else None
                rows = mcqs_to_export_rows(qs, bloom)
                
                st.markdown("#### Preview")
                st.dataframe(rows, use_container_width=True, hide_index=True)

                st.markdown("#### Download")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "📥 Download JSON",
                        data=json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8"),
                        file_name="mcqs.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                
                with col2:
                    try:
                        import pandas as pd
                        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download CSV",
                            data=csv,
                            file_name="mcqs.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    except ImportError:
                        st.caption("Install pandas for CSV export: `pip install pandas`")

        # ---------- Reports tab ----------
        with tabs[5]:
            st.markdown("### 📈 Student Performance Reports")
            quizzes = load_quizzes()
            attempts = load_attempts()
            my_quizzes = {qid: q for qid, q in quizzes.items() if q.get("teacher_name", "") == teacher_name}
            
            if not my_quizzes:
                st.info("📝 No quizzes created yet. Create and share quizzes to see student reports.")
            else:
                for qid, qmeta in my_quizzes.items():
                    with st.expander(f"📊 {qmeta.get('title', qid)}", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            created = qmeta.get("created_at", "")[:19].replace("T", " ")
                            st.caption(f"📅 Created: {created}")
                        with col2:
                            st.code(f"?quiz={qid}")
                        
                        atts = attempts.get(qid, [])
                        if not atts:
                            st.info("👥 No student attempts yet")
                        else:
                            st.markdown("#### Student Attempts")
                            for a in atts:
                                pct = (100 * a['score']) // max(1, a['total'])
                                status = "🟢" if pct >= 70 else "🟡" if pct >= 50 else "🔴"
                                st.markdown(f"{status} **{a['student_name']}**: {a['score']}/{a['total']} ({pct}%) — {a['completed_at'][:10]}")


if __name__ == "__main__":
    main()