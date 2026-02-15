# MCQ + Toxicity/Bias + Cognitive Level (UI)

This is a UI on top of your existing POC (MCQ generation + toxicity + bias), with an extra feature: **Bloom (cognitive level) estimation** for each generated question.

## Setup

From `C:\Users\viswa\Downloads\mcq-safety-ui`:

```bash
python -m pip install -r requirements.txt
```

If `torch` install fails on Windows, install PyTorch first from the official selector, then re-run the requirements install.

## Run

```bash
python -m streamlit run streamlit_app_mcq_safety.py
```

## What you get

- **Generate tab**: upload a file (.txt, .docx, .pdf) or paste a paragraph, then generate MCQs. Choose **number of questions**, **class/grade** (1st–12th), and **difficulty** (Easy/Medium/Hard) for age-appropriate, level-targeted questions. Optional AI-generated explanations for each answer.
- **Practice Quiz tab**: take the generated MCQs interactively—select answers, submit, and see your score plus explanations.
- **Safety & Bias tab**: toxicity and bias checks on the passage, optional sentence-level flags, and **MCQ-level safety** (checks generated questions and options).
- **Cognitive level tab**: Bloom estimate (heuristic POC).
- **Export tab**: download MCQs as JSON/CSV (includes explanations when enabled).

