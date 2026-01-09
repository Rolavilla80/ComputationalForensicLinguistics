# Computational_Forensic_Linguistics
Final Paper and Repo

# Robustness of Stylometric AI-Text Detection Under Generator Shift

This repository contains code and analysis for a computational forensics study on AI-generated text detection.
The focus is on robustness under distribution shift, using character-level stylometric features and semantic embeddings.

## Summary
- Task: Binary classification (Human vs AI-generated text)
- Dataset: AH-AITD (Arslan’s Human and AI Text Database)
- Evaluation:
  - Random 80/20 split (in-distribution)
  - Hold-out generator experiments (out-of-distribution)
- Models:
  - Character TF–IDF (3–5 char n-grams) + Logistic Regression (calibrated)
  - Character TF–IDF + Random Forest
  - SBERT embeddings + Logistic Regression

## Main Findings
- Character n-grams perform strongly in-distribution
- Performance degrades sharply under generator shift
- Paraphrase-generated text defeats stylometric detectors
- SBERT generalizes more uniformly but with weaker discrimination

