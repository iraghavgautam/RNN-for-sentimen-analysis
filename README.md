# IMDB Movie Review Sentiment Analysis using RNN

This repository contains a PyTorch implementation of a Recurrent Neural Network (RNN) designed to perform binary sentiment analysis on movie reviews. The model processes raw text data, extracts features using TF-IDF vectorization, and classifies the sentiment as either positive or negative.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Setup & Installation](#setup--installation)
- [Model Performance](#model-performance)

---

## Project Overview
The goal of this project is to build a deep learning pipeline that cleans natural language text, applies classic text preprocessing techniques (lowercasing, punctuation removal, HTML tag stripping, stopword filtering, and stemming), and trains a basic Recurrent Neural Network (RNN) in PyTorch.

## Dataset
The project utilizes the classic **IMDB Dataset**, which consists of 50,000 highly polar movie reviews for binary sentiment classification. 
* **Features:** `review` (Text string containing user reviews)
* **Target:** `sentiment` (Binary value: `1` for positive, `0` for negative)

*Note: Duplicate records are dropped during preprocessing, reducing the dataset size slightly to 49,582 reviews.*

---

## Project Architecture

1. **Text Preprocessing:**
   * Case folding (lowercasing)
   * URL and HTML tag removal using regular expressions (`re`)
   * Punctuation removal
   * Tokenization & Stopword exclusion via `nltk`
   * Stemming via `PorterStemmer`
2. **Feature Extraction:**
   * Text is converted into numerical values using a `TfidfVectorizer` configured for the top 5,000 most frequent features.
3. **Deep Learning Model:**
   * Built using PyTorch (`nn.RNN`), followed by a fully connected linear layer (`nn.Linear`) mapped to a single output.
   * Optimized using Binary Cross-Entropy Loss (`nn.BCELoss`) and the Adam optimizer.

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-directory>
