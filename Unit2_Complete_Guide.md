# UNIT 2: INTRODUCTION TO FEATURE EXTRACTION
## Comprehensive Study Guide with Complete Answers & Revision Sheet

---

## TABLE OF CONTENTS
1. One-Hot Encoding Methods
2. Bag of Words (BoW) & Skip-Grams
3. CountVectorizer & TF-IDF
4. Probabilistic Language Modeling
5. Naive Bayes Classifier
6. Markov Models & N-Grams
7. Word Probability Estimation & Smoothing
8. Generative Models of Language
9. Predicted Exam Questions with Answers
10. Quick Revision Sheet

---

# PART 1: DETAILED EXPLANATIONS

## 1. ONE-HOT ENCODING

### 1.1 Definition & Concept

**One-Hot Encoding**: Method of representing categorical variables (like words) as binary vectors where exactly one element is 1 and all others are 0.

**Purpose**: Convert discrete words into numerical format that machines can process.

**Formula**:
```
Word at position i in vocabulary of size N:
[0, 0, ..., 1, ..., 0]
           ↑
        position i (1)
        rest are 0s
```

### 1.2 How One-Hot Encoding Works

**Step 1: Create Vocabulary**
```
Corpus: "The cat sat on the mat"
Unique words (vocabulary): {the, cat, sat, on, mat}
Vocabulary size: 5

Mapping:
- the: index 0
- cat: index 1
- sat: index 2
- on: index 3
- mat: index 4
```

**Step 2: Create Binary Vectors**
```
Word "the" (index 0):   [1, 0, 0, 0, 0]
Word "cat" (index 1):   [0, 1, 0, 0, 0]
Word "sat" (index 2):   [0, 0, 1, 0, 0]
Word "on" (index 3):    [0, 0, 0, 1, 0]
Word "mat" (index 4):   [0, 0, 0, 0, 1]
```

**Step 3: Encode Sentence**
```
Sentence: "cat on mat"
Result: 
[
  [0, 1, 0, 0, 0],  # cat
  [0, 0, 0, 1, 0],  # on
  [0, 0, 0, 0, 1]   # mat
]
```

### 1.3 Advantages & Disadvantages

**Advantages**:
- Simple to understand and implement
- No information loss (each word gets unique representation)
- Works with any categorical data
- No assumptions about word relationships

**Disadvantages**:
- **High Dimensionality**: Vocabulary of 50,000 words → 50,000-dimensional vectors
- **Sparsity**: Each vector has only one 1, rest are zeros (99.998% zeros)
- **No Semantic Meaning**: Cannot capture similarity (cat and dog equally different)
- **Computational Inefficiency**: Large, sparse matrices hard to process
- **No Context**: Doesn't consider word meaning or relationships

**Example of Inefficiency**:
```
One-hot encoding for 50,000 word vocabulary:
Each word: [0, 0, ..., 1, ..., 0, 0]  (50,000 dimensions)
Memory: 50,000 × 50,000 matrix for documents
Sparsity: 99.998% zeros!

Word2Vec embedding (alternative):
Each word: [0.2, -0.5, 0.8, ...] (300 dimensions)
Memory: 50,000 × 300 matrix
Dense vectors with semantic meaning
```

### 1.4 Mathematical Representation

**One-Hot Vector for word w in vocabulary V of size |V|**:
```
e_w = [e_1, e_2, ..., e_|V|]

where:
e_i = 1 if word w is at position i
e_i = 0 otherwise

∀i: Σ e_i = 1 (exactly one element is 1)
```

---

## 2. BAG OF WORDS (BOW) & SKIP-GRAMS

### 2.1 Bag of Words (BoW)

**Definition**: Representation of text as an unordered collection of words with their frequencies, ignoring word order and grammar.

**Key Principle**: Text is represented by the count of words it contains, not by their sequence.

**How BoW Works**:

**Step 1: Tokenization**
```
Sentence: "The cat sat on the mat"
Tokens: ["The", "cat", "sat", "on", "the", "mat"]
```

**Step 2: Vocabulary Creation**
```
Unique words: {the, cat, sat, on, mat}
(Note: "The" and "the" treated as same after lowercasing)
```

**Step 3: Count Words**
```
Word Counts:
- the: 2
- cat: 1
- sat: 1
- on: 1
- mat: 1
```

**Step 4: Create Vector**
```
BoW vector: [2, 1, 1, 1, 1]
Ordering: [the, cat, sat, on, mat]

Or sparse representation: {the:2, cat:1, sat:1, on:1, mat:1}
```

### 2.2 Advantages & Disadvantages of BoW

**Advantages**:
- Simple to implement
- Computationally efficient
- Works well for simple text classification
- Interpretable (can see which words matter)

**Disadvantages**:
- **Loses Word Order**: "The cat sat" vs "Sat cat the" same representation
- **Loses Context**: "not good" and "good" have same words
- **High Dimensionality**: Vocabulary size can be huge
- **Sparsity**: Most documents don't contain most words
- **No Semantic Relationship**: Cannot understand similar words

**Example - Lost Information**:
```
Document 1: "The dog bit the cat"
Document 2: "The cat bit the dog"

BoW representation (same!):
D1: {the:2, dog:1, bit:1, cat:1}
D2: {the:2, dog:1, bit:1, cat:1}

But meanings are different! Dogs biting cats ≠ Cats biting dogs
```

### 2.3 Skip-Grams

**Definition**: Model that learns to predict context words given a target word. Word "skips" some words to capture broader context.

**Key Difference from BoW**:
- BoW: What words occur in document?
- Skip-gram: What words appear near each other?

**Skip-Gram Model (Word2Vec)**:

**Architecture**:
```
Input Layer: One-hot encoded word (vocab_size)
      ↓
Hidden Layer: (embedding_dim, typically 300)
      ↓
Output Layer: Softmax over vocabulary (vocab_size)

Task: Given word "dog", predict surrounding words
```

**How Skip-Gram Works**:

```
Sentence: "The quick brown fox jumps"
Target word: "fox" (position 3)

Context window size: 2 (two words on each side)

Context words: [quick, brown, jumps, ...]
(words within 2 positions of "fox")

Skip-gram learns:
P(quick | fox) ≈ 0.8
P(brown | fox) ≈ 0.7
P(jumps | fox) ≈ 0.8
P(the | fox) ≈ 0.1  (further away)
```

**Training Process**:
```
For each word w in vocabulary:
  1. Get one-hot encoding of w
  2. Pass through hidden layer → embedding
  3. Output layer produces context word probabilities
  4. Compare with actual context words
  5. Update weights to maximize P(context | word)
```

**Output - Word Embeddings**:
```
After training, hidden layer weights are word embeddings:

dog = [0.2, -0.5, 0.8, 0.1, ..., 0.3]  (300 dims)
cat = [0.3, -0.6, 0.7, 0.2, ..., 0.4]  (300 dims)
fox = [0.1, -0.4, 0.9, 0.0, ..., 0.2]  (300 dims)

Vector properties:
- cat ≈ dog (similar meaning)
- cat ≠ computer (different meanings)
- Semantic relationships: king - man + woman ≈ queen
```

### 2.4 Skip-Gram vs CBOW

**Continuous Bag of Words (CBOW)**:
```
Predicts target word from context words

Input: [quick, brown, jumps]
Output: fox

Better for: Common words, faster training
```

**Skip-Gram**:
```
Predicts context words from target word

Input: fox
Output: [quick, brown, jumps]

Better for: Rare words, captures nuances
```

---

## 3. COUNTVECTORIZER & TF-IDF

### 3.1 CountVectorizer

**Definition**: Converts collection of text documents into a matrix of token counts.

**Process**:

**Step 1: Build Vocabulary**
```
Documents:
D1: "I love machine learning"
D2: "Machine learning is great"
D3: "I love great learning"

Unique words: {i, love, machine, learning, is, great}
Vocabulary: [i, love, machine, learning, is, great] (size: 6)
```

**Step 2: Count Occurrences**
```
Document-Term Matrix:

      i  love  machine  learning  is  great
D1 [  1   1      1        1       0    0  ]
D2 [  0   0      1        1       1    1  ]
D3 [  1   1      0        1       0    1  ]
```

**Step 3: Output**
```
Each document represented as row with word counts

D1 = [1, 1, 1, 1, 0, 0]
D2 = [0, 0, 1, 1, 1, 1]
D3 = [1, 1, 0, 1, 0, 1]
```

**Sparse Representation**:
```
D1: {i:1, love:1, machine:1, learning:1}
(omit zeros to save space)
```

**Code Example**:
```python
from sklearn.feature_extraction.text import CountVectorizer

documents = ["I love machine learning",
             "Machine learning is great"]

vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(documents)

# Result: sparse matrix of word counts
```

### 3.2 TF-IDF (Term Frequency-Inverse Document Frequency)

**Definition**: Numerical statistic that reflects how important a word is to a document in a collection of documents.

**Components**:

#### A. **Term Frequency (TF)**
```
TF(t, d) = (Count of term t in document d) / (Total words in document d)

Example:
Document: "machine learning machine"
Total words: 3

TF(machine) = 2/3 ≈ 0.667
TF(learning) = 1/3 ≈ 0.333
```

#### B. **Inverse Document Frequency (IDF)**
```
IDF(t, D) = log(Total documents in collection / Documents containing term t)

Example:
Collection: 10 documents
Term "machine" appears in 5 documents
Term "the" appears in 10 documents (stop word!)

IDF(machine) = log(10/5) = log(2) ≈ 0.301
IDF(the) = log(10/10) = log(1) = 0
```

**Key Insight**: Common words (the, is, and) get low IDF, rare words get high IDF

#### C. **TF-IDF**
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

Combines:
- How frequent term is in document (TF)
- How rare term is across documents (IDF)

Higher TF-IDF = More important word for this document
```

### 3.3 Detailed TF-IDF Example

```
Corpus: 
D1: "machine learning is great"
D2: "machine learning is useful"
D3: "deep learning is powerful"

Step 1: Calculate TF for each term in each document

D1 word counts: {machine:1, learning:1, is:1, great:1} (4 words total)
TF(machine, D1) = 1/4 = 0.25
TF(learning, D1) = 1/4 = 0.25
TF(is, D1) = 1/4 = 0.25
TF(great, D1) = 1/4 = 0.25

D2 word counts: {machine:1, learning:1, is:1, useful:1} (4 words total)
TF(machine, D2) = 1/4 = 0.25
TF(is, D2) = 1/4 = 0.25
...

D3 word counts: {deep:1, learning:1, is:1, powerful:1} (4 words total)
TF(deep, D3) = 1/4 = 0.25
...

Step 2: Calculate IDF for each term

Total documents: 3

IDF(machine) = log(3/2) ≈ 0.176  (appears in 2 docs)
IDF(learning) = log(3/3) = 0      (appears in all 3 docs) - COMMON!
IDF(is) = log(3/3) = 0            (appears in all 3 docs) - COMMON!
IDF(great) = log(3/1) ≈ 1.099     (appears in 1 doc) - RARE!
IDF(useful) = log(3/1) ≈ 1.099    (appears in 1 doc) - RARE!
IDF(deep) = log(3/1) ≈ 1.099      (appears in 1 doc) - RARE!
IDF(powerful) = log(3/1) ≈ 1.099  (appears in 1 doc) - RARE!

Step 3: Calculate TF-IDF

For D1:
TF-IDF(machine, D1) = 0.25 × 0.176 ≈ 0.044
TF-IDF(learning, D1) = 0.25 × 0 = 0
TF-IDF(is, D1) = 0.25 × 0 = 0
TF-IDF(great, D1) = 0.25 × 1.099 ≈ 0.275 ← HIGHEST!

For D3:
TF-IDF(deep, D3) = 0.25 × 1.099 ≈ 0.275
TF-IDF(learning, D3) = 0.25 × 0 = 0
TF-IDF(is, D3) = 0.25 × 0 = 0
TF-IDF(powerful, D3) = 0.25 × 1.099 ≈ 0.275

Observation:
- Common words "learning", "is" have TF-IDF = 0
- Unique words "great", "deep", "powerful" have high TF-IDF
```

### 3.4 TF-IDF Advantages

**Advantages**:
- Reduces weight of common words (stop words)
- Highlights important, discriminative words
- Better than raw counts for text classification
- Works well with simple ML algorithms

**Disadvantages**:
- Still loses word order
- Doesn't capture word meaning
- Large, sparse matrices for big vocabularies
- No semantic relationships

---

## 4. PROBABILISTIC LANGUAGE MODELING

### 4.1 Definition

**Language Model**: Probability distribution over sequences of words that assigns probability P(W) to any sequence W.

**Purpose**: 
- Assign probability to text
- Generate text
- Perform language tasks (translation, speech recognition)

**Core Question**: What is P(sentence)?

```
P("The cat sat") = ?
P("Sat cat the") = ?

Should be: P("The cat sat") >> P("Sat cat the")
Language model captures English language patterns
```

### 4.2 Chain Rule for Probability

**Joint Probability**:
```
P(w_1, w_2, ..., w_n) = Probability of entire sequence
```

**Chain Rule Decomposition**:
```
P(w_1, w_2, w_3, ..., w_n) 
= P(w_1) × P(w_2 | w_1) × P(w_3 | w_1, w_2) × ... × P(w_n | w_1,...,w_{n-1})

Each word probability conditioned on ALL previous words
```

**Example**:
```
Sentence: "I love cats"

P(I, love, cats) 
= P(I) × P(love | I) × P(cats | I, love)

P(I) = 0.1 (word "I" common)
P(love | I) = 0.5 (after "I", "love" likely)
P(cats | I, love) = 0.7 (after "I love", "cats" very likely)

P(I, love, cats) = 0.1 × 0.5 × 0.7 = 0.035
```

---

## 5. NAIVE BAYES CLASSIFIER

### 5.1 Definition & Bayes' Theorem

**Naive Bayes**: Probabilistic classifier based on Bayes' theorem with strong independence assumptions.

**Bayes' Theorem**:
```
P(C | D) = P(D | C) × P(C) / P(D)

Where:
P(C | D) = Posterior probability (what we want)
           Probability of class C given document D
           
P(D | C) = Likelihood
           Probability of observing document D given class C
           
P(C) = Prior probability
       Probability of class C before seeing any document
       
P(D) = Evidence
       Probability of observing document D
       (constant for all classes)
```

### 5.2 Naive Independence Assumption

**The "Naive" Part**:

Naive Bayes assumes all features (words) are independent given the class.

```
Assumption: P(w_1, w_2, ..., w_n | C) = ∏ P(w_i | C)

Simplifies calculation dramatically!

Without assumption:
P("I love cats" | POSITIVE) = 
  P(I, love, cats | POSITIVE)  [hard to estimate]

With Naive assumption:
P("I love cats" | POSITIVE) = 
  P(I | POSITIVE) × P(love | POSITIVE) × P(cats | POSITIVE)
  [easy to estimate from data]
```

### 5.3 Classification Process

**For Spam Classification**:

```
Document: "Free money click here!"
Classes: {SPAM, NOT_SPAM}

Goal: Find class C that maximizes P(C | D)

P(SPAM | D) ∝ P(D | SPAM) × P(SPAM)
P(NOT_SPAM | D) ∝ P(D | NOT_SPAM) × P(NOT_SPAM)

Using Naive assumption:
P(D | SPAM) = P(free | SPAM) × P(money | SPAM) × P(click | SPAM) × P(here | SPAM)

From training data:
P(free | SPAM) = 0.8 (80% of spam emails have "free")
P(money | SPAM) = 0.7
P(click | SPAM) = 0.6
P(here | SPAM) = 0.4

P(D | SPAM) = 0.8 × 0.7 × 0.6 × 0.4 = 0.134

P(NOT_SPAM | D) = P(free | NOT_SPAM) × ... 
P(free | NOT_SPAM) = 0.01 (only 1% of legitimate emails have "free")
P(money | NOT_SPAM) = 0.02
P(click | NOT_SPAM) = 0.05
P(here | NOT_SPAM) = 0.15

P(D | NOT_SPAM) = 0.01 × 0.02 × 0.05 × 0.15 = 0.000015

Decision:
P(SPAM | D) ∝ 0.134 × P(SPAM)
P(NOT_SPAM | D) ∝ 0.000015 × P(NOT_SPAM)

SPAM probability much higher → Classify as SPAM
```

### 5.4 Variants of Naive Bayes

**1. Multinomial Naive Bayes**:
```
Used when: Word frequencies matter
Example: Spam classification, sentiment analysis

Each word can appear multiple times in document
P(w_i | C) = (count of w_i in documents of class C) / (total words in class C)
```

**2. Bernoulli Naive Bayes**:
```
Used when: Word presence/absence matters
Example: Binary text classification

Each word is either present (1) or absent (0)
P(w_i | C) = (documents of class C with w_i) / (total documents of class C)
```

**3. Gaussian Naive Bayes**:
```
Used when: Features are continuous
Example: Iris flower classification

Assumes continuous features follow Gaussian distribution
```

### 5.5 Advantages & Applications

**Advantages**:
- Simple and fast
- Works well with small datasets
- Handles high-dimensional data
- Good for text classification
- Interpretable

**Applications**:
- Spam email filtering (spam vs. not spam)
- Sentiment analysis (positive vs. negative)
- Text classification
- Medical diagnosis
- Credit scoring

---

## 6. MARKOV MODELS & N-GRAMS

### 6.1 Markov Assumption

**Markov Property**: Future state depends only on present state, not on entire history.

**Application to Language**:
```
Full context: P(w_n | w_1, w_2, ..., w_{n-1})
              Depends on ALL previous words

Markov assumption (limited context):
P(w_n | w_{n-1})  for bigram
P(w_n | w_{n-2}, w_{n-1})  for trigram

SIMPLIFICATION: Only look back n-1 words, not entire history
```

**Why Use Markov Assumption?**:
- Data sparsity: Not enough data to estimate all contexts
- Computational efficiency: Fewer parameters to learn
- Practical: Often sufficient to capture language patterns

### 6.2 N-Gram Models

**Definition**: Sequence of N words

**Types**:
```
Unigram (1-gram): Single word
"cat", "dog", "the"

Bigram (2-gram): Two words
"the cat", "cat sat", "sat on"

Trigram (3-gram): Three words
"the cat sat", "cat sat on"

4-gram: "the cat sat on"

N-gram: General sequence of N words
```

### 6.3 N-Gram Probability

**Unigram (1-gram)**:
```
P(word) = count(word) / total_words

P(the) = 1000 / 10000 = 0.1
P(cat) = 50 / 10000 = 0.005
```

**Bigram (2-gram)**:
```
P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})

Given word "the", what's probability next word is "cat"?
P(cat | the) = count("the cat") / count("the")
            = 200 / 1000 = 0.2
```

**Trigram (3-gram)**:
```
P(w_i | w_{i-2}, w_{i-1}) = count(w_{i-2}, w_{i-1}, w_i) / count(w_{i-2}, w_{i-1})

P(sat | the cat) = count("the cat sat") / count("the cat")
                = 150 / 200 = 0.75
```

### 6.4 Sentence Probability Using N-Grams

**Using Bigrams**:
```
Sentence: "I love cats"

P(I, love, cats) ≈ P(I) × P(love | I) × P(cats | love)

P(I) = 0.1 (unigram probability)
P(love | I) = count("I love") / count("I") = 0.5
P(cats | love) = count("love cats") / count("love") = 0.6

P(I, love, cats) ≈ 0.1 × 0.5 × 0.6 = 0.03
```

### 6.5 N-Gram Model Examples

**Bigram Model Training**:
```
Corpus:
"The cat sat on the mat"
"The dog sat on the ground"
"The cat loves the dog"

Bigram counts:
<START> the: 3
the cat: 2
the dog: 1
cat sat: 1
cat loves: 1
sat on: 2
on the: 2
the mat: 1
the ground: 1
loves the: 1
dog sat: 0 (not observed)

Bigram probabilities:
P(the | <START>) = 3/3 = 1.0
P(cat | the) = 2/3 = 0.667
P(dog | the) = 1/3 = 0.333
P(sat | cat) = 1/2 = 0.5
P(loves | cat) = 1/2 = 0.5
...
```

---

## 7. ESTIMATING WORD PROBABILITY & SMOOTHING

### 7.1 Problem: Zero Probabilities

**The Problem**:
```
If we never see bigram "dog jumps" in training data:
P(jumps | dog) = count("dog jumps") / count("dog") = 0/5 = 0

Problem: New sentences can't use this bigram at all!

More generally:
For 10,000 word vocabulary:
Possible bigrams: 10,000² = 100 million
Training data usually has < 1 billion words
Most bigrams never seen!

Result: Many probabilities will be 0
```

**Data Sparsity**:
```
Most n-grams never appear in training data
Model cannot handle unseen n-grams
Need technique to assign non-zero probability to unseen data
```

### 7.2 Smoothing Techniques

**Purpose**: Assign small probability to unseen n-grams

#### A. **Laplace Smoothing (Add-One)**

**Formula**:
```
P_Laplace(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)

Where:
count = observed count in data
V = vocabulary size
+1 = add one count to all n-grams (seen or unseen)
```

**Example**:
```
Vocabulary size: 10
count("dog jumps") = 0 (never seen)
count("dog") = 5

Without smoothing:
P(jumps | dog) = 0/5 = 0 (problem!)

With Laplace smoothing:
P(jumps | dog) = (0 + 1) / (5 + 10) = 1/15 ≈ 0.067

Now unseen bigram has probability 1/15 instead of 0
```

**Advantages**:
- Simple
- Guarantees non-zero probabilities

**Disadvantages**:
- Gives too much probability to unseen n-grams
- May overestimate rare events

#### B. **Add-k Smoothing**

**Formula**:
```
P_AddK(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k×V)

Where k = 0.1, 0.5, etc. (hyperparameter)
```

**Advantage**: Less aggressive than Add-One

```
With k = 0.5:
P(jumps | dog) = (0 + 0.5) / (5 + 0.5×10) = 0.5 / 10 = 0.05

Less probability given to unseen than Add-One
```

#### C. **Backoff**

**Idea**: If n-gram unseen, use shorter n-gram

```
For trigram P(w_i | w_{i-2}, w_{i-1}):
If P(w_i | w_{i-2}, w_{i-1}) = 0 (unseen):
  Use bigram: P(w_i | w_{i-1})
  
If bigram also unseen:
  Use unigram: P(w_i)
```

**Example**:
```
Never seen: "New York City" (trigram)
Check bigram: P(City | York) ← if exists, use this
If not, check unigram: P(City) ← fallback
```

#### D. **Interpolation**

**Idea**: Combine probabilities from different n-grams

```
P_interp(w_i | w_{i-2}, w_{i-1}) = 
  λ_3 × P(w_i | w_{i-2}, w_{i-1}) +
  λ_2 × P(w_i | w_{i-1}) +
  λ_1 × P(w_i)

Where: λ_1 + λ_2 + λ_3 = 1  (weights sum to 1)
Typical: λ_1 = 0.1, λ_2 = 0.3, λ_3 = 0.6
```

**Advantage**: Uses all levels of information

```
If unseen trigram and unseen bigram:
Still contributes from unigram through interpolation
```

---

## 8. GENERATIVE MODELS OF LANGUAGE

### 8.1 Definition

**Generative Model**: Model that generates new text samples by learning the underlying probability distribution of a language.

**vs Discriminative Model**:
```
Generative: Learn P(text)
           Generate new text samples
           Example: N-gram models, language models

Discriminative: Learn P(class | text)
                Classify text
                Example: Naive Bayes classifier
```

### 8.2 How Generative Models Work

**Training**:
1. Learn probability distribution from training data
2. Estimate P(word sequence)

**Generation**:
1. Sample word probabilities
2. Generate new sequences

**Example - N-gram Generation**:
```
Trigram model trained on English text

Generation process:
Start: "<START> The"

Step 1: Generate word after "The"
  Options: cat (0.4), dog (0.3), car (0.2), house (0.1)
  Sample randomly: "cat"
  
Result so far: "The cat"

Step 2: Generate word after "cat"
  Options: sat (0.5), loves (0.3), is (0.2)
  Sample: "sat"
  
Result so far: "The cat sat"

Step 3: Generate word after "sat"
  Options: on (0.6), down (0.3), <END> (0.1)
  Sample: "on"
  
Result so far: "The cat sat on"

Continue until <END> token
Generated sentence: "The cat sat on..."
```

### 8.3 Applications of Generative Models

**Text Generation**:
- Story writing
- Poetry generation
- Email/code completion

**Machine Translation**:
- Generate target language from source

**Speech Recognition**:
- Generate text from acoustic signal

**Data Augmentation**:
- Generate synthetic training data

### 8.4 Evaluation of Language Models

**Perplexity**: Main metric for language models

```
Perplexity = 2^(-L)

Where L = average log probability of test set

Lower perplexity = better model

Example:
Model A: Perplexity = 100
Model B: Perplexity = 50
→ Model B is better (assigns higher probability to test data)
```

---

# PART 2: PREDICTED EXAM QUESTIONS WITH ANSWERS

## QUESTION 1: ONE-HOT ENCODING & FEATURE EXTRACTION (PROBABILITY: 90%)

### Question:
**"Explain one-hot encoding method for feature extraction. What are its advantages and disadvantages? Discuss why other encoding methods like word embeddings are preferred."**

### Complete Answer (6 marks):

**One-Hot Encoding Explanation (1.5 marks)**

One-hot encoding is a method of representing categorical variables as binary vectors where exactly one element is 1 and all others are 0. 

**Process**:
1. Create vocabulary of unique words
2. Assign index to each word
3. For each word, create vector with 1 at that index position, 0s elsewhere

**Example**:
```
Vocabulary: {the, cat, sat}
Word "cat" (index 1) → [0, 1, 0]
Word "sat" (index 2) → [0, 0, 1]
```

**Advantages (1 mark)**:
- Simple to understand and implement
- No information loss (each word unique)
- Works with any categorical data
- No assumptions about relationships

**Disadvantages (1.5 marks)**:
1. **High Dimensionality**: 50,000 word vocabulary = 50,000-dimensional vectors
2. **Sparsity**: 99.998% zeros in each vector
3. **No Semantic Meaning**: Cannot capture that "cat" and "dog" are similar
4. **Computational Inefficiency**: Large sparse matrices hard to process
5. **Memory Intensive**: Requires storing large matrices

**Example of Problem**:
```
One-hot: cat = [0,0,0,0,1,0,...] (50,000 dims)
         dog = [0,0,0,0,0,1,...] (50,000 dims)
         
Similarity: 0 (orthogonal vectors)
But semantically cat and dog are similar!
```

**Why Word Embeddings Preferred (1.5 marks)**:

Word embeddings (like Word2Vec) solve these problems:

```
Word2Vec: cat = [0.2, -0.5, 0.8, ...] (300 dims)
          dog = [0.3, -0.4, 0.9, ...]  (300 dims)

Similarity: High (vectors close together)
Semantic meaning: Captured in dimensions
Dense representation: Only 300 dimensions, no zeros
Efficient: Can process quickly
```

**Advantages of embeddings**:
- Low dimensionality (300-500 vs 50,000)
- Dense vectors (meaningful information in each dimension)
- Captures semantic relationships
- Transfer learning: Pre-trained embeddings help new tasks
- Compositional: dog + pet ≈ cat

---

## QUESTION 2: BAG OF WORDS & TF-IDF (PROBABILITY: 90%)

### Question:
**"Compare Bag of Words and TF-IDF methods. Explain how TF-IDF improves upon BoW. Provide a detailed example showing TF and IDF calculations."**

### Complete Answer (6 marks):

**Bag of Words Explanation (1 mark)**

BoW represents text as unordered collection of words with frequencies, ignoring word order and grammar.

**Process**:
1. Tokenize and lowercase
2. Count word frequencies
3. Create vector of counts

**Example**:
```
Text: "The cat sat on the mat"
BoW: {the:2, cat:1, sat:1, on:1, mat:1}
Vector: [2, 1, 1, 1, 1]
```

**BoW Limitations (1 mark)**

1. **Loses Word Order**: "dog ate cat" vs "cat ate dog" same representation
2. **Loses Context**: "not good" and "good" indistinguishable
3. **High Dimensionality**: Huge vocabulary size
4. **Sparsity**: Mostly zeros
5. **No Semantic Relationship**: Similar words treated as unrelated

**TF-IDF Explanation (1.5 marks)**

TF-IDF combines two statistics to identify important words:

**Term Frequency (TF)**:
```
TF(t, d) = (Count of term t in doc d) / (Total words in doc d)

Measures: How often word appears in specific document
Intuition: Frequent words more relevant to document
```

**Inverse Document Frequency (IDF)**:
```
IDF(t, D) = log(Total documents / Documents containing term t)

Measures: How rare term is across collection
Intuition: Rare words more discriminative
```

**TF-IDF Score**:
```
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)

High score: Appears frequently in specific document AND rare overall
Low score: Common word or not in document
```

**Detailed Example (2.5 marks)**

**Corpus**:
```
D1: "machine learning is powerful"
D2: "machine learning is efficient"
D3: "deep learning is interesting"
Total docs: 3
Words per doc: 4
```

**Step 1: Calculate TF**

Document D1: 4 words total
```
TF(machine, D1) = 1/4 = 0.25
TF(learning, D1) = 1/4 = 0.25
TF(is, D1) = 1/4 = 0.25
TF(powerful, D1) = 1/4 = 0.25
```

**Step 2: Calculate IDF**

```
machine: appears in D1, D2 → 2 docs → IDF = log(3/2) ≈ 0.176
learning: appears in D1, D2, D3 → 3 docs → IDF = log(3/3) = 0
is: appears in D1, D2, D3 → 3 docs → IDF = log(3/3) = 0
powerful: appears in D1 → 1 doc → IDF = log(3/1) ≈ 1.099
```

**Step 3: Calculate TF-IDF**

For Document D1:
```
TF-IDF(machine, D1) = 0.25 × 0.176 ≈ 0.044
TF-IDF(learning, D1) = 0.25 × 0 = 0 (too common)
TF-IDF(is, D1) = 0.25 × 0 = 0 (too common)
TF-IDF(powerful, D1) = 0.25 × 1.099 ≈ 0.275 ← HIGHEST!
```

**Comparison**:
```
BoW D1: [1, 1, 1, 1] (all words equal importance)
TF-IDF D1: [0.044, 0, 0, 0.275] (unique words weighted more)

"powerful" gets highest score - correctly identifies important word
"learning" "is" get 0 - correctly identifies common words
```

---

## QUESTION 3: NAIVE BAYES CLASSIFIER (PROBABILITY: 85%)

### Question:
**"Explain Naive Bayes classifier. Describe Bayes' theorem and the naive independence assumption. Provide a spam classification example."**

### Complete Answer (6 marks):

**Bayes' Theorem (1 mark)**

```
P(C | D) = P(D | C) × P(C) / P(D)

Where:
P(C | D) = Posterior: Probability of class C given document D
P(D | C) = Likelihood: Probability of document D given class C
P(C) = Prior: Probability of class C
P(D) = Evidence: Probability of document D
```

**Intuition**: Update prior belief based on observed evidence

**Naive Independence Assumption (1.5 marks)**

"Naive" = Assumes words are independent given class

```
Without assumption (hard):
P(w1, w2, w3 | C) = requires huge dataset

With assumption (easy):
P(w1, w2, w3 | C) = P(w1|C) × P(w2|C) × P(w3|C)

Each word probability independent
Drastically simplifies calculation
```

**Why works**: Despite naive assumption, empirically works well for text classification!

**Multinomial Naive Bayes Process (1 mark)**

For text classification:
```
1. Count word frequencies in each class
2. Calculate P(word | class) from frequencies
3. For new document, multiply all word probabilities
4. Choose class with highest probability
```

**Spam Classification Example (2.5 marks)**

**Training Data**:
```
SPAM emails (10 total):
- Email1: "Free money now"
- Email2: "Click here free"
... (10 spam emails)

NOT_SPAM emails (10 total):
- Email1: "Meeting at 3pm"
- Email2: "Please review document"
... (10 legitimate emails)
```

**Probability Estimation**:

```
From SPAM emails:
P(free | SPAM) = 8/10 = 0.8 (appears in 8 spam emails)
P(money | SPAM) = 6/10 = 0.6
P(click | SPAM) = 7/10 = 0.7
P(meeting | SPAM) = 0/10 = 0 (never in spam)

From NOT_SPAM emails:
P(free | NOT_SPAM) = 1/10 = 0.1
P(money | NOT_SPAM) = 1/10 = 0.1
P(click | NOT_SPAM) = 2/10 = 0.2
P(meeting | NOT_SPAM) = 4/10 = 0.4

Prior probabilities:
P(SPAM) = 10/20 = 0.5
P(NOT_SPAM) = 10/20 = 0.5
```

**Classification of new email**:
```
New email: "Free money click"

P(SPAM | "free money click") ∝ 
  P("free money click" | SPAM) × P(SPAM)
  = P(free|SPAM) × P(money|SPAM) × P(click|SPAM) × P(SPAM)
  = 0.8 × 0.6 × 0.7 × 0.5
  = 0.168

P(NOT_SPAM | "free money click") ∝ 
  P(free|NOT_SPAM) × P(money|NOT_SPAM) × P(click|NOT_SPAM) × P(NOT_SPAM)
  = 0.1 × 0.1 × 0.2 × 0.5
  = 0.001

Decision:
0.168 > 0.001
→ Classify as SPAM
```

**Advantages**:
- Fast computation
- Works with small datasets
- Good for text classification
- Interpretable

---

## QUESTION 4: N-GRAMS & MARKOV MODELS (PROBABILITY: 85%)

### Question:
**"Explain N-gram models and Markov assumption. Provide examples of unigram, bigram, and trigram probabilities. Discuss smoothing techniques for handling unseen n-grams."**

### Complete Answer (6 marks):

**Markov Assumption (1 mark)**

The Markov property states that future probability depends only on present state, not entire history.

```
Full: P(w_n | w_1, w_2, ..., w_{n-1})
Markov: P(w_n | w_{n-1})  (bigram)
        P(w_n | w_{n-2}, w_{n-1})  (trigram)

Reduces data sparsity and computational cost
```

**N-Gram Definition (1 mark)**

N-gram: Sequence of N consecutive words

```
Unigram (1-gram): "cat"
Bigram (2-gram): "the cat"
Trigram (3-gram): "the cat sat"
```

**N-Gram Probabilities (2 marks)**

**Unigram**:
```
P(w) = count(w) / total_words
P(the) = 1000 / 10000 = 0.1
```

**Bigram**:
```
P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
P(cat | the) = count("the cat") / count("the")
            = 200 / 1000 = 0.2
```

**Trigram**:
```
P(w_i | w_{i-2}, w_{i-1}) = count(w_{i-2}, w_{i-1}, w_i) / count(w_{i-2}, w_{i-1})
P(sat | the cat) = count("the cat sat") / count("the cat")
                = 150 / 200 = 0.75
```

**Smoothing Techniques (2 marks)**

**1. Laplace Smoothing**:
```
P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)

Where V = vocabulary size

Advantage: Simple, guarantees non-zero probability
Disadvantage: Overestimates unseen n-grams
```

**2. Add-k Smoothing**:
```
P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + k) / (count(w_{i-1}) + k×V)

k < 1 (e.g., 0.1, 0.5)
Less aggressive than Laplace
```

**3. Backoff**:
```
If P(w_i | w_{i-2}, w_{i-1}) = 0:
  Use P(w_i | w_{i-1})
If still 0:
  Use P(w_i)

Combines different n-gram levels
```

**Example - Laplace Smoothing**:
```
Never seen: "dog jumps"
count("dog jumps") = 0
count("dog") = 5
Vocabulary size: 10

Without smoothing:
P(jumps | dog) = 0/5 = 0

With Laplace:
P(jumps | dog) = (0 + 1) / (5 + 10) = 1/15 ≈ 0.067

Now unseen bigram has small probability instead of 0
```

---

# PART 3: QUICK REVISION SHEET FOR UNIT 2

---

## UNIT 2 QUICK REVISION SHEET
## Feature Extraction & Language Modeling

---

### 1. ONE-HOT ENCODING - SUMMARY

```
Definition: Binary vector with single 1, rest 0s

Example:
Vocabulary: {cat, dog, bird}
cat: [1, 0, 0]
dog: [0, 1, 0]
bird: [0, 0, 1]

Problems:
- High dimensional (vocab_size)
- Sparse (99% zeros)
- No semantic meaning
- Memory intensive

Solution: Word embeddings (dense vectors)
```

---

### 2. BAG OF WORDS - KEY POINTS

```
What: Unordered collection of word counts
How: 1. Tokenize 2. Count words 3. Create vector

Example:
"cat sat" → {cat:1, sat:1} → [1, 1]

Problem: Loses word order and context
"I hate dogs" vs "I love dogs" same words!

Better: TF-IDF, word embeddings
```

---

### 3. TF-IDF - FORMULAS & INTUITION

```
TF (Term Frequency):
TF = count(word in doc) / total_words_in_doc
How often word appears in document

IDF (Inverse Document Frequency):
IDF = log(total_docs / docs_containing_word)
How rare word is overall

TF-IDF = TF × IDF
High score: Frequent in doc + Rare overall
Low score: Stop words (common everywhere)

Benefits:
- Down-weights common words
- Highlights important words
- Better than raw counts
```

---

### 4. SKIP-GRAM WORD2VEC - QUICK

```
Learns: Word embeddings (dense vectors)

How: 
- Input: One-hot encoded word
- Hidden layer: Embedding (300 dims)
- Output: Context word probabilities
- Task: Predict context from word

Result: Words with similar contexts → similar embeddings
dog ≈ cat (both appear with similar words)

Better than One-Hot:
- Dense vectors (300 vs 50,000 dims)
- Semantic meaning captured
- Efficient computation
```

---

### 5. PROBABILISTIC LANGUAGE MODELING

```
Goal: P(word sequence)

Chain rule:
P(w1, w2, w3) = P(w1) × P(w2|w1) × P(w3|w1,w2)

Markov assumption (bigram):
P(w3|w1,w2) ≈ P(w3|w2)

Only look back 1 word instead of all history
Much simpler to estimate from data!
```

---

### 6. NAIVE BAYES - CORE CONCEPTS

```
Bayes' Theorem:
P(C|D) = P(D|C) × P(C) / P(D)

Naive assumption:
P(w1,w2,w3|C) = P(w1|C) × P(w2|C) × P(w3|C)

Words independent given class (not realistic!)
But works well empirically

Classification:
P(SPAM|"free money") ∝ P(free|SPAM) × P(money|SPAM) × P(SPAM)
P(HAM|"free money") ∝ P(free|HAM) × P(money|HAM) × P(HAM)
Choose class with higher probability
```

---

### 7. N-GRAMS & MARKOV MODELS

```
N-gram: Sequence of N words

Unigram: P(w) = freq(w) / total
Bigram: P(w2|w1) = count(w1,w2) / count(w1)
Trigram: P(w3|w1,w2) = count(w1,w2,w3) / count(w1,w2)

Markov assumption:
Only look back n-1 words, not entire history
Reduces data sparsity + computation

Problem: Unseen n-grams have P = 0!
```

---

### 8. SMOOTHING TECHNIQUES - COMPARISON

| Technique | Formula | Use |
|-----------|---------|-----|
| **Laplace** | (count+1)/(total+V) | Simple, aggressive |
| **Add-k** | (count+k)/(total+kV) | Moderate, k<1 |
| **Backoff** | Use lower-order n-gram | Hierarchical |
| **Interpolation** | Combine different orders | Weighted blend |

All solutions: Assign non-zero P to unseen n-grams

---

### 9. FEATURE EXTRACTION COMPARISON TABLE

| Method | Dimensionality | Sparsity | Semantic | Use Case |
|--------|---|---|---|---|
| **One-Hot** | High | 99%+ | None | Baseline only |
| **BoW** | High | High | None | Text classification |
| **TF-IDF** | High | Medium | None | Search, classification |
| **Skip-gram** | Low (300) | Dense | Yes | Modern NLP |

Trend: Higher dimensional → More sparse → Less efficient

---

### 10. ONE-LINER SUMMARIES

- **One-Hot**: One 1, rest 0s (simple but inefficient)
- **BoW**: Count words, ignore order
- **TF-IDF**: Frequent in doc + rare overall
- **Skip-Gram**: Learn embeddings by predicting context
- **Language Model**: P(sequence) using chain rule
- **Naive Bayes**: Classify using Bayes' theorem + independence
- **N-gram**: Sequence of N words
- **Markov**: Only look back n-1 words
- **Smoothing**: Assign probability to unseen n-grams
- **Generative**: Generate new text samples

---

### 11. FORMULAS TO MEMORIZE

```
TF = count(word,doc) / total_words_in_doc

IDF = log(total_docs / docs_with_word)

TF-IDF = TF × IDF

P(C|D) = P(D|C) × P(C) / P(D)  [Bayes]

P(w2|w1) = count(w1,w2) / count(w1)  [Bigram]

P_Laplace(w|w') = (count(w',w) + 1) / (count(w') + V)
```

---

### 12. PREDICTED EXAM QUESTIONS SUMMARY

| Question | Probability | Key Topics |
|----------|-----------|-----------|
| One-Hot & Embeddings | 90% | Definition, advantages, disadvantages |
| BoW & TF-IDF | 90% | Calculation, comparison, example |
| Naive Bayes | 85% | Bayes, naive assumption, classification |
| N-grams | 85% | Probabilities, Markov, smoothing |
| Skip-gram | 70% | Architecture, output embeddings |
| Smoothing | 75% | Techniques, Laplace, backoff |

---

### 13. COMMON MISTAKES TO AVOID

❌ Confusing one-hot with embeddings
✓ One-hot: binary vectors. Embeddings: dense semantic

❌ BoW and TF-IDF same thing
✓ BoW: counts. TF-IDF: weighted by rarity

❌ Naive Bayes assumes words independent in reality
✓ Words NOT independent, but naive assumption makes model tractable

❌ Unseen n-gram should have P = 0
✓ Need smoothing to handle unseen data

❌ Higher n-gram always better
✓ Higher n → more data sparsity → need more smoothing

---

### 14. PRACTICE CALCULATION

**Quick TF-IDF Calculation**:
```
3 documents, 1000 total words

Doc: "machine learning" (2 words)
"machine" count in doc: 1
TF(machine) = 1/2 = 0.5

Docs containing "machine": 2 (appears in this doc + 1 other)
IDF(machine) = log(3/2) ≈ 0.176

TF-IDF = 0.5 × 0.176 = 0.088
```

---

**End of Unit 2 Quick Revision Sheet**
**Total Study Material: 4 comprehensive documents**

