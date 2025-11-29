# UNIT 2: VISUAL DIAGRAMS & FLOWCHARTS
## Feature Extraction & Language Modeling Illustrated

---

## 1. FEATURE EXTRACTION EVOLUTION

```
EVOLUTION OF TEXT REPRESENTATION:

One-Hot Encoding
─────────────────
cat = [1, 0, 0, 0, 0]
dog = [0, 1, 0, 0, 0]
bird = [0, 0, 1, 0, 0]
Pros: Simple
Cons: High dim, 99% sparse, no meaning

                    ↓

Bag of Words (BoW)
──────────────────
"cat dog cat" = [2, 1, 0]  (cat:2, dog:1, bird:0)
Pros: Simple, meaningful
Cons: Order lost, context lost

                    ↓

TF-IDF
──────────────────
cat = 0.5  (frequent in doc, somewhat common overall)
dog = 0.3
bird = 0.8 (rare overall, very distinctive)
Pros: Weighs by importance
Cons: Still no word relationships

                    ↓

Word Embeddings (Skip-Gram, CBOW)
──────────────────────────────────
cat = [0.2, -0.5, 0.8, 0.1, ...]  (300 dims)
dog = [0.3, -0.4, 0.9, 0.0, ...]  (300 dims)
bird = [0.1, 0.2, 0.5, -0.1, ...] (300 dims)
Pros: Dense, semantic, efficient
Cons: Less interpretable than TF-IDF
```

---

## 2. ONE-HOT ENCODING VISUALIZATION

```
VOCABULARY: {the, cat, dog, sat, on}
Size: 5

Word Index Mapping:
┌────────┬───────┐
│ Word   │ Index │
├────────┼───────┤
│ the    │ 0     │
│ cat    │ 1     │
│ dog    │ 2     │
│ sat    │ 3     │
│ on     │ 4     │
└────────┴───────┘

One-Hot Encoded Vectors:

the:  [1, 0, 0, 0, 0]  ← Only position 0 is 1
cat:  [0, 1, 0, 0, 0]  ← Only position 1 is 1
dog:  [0, 0, 1, 0, 0]  ← Only position 2 is 1
sat:  [0, 0, 0, 1, 0]  ← Only position 3 is 1
on:   [0, 0, 0, 0, 1]  ← Only position 4 is 1

DOCUMENT ENCODING:
Sentence: "cat sat on the mat"
Missing "mat" - not in vocabulary

Result:
[
  [0, 1, 0, 0, 0],  # cat
  [0, 0, 0, 1, 0],  # sat
  [0, 0, 0, 0, 1],  # on
  [1, 0, 0, 0, 0]   # the
]

PROBLEM: 50,000 vocabulary
→ 50,000-dimensional vectors!
→ Each vector 99.998% zeros
→ Huge memory waste
```

---

## 3. BAG OF WORDS PROCESS FLOWCHART

```
INPUT: "I love machine learning. Learning is fun"

        ↓ TOKENIZATION
["I", "love", "machine", "learning", "Learning", "is", "fun"]

        ↓ LOWERCASING
["i", "love", "machine", "learning", "learning", "is", "fun"]

        ↓ BUILD VOCABULARY
{i, love, machine, learning, is, fun}

        ↓ COUNT OCCURRENCES
i: 1
love: 1
machine: 1
learning: 2
is: 1
fun: 1

        ↓ CREATE BOW VECTOR
Ordering: [i, love, machine, learning, is, fun]
BoW: [1, 1, 1, 2, 1, 1]

OUTPUT: Document represented as count vector
        (order doesn't matter, only counts)

PROBLEM:
Sentence 1: "I love dogs" → [1, 1, ..., dogs:1, ...]
Sentence 2: "Dogs love me" → [1, 1, ..., dogs:1, ...]
Same representation! But meanings different.
```

---

## 4. TF-IDF CALCULATION FLOWCHART

```
INPUT: Collection of documents

STEP 1: TERM FREQUENCY (within document)
────────────────────────────────────────

Document: "machine learning machine"
Total words: 3

For each word:
TF(machine) = 2/3 ≈ 0.667
TF(learning) = 1/3 ≈ 0.333

Result: Normalized word frequency


STEP 2: INVERSE DOCUMENT FREQUENCY (across collection)
───────────────────────────────────────────────────────

Collection: 10,000 documents

For each word:
machine appears in 500 docs → IDF = log(10000/500) = 1.30
the appears in 9500 docs → IDF = log(10000/9500) = 0.005
learning appears in 4000 docs → IDF = log(10000/4000) = 0.92

Key insight:
- Common words (the, is): LOW IDF
- Rare words (machine-specific): HIGH IDF


STEP 3: MULTIPLY TF × IDF
──────────────────────────

TF-IDF(machine) = 0.667 × 1.30 ≈ 0.87  (HIGHEST!)
TF-IDF(learning) = 0.333 × 0.92 ≈ 0.31
TF-IDF(the) = ? × 0.005 ≈ ~0 (LOWEST)

OUTPUT: TF-IDF vector highlighting important words
        Common words downweighted
        Unique words emphasized
```

---

## 5. SKIP-GRAM WORD2VEC ARCHITECTURE

```
SKIP-GRAM MODEL:

Given word: "king"
Goal: Predict surrounding context words

Input Layer:
─────────────
One-hot: [0,0,0,1,0,...,0]  (word index for "king")
         (vocabulary_size dimensions: 10,000)


Hidden Layer:
─────────────
Embedding transformation
[0,0,0,1,0,...,0] × Weight Matrix (10k × 300)
                 ↓
         [0.25, -0.5, 0.8, ..., 0.3]  (300 dimensions)
         
         ^ This is the WORD EMBEDDING!
         (Dense vector capturing meaning)


Output Layer:
─────────────
[0.25, -0.5, 0.8, ...] × Weight Matrix (300 × 10k)
                      ↓
         [0.2, 0.05, 0.8, 0.1, ...] (softmax probabilities)
         (probability for each word to appear in context)


Context Prediction:
───────────────────
P(man | king) = 0.8   (high probability)
P(queen | king) = 0.7 (high probability)
P(cat | king) = 0.01  (low probability)


TRAINING:
─────────
For sentence: "the king sat on his throne"
Window size: 2 (2 words on each side)
Target: king
Context: {the, sat, on, his}

Adjust weights to:
- Increase P(the | king), P(sat | king), P(on | king), P(his | king)
- Decrease P(cat | king), P(dog | king), etc.

After training:
Words in similar contexts have similar embeddings!
king ≈ queen (both appear with royal context)
king ≈ prince (both appear in similar contexts)
```

---

## 6. BAG OF WORDS vs SKIP-GRAM vs CBOW

```
THREE WORD EMBEDDING APPROACHES:

BAG OF WORDS (Document Level):
────────────────────────────────
"the cat sat on the mat"
Order: irrelevant
Representation: {the:2, cat:1, sat:1, on:1, mat:1}
Vector: [2, 1, 1, 1, 1]
Use: Document classification
Loss: Word order and relationships


SKIP-GRAM (Prediction Task):
────────────────────────────
Target word: "cat"
Predict context: {sat, on, the, mat, ...}

Neural network learns:
- What words appear near "cat"
- How similar "cat" and "dog" contexts are
- Result: cat ≈ dog (similar embeddings)


CBOW (Context-to-Target):
─────────────────────────
Context words: {the, sat, on}
Predict target: "cat"

Neural network learns:
- What word likely given context
- Similar to Skip-gram but reversed task
- Better for common words


COMPARISON:
Skip-gram:
  ✓ Better for rare words
  ✓ Captures nuances
  ✗ Slower training
  
CBOW:
  ✓ Better for common words
  ✓ Faster training
  ✗ Less nuanced
```

---

## 7. NAIVE BAYES CLASSIFICATION PROCESS

```
SPAM CLASSIFICATION EXAMPLE:

STEP 1: TRAINING
─────────────────

Training data:
SPAM: "free money click here"
SPAM: "buy now free"
HAM: "meeting at 3pm"
HAM: "please review document"

Extract probabilities:
P(free | SPAM) = 2/2 = 1.0
P(money | SPAM) = 1/2 = 0.5
P(click | SPAM) = 1/2 = 0.5
P(here | SPAM) = 1/2 = 0.5

P(free | HAM) = 0/2 = 0
P(meeting | HAM) = 1/2 = 0.5
P(review | HAM) = 1/2 = 0.5


STEP 2: NEW EMAIL CLASSIFICATION
──────────────────────────────────

New email: "free money"
Goal: Is it SPAM or HAM?

Calculate: P(SPAM | "free money")
────────────────────────────────
= P("free money" | SPAM) × P(SPAM) / P("free money")
= P(free|SPAM) × P(money|SPAM) × P(SPAM) / P("free money")
= 1.0 × 0.5 × 0.5 / P("free money")
= 0.25 / P("free money")

Calculate: P(HAM | "free money")
───────────────────────────────
= P(free|HAM) × P(money|HAM) × P(HAM) / P("free money")
= 0 × ? × 0.5 / P("free money")
= 0

DECISION:
─────────
P(SPAM | "free money") >> P(HAM | "free money")
→ Classify as SPAM ✓

NAIVE ASSUMPTION:
─────────────────
Assumes: P(free, money | SPAM) = P(free|SPAM) × P(money|SPAM)
Reality: Words not independent!
But: Despite assumption, works well empirically!
```

---

## 8. MARKOV ASSUMPTION & N-GRAMS

```
WITHOUT MARKOV ASSUMPTION:
──────────────────────────
P(cat | the dog is running very fast)
Need entire history of words
PROBLEM: Exponential possibilities!
Huge data sparsity

WITH MARKOV ASSUMPTION (BIGRAM):
─────────────────────────────────
P(cat | fast)  ← Only look at previous 1 word
Much simpler to estimate!

WITH HIGHER N (TRIGRAM):
────────────────────────
P(cat | very fast)  ← Look back 2 words
Better approximation than bigram
But still manageable complexity


N-GRAM TYPES:

Unigram (1-gram):
P(word) independent
"cat", "dog", "the"
Example: P(cat) = 0.01


Bigram (2-gram):
P(word | previous)
"the cat", "dog ran"
Example: P(cat | the) = P("the cat") / P("the") = 0.5


Trigram (3-gram):
P(word | prev2, prev1)
"the quick cat", "dog ran away"
Example: P(sat | the cat) = P("the cat sat") / P("the cat") = 0.8


MARKOV CHAIN VISUALIZATION:

States: Words
Transitions: Probabilities

Example: "I like cats. I like dogs."

From "I" → "like" (prob 1.0, always follows "I")
From "like" → "cats" (prob 0.5) or "dogs" (prob 0.5)
From "cats" → END (prob 1.0)
From "dogs" → END (prob 1.0)

Generation: Start → I → like → cats/dogs → END
```

---

## 9. SMOOTHING TECHNIQUES COMPARISON

```
PROBLEM: Unseen N-grams
─────────────────────────
Training data: 1 million words
Vocabulary: 10,000 words

Possible bigrams: 10,000² = 100 million
Observed bigrams: ~1 million

Result: 99 million bigrams UNSEEN
These have P = 0 (problem!)

SOLUTIONS:

1. LAPLACE SMOOTHING (Add-1)
───────────────────────────
P = (count + 1) / (total + V)

Unseen bigram:
P = (0 + 1) / (total + 10,000)
  = 1 / 10,050
  (Small but non-zero!)

Problem: Gives too much probability to unseen


2. ADD-K SMOOTHING
──────────────────
P = (count + k) / (total + k×V)
k = 0.1 to 0.5 (less than 1)

Less aggressive than Laplace
Better calibration

Unseen with k=0.1:
P = (0 + 0.1) / (total + 0.1×10,000)
  = 0.1 / 10,010
  (Even smaller than Laplace)


3. BACKOFF
──────────
If P(trigram) = 0:
  Use P(bigram)
If P(bigram) = 0:
  Use P(unigram)

Hierarchical: Combine different n-gram levels


4. INTERPOLATION
─────────────────
Combine all n-gram levels:

P = 0.6 × P(trigram) + 0.3 × P(bigram) + 0.1 × P(unigram)

Weights: λ1 + λ2 + λ3 = 1
Can learn optimal weights from data

Advantage: Uses all information available
```

---

## 10. LANGUAGE MODELS & PROBABILITY

```
CHAIN RULE:
───────────
P(word sequence) = Product of conditional probabilities

P(I love cats) = P(I) × P(love|I) × P(cats|love)

Example:
P(I) = 0.1
P(love|I) = 0.5
P(cats|love) = 0.6

P(I love cats) = 0.1 × 0.5 × 0.6 = 0.03


FULL vs MARKOV:
───────────────

Full context (expensive):
P(cat | I love my) = count(I love my cat) / count(I love my)
Need huge dataset

Bigram (Markov):
P(cat | my) = count(my cat) / count(my)
Much easier!


PERPLEXITY METRIC:
──────────────────
Measures: How good language model is

Lower perplexity = Better model

Perplexity = 2^(-L)
where L = average log probability on test set

Example:
Model A: Perplexity = 100
Model B: Perplexity = 50
→ Model B is better (assigns higher probability to test data)
```

---

## 11. GENERATIVE VS DISCRIMINATIVE

```
GENERATIVE MODELS:
──────────────────
Learn: P(text) or P(words)
Generate: Create new samples

Examples:
- N-gram models
- Language models
- Generative models

Process: Learn distribution → Sample from it

Can generate:
"The quick brown fox jumps over..."


DISCRIMINATIVE MODELS:
──────────────────────
Learn: P(class | text)
Classify: Assign labels to data

Examples:
- Naive Bayes classifier
- SVM
- Neural networks for classification

Process: Learn decision boundaries

Can classify:
"The quick brown fox..." → DOCUMENT_TYPE = ADVENTURE

COMPARISON:
───────────────────────────────────────────
Generative:
✓ Can generate new data
✓ Works with less labeled data
✗ May be less accurate for classification

Discriminative:
✓ Better for classification
✓ More accurate typically
✗ Cannot generate new data
✗ Needs more labeled data
```

---

## 12. COUNTVECTORIZER VS TF-IDF EXAMPLE

```
INPUT DOCUMENTS:
───────────────
D1: "machine learning is powerful"
D2: "machine learning is useful"

COUNTVECTORIZER (Raw Counts):
─────────────────────────────
Vocabulary: {machine, learning, is, powerful, useful}

Count Matrix:
        machine  learning  is  powerful  useful
D1      1        1         1   1         0
D2      1        1         1   0         1

Vector D1: [1, 1, 1, 1, 0]
Vector D2: [1, 1, 1, 0, 1]

All words treated equally


TF-IDF VECTORIZER (Weighted):
─────────────────────────────

TF (frequency in document):
All words: 1/4 = 0.25

IDF (rarity across documents):
machine: log(2/2) = 0 (in both docs - common!)
learning: log(2/2) = 0 (in both docs - common!)
is: log(2/2) = 0 (in both docs - common!)
powerful: log(2/1) = 0.30 (in 1 doc - unique!)
useful: log(2/1) = 0.30 (in 1 doc - unique!)

TF-IDF:
       machine  learning  is  powerful  useful
D1     0        0         0   0.075    0
D2     0        0         0   0        0.075

Vector D1: [0, 0, 0, 0.075, 0]
Vector D2: [0, 0, 0, 0, 0.075]

Key insight:
Common words: 0 weight
Unique words: High weight
Much more interpretable!
```

---

**End of Unit 2 Visual Diagrams**

