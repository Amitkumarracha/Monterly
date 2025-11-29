# UNIT 3: INTRODUCTION TO WORD EMBEDDINGS & NEURAL NETWORKS
## Comprehensive Study Guide with Complete Answers & Revision Sheet

---

## TABLE OF CONTENTS
1. Word Embeddings & Word Vectors (Word2Vec)
2. Gensim Word2Vec Example
3. Word Vectors 2 & Word Senses (GloVe)
4. Word Window Classification
5. Neural Network Basics
6. Matrix Calculus & Backpropagation
7. Dependency Parsing
8. Negative Sampling
9. Predicted Exam Questions with Answers
10. Quick Revision Sheet

---

# PART 1: DETAILED EXPLANATIONS

## 1. WORD EMBEDDINGS & WORD VECTORS (WORD2VEC)

### 1.1 Definition & Concept

**Word Embeddings**: Dense vector representations of words that capture semantic and syntactic relationships, where similar words have similar vectors.

**Word2Vec**: Algorithm that learns word embeddings using neural networks with shallow architecture.

**Key Innovation**: Instead of learning word embeddings as byproduct, Word2Vec directly learns representations using self-supervised learning (learning from unlabeled data).

### 1.2 Core Principles

**Distributional Hypothesis**: Words that appear in similar contexts have similar meanings.

**Example**:
```
"The king wears a crown"
"The queen wears a crown"

"king" and "queen" appear in similar contexts
→ king ≈ queen (similar embeddings)
```

### 1.3 Two Architectures of Word2Vec

#### A. **Skip-Gram Model**

**Objective**: Given a word, predict surrounding context words

**Architecture**:
```
Input: One-hot encoded word (vocabulary_size)
         ↓
Hidden Layer: Embedding (embedding_size, typically 300)
         ↓
Output: Probability distribution over vocabulary (softmax)
         ↓
Prediction: Context words
```

**Process**:
```
Sentence: "The quick brown fox jumps"
Target word: "fox"
Context window: 2 (predict words 2 positions away)

Input: fox (one-hot)
       ↓
Hidden: [0.2, -0.5, 0.8, ...]  (300-dim embedding)
       ↓
Output probabilities:
P(quick | fox) = 0.7
P(brown | fox) = 0.6
P(jumps | fox) = 0.8
P(car | fox) = 0.01 (unrelated)
```

**Advantages**:
- Better for rare words
- Higher quality embeddings on large corpora
- Captures semantic relationships well

**Disadvantages**:
- Slower training (multiple predictions per word)
- More computationally expensive

#### B. **Continuous Bag of Words (CBOW)**

**Objective**: Given context words, predict center word

**Architecture**:
```
Input: Multiple one-hot encoded context words
         ↓
Hidden Layer: Average of embeddings
         ↓
Output: Probability for center word (softmax)
```

**Process**:
```
Sentence: "The quick brown fox jumps"
Target word: "fox"
Context window: 2

Input: {quick, brown, jumps} (context words)
       ↓
Hidden: Average of embeddings = [0.15, -0.35, 0.7, ...]
       ↓
Output probability:
P(fox | quick, brown, jumps) = 0.95 (high!)
P(cat | quick, brown, jumps) = 0.02 (low)
```

**Advantages**:
- Faster training
- Better for frequent words
- Needs less data

**Disadvantages**:
- Lower quality embeddings on large corpora
- Less good for rare words

### 1.4 Skip-Gram vs CBOW Comparison

```
                Skip-Gram           CBOW
─────────────────────────────────────────────
Direction       Word → Context      Context → Word
Input           Single word         Multiple words
Output          Multiple words      Single word
Training speed  Slower              Faster
Rare words      Better              Worse
Semantic        Excellent           Good
Syntactic       Good                Excellent
Data needed     More                Less
Model size      Larger              Smaller

When to use:
- Skip-Gram: Need good quality embeddings, have large dataset
- CBOW: Need speed, small dataset, good syntactic relations
```

### 1.5 Vector Semantics

**Key Property**: Vector arithmetic captures semantic relationships

```
king - man + woman ≈ queen

Why this works:
- king vector encodes "royalty" + "male"
- man vector encodes "male"
- Remove "male" from king: get "royalty"
- Add "woman" (female + human): get queen

This emerges naturally from Skip-gram training!
```

**Other Examples**:
```
paris - france + germany ≈ berlin
  (country - specific + other country ≈ other capital)

good - bad + worse ≈ terrible
  (antonym relationships encoded)
```

---

## 2. GENSIM WORD2VEC EXAMPLE

### 2.1 Installation & Setup

```python
# Installation
pip install gensim

# Import
from gensim.models import Word2Vec
import numpy as np
```

### 2.2 Training Skip-Gram Model

```python
# Sample corpus
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["a", "dog", "ran", "in", "the", "park"],
    ["the", "cat", "and", "the", "dog", "played"],
    ["cats", "and", "dogs", "are", "animals"]
]

# Train Skip-Gram Model
model = Word2Vec(
    sentences,
    vector_size=100,      # Embedding dimension
    window=2,             # Context window size
    min_count=1,          # Minimum word frequency
    sg=1,                 # 1 = Skip-gram, 0 = CBOW
    workers=4             # Number of threads
)

# Access word embedding
cat_vector = model.wv['cat']
print(cat_vector)  # Output: [0.2, -0.5, 0.8, ...]
```

### 2.3 Key Operations

**Word Similarity**:
```python
# Cosine similarity between words
similarity = model.wv.similarity('cat', 'dog')
print(similarity)  # Output: 0.85 (high, similar)

similarity = model.wv.similarity('cat', 'car')
print(similarity)  # Output: 0.15 (low, different)
```

**Most Similar Words**:
```python
# Find similar words
similar = model.wv.most_similar('king', topn=5)
# Output: [('queen', 0.92), ('prince', 0.89), ('throne', 0.87), ...]

# Analogies
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=5
)
# Output: [('queen', 0.95), ('princess', 0.92), ...]
```

**Vector Operations**:
```python
# Arithmetic on vectors
king = model.wv['king']
man = model.wv['man']
woman = model.wv['woman']

queen_approx = king - man + woman
# Now find closest word to this vector
similar_to_queen = model.wv.most_similar([queen_approx], topn=1)
print(similar_to_queen)  # Output: [('queen', 0.95)]
```

### 2.4 CBOW Model Example

```python
# Train CBOW Model (sg=0)
cbow_model = Word2Vec(
    sentences,
    vector_size=100,
    window=2,
    sg=0,           # 0 = CBOW
    min_count=1
)

# Usage is same, embeddings just learned differently
```

### 2.5 Model Evaluation

```python
# Check if words are in vocabulary
'cat' in model.wv  # True
'xyz' in model.wv  # False

# Vocabulary size
vocab_size = len(model.wv)

# Save and load model
model.save('word2vec_model.bin')
loaded_model = Word2Vec.load('word2vec_model.bin')

# Training parameters
print(model.vector_size)     # 100
print(model.window)          # 2
print(model.min_count)       # 1
```

---

## 3. WORD VECTORS 2 & WORD SENSES (GLOVE)

### 3.1 Limitations of Word2Vec

**Problem 1: One Vector Per Word**
```
"bank" has multiple meanings:
- Financial institution
- River side

Word2Vec assigns single vector → averages meanings
Result: Vector captures mix of both meanings
Lose ability to distinguish senses!
```

**Problem 2: Local Context Only**
```
Word2Vec uses context window (e.g., 5 words)
Ignores global co-occurrence statistics
Missing information about overall word relationships
```

### 3.2 GloVe (Global Vectors for Word Representation)

**Definition**: Unsupervised learning algorithm combining local and global information

**Key Idea**: Words appearing together frequently should have similar vectors

**Two-Part Training**:
1. Count how often words co-occur in corpus (global)
2. Learn vectors to match these co-occurrence statistics

### 3.3 GloVe Training Process

**Step 1: Build Co-occurrence Matrix**

```
Corpus:
D1: "the cat sat on the mat"
D2: "the dog sat on the ground"

Co-occurrence Matrix (window size 2):
       the  cat  sat  on  dog  mat ground
the    0    2    1    1   1    1   0
cat    2    0    1    0   0    1   0
sat    1    1    0    2   0    0   1
on     1    0    2    0   0    1   1
dog    1    0    0    0   0    0   1
mat    1    1    0    1   0    0   0
ground 0    0    1    1   1    0   0

X[i][j] = how often words i and j appear together
```

**Step 2: Optimize Vectors**

```
For each word pair (i, j):
  target = log(X[i][j])  (log of co-occurrence count)
  prediction = w_i · w_j  (dot product of vectors)
  
Minimize: (w_i · w_j - log(X[i][j]))^2

After training:
- Words appearing together have similar vectors
- Word vectors encode co-occurrence patterns
```

**Step 3: Output Embeddings**

```
glove_vectors = {
    'the': [0.2, -0.5, 0.8, ...],
    'cat': [0.3, -0.4, 0.9, ...],  # Similar to 'dog'
    'dog': [0.29, -0.42, 0.91, ...], # Similar to 'cat'
    ...
}
```

### 3.4 GloVe vs Word2Vec

```
                Word2Vec         GloVe
──────────────────────────────────────────
Information     Local context    Global co-occurrence
Method          Neural network   Matrix factorization
Training        Predictive       Regression
Computation     Fast             Moderate
Embeddings      Good             Excellent
Context focus   Window-based     Corpus-wide

GloVe Advantages:
- Better semantic relationships
- Uses global information
- Typically better quality
- Can incorporate word frequency

Word2Vec Advantages:
- Faster training
- Works online (streaming)
- Flexible architecture
```

### 3.5 Word Senses

**Problem**: Same word, multiple meanings

**Word2Vec Limitation**:
```
bank → single vector
  Mixes: financial institution + river bank
  Result: Vector is average of senses
```

**Solutions**:

**1. Sense2Vec**: Learn separate vectors for each sense
```
bank#1 (financial) → [0.5, 0.3, 0.8, ...]
bank#2 (river) → [0.2, 0.7, 0.1, ...]

Disambiguate using context
```

**2. Contextual Embeddings (BERT, GPT)**:
```
Same word, different context → different vectors

"I went to the bank" 
  → bank_vector = [0.5, 0.3, 0.8, ...] (financial)

"I sat on the bank"
  → bank_vector = [0.2, 0.7, 0.1, ...] (river)

Solution: Generate embeddings based on context!
```

---

## 4. WORD WINDOW CLASSIFICATION

### 4.1 Definition

**Word Window Classification**: Classify a target word based on surrounding context words using neural network.

**Task**: Assign class label to word using its context

**Example Applications**:
- Named Entity Recognition: Is this word an entity?
- Sentiment: Is this word expressing sentiment?
- POS tagging: What is part of speech?

### 4.2 Architecture

```
INPUT LAYER:
┌─────────────────────────────────┐
│ Context words (one-hot encoded) │
│ window size = 2 (example)       │
│                                 │
│ Word i-2: [0, 0, 1, 0, ...]    │
│ Word i-1: [1, 0, 0, 0, ...]    │
│ Target i: [0, 1, 0, 0, ...]    │
│ Word i+1: [0, 0, 0, 1, ...]    │
│ Word i+2: [0, 0, 0, 0, 1...]   │
└──────────────────────────────────┘
              ↓ (convert to embeddings)
┌──────────────────────────────────┐
│ Embedding vectors (100-dim each) │
│ 5 words × 100 = 500-dim vector   │
│                                  │
│ [0.2, -0.5, ..., 0.1] × 5       │
│        ↓                          │
│ Concatenate: 500-dim vector      │
└──────────────────────────────────┘
              ↓
┌──────────────────────────────────┐
│ HIDDEN LAYER(s)                  │
│ Dense layer with ReLU            │
│ 500 → 100 neurons               │
│ Output: 100-dim representation   │
└──────────────────────────────────┘
              ↓
┌──────────────────────────────────┐
│ OUTPUT LAYER                     │
│ Dense layer with Softmax         │
│ 100 → number_of_classes          │
│                                  │
│ For 3-class problem: 100 → 3    │
│ Output: [0.7, 0.2, 0.1]         │
│ Class probabilities              │
└──────────────────────────────────┘
```

### 4.3 Training Process

```
Step 1: Get context words
  Sentence: "The quick brown fox"
  Target: "fox"
  Context window 2: ["brown", "fox", "?"]  (padding at edge)

Step 2: Convert to embeddings
  brown → [0.1, -0.2, 0.3, ...]
  fox → [0.2, -0.5, 0.8, ...]
  ... → [0, 0, 0, ...] (padding)
  
Step 3: Concatenate embeddings
  Vector = [0.1, -0.2, 0.3, ..., 0.2, -0.5, 0.8, ..., 0, 0, 0, ...]
           (500-dim)

Step 4: Forward pass through network
  Input 500-dim → Hidden 100-dim → Output 3-dim (class probs)

Step 5: Compare with true label (one-hot)
  Prediction: [0.7, 0.2, 0.1]
  True label: [1, 0, 0] (class 0)
  Loss = CrossEntropy([0.7, 0.2, 0.1], [1, 0, 0])

Step 6: Backpropagation
  Update all weights to minimize loss

Step 7: Repeat for next example
```

### 4.4 Example: NER Classification

```
Sentence: "John Smith works at Google in New York"
Task: Identify named entities

Word-by-word:
- John: context=[<START>, John, Smith, works, at]
  Classifier → PERSON (John is name)
  
- Smith: context=[John, Smith, works, at, Google]
  Classifier → PERSON (Smith is surname)
  
- works: context=[Smith, works, at, Google, in]
  Classifier → O (not entity)
  
- Google: context=[at, Google, in, New, York]
  Classifier → ORG (Google is company)
  
- New: context=[Google, New, York, <END>, <END>]
  Classifier → LOCATION (New is part of location)
  
- York: context=[New, York, <END>, <END>, <END>]
  Classifier → LOCATION
```

---

## 5. NEURAL NETWORK BASICS

### 5.1 Architecture

**Single Layer Neural Network**:
```
x₁ ─┐
x₂ ─┼─→ [w₁, w₂, w₃] ─→ y
x₃ ─┘

Forward pass: y = w₁x₁ + w₂x₂ + w₃x₃ + b (linear)
With activation: y = activation(w₁x₁ + w₂x₂ + w₃x₃ + b) (nonlinear)
```

**Multi-Layer Network**:
```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
   (5)            (100)           (50)             (3)
```

### 5.2 Activation Functions

**Linear**: y = x
- Problem: Network becomes linear, loses expressiveness

**ReLU** (Rectified Linear Unit): y = max(0, x)
```
   y
   |    /
   |   /
   |  /
   | /___________
   |_________________ x
   0
   
When x < 0: output 0
When x > 0: output x
Advantage: Computationally efficient, helps with vanishing gradient
```

**Sigmoid**: y = 1 / (1 + e^(-x))
```
   y
   |  ___
   | /
   |/____________
   |_________________ x
   0              1
   
Output range: 0 to 1
Problem: Vanishing gradient (derivative → 0 at extremes)
Use: For binary classification output
```

**Tanh**: y = (e^x - e^-x) / (e^x + e^-x)
```
   y
   |    ___
   |   /
   |  /
   | /
   |_________________ x
  -1              1
   
Output range: -1 to 1
Better than sigmoid
Use: Hidden layers
```

**Softmax**: For multiclass output
```
y_i = e^(z_i) / Σ_j e^(z_j)

Converts logits to probability distribution
Σ y_i = 1 (probabilities sum to 1)
```

### 5.3 Forward Propagation

**Simple Network**:
```
Input: x = [1, 2]

Layer 1:
  z₁ = W₁ · x + b₁
  z₁ = [[0.5, 0.3], [0.2, -0.1]] · [1, 2] + [0.1, -0.2]
  z₁ = [0.5×1 + 0.3×2 + 0.1, 0.2×1 + (-0.1)×2 + (-0.2)]
  z₁ = [1.2, -0.2]
  
  a₁ = ReLU(z₁) = [max(0, 1.2), max(0, -0.2)] = [1.2, 0]

Layer 2:
  z₂ = W₂ · a₁ + b₂
  z₂ = [[0.4, 0.6], [-0.3, 0.5]] · [1.2, 0] + [0.1, -0.1]
  z₂ = [0.4×1.2 + 0.6×0 + 0.1, -0.3×1.2 + 0.5×0 + (-0.1)]
  z₂ = [0.58, -0.46]
  
  a₂ = softmax(z₂) = [0.6, 0.4]  (probabilities for 2 classes)

Output: Prediction = class 0 (probability 0.6)
```

---

## 6. MATRIX CALCULUS & BACKPROPAGATION

### 6.1 Gradients & Derivatives

**Derivative**: Rate of change of function

```
f(x) = x²
f'(x) = 2x

At x = 3: f'(3) = 6 (steep)
At x = 0: f'(0) = 0 (flat)
```

**Partial Derivative**: Derivative with respect to one variable

```
f(x, y) = xy + x²
∂f/∂x = y + 2x  (treating y as constant)
∂f/∂y = x       (treating x as constant)
```

**Gradient**: Vector of all partial derivatives

```
∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z, ...]

Example: f(x, y) = x² + 2y
∇f = [2x, 2]

At point (3, 4): ∇f = [6, 2]
Direction of steepest increase
```

### 6.2 Chain Rule

**Key for Backpropagation**:

```
If: y = f(g(x))
Then: dy/dx = (df/dg) × (dg/dx)

Example:
y = sin(x²)
Let u = x², then y = sin(u)

dy/dx = (dy/du) × (du/dx) = cos(u) × 2x = cos(x²) × 2x
```

### 6.3 Backpropagation Algorithm

**Goal**: Compute gradient of loss with respect to all weights

**Process**: Apply chain rule backward through network

```
Forward Pass:
Input x → Layer 1 → a₁ → Layer 2 → a₂ → Output y

Loss = CrossEntropy(y, true_label)

Backward Pass:
Loss ← ∂Loss/∂y ← ∂Loss/∂a₂ ← ∂Loss/∂z₂ ← ∂Loss/∂W₂
              ↓
         ∂Loss/∂a₁ ← ∂Loss/∂z₁ ← ∂Loss/∂W₁
         
∂Loss/∂W₁ tells us how much to update W₁
```

### 6.4 Detailed Backpropagation Example

```
Network:
Input (2) → W₁, b₁ → Hidden (3) → W₂, b₂ → Output (2)

Forward:
z₁ = W₁x + b₁
a₁ = ReLU(z₁)
z₂ = W₂a₁ + b₂
ŷ = softmax(z₂)

Loss = -Σ log(ŷ)  (cross-entropy)

Backward:

Step 1: ∂Loss/∂ŷ (from cross-entropy)
  = [-1/ŷ for true class, others depend on formula]

Step 2: ∂Loss/∂z₂ (chain rule through softmax)
  = ŷ - true_label

Step 3: ∂Loss/∂W₂ (chain rule through matrix multiplication)
  = ∂Loss/∂z₂ · a₁ᵀ

Step 4: ∂Loss/∂b₂
  = ∂Loss/∂z₂

Step 5: ∂Loss/∂a₁ (backprop to previous layer)
  = W₂ᵀ · ∂Loss/∂z₂

Step 6: ∂Loss/∂z₁ (chain rule through ReLU)
  = ∂Loss/∂a₁ ⊙ ReLU'(z₁)
  
  where ReLU'(z₁) = 1 if z₁ > 0, else 0

Step 7: ∂Loss/∂W₁
  = ∂Loss/∂z₁ · xᵀ

Step 8: ∂Loss/∂b₁
  = ∂Loss/∂z₁
```

### 6.5 Weight Update (Gradient Descent)

```
After computing gradients:

W_new = W_old - learning_rate × ∂Loss/∂W
b_new = b_old - learning_rate × ∂Loss/∂b

Example:
W = [[0.5, 0.3], [0.2, -0.1]]
∂Loss/∂W = [[0.1, -0.05], [0.02, 0.01]]
learning_rate = 0.01

W_new = [[0.5, 0.3], [0.2, -0.1]] - 0.01 × [[0.1, -0.05], [0.02, 0.01]]
      = [[0.499, 0.3005], [0.1998, -0.1001]]

Weights move in direction opposite to gradient (downhill)
```

---

## 7. DEPENDENCY PARSING

### 7.1 Definition

**Dependency Parsing**: Analyzing sentence structure by identifying syntactic relationships (dependencies) between words.

**Output**: Dependency tree showing head-dependent relationships

### 7.2 Dependency vs Constituency Parsing

```
CONSTITUENCY PARSING:
Focus: Phrase structure
Tree structure by phrase groups

Sentence: "The cat sat on the mat"
     S
    /|\
   NP VP PP
   |  |  |
   Det N V ...

Hierarchical grouping of phrases


DEPENDENCY PARSING:
Focus: Word-to-word relationships
Direct relationships between words

Sentence: "The cat sat on the mat"
    det  |   case
    |    |  / |  /
   The  cat sat on the mat
              /
            root
            
"cat" is subject (dependent on "sat")
"sat" is verb (root)
"on" indicates location (dependent on "sat")
```

### 7.3 Dependency Relations

**Common Dependency Types**:

```
nsubj (nominal subject):
"The cat sat" → cat depends on sat (cat is subject)

det (determiner):
"The cat" → "The" depends on "cat" (article)

prep (preposition):
"in Paris" → "in" depends on head word, "Paris" depends on "in"

compound (compound noun):
"New York" → "New" depends on "York"

amod (adjectival modifier):
"beautiful day" → "beautiful" modifies "day"

advmod (adverbial modifier):
"very quickly" → "very" modifies "quickly"

obj (direct object):
"eat apples" → "apples" is object of "eat"

iobj (indirect object):
"give me apples" → "me" is indirect object

cop (copula):
"is good" → "is" links subject to predicate
```

### 7.4 Dependency Parsing Algorithms

**Transition-Based Parsing**:
```
Uses stack and buffer to build tree incrementally

Operations:
1. SHIFT: Move word from buffer to stack
2. LEFT-ARC: Create left-pointing arc (previous words depend on current)
3. RIGHT-ARC: Create right-pointing arc (current depends on previous)

Process:
Initial: Stack=[], Buffer=[The, cat, sat, ...]

Step 1: SHIFT → Stack=[The], Buffer=[cat, sat, ...]
Step 2: SHIFT → Stack=[The, cat], Buffer=[sat, ...]
Step 3: LEFT-ARC(det) → "The" depends on "cat"
        Stack=[cat], Buffer=[sat, ...]
Step 4: SHIFT → Stack=[cat, sat], Buffer=[...]
Step 5: RIGHT-ARC(nsubj) → "cat" depends on "sat"
        And so on...

Result: Complete dependency tree
```

**Graph-Based Parsing**:
```
Creates probability score for each possible arc
Finds maximum spanning tree (best overall structure)

More accurate but slower than transition-based
```

### 7.5 Example Parse

```
Sentence: "John ate the apple"

Dependency Tree:
        ate (root)
       /   \
      /     \
    John   apple
    (nsubj) (obj)
             |
            the
           (det)

Interpretation:
- "ate" is main verb (root)
- "John" is subject of "ate" (nsubj relation)
- "apple" is object of "ate" (obj relation)
- "the" is determiner of "apple" (det relation)

Typed Dependencies:
nsubj(ate, John)
obj(ate, apple)
det(apple, the)
```

---

## 8. NEGATIVE SAMPLING

### 8.1 Problem: Computational Complexity

**Standard Word2Vec Training**:

```
For each training example (target word, context word):
  1. Compute probability for EVERY word in vocabulary
  2. Calculate softmax over all vocabulary words
  
Vocabulary size: 1 million words
Computation per example: 1 million operations!

Training corpus: 1 billion examples
Total: 1 trillion operations → Too slow!
```

**Mathematical Problem**:
```
Standard objective (Skip-gram):
Maximize: log P(context | target)
        = log softmax(v_context · v_target)
        = log [ exp(v_context · v_target) / Σ_w exp(v_w · v_target) ]

The denominator sum requires computing all vocabulary!
```

### 8.2 Negative Sampling Solution

**Idea**: Instead of computing probabilities for all words, only update a few "negative" samples.

**Modified Objective**:
```
Instead of:
  Maximize: P(context | target) = softmax(v_context · v_target)

Do:
  Maximize: P(D=1 | target, context) × Product_negative P(D=0 | target, negative)
  
  Where:
  D=1: pair is real (target, context) co-occurs
  D=0: pair is fake (target, negative random word)
```

### 8.3 How Negative Sampling Works

```
Training Example: ("fox", "quick")  [positive pair]

Step 1: Keep positive sample
  Positive: ("fox", "quick") with label D=1

Step 2: Sample negative words
  Sample 5 random words that DON'T appear near "fox"
  Examples: ("fox", "car"), ("fox", "table"), ("fox", "book"), ("fox", "computer"), ("fox", "phone")
  
  Each with label D=0

Step 3: Binary Classification
  For each pair (word1, word2), predict: does it co-occur?
  
  ("fox", "quick"): should predict 1 (yes)
  ("fox", "car"): should predict 0 (no)
  ("fox", "table"): should predict 0 (no)
  etc.

Step 4: Update weights
  Update weights for: target word, positive context word, and 5 negative words
  Total updates: 6 words instead of 1 million!

Result: 99.4% reduction in computation!
```

### 8.4 Sampling Strategy

**How to select negative samples?**

**Unigram Distribution**:
```
P(w) ∝ count(w)^0.75

More frequent words more likely to be sampled as negative

Why 0.75?
- Prevents common words from dominating
- Allows rare words to appear in negative samples
- Empirically works well
```

**Example**:
```
Word frequencies:
the: 100,000 → P = 100,000^0.75 = 1000
cat: 1,000 → P = 1,000^0.75 = 31.6
dog: 1,000 → P = 1,000^0.75 = 31.6
car: 100 → P = 100^0.75 = 3.2

Normalize to probabilities:
the: 1000/(1000+31.6+31.6+3.2) ≈ 0.96
cat: 31.6/... ≈ 0.03
dog: 31.6/... ≈ 0.03
car: 3.2/... ≈ 0.003

Result: "the" very likely to be negative sample, "car" unlikely
```

### 8.5 Number of Negative Samples

```
Typical values: 5-15 negative samples per positive

Fewer samples (5):
✓ Faster training
✗ Less stable

More samples (15):
✓ More stable, better quality
✗ Slower

Trade-off: Usually 5-15 works well
```

---

# PART 2: PREDICTED EXAM QUESTIONS WITH ANSWERS

[Due to length, I'll create comprehensive answers for 4 key questions]

## QUESTION 1: WORD2VEC ARCHITECTURES (PROBABILITY: 90%)

### Question:
**"Explain Skip-Gram and CBOW models in Word2Vec. Compare their architectures, training objectives, and when to use each."**

### Complete Answer (6 marks):

**Skip-Gram Model (2 marks)**

Skip-Gram predicts context words given a target word.

**Architecture**:
```
Input: One-hot encoded target word (vocabulary_size)
Hidden: Embedding layer (embedding_size = 300)
Output: Softmax over vocabulary (vocabulary_size)
```

**Training Objective**:
- Maximize probability of observing context words near target
- For each target word, predict all context words in window

**Process**:
```
Sentence: "The quick brown fox jumps"
Target: "fox"
Context words (window=2): quick, brown, jumps, ...

Maximize: P(quick|fox) × P(brown|fox) × P(jumps|fox) × ...
```

**CBOW Model (2 marks)**

CBOW predicts target word from context words.

**Architecture**:
```
Input: One-hot encoded context words
Hidden: Average of embeddings
Output: Softmax for target word (vocabulary_size)
```

**Training Objective**:
- Maximize probability of target word given context
- Use average of context word embeddings

**Process**:
```
Sentence: "The quick brown fox jumps"
Context: {quick, brown} (window=2)
Target: fox

Maximize: P(fox | quick, brown)
```

**Comparison & When to Use (2 marks)**

| Aspect | Skip-Gram | CBOW |
|--------|-----------|------|
| Direction | Word → Context | Context → Word |
| Training | Slower | Faster |
| Rare Words | Better | Worse |
| Common Words | Okay | Better |
| Data Required | More | Less |
| Embeddings | Higher quality (large corpus) | Lower quality (large corpus) |
| Use When | Need quality, have large corpus | Need speed, limited data |

**Decision**:
- Skip-Gram: Recommended for most NLP tasks with sufficient data
- CBOW: Use when training time critical or data limited

---

## QUESTION 2: GLOVE & WORD SENSES (PROBABILITY: 85%)

### Question:
**"Explain GloVe algorithm and how it differs from Word2Vec. What are word senses and why are they important?"**

### Complete Answer (6 marks):

**GloVe Algorithm (2 marks)**

GloVe (Global Vectors) combines global statistical information with local context.

**Key Idea**: Words appearing together frequently should have similar vectors

**Process**:
1. **Build Co-occurrence Matrix**: Count how often each word pair appears together
2. **Optimize Vectors**: Learn vectors such that their dot product approximates log(co-occurrence count)
3. **Output**: Dense embeddings capturing co-occurrence patterns

**Mathematical Objective**:
```
Minimize: Σ_ij (w_i · w_j + b_i + b_j - log(X_ij))^2

Where:
w_i, w_j = word vectors
b_i, b_j = bias terms
X_ij = co-occurrence count
```

**Difference from Word2Vec (1.5 marks)**

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| Information | Local context only | Global + local |
| Method | Predictive neural network | Regression on matrix |
| Training | Forward-backward | Matrix optimization |
| Computation | Fast | Moderate |
| Embeddings | Good | Excellent |
| Window Dependent | Yes | Incorporates global stats |

**Word Senses (1.5 marks)**

**Problem**: Same word, multiple meanings

```
Example: "bank"
Meaning 1: Financial institution
Meaning 2: River side

Word2Vec Problem:
- Assigns single vector to "bank"
- Vector averages both meanings
- Cannot distinguish contexts

Solution: Multiple vectors per word sense
bank#1 ≈ "financial", "money", "account"
bank#2 ≈ "river", "water", "flowing"
```

**Contextual Approaches**:
- Sense2Vec: Learn separate embeddings per sense
- BERT/ELMo: Generate different embeddings based on context
- Better: "I went to the bank" vs "Sat on the bank" → different vectors

---

## QUESTION 3: NEURAL NETWORKS & BACKPROPAGATION (PROBABILITY: 85%)

### Question:
**"Explain neural network architecture, forward propagation, backpropagation, and matrix calculus concepts."**

### Complete Answer (6 marks):

**Neural Network Architecture (1 mark)**

Layers connected with weights:
```
Input Layer (5 neurons) → Hidden (100) → Hidden (50) → Output (3)

Forward information flow through weights and activations
```

**Forward Propagation (1.5 marks)**

Computing output given input:

```
Layer 1:
z₁ = W₁x + b₁
a₁ = activation(z₁)  [e.g., ReLU]

Layer 2:
z₂ = W₂a₁ + b₂
a₂ = activation(z₂)

Output:
z₃ = W₃a₂ + b₃
ŷ = softmax(z₃)  [for classification]

Loss = CrossEntropy(ŷ, true_label)
```

**Backpropagation (2 marks)**

Computing gradients for weight update:

```
Chain rule applied backward:
∂Loss/∂W₃ = ∂Loss/∂ŷ · ∂ŷ/∂z₃ · ∂z₃/∂W₃
∂Loss/∂W₂ = (∂Loss/∂z₃ · ∂z₃/∂a₂) · ∂a₂/∂z₂ · ∂z₂/∂W₂
∂Loss/∂W₁ = ((∂Loss/∂z₃ · ∂z₃/∂a₂ · ∂a₂/∂z₂) · ∂z₂/∂a₁) · ∂a₁/∂z₁ · ∂z₁/∂W₁

Result: ∂Loss/∂W for all weights
```

**Weight Update**:
```
W_new = W_old - learning_rate × ∂Loss/∂W
```

**Matrix Calculus (1.5 marks)**

**Gradient**: Vector of partial derivatives
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃]
```

**Jacobian**: Matrix of all first-order derivatives
```
For function f: ℝⁿ → ℝᵐ
J = ∂f/∂x (m × n matrix)
```

**Chain Rule** (key for backprop):
```
If y = f(g(x)):
∂y/∂x = ∂f/∂g · ∂g/∂x

For matrices:
∂Loss/∂W₁ = ∂Loss/∂a₂ · ∂a₂/∂W₁ (etc., applied through chain)
```

---

## QUESTION 4: DEPENDENCY PARSING & NEGATIVE SAMPLING (PROBABILITY: 80%)

### Question:
**"Explain dependency parsing and negative sampling. How does negative sampling solve Word2Vec's computational problem?"**

### Complete Answer (6 marks):

**Dependency Parsing (3 marks)**

**Definition**: Analyzing sentence structure by identifying word-to-word syntactic relationships

**Output**: Dependency tree where each word has a head

**Process**:
1. Identify head word (governor)
2. Identify dependent word
3. Label relationship type (nsubj, obj, det, etc.)

**Example**:
```
"The cat sat on the mat"

sat (root)
├── cat (nsubj)
│   └── The (det)
├── mat (obl)
    ├── on (case)
    └── the (det)

Dependencies:
nsubj(sat, cat): "cat" is subject
det(cat, the): "the" modifies "cat"
obl(sat, mat): "mat" is oblique argument
case(mat, on): "on" is preposition
det(mat, the): "the" modifies "mat"
```

**Algorithms**:
- **Transition-based**: Build incrementally using stack + buffer
- **Graph-based**: Find maximum spanning tree

**Applications**:
- Information extraction
- Machine translation
- Semantic analysis

**Negative Sampling (3 marks)**

**Problem**: Standard softmax requires computing probabilities for all words
```
Vocabulary: 1 million words
For each training example, update 1 million weights!
Corpus: 1 billion examples → 1 trillion operations (too slow!)
```

**Solution**: Only update weights for positive + few random "negative" words

**Process**:
```
Positive: ("fox", "quick") - they co-occur

Negatives (randomly sampled):
("fox", "car")
("fox", "table")
("fox", "book")
("fox", "computer")
("fox", "phone")

Train binary classifier:
("fox", "quick") → 1 (is real pair)
("fox", "car") → 0 (not real pair)
... etc.

Update only 6 words instead of 1 million!
99.4% computation reduction!
```

**Sampling Strategy**:
```
Use unigram distribution: P(w) ∝ count(w)^0.75

Frequent words more likely to be sampled
Rare words still have chance to appear
```

**Results**:
- 100-300x faster training
- Comparable or better embeddings
- Practical for large corpora

---

# PART 3: QUICK REVISION SHEET FOR UNIT 3

[Quick revision sheet content follows - comprehensive summary of all topics]

---

## UNIT 3 QUICK REVISION SHEET
## Word Embeddings & Neural Networks

---

### 1. WORD2VEC ARCHITECTURES

```
SKIP-GRAM:
Word → Context words
Input: Single word (one-hot)
Output: Multiple context words (softmax over vocabulary)
When: Large corpus, need quality, rare words important
Speed: Slower

CBOW:
Context words → Single word
Input: Multiple context words (average)
Output: Single target word (softmax)
When: Limited data, need speed, frequent words important
Speed: Faster

KEY PROPERTY: Vector arithmetic
king - man + woman ≈ queen
```

---

### 2. WORD EMBEDDINGS vs ONE-HOT

```
One-Hot:
[1,0,0,0,0] - 50,000 dims, 99.998% sparse, no meaning

Word2Vec Embedding:
[0.2, -0.5, 0.8, ..., 0.1] - 300 dims, dense, semantic

GloVe Embedding:
[0.3, -0.4, 0.9, ..., 0.2] - 300 dims, dense, global stats

Advantage: Dense, meaningful, efficient
```

---

### 3. GENSIM WORD2VEC

```python
from gensim.models import Word2Vec

# Train
model = Word2Vec(sentences, vector_size=100, window=2, sg=1)

# Operations
similarity = model.wv.similarity('king', 'queen')
similar = model.wv.most_similar('king', topn=5)
analogy = model.wv.most_similar(positive=['king', 'woman'], 
                                 negative=['man'])

# Save/Load
model.save('model.bin')
loaded = Word2Vec.load('model.bin')
```

---

### 4. GLOVE vs WORD2VEC

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| Info | Local context | Global + local |
| Method | Neural | Matrix factorization |
| Speed | Fast | Moderate |
| Quality | Good | Excellent |
| Word senses | Single vector | Better with context |

---

### 5. WORD WINDOW CLASSIFICATION

```
Task: Classify target word using context

Architecture:
Context words → Embeddings → Concatenate → Hidden → Output

Example: NER
"John works at Google"
Classify each word: PERSON / ORGANIZATION / O (other)

Process:
1. Get context window (e.g., 2 words each side)
2. Convert to embeddings
3. Concatenate embeddings
4. Pass through neural network
5. Output: class probabilities
```

---

### 6. NEURAL NETWORK BASICS

```
Forward Pass:
x → [W₁,b₁] → ReLU → [W₂,b₂] → Softmax → ŷ

Activation Functions:
- ReLU: max(0, x) - most popular
- Sigmoid: 1/(1+e^-x) - for binary output
- Tanh: between -1 and 1
- Softmax: multiclass probability distribution
```

---

### 7. BACKPROPAGATION

```
Backward Pass:
Compute ∂Loss/∂W for all weights

Chain Rule:
∂Loss/∂W₁ = ∂Loss/∂output · ∂output/∂hidden · ∂hidden/∂W₁

Weight Update:
W_new = W_old - learning_rate × ∂Loss/∂W

Key: Gradients flow backward through network
```

---

### 8. MATRIX CALCULUS

```
Gradient: ∇f = [∂f/∂x₁, ∂f/∂x₂, ...]

Jacobian: Matrix of all first derivatives

Chain Rule: ∂y/∂x = ∂f/∂g · ∂g/∂x

Softmax: y_i = e^z_i / Σ_j e^z_j

Cross-Entropy Loss: -Σ_i y_i × log(ŷ_i)
```

---

### 9. DEPENDENCY PARSING

```
Definition: Identify word-to-word syntactic relationships

Output: Dependency tree

Common Relations:
- nsubj: nominal subject
- obj: direct object
- det: determiner
- prep: preposition
- amod: adjectival modifier

Algorithms:
- Transition-based: Stack + buffer, incremental
- Graph-based: Maximum spanning tree
```

---

### 10. NEGATIVE SAMPLING

```
Problem: Softmax computes probabilities for all vocab
Solution: Binary classification on small set of words

Process:
1. Keep positive pair (target, context) with label 1
2. Sample 5-15 random words as negatives with label 0
3. Train binary classifier: does pair co-occur?
4. Update only ~6-16 words instead of million!

Efficiency:
- 100-300x speedup
- 99%+ fewer weight updates
- Better embeddings!

Sampling:
P(w) ∝ count(w)^0.75
(More frequent words more likely)
```

---

### 11. FORMULAS TO MEMORIZE

```
Forward: z = Wx + b, a = activation(z)

Chain Rule: ∂Loss/∂W = ∂Loss/∂a · ∂a/∂z · ∂z/∂W

Backprop: W_new = W_old - η × ∂Loss/∂W

Softmax: y_i = exp(z_i) / Σ_j exp(z_j)

Cross-Entropy: Loss = -Σ_i y_i log(ŷ_i)

Skip-gram: Maximize Σ log P(context | word)

Negative Sampling: Minimize [(word·context-1)² + Σ_neg (word·neg)²]
```

---

### 12. PREDICTED EXAM QUESTIONS

| Question | Probability | Topics |
|----------|-----------|--------|
| Skip-Gram vs CBOW | 90% | Architectures, when to use |
| GloVe vs Word2Vec | 85% | Algorithms, differences |
| Backpropagation | 85% | Forward/backward, chain rule |
| Dependency Parsing | 80% | Structure, algorithms |
| Negative Sampling | 80% | Problem, solution, efficiency |
| Neural Networks | 75% | Architecture, forward pass |
| Word Window Classification | 70% | Task, process, application |

---

### 13. COMMON MISTAKES

❌ Skip-gram for small datasets
✓ Use CBOW for small data

❌ Word2Vec captures multiple senses
✓ One vector per word; use context-aware for senses

❌ Backprop updates all weights equally
✓ Different layers get different gradients

❌ Negative sampling reduces quality
✓ Actually improves quality + speed

❌ Dependency parsing same as constituency
✓ Dependency: word relationships, Constituency: phrases

---

### 14. ONE-LINERS

- **Skip-gram**: Word predicts context (good for rare)
- **CBOW**: Context predicts word (good for frequent)
- **GloVe**: Global + local information (best quality)
- **Backprop**: Chain rule applied backward through network
- **Negative Sampling**: Binary classify real vs random pairs
- **Dependency Parsing**: Extract word-to-word relationships
- **Word Vectors**: Capture meaning in geometric space

---

**End of Unit 3 Quick Revision Sheet**

