# UNIT 3: VISUAL DIAGRAMS & FLOWCHARTS
## Word Embeddings & Neural Networks Illustrated

---

## 1. SKIP-GRAM vs CBOW ARCHITECTURE

```
SKIP-GRAM MODEL:
────────────────

Target word: "fox"
Context window: 2

Input Layer (One-Hot):
───────────────────
[0,0,0,0,1,0,0,...] ← "fox" (index 4)
    (vocabulary_size)

         ↓ × Weight Matrix W (vocab × embedding_dim)

Hidden Layer (Embedding):
───────────────────────
[0.2, -0.5, 0.8, 0.1, ..., 0.3]
           ↓
    (embedding_size = 300)

         ↓ × Weight Matrix W' (embedding_dim × vocab)

Output Layer (Softmax):
──────────────────────
[0.7, 0.2, 0.05, 0.01, 0.04, ...]
Probabilities for each word as context

Training:
- Positive contexts: quick (0.7), brown (0.6), jumps (0.8)
- Update W to maximize these probabilities
- Minimize others


CBOW MODEL (OPPOSITE):
──────────────────────

Context words: ["quick", "brown", "jumps"]

Input Layer (One-Hot × 3):
────────────────────────
[0,1,0,0,0] quick
[0,0,1,0,0] brown
[0,0,0,0,1] jumps
          ↓ Convert to embeddings and average

Hidden Layer (Averaged Embedding):
──────────────────────────────────
Average([quick_emb, brown_emb, jumps_emb])
= [0.25, -0.4, 0.8, ...]

         ↓ × Weight Matrix

Output Layer (Softmax):
──────────────────────
[0.95, 0.02, 0.01, 0.01, 0.01]
High probability for "fox" (target)!

Training:
- Input: context words
- Output: maximize probability of "fox"
- Lower: probability of other words
```

---

## 2. WORD EMBEDDING PROPERTIES

```
SEMANTIC RELATIONSHIPS IN VECTOR SPACE:

Dimension 1 (Gender):
  Male ←────────────→ Female
   ↑                    ↑
  man                 woman
  king                queen
  prince             princess
   ↓                    ↓
Dimension 2 (Royalty)

Visualization:
───────────────

            queen
            /  \
        prince  princess
           |
        king
        /   \
     man    woman

Direction 1: Male ← → Female
Direction 2: Royalty (higher = royal)

Vector Arithmetic:
king - man + woman ≈ queen

Why?
king - man removes "male" direction
+ woman adds "female" direction
Result: royal female = queen!


CONTEXTUAL SIMILARITY:

Words appearing together:
{king, prince, throne, royal} 
{queen, princess, crown, royal}

Their vectors point in similar directions
Cosine similarity: high!

Unrelated words:
{dog, cat, animal}
{car, vehicle, engine}

Vectors point in different directions
Cosine similarity: low!
```

---

## 3. GLOVE ALGORITHM VISUALIZATION

```
STEP 1: BUILD CO-OCCURRENCE MATRIX
──────────────────────────────────

Corpus: "the cat sat on the mat"
        "the dog sat on the ground"

Window size: 2

Co-occurrence counts:
       the  cat  sat  on   mat  dog ground
the    0    2    1    1    1    1   0
cat    2    0    1    0    1    0   0
sat    1    1    0    2    0    0   1
on     1    0    2    0    1    0   1
mat    1    1    0    1    0    0   0
dog    1    0    0    0    0    0   1
ground 0    0    1    1    0    1   0

X[i][j] = how often words i and j co-occur


STEP 2: OPTIMIZE VECTORS
─────────────────────────

For each pair (i, j):
Target: log(X[i][j])
Prediction: w_i · w_j

Loss: Minimize (w_i · w_j - log(X[i][j]))^2

Example:
Pair (cat, sat):
  X[cat][sat] = 1
  log(1) = 0
  Optimize: w_cat · w_sat ≈ 0

Pair (the, cat):
  X[the][cat] = 2
  log(2) ≈ 0.69
  Optimize: w_the · w_cat ≈ 0.69


STEP 3: RESULT
──────────────

Trained vectors:
cat ≈ [0.3, -0.4, 0.9, ...]
dog ≈ [0.29, -0.42, 0.88, ...] (similar to cat!)
the ≈ [0.2, -0.5, 0.8, ...]

Words co-occurring → similar vectors
Words not co-occurring → different vectors
```

---

## 4. WORD SENSES PROBLEM

```
SINGLE VECTOR PER WORD (PROBLEM):

Word: "bank"
Meaning 1: Financial institution
Meaning 2: River side

word2vec assigns single vector:
bank = [0.5, 0.3, 0.8, ..., 0.2]
  ↓ Mixes both meanings!

Contexts:
"I went to the bank" (financial)
bank vector ≠ context
"Sat on the bank" (river)
bank vector ≠ context

Both contexts use SAME vector!
Cannot distinguish!


MULTIPLE SENSES (SOLUTION 1):

bank#1 = [0.6, 0.2, 0.8, ...] (financial)
         ↑ Similar to: money, account, business
         
bank#2 = [0.2, 0.7, 0.1, ...] (river)
         ↑ Similar to: water, flow, side


CONTEXTUAL EMBEDDINGS (SOLUTION 2):

Same word → different embeddings based on context

"I went to the bank"
  → bank_vec = [0.6, 0.2, 0.8, ...]  (financial sense)

"Sat on the bank"
  → bank_vec = [0.2, 0.7, 0.1, ...]  (river sense)

BERT/GPT generate embeddings on-the-fly
Based on surrounding context words
Better contextual understanding
```

---

## 5. WORD WINDOW CLASSIFICATION ARCHITECTURE

```
TASK: Classify target word using context

Sentence: "John loves apples"
Target: "loves"
Task: POS tagging - What part of speech?

INPUT LAYER:
────────────
Context window (2 words each side):

Word i-2: "John"    → one-hot: [1,0,0,0,0]
Word i-1: "loves"   → one-hot: [0,1,0,0,0]
Target i: "loves"   → one-hot: [0,1,0,0,0]
Word i+1: "apples"  → one-hot: [0,0,1,0,0]
Word i+2: "(END)"   → one-hot: [0,0,0,0,1]

         ↓ Convert to embeddings using embedding matrix E

EMBEDDING CONVERSION:
──────────────────────

[1,0,0,0,0] × E = [0.2, -0.5, 0.8, ...]  (100-dim)
[0,1,0,0,0] × E = [0.1, 0.3, -0.2, ...]
[0,0,1,0,0] × E = [0.4, -0.3, 0.6, ...]
[0,0,0,1,0] × E = [0.0, 0.5, -0.1, ...]
[0,0,0,0,1] × E = [0, 0, 0, ...]  (padding)

All embeddings: 5 × 100-dim

         ↓ Concatenate into single vector

CONCATENATION:
───────────────
[0.2, -0.5, 0.8, ..., 0.1, 0.3, -0.2, ..., 0.4, -0.3, 0.6, ...]
                    (500-dimensional vector)

         ↓ Pass through hidden layers

HIDDEN LAYER 1:
───────────────
z₁ = W₁ × [500-dim] + b₁
   = 500 × 100 matrix × 500-dim = 100-dim

a₁ = ReLU(z₁)
   = [max(0, z₁[0]), max(0, z₁[1]), ..., max(0, z₁[99])]

         ↓

HIDDEN LAYER 2:
───────────────
z₂ = W₂ × a₁ + b₂
   = 100 × 50 matrix × 100-dim = 50-dim

a₂ = ReLU(z₂)

         ↓

OUTPUT LAYER:
──────────────
z₃ = W₃ × a₂ + b₃
   = 50 × 5 matrix × 50-dim = 5-dim
   (5 POS classes)

ŷ = softmax(z₃)
  = [0.05, 0.8, 0.1, 0.03, 0.02]
    (VERB likely: 0.8)

TARGET: [0, 1, 0, 0, 0] (VERB)

Loss = CrossEntropy([0.05, 0.8, 0.1, 0.03, 0.02], [0, 1, 0, 0, 0])
     ≈ low loss (good prediction!)
```

---

## 6. FORWARD PROPAGATION DETAILED EXAMPLE

```
NETWORK STRUCTURE:
Input (2) → W₁, b₁ → Hidden (3) → W₂, b₂ → Output (2)

INITIALIZATION:
───────────────
W₁ = [[0.5, 0.3, 0.2],
      [0.2, -0.1, 0.4]]

b₁ = [0.1, -0.2, 0.1]

W₂ = [[0.4, 0.6, -0.2],
      [-0.3, 0.5, 0.3]]

b₂ = [0.1, -0.1]

Input: x = [1, 2]


FORWARD PASS:
──────────────

Step 1: Compute z₁
z₁ = W₁ · x + b₁
   = [[0.5, 0.3, 0.2],  · [1]  + [0.1  ]
      [0.2, -0.1, 0.4]]   [2]    [-0.2 ]
                                  [0.1  ]

z₁[0] = 0.5×1 + 0.3×2 + 0.1 = 0.5 + 0.6 + 0.1 = 1.2
z₁[1] = 0.2×1 + (-0.1)×2 + (-0.2) = 0.2 - 0.2 - 0.2 = -0.2
z₁[2] = 0.2×1 + 0.4×2 + 0.1 = 0.2 + 0.8 + 0.1 = 1.1

z₁ = [1.2, -0.2, 1.1]

Step 2: Apply ReLU
a₁ = ReLU(z₁) = [max(0, 1.2), max(0, -0.2), max(0, 1.1)]
   = [1.2, 0, 1.1]

Step 3: Compute z₂
z₂ = W₂ · a₁ + b₂
   = [[0.4, 0.6, -0.2],  · [1.2]  + [0.1 ]
      [-0.3, 0.5, 0.3]]    [0  ]    [-0.1]
                            [1.1]

z₂[0] = 0.4×1.2 + 0.6×0 + (-0.2)×1.1 + 0.1 = 0.48 + 0 - 0.22 + 0.1 = 0.36
z₂[1] = (-0.3)×1.2 + 0.5×0 + 0.3×1.1 + (-0.1) = -0.36 + 0 + 0.33 - 0.1 = -0.13

z₂ = [0.36, -0.13]

Step 4: Apply Softmax
e^z₂ = [e^0.36, e^-0.13] = [1.434, 0.878]
sum = 1.434 + 0.878 = 2.312

ŷ = softmax(z₂) = [1.434/2.312, 0.878/2.312]
                 = [0.620, 0.380]

PREDICTION: Class 0 with probability 62.0%
```

---

## 7. BACKPROPAGATION FLOW

```
BACKWARD PASS:

From Loss → Output
Loss = -log(ŷ[true_class])
     = -log(0.620)  (if true class is 0)
     ≈ 0.479

∂Loss/∂ŷ = [-1/ŷ[0], 0, 0, ...] at true class position


Through Softmax → z₂
∂Loss/∂z₂ = ŷ - one_hot_true
          = [0.620, 0.380] - [1, 0]
          = [-0.380, 0.380]


Through W₂ → a₁
∂Loss/∂W₂ = ∂Loss/∂z₂ · a₁ᵀ

∂Loss/∂W₂ = [-0.380] · [1.2, 0, 1.1]
            [0.380]

∂Loss/∂W₂ = [[-0.456, 0, -0.418],
             [0.456, 0, 0.418]]

(These gradients update W₂)

∂Loss/∂a₁ = W₂ᵀ · ∂Loss/∂z₂
          = [[0.4, -0.3],     · [-0.380]
             [0.6, 0.5],        [0.380]
             [-0.2, 0.3]]

∂Loss/∂a₁ = [-0.252, 0.038, -0.191]


Through ReLU → z₁
∂Loss/∂z₁ = ∂Loss/∂a₁ ⊙ ReLU'(z₁)

ReLU'(z₁) = [1 if z₁ > 0 else 0] = [1, 0, 1]

∂Loss/∂z₁ = [-0.252, 0.038, -0.191] ⊙ [1, 0, 1]
          = [-0.252, 0, -0.191]


Through W₁ → x
∂Loss/∂W₁ = ∂Loss/∂z₁ · xᵀ

∂Loss/∂W₁ = [[-0.252],      · [1, 2]
             [0    ],
             [-0.191]]

∂Loss/∂W₁ = [[-0.252, -0.504],
             [0, 0],
             [-0.191, -0.382]]

(These gradients update W₁)

WEIGHT UPDATE:
──────────────
W₁_new = W₁_old - learning_rate × ∂Loss/∂W₁
W₂_new = W₂_old - learning_rate × ∂Loss/∂W₂
(similar for biases)
```

---

## 8. DEPENDENCY PARSING TREE

```
SENTENCE: "The quick brown fox jumps over the lazy dog"

PARSE TREE:
───────────

                jumps
               /  |  \
              /   |   \
            fox  over  dog
           / |          / |
          /  |         /  |
       quick brown   the lazy
        |    |
        |    |
       the   adj

CONSTITUENCY VIEW:
──────────────────
            S
          / | \
        NP  VP PP
       /|   |  /|
      DET ADJ N V P NP
      The quick brown jumps over the dog


DEPENDENCY VIEW:
────────────────

ROOT: jumps

Children of jumps:
- fox (nsubj - nominal subject)
- dog (obl - oblique: over the dog)

Children of fox:
- quick (amod - adjectival modifier)
- brown (amod)

Children of dog:
- lazy (amod)
- the (det)

Full Dependencies:
nsubj(jumps, fox)
amod(fox, quick)
amod(fox, brown)
obl(jumps, dog)
case(dog, over)
det(dog, the)
amod(dog, lazy)

TRANSITION-BASED PARSING:
─────────────────────────

Initial: Stack=[], Buffer=[the, quick, brown, fox, jumps, ...]

Step 1: SHIFT
       Stack=[the], Buffer=[quick, brown, fox, jumps, ...]

Step 2: SHIFT
       Stack=[the, quick], Buffer=[brown, fox, jumps, ...]

Step 3: LEFT-ARC(det)
       Quick word "the" depends on next word
       Stack=[quick], Buffer=[brown, fox, jumps, ...]
       (the points to quick)

Continue until full tree built...
```

---

## 9. NEGATIVE SAMPLING VS SOFTMAX

```
STANDARD SOFTMAX (INEFFICIENT):
────────────────────────────────

Target word: "fox"
Vocabulary size: 1,000,000 words

P(quick | fox) = exp(v_quick · v_fox) / Σ_i exp(v_i · v_fox)

Must compute dot product with ALL 1 million words!

Operation: 1 million multiplications per example
Dataset: 1 billion examples
Total: 1 trillion operations → SLOW!


NEGATIVE SAMPLING (EFFICIENT):
───────────────────────────────

Target: fox
Positive: quick (target co-occurs with this)

Negatives (random sample):
car, tree, book, computer, phone

Binary Classification:
("fox", "quick") → 1 (does co-occur)
("fox", "car") → 0 (doesn't co-occur)
("fox", "tree") → 0
("fox", "book") → 0
("fox", "computer") → 0
("fox", "phone") → 0

Train binary classifier:
Maximize: σ(v_quick · v_fox)
Minimize: σ(v_car · v_fox)
          σ(v_tree · v_fox)
          ... (5-15 negatives total)

Operations: ~6-16 multiplications per example
Speedup: 1,000,000 / 10 = 100,000x faster!

With batching: ~100-300x speedup in practice


SAMPLING DISTRIBUTION:
──────────────────────

How to select negatives?

Unigram distribution raised to 0.75 power:
P(w) ∝ count(w)^0.75

Example counts:
the: 100,000 → 100,000^0.75 = 1000
cat: 1,000 → 1,000^0.75 ≈ 32
dog: 1,000 → 1,000^0.75 ≈ 32
car: 100 → 100^0.75 ≈ 3.2

Probabilities (normalized):
the: 96%  (very likely to sample)
cat: 1.5%
dog: 1.5%
car: 0.15% (unlikely)

Result: Common words sampled more, rare words still included
```

---

## 10. MATRIX CALCULUS CHAIN RULE

```
FUNCTION COMPOSITION:

Layer 1: z₁ = W₁x + b₁
Layer 2: a₁ = ReLU(z₁)
Layer 3: z₂ = W₂a₁ + b₂
Layer 4: ŷ = softmax(z₂)
Layer 5: Loss = CrossEntropy(ŷ, true_y)

FORWARD PASS:
─────────────
x → [W₁] → z₁ → [ReLU] → a₁ → [W₂] → z₂ → [softmax] → ŷ → [Loss]

CHAIN RULE (Backward):
──────────────────────

∂Loss/∂W₂ = ∂Loss/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂W₂

Let's compute each part:

∂Loss/∂ŷ = derivatives from CrossEntropy
∂ŷ/∂z₂ = Jacobian of softmax
∂z₂/∂W₂ = a₁ᵀ

Multiply together (matrix chain rule):
∂Loss/∂W₂ = (scalar) × (matrix) × (matrix) = (matrix of same shape as W₂)

Similarly:
∂Loss/∂W₁ = ∂Loss/∂ŷ · ∂ŷ/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
             └─────────────────────────────────────────────────┘
                    Propagating backward through all layers
```

---

**End of Unit 3 Visual Diagrams**

