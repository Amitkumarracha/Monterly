# UNIT 5: VISUAL DIAGRAMS & FLOWCHARTS

---

## 1. SEQ2SEQ ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                    SEQ2SEQ MODEL                             │
└─────────────────────────────────────────────────────────────┘

INPUT: "Hello world"
  │
  ├─→ Tokenization: ["Hello", "world"]
  │
  ├─→ Embedding layer: 
  │    hello → [0.2, -0.5, 0.8, ...]  (512-dim)
  │    world → [0.1, 0.3, -0.2, ...]  (512-dim)
  │
  └──────────────────────────┬──────────────────────────
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
   ┌─────────────┐                         ┌─────────────┐
   │  ENCODER    │                         │  DECODER    │
   │  (LSTM)     │                         │  (LSTM)     │
   └─────────────┘                         └─────────────┘
        │                                         │
   Input: embedded words                    Input: context vector
        │                                        + [START] token
   Process each word:                       │
   Time 1: hello → h₁                       Generate output words:
   Time 2: world → h₂                       Time 1: predict "Hola"
                                            Time 2: predict "mundo"
   Context Vector: C = h₂                   Time 3: predict [STOP]
   (compressed representation)
        │                                         │
        └────────────────────────────────────────┘
                      │
                      ▼
              Loss = CrossEntropy(
                predicted_words,
                actual_output_words
              )
                      │
                      ▼
              Backpropagation updates
              all weights
                      │
                      ▼
              OUTPUT: "Hola mundo"
```

---

## 2. ATTENTION MECHANISM - STEP BY STEP

```
ATTENTION COMPUTATION FLOWCHART:

┌──────────────────────────────────────────────────────────┐
│ ENCODER outputs hidden states: h₁, h₂, h₃, h₄           │
│ Example: ["Hello", "world", "how", "are"]               │
│                                                          │
│ h₁=[...], h₂=[...], h₃=[...], h₄=[...]                 │
└──────────────────────────────────────────────────────────┘
           │
           │ Decoder working at time step 2
           ▼
┌──────────────────────────────────────────────────────────┐
│ Decoder hidden state: s₂                                │
│ Task: Generate 2nd output word                          │
└──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │ STEP 1: Compute Alignment Scores        │
    │                                         │
    │ e₁ = alignment_score(s₂, h₁)            │
    │ e₂ = alignment_score(s₂, h₂)            │
    │ e₃ = alignment_score(s₂, h₃)            │
    │ e₄ = alignment_score(s₂, h₄)            │
    │                                         │
    │ Result: [0.2, 0.8, 0.3, 0.1]            │
    └─────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │ STEP 2: Softmax Normalization           │
    │                                         │
    │ α₁ = e^0.2 / (e^0.2 + e^0.8 + e^0.3 + e^0.1) ≈ 0.10
    │ α₂ = e^0.8 / (same denominator)        ≈ 0.75
    │ α₃ = e^0.3 / (same denominator)        ≈ 0.10
    │ α₄ = e^0.1 / (same denominator)        ≈ 0.05
    │                                         │
    │ Attention weights: [0.10, 0.75, 0.10, 0.05]
    │ Sum = 1.0 ✓                             │
    └─────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │ STEP 3: Compute Context Vector          │
    │                                         │
    │ c₂ = 0.10×h₁ + 0.75×h₂ + 0.10×h₃ + 0.05×h₄
    │                                         │
    │ c₂ ≈ 0.75×h₂ (HEAVILY WEIGHTED TO h₂)
    │                                         │
    │ Meaning: Focus 75% on word "world"      │
    └─────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────┐
    │ STEP 4: Generate Output                 │
    │                                         │
    │ ỹ₂ = combine(s₂, c₂)                    │
    │ prediction = softmax(W × ỹ₂)            │
    │                                         │
    │ Most likely word at position 2          │
    └─────────────────────────────────────────┘
```

---

## 3. ATTENTION WEIGHTS VISUALIZATION

```
ENGLISH → GERMAN TRANSLATION WITH ATTENTION WEIGHTS

English (Input):     The  dog  bit  the  cat
                      ↑    ↑    ↑    ↑    ↑
                    0.05  0.8  0.1  0.03 0.02
                      ↓    ↓    ↓    ↓    ↓
German (Output):    Der Hund biss die Katze
                     [predicting "Hund"]

Interpretation:
- When generating German word "Hund" (dog), model focuses 80% on English "dog"
- Correctly learned source-target alignment through attention
- 5% on "The", 10% on "bit", 3% on "the", 2% on "cat"

HEATMAP (darker = more attention):
          The  dog  bit  the  cat
Der       ███░░░░░░░░░░░░░░░░░░
Hund      ░░███████░░░░░░░░░░░░
biss      ░░░░░░░░░███░░░░░░░░░
die       ░░░░░░░░░░░░███░░░░░░
Katze     ░░░░░░░░░░░░░░░███░░░
```

---

## 4. TRANSFORMER ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

INPUT PROCESSING:
┌────────────────────────────┐
│ Tokenization & Embedding   │
│ "Hello world" → [h_1, h_2] │
└────────────────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Positional Encoding        │
│ Add position info to each  │
│ embedding (since no RNN)   │
└────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────┐
    │ ENCODER STACK (N = 6-12 layers)          │
    │                                          │
    │ ┌────────────────────────────────────┐   │
    │ │ ENCODER LAYER 1                    │   │
    │ │                                    │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Multi-Head Self-Attention   │   │   │
    │ │ │ (8 heads, unmasked)         │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ │           ↓                        │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Add & Normalize              │   │   │
    │ │ │ (residual connection + LN)   │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ │           ↓                        │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Feed-Forward Network        │   │   │
    │ │ │ 512 → 2048 → 512            │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ │           ↓                        │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Add & Normalize              │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ └────────────────────────────────────┘   │
    │           ↓ (repeat 12 times)           │
    │ ┌────────────────────────────────────┐   │
    │ │ ENCODER LAYER 2...12               │   │
    │ │ (same structure)                   │   │
    │ └────────────────────────────────────┘   │
    │           ↓                              │
    │    ENCODER OUTPUT                        │
    └──────────────────────────────────────────┘
           │
           │  Encoder representation passed to decoder
           │
           ▼
    ┌──────────────────────────────────────────┐
    │ DECODER STACK (N = 6-12 layers)          │
    │                                          │
    │ ┌────────────────────────────────────┐   │
    │ │ DECODER LAYER 1                    │   │
    │ │                                    │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Masked Self-Attention       │   │   │
    │ │ │ (CAN'T see future positions) │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ │           ↓                        │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Add & Normalize              │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ │           ↓                        │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Cross-Attention             │   │   │
    │ │ │ (Query: decoder             │   │   │
    │ │ │  Key/Value: encoder)        │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ │           ↓                        │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Add & Normalize              │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ │           ↓                        │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Feed-Forward Network        │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ │           ↓                        │   │
    │ │ ┌─────────────────────────────┐   │   │
    │ │ │ Add & Normalize              │   │   │
    │ │ └─────────────────────────────┘   │   │
    │ └────────────────────────────────────┘   │
    │           ↓ (repeat 12 times)           │
    │ ┌────────────────────────────────────┐   │
    │ │ DECODER LAYER 2...12               │   │
    │ │ (same structure)                   │   │
    │ │                                    │   │
    │ └────────────────────────────────────┘   │
    └──────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────┐
    │ Linear Layer (512 → vocabulary_size)     │
    │ Maps to all possible output words        │
    └──────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────┐
    │ Softmax (normalize to probabilities)     │
    │ Output: [p(word1), p(word2), ...]        │
    └──────────────────────────────────────────┘
           │
           ▼
    OUTPUT PREDICTION
```

---

## 5. MULTI-HEAD ATTENTION MECHANISM

```
MULTI-HEAD SELF-ATTENTION (8 heads):

Input: X (sequence of embeddings)
        │
        ├─→ Linear(Wq) → Q (query)
        ├─→ Linear(Wk) → K (key)
        └─→ Linear(Wv) → V (value)
        │
        ├─────────────────────────────────────────┐
        │                                         │
        ▼ (split into 8 heads)                    ▼ (split into 8 heads)
    Q1, Q2, ... Q8                          K1, K2, ... K8, V1, V2, ... V8
        │                                         │
        ▼ For each head h:                        ▼
    Attention_h = Softmax(Qh × Kh^T / √d) × Vh
        │
        ├─→ Head_1_output
        ├─→ Head_2_output
        ├─→ Head_3_output
        ...
        └─→ Head_8_output
            │
            ▼
        Concatenate all heads
        │
        ▼
        Linear(W^o) → Final Output

EXAMPLE - What Each Head Learns:
┌────────────────────────────────────────────────────┐
│ Input: "The dog bit the cat"                       │
├────────────────────────────────────────────────────┤
│ Head 1: Subject-Verb relationships                 │
│         dog → bit (high attention)                 │
│         cat → [unknown verb] (lower)               │
├────────────────────────────────────────────────────┤
│ Head 2: Object relationships                       │
│         bit → cat (high attention)                 │
│         dog → [not object] (lower)                 │
├────────────────────────────────────────────────────┤
│ Head 3: Syntax/Grammar                             │
│         The ↔ dog (article-noun)                   │
│         The ↔ cat (article-noun)                   │
├────────────────────────────────────────────────────┤
│ Head 4-8: Other patterns                           │
│         Long-range deps, tense, etc.               │
└────────────────────────────────────────────────────┘
```

---

## 6. BERT vs GPT COMPARISON

```
┌─────────────────────────────────────────────────────────┐
│             BERT vs GPT - Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  BERT:                        GPT:                      │
│  ┌─────────────┐              ┌─────────────┐          │
│  │   INPUT:    │              │   INPUT:    │          │
│  │ "I ____ you"│              │ "I love"    │          │
│  └─────────────┘              └─────────────┘          │
│       │                            │                    │
│  CAN READ BOTH                 CAN ONLY READ            │
│  DIRECTIONS:                   LEFT-TO-RIGHT:          │
│  "I" AND "you" CONTEXT        "I" AND "love" CONTEXT   │
│       │                            │                    │
│  ┌─────────────────┐          ┌─────────────────┐      │
│  │    ENCODER      │          │     DECODER     │      │
│  │  12-24 layers   │          │   12-48 layers  │      │
│  │  (bidirectional)│          │   (causal mask) │      │
│  └─────────────────┘          └─────────────────┘      │
│       │                            │                    │
│   PREDICT: "love"              PREDICT: "you"          │
│   (fill mask)                  (next word)             │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  Best For:                    Best For:                │
│  - Classification             - Generation             │
│  - Understanding              - Text completion        │
│  - NER                         - Summarization          │
│  - Question Answering          - Machine Translation   │
└─────────────────────────────────────────────────────────┘
```

---

## 7. TRAINING OBJECTIVES - BERT vs GPT

```
BERT TRAINING:

1. MASKED LANGUAGE MODELING (MLM)
   Original: "The quick brown fox jumps"
   Masked:   "The [MASK] brown [MASK] jumps"
   Predict:  "quick" and "fox"
   
   Model learns bidirectional context understanding

2. NEXT SENTENCE PREDICTION (NSP)
   Sentence A: "I went to the store"
   Sentence B: "I bought milk"
   Label: IsNext (model predicts yes)
   
   Model learns sentence-level relationships

═══════════════════════════════════════════════════════════

GPT TRAINING:

LANGUAGE MODELING (Autoregressive)
   Given: "The quick brown"
   Predict: "fox"
   
   Given: "The quick brown fox"
   Predict: "jumps"
   
   Given: "The quick brown fox jumps"
   Predict: "over"
   
   Model learns to generate text one word at a time
   
   Only sees previous context (left-to-right)
```

---

## 8. CONVERSATIONAL AI - SYSTEM FLOW

```
USER INPUT: "Book me a flight to New York tomorrow"
            │
            ▼
    ┌──────────────────────┐
    │  NATURAL LANGUAGE    │
    │  UNDERSTANDING (NLU) │
    └──────────────────────┘
            │
            ├─→ Intent Recognition
            │   Intent: BOOK_FLIGHT
            │
            ├─→ Entity Extraction
            │   Destination: New York
            │   Date: Tomorrow
            │
            └─→ Context Understanding
                Previous context: Looking for travel
                │
                ▼
        ┌──────────────────────┐
        │ DIALOGUE MANAGEMENT  │
        └──────────────────────┘
            │
            ├─→ State Tracking
            │   State: destination_confirmed, date_confirmed
            │
            ├─→ Action Selection
            │   Action: SEARCH_FLIGHTS
            │
            └─→ Response Selection
                Next: Show flight options
                │
                ▼
        ┌──────────────────────┐
        │ NATURAL LANGUAGE     │
        │ GENERATION (NLG)     │
        └──────────────────────┘
            │
            ├─→ Response Formulation
            │   Generate human-friendly message
            │
            ├─→ Ensure Naturalness
            │   Make it conversational, not robotic
            │
            └─→ Maintain Consistency
                Style and tone match previous responses
                │
                ▼
BOT OUTPUT: "Great! I found 3 flights to New York
            for tomorrow. Here are your options:
            1. Delta $300 (8:00 AM - 4:30 PM)
            2. United $350 (10:15 AM - 6:45 PM)
            3. Southwest $250 (12:30 PM - 9:00 PM)
            
            Which one interests you?"
```

---

## 9. SEQ2SEQ WITH ATTENTION - SIDE BY SIDE

```
WITHOUT ATTENTION:              WITH ATTENTION:

Input: [w1, w2, w3, w4]        Input: [w1, w2, w3, w4]
         │    │    │    │               │    │    │    │
         └────┴────┴────┘               └────┴────┴────┘
              │                              │
              ▼                              ▼
    ENCODER (RNN)                  ENCODER (RNN)
    Processes all words            Processes all words
              │                              │
              ▼                              ▼
    Context Vector C               Encoder Hidden States
    (bottleneck)                   [h1, h2, h3, h4]
              │                              │
              ▼                              ▼
    DECODER                         DECODER
    Generates output                Generates output
    using ONLY C                    using attention weights
         │                               │
         ├─→ At time t:             ├─→ Compute alignment:
         │   S_t = f(C, y_{t-1})    │   e_i = score(s_t, h_i)
         │                          │
         ├─→ Limited context        ├─→ Softmax weights:
         │   Can't access           │   α_i = softmax(e_i)
         │   individual encoder     │
         │   states                 ├─→ Context vector:
         │                          │   c_t = Σ α_i × h_i
         └─→ BLEU Score: 25         │
             (for 50-word sent)    └─→ BLEU Score: 35-40
                                       (same 50-word sent)

KEY DIFFERENCE:
- Without: Single pathway (bottleneck)
- With: Multiple parallel pathways (direct access)
```

---

## 10. POSITIONAL ENCODING - EXPLANATION

```
WHY POSITIONAL ENCODING IS NEEDED:

TRANSFORMERS:
- No recurrence (unlike RNNs)
- Process all words in parallel
- No inherent position information

PROBLEM:
- Without position info: "dog bit cat" = "cat bit dog"?
- How does model know "dog" is first word?

SOLUTION: Positional Encoding

FORMULA (for position pos, dimension i):
┌──────────────────────────────────────────────┐
│ PE(pos, 2i)   = sin(pos / 10000^(2i/d))      │
│ PE(pos, 2i+1) = cos(pos / 10000^(2i/d))      │
└──────────────────────────────────────────────┘

EXAMPLE (simplified):
Position 0: PE = [sin(0), cos(0), sin(0/10000), cos(0/10000), ...]
            PE = [0, 1, 0, 1, ...]

Position 1: PE = [sin(1), cos(1), sin(1/10000), cos(1/10000), ...]
            PE ≈ [0.84, 0.54, 0.0001, 1, ...]

Position 2: PE = [sin(2), cos(2), sin(2/10000), cos(2/10000), ...]
            PE ≈ [0.91, -0.42, 0.0002, 1, ...]

EACH POSITION HAS UNIQUE ENCODING!

APPLICATION:
Input Embedding + Positional Encoding = Final Input
[embeddings] + [position info] → [contextualized input]
```

---

## 11. ATTENTION COMPLEXITY COMPARISON

```
COMPUTATIONAL COMPLEXITY:

SEQ2SEQ (Basic):
├─ Encoder: O(m) time steps × O(d²) per step
└─ Total: O(m × d²) + O(t × d²) = O((m+t) × d²)
  ✓ Linear in sequence length
  ✓ Cannot parallelize (sequential)

SEQ2SEQ + ATTENTION:
├─ Encoder: O(m × d²)
├─ For each decoder step:
│  ├─ Compute m alignment scores: O(m×d)
│  ├─ Softmax: O(m)
│  └─ Weighted sum: O(m×d)
├─ Total decoder: O(t × m × d)
└─ Total: O((m+t) × d²) + O(t × m × d) ≈ O(m × t × d)
  ⚠ Quadratic in sequence length
  ✓ Can parallelize across t steps

TRANSFORMERS:
├─ Self-attention: O(n² × d) for n tokens
├─ FFN: O(n × d²)
└─ Total: O(n²d + nd²) but highly parallelizable
  ⚠ Quadratic complexity
  ✓ Much faster wall-clock time due to parallelization

MEMORY COMPARISON (for 50-word sentence):
┌──────────────────────────┐
│ Seq2Seq basic: 2,500 ops │
│ Seq2Seq+Attn: 2,500 ops  │
│ Transformer:  62,500 ops │
└──────────────────────────┘
But Transformers can use GPUs effectively!
```

---

## 12. QUICK DECISION TREE - Which Model to Use?

```
                      TASK?
                      │
          ┌───────────┼───────────┐
          │           │           │
      GENERATION   UNDERSTANDING  DIALOGUE
          │           │           │
          ▼           ▼           ▼
        GPT         BERT       LaMDA/
        ├─ Text    ├─ Text    ├─ Conversational
        │  Completion Classification Understanding
        ├─ Story    ├─ NER    ├─ Dialogue
        │  Generation├─ QA     │  Management
        ├─ Machine  ├─ NLI    ├─ Response
        │  Translation ├─ Similarity  Generation
        └─ Summarization └─ Semantic Search


DETAILED SELECTION:

Task                          → Model
────────────────────────────────────────────
Machine Translation           → Seq2Seq + Attention / Transformer
Text Classification           → BERT / GPT
Named Entity Recognition      → BERT (fine-tune)
Question Answering            → BERT / GPT
Sentiment Analysis            → BERT
Text Generation               → GPT
Summarization                 → Seq2Seq / Transformer
Chatbot / Conversational AI   → LaMDA / GPT / BERT
Semantic Similarity           → BERT embeddings
Information Extraction        → BERT (fine-tune)
Text Completion               → GPT
Code Generation               → GPT
Image Captioning              → Seq2Seq / Transformer (vision+language)
```

---

**End of Visual Diagrams & Flowcharts**
