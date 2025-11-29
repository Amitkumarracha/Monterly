# UNIT 5: QUICK REVISION SHEET
## Machine Translation, Transformers & Modern NLP

---

## 1. SEQ2SEQ MODEL - At a Glance

### What is it?
Neural network that converts input sequence to output sequence of different length.

### Two Main Parts:
- **Encoder**: LSTM/GRU reads input word-by-word → produces context vector
- **Decoder**: LSTM/GRU reads context vector → generates output word-by-word

### Process Flow:
```
Input: "hello world" 
  ↓
Encoder: Process each word → Final hidden state (context)
  ↓
Context Vector: Compressed representation [0.2, -0.5, 0.8, ...]
  ↓
Decoder: Use context + [START] token → predict "hola" → predict "mundo" → [STOP]
  ↓
Output: "hola mundo"
```

### Main Problem:
Single context vector = bottleneck. Long sequences lose information.

### Evaluation:
**BLEU Score**: Measures translation quality (0-1, higher is better)

---

## 2. ATTENTION MECHANISM - Simplified

### The Problem It Solves:
Can't fit all information in one context vector for long sentences.

### The Solution:
Let decoder look at ALL encoder hidden states, not just final one.

### How It Works (3 Steps):
```
Step 1: Compute Alignment
eᵢⱼ = score(decoder_state_j, encoder_state_i)

Step 2: Normalize (Softmax)
αᵢⱼ = e^eᵢⱼ / Σ(e^eₖⱼ)
Result: weights between 0 and 1

Step 3: Weighted Sum
context_j = Σ(αᵢⱼ × encoder_state_i)
```

### Why It's Better:
- No information loss
- Can visualize which inputs matter (interpretability)
- Works for long sentences (BLEU doesn't drop)

### Complexity:
- Without attention: O(m + t) 
- With attention: O(m × t) ← slower but worth it!

---

## 3. SINGLE-HEAD vs MULTI-HEAD ATTENTION

### Single-Head:
- ONE set of Query, Key, Value matrices
- ONE attention mechanism
- Limited to one "view" of the sequence

### Multi-Head (8 heads typical):
- 8 different sets of Q, K, V matrices
- 8 parallel attention mechanisms
- Each head learns different aspects

### Why Multi-Head is Better:
```
Head 1: Focus on subject-verb relationships
Head 2: Focus on object relationships
Head 3: Focus on syntax
Head 4: Focus on long-range dependencies
Heads 5-8: Other linguistic patterns

Together = Rich, comprehensive understanding
```

### Formula:
```
MultiHead = Concat(head1, head2, ..., head8) × W^out
```

---

## 4. TRANSFORMER ARCHITECTURE - Core Concepts

### Why Transformers?
- RNNs process sequentially (slow)
- Transformers process ALL positions in parallel (fast)
- No vanishing gradient problem

### Main Components:
1. **Embedding + Positional Encoding** (add position info)
2. **Encoder Stack** (6-12 layers)
3. **Decoder Stack** (6-12 layers)

### Each Encoder Layer:
```
Multi-Head Self-Attention
         ↓
  Add & Normalize
         ↓
  Feed-Forward Network (dense layers)
         ↓
  Add & Normalize
```

### Each Decoder Layer:
```
Masked Self-Attention (can't see future)
         ↓
  Add & Normalize
         ↓
Cross-Attention (decoder attends to encoder)
         ↓
  Add & Normalize
         ↓
Feed-Forward Network
         ↓
Add & Normalize
```

### Positional Encoding (tells position info):
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
Added to embeddings before processing.

### Key Advantage:
All positions processed in parallel → massive speedup

---

## 5. BERT - Bidirectional Encoder Representations from Transformers

### What is it?
Pre-trained encoder-only transformer that reads text BOTH directions.

### Architecture:
- **BERT-Base**: 12 layers, 768 hidden units, 12 heads
- **BERT-Large**: 24 layers, 1024 hidden units, 16 heads

### Input Representation (3 components):
```
Final = Token_Embedding + Segment_Embedding + Position_Embedding
```

### Training Methods:

**1. Masked Language Modeling (MLM)**
- Mask 15% of words randomly
- 80% → [MASK], 10% → random word, 10% → unchanged
- Predict original words
- Purpose: Learn context-dependent representations

Example:
```
Original: "The brown dog jumped"
Masked: "The [MASK] dog [MASK]"
Predict: "brown" and "jumped"
```

**2. Next Sentence Prediction (NSP)**
- Given two sentences, predict if second follows first
- 50% IsNext, 50% NotNext
- Purpose: Learn sentence relationships

### Applications:
| Task | Method |
|------|--------|
| Text Classification | [CLS] token output → classifier |
| NER | Each token output → classifier |
| Question Answering | Predict answer span positions |
| Semantic Similarity | Cosine similarity of [CLS] vectors |

---

## 6. GPT - Generative Pre-trained Transformer

### What is it?
Decoder-only transformer for autoregressive text generation.

### Key Difference from BERT:
- **BERT**: Bidirectional, encoder, for understanding
- **GPT**: Unidirectional (left-to-right), decoder, for generation

### How GPT Works:
```
Input: "The cat sat on the"
↓
Predict: "mat"
↓
Input: "The cat sat on the mat"
↓
Predict: "and"
↓
Continue until [END] token
```

### Autoregressive Principle:
```
P(x₁,x₂,...,xₙ) = Π P(xᵢ | x₁,...,xᵢ₋₁)
(probability = product of conditional probabilities)
```

### Versions:
- GPT-2: 1.5 billion parameters
- GPT-3: 175 billion parameters
- GPT-4: Even larger, multimodal

### Applications:
- Text generation
- Translation
- Summarization
- Code generation
- Question answering

---

## 7. LAMDA - Language Model for Dialogue Applications

### What is it?
Google's conversational AI model (137B parameters).

### Key Feature:
Specialized fine-tuning for dialogue, not just general language.

### Two-Stage Training:
1. **Pre-training**: Learn general language (decoder-only transformer)
2. **Fine-tuning**: Optimize for dialogue quality
   - Score responses: sensibleness, specificity, interestingness
   - Learn to maximize scores

### Groundedness:
- Consults search engines for factual claims
- Reduces hallucination problem
- Improves accuracy

### vs GPT-3:
- LaMDA: Dialogue specialist
- GPT-3: General purpose text generator

---

## 8. ATTENTION MECHANISM - COMPARISON TABLE

| Type | Formula | Where Used |
|------|---------|-----------|
| **Additive (Bahdanau)** | v^T tanh(W_s s + W_h h) | Early seq2seq |
| **Multiplicative (Luong)** | s^T h | Luong attention |
| **Scaled Dot-Product** | (Q K^T / √d) V | Transformers |
| **Self-Attention** | (Q K^T / √d) V, same sequence | Vision, NLP |
| **Cross-Attention** | Q from decoder, K,V from encoder | Transformer decoder |

---

## 9. CONVERSATIONAL AI - Key Components

### 3 Main Stages:

**1. Natural Language Understanding (NLU)**
- Intent recognition: What does user want?
- Entity extraction: What entities mentioned?
- Context tracking: What was discussed before?

**2. Dialogue Management**
- State tracking: What's the conversation state?
- Response selection: What should bot say?
- Context maintenance: Remember what happened

**3. Natural Language Generation (NLG)**
- Response formulation: Create human-like response
- Ensure naturalness: Don't sound robotic
- Maintain consistency: Coherent style

### Example Flow:
```
User: "Book me a flight to New York"
NLU: Intent=BOOK_FLIGHT, Destination=NewYork
Dialogue Mgmt: Ask for date
NLG: "When would you like to travel?"

User: "Tomorrow"
NLU: Intent=CONFIRM, Date=Tomorrow
Dialogue Mgmt: Check availability
NLG: "I found 3 flights for $300-500. Which one?"
```

### Types:
- **Retrieval-based**: Select from pre-defined responses
- **Generative**: Generate new responses with neural nets
- **Hybrid**: Combine both approaches

---

## 10. QUICK FORMULAS REFERENCE

| Formula | Purpose |
|---------|---------|
| `eᵢⱼ = s^T h` | Attention score (dot product) |
| `αᵢⱼ = e^eᵢⱼ / Σ e^eₖⱼ` | Attention weights (softmax) |
| `cⱼ = Σ αᵢⱼ hᵢ` | Context vector (weighted sum) |
| `Attention = softmax(QK^T/√d)V` | Scaled dot-product attention |
| `MultiHead = Concat(h₁,...,h₈)W^o` | Multi-head concatenation |
| `Loss = CrossEntropy(predicted, actual)` | Training loss |

---

## 11. COMMON EXAM PATTERNS

### High Probability Questions (90-95%):
1. Explain Seq2Seq with encoder-decoder
2. How does attention help Seq2Seq?
3. Multi-head vs single-head attention
4. Seq2Seq + Attention for machine translation

### Medium Probability Questions (75-85%):
1. BERT architecture and training objectives
2. Transformer model structure
3. Conversational AI components
4. GPT vs BERT comparison

### Lower Probability but Important (60-70%):
1. Positional encoding formula
2. LaMDA and paradigm shift
3. NLP applications
4. Dialogue management strategies

---

## 12. MUST-REMEMBER POINTS

✓ Seq2Seq: Encoder → Context → Decoder
✓ Attention: Look at ALL encoder states, not just context
✓ Multi-head: Different heads capture different aspects
✓ Transformers: No RNNs, pure attention, parallel processing
✓ BERT: Bidirectional, MLM+NSP training, for understanding
✓ GPT: Unidirectional, autoregressive, for generation
✓ Attention complexity: O(m×t) where m=input, t=output length
✓ Conversational AI: NLU → Dialogue Mgmt → NLG

---

## 13. COMMON MISTAKES TO AVOID

❌ Saying Seq2Seq is the same as Seq2Seq+Attention
✓ Seq2Seq uses context vector; Seq2Seq+Attention uses weighted sum of all states

❌ Confusing BERT (bidirectional) with GPT (unidirectional)
✓ BERT reads both directions; GPT reads left-to-right

❌ Saying transformers use RNNs
✓ Transformers use ONLY attention mechanism, no recurrence

❌ Multi-head attention has SAME complexity as single-head
✓ Actually similar O(n²d) but distributed across heads

❌ BLEU score ranges from -1 to 1
✓ BLEU score ranges from 0 to 1

---

## 14. LAST-MINUTE PREP

### 5-Minute Review:
1. Seq2Seq = Encoder-Decoder
2. Attention = Weighted focus on all states
3. Transformers = All-attention, no RNN
4. BERT = Bidirectional, MLM training
5. GPT = Autoregressive generation

### 10-Minute Review:
Add process flows:
- Seq2Seq flow
- Attention computation (3 steps)
- Transformer layer structure
- BERT training objectives
- Conversational AI components

### 15-Minute Review:
Add comparison tables:
- BERT vs GPT
- Single-head vs Multi-head
- Architectures comparison
- Attention types

---

**Last Updated**: November 2025
**Exam Focus**: Unit 5 - Machine Translation, Transformers & Modern NLP
**Total Questions Pattern**: Mostly 5-6 mark questions on Seq2Seq and Attention
