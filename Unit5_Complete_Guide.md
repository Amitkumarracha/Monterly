# UNIT 5: MACHINE TRANSLATION, TRANSFORMERS & MODERN NLP
## Comprehensive Study Guide with Predicted Questions and Answers

---

## TABLE OF CONTENTS
1. Detailed Topic Explanations
2. Key Concepts & Definitions
3. Predicted Exam Questions with Complete Answers
4. Quick Revision Sheet
5. Important Diagrams & Flow Charts

---

# PART 1: DETAILED TOPIC EXPLANATIONS

## 1. SEQUENCE-TO-SEQUENCE (SEQ2SEQ) MODEL

### 1.1 Definition & Overview
**Seq2Seq** (Sequence-to-Sequence) is a neural network architecture designed to transform an input sequence into an output sequence of different length. It is widely used for tasks where both input and output are sequential data, such as machine translation, text summarization, and question-answering systems.

### 1.2 Architecture Components

#### A. **Encoder**
- The encoder is a Recurrent Neural Network (RNN), typically LSTM or GRU
- **Function**: Processes the entire input sequence word by word
- **Process**:
  - Takes input sequence: e.g., "Hello how are you" (English)
  - Each word is converted to a word embedding (vector representation)
  - Each time step: processes one word embedding and updates hidden state
  - Final output: Context vector (final hidden state)
- **Context Vector**: A fixed-size vector containing compressed information about the entire input sequence

**Mathematical Representation:**
For input sequence x = (x₁, x₂, ..., xₘ):
- h₀ = 0 (initial hidden state)
- hᵢ = RNN(xᵢ, hᵢ₋₁) for i = 1 to m
- Context vector C = hₘ (final hidden state)

#### B. **Decoder**
- Also an RNN (LSTM/GRU) that generates the output sequence
- **Function**: Generates output word by word using the context vector from encoder
- **Process**:
  - Receives context vector as initial hidden state
  - At each time step, generates one output word
  - Uses previously generated word as input for next time step (teacher forcing during training)
  - Stops when end-of-sequence token is generated
- **Start Token**: Special token [START] marks the beginning
- **Stop Token**: Special token [STOP] marks the end

**Mathematical Representation:**
For output sequence y = (y₁, y₂, ..., yₙ):
- s₀ = C (context vector from encoder)
- sⱼ = RNN(yⱼ₋₁, sⱼ₋₁) for j = 1 to n
- yⱼ = softmax(Ws × sⱼ) (probability distribution over vocabulary)

### 1.3 Working Process Step-by-Step

**Example: Translating "go away" to German "geh weg"**

**Step 1: Tokenization**
- Input: "go away" → Tokens: ['g','o','_','a','w','a','y']
- Output: "geh weg" → Tokens: ['g','e','h','_','w','e','g']

**Step 2: Build Dictionary**
- English Dictionary: {g:1, o:2, ..., _:27}
- German Dictionary: {g:1, e:2, h:3, _:4, w:5}

**Step 3: One-Hot Encoding**
- Convert token indices to one-hot vectors
- Example: 'g' → [1,0,0,0,0,...]

**Step 4: Embedding Layer**
- Convert one-hot vectors to dense embeddings
- Each word gets a continuous vector representation
- Embedding dimension typically 128-512

**Step 5: Encoder Processing**
```
Input: "go away"
Time Step 1: Process 'g'
    - Input embedding: [0.34, -0.17, ...]
    - Hidden state h₁ updated
Time Step 2: Process 'o'
    - Uses h₁ and embedding of 'o'
    - Hidden state h₂ updated
... continue for all tokens
Final Context: C = h₇ (contains meaning of "go away")
```

**Step 6: Decoder Processing**
```
Start with context vector C and [START] token
Time Step 1: Predict 'g'
    - Loss = CrossEntropy(predicted_g, actual_g)
    - Use predicted 'g' as input to next step
Time Step 2: Predict 'e'
    - Loss = CrossEntropy(predicted_e, actual_e)
... continue until [STOP] token
```

### 1.4 Loss Function & Training
**Loss Calculation**: CrossEntropy loss for each time step
```
Total Loss = Σ CrossEntropy(predicted_token, actual_token)
```

Backpropagation Through Time (BPTT) updates all weights in encoder and decoder.

### 1.5 Limitations of Basic Seq2Seq

| Limitation | Impact | Example |
|-----------|--------|---------|
| Fixed context vector | Loses information from long sequences | For 50-word sentences, BLEU score drops significantly |
| Bottleneck problem | Single vector cannot capture all information | Long-range dependencies are forgotten |
| Sequential processing | Cannot parallelize | Slow for training on large datasets |

### 1.6 Performance Metrics for Seq2Seq

**BLEU Score (Bilingual Evaluation Understudy)**
- Measures similarity between predicted and reference translation
- Range: 0-1 (higher is better)
- Formula considers n-gram precision
- Example: If predicted = "the cat" and reference = "the cat", BLEU = 1.0

---

## 2. ATTENTION MECHANISM

### 2.1 Problem with Basic Seq2Seq
The context vector alone is insufficient for long sequences. The attention mechanism addresses this by allowing the decoder to focus on different parts of the input sequence at each decoding step.

### 2.2 Attention Mechanism Concept

**Core Idea**: Instead of compressing entire input into single context vector, allow decoder to look at ALL encoder hidden states at each step.

**Key Insight**: For each output token generation, compute which input tokens are most relevant (important).

### 2.3 Attention Computation Process

**Step 1: Alignment Score Calculation**
For each decoder hidden state sⱼ and encoder hidden state hᵢ:
```
eᵢⱼ = attention_function(sⱼ, hᵢ)
```

Common attention functions:
- **Additive (Bahdanau)**: eᵢⱼ = vᵀ tanh(Wₛsⱼ + Wₕhᵢ)
- **Multiplicative (Luong)**: eᵢⱼ = sⱼᵀ hᵢ
- **Scaled Dot-Product**: eᵢⱼ = (sⱼᵀ hᵢ) / √d

**Step 2: Convert to Attention Weights (Softmax)**
```
αᵢⱼ = exp(eᵢⱼ) / Σₖ exp(eₖⱼ)
```
This ensures weights sum to 1 and range from 0 to 1.

**Step 3: Compute Context Vector**
```
cⱼ = Σᵢ αᵢⱼ × hᵢ
```
Context vector is weighted sum of all encoder hidden states.

**Step 4: Combine with Decoder State**
```
c̃ⱼ = tanh(W[cⱼ ; sⱼ])  # Concatenate and transform
```

**Step 5: Generate Output**
```
yⱼ = softmax(Wₒ c̃ⱼ)
```

### 2.4 Attention Weights Interpretation

**Example: English→German "I love cats"**

When generating German word at position 2:
```
Attention weights: [0.05, 0.85, 0.10]
                  (on "I") (on "love") (on "cats")
```
This means the model focuses 85% on the word "love" for generating the corresponding German word.

### 2.5 Benefits of Attention

| Benefit | Explanation |
|---------|-------------|
| No information loss | All encoder states accessible | 
| Interpretability | Can visualize which input tokens are important |
| Handles long sequences | BLEU scores remain high even for 50+ word sentences |
| Parallel computation | Attention weights can be computed in parallel |

### 2.6 Computational Complexity

- **Seq2Seq without Attention**: O(m + t) where m=input length, t=output length
- **Seq2Seq with Attention**: O(m × t) - linear increase in computation

---

## 3. SELF-ATTENTION MECHANISM

### 3.1 Definition
**Self-Attention**: A mechanism where each element in a sequence attends to all other elements in the SAME sequence. Used in Transformers.

### 3.2 Query-Key-Value (QKV) Framework

Each input element is transformed into three vectors:

**Query (Q)**: "What am I looking for?"
```
Q = Wq × x
```

**Key (K)**: "What do I contain?"
```
K = Wk × x
```

**Value (V)**: "What information do I provide?"
```
V = Wv × x
```

Where Wq, Wk, Wv are learnable weight matrices.

### 3.3 Self-Attention Computation

**Step 1: Compute Attention Scores**
```
Scores = (Q × Kᵀ) / √d
```
Divided by √d for numerical stability (scaled dot-product).

**Step 2: Apply Softmax**
```
Attention_weights = softmax(Scores)
```

**Step 3: Apply to Values**
```
Self_Attention_Output = Attention_weights × V
```

### 3.4 Single-Head vs Multi-Head Attention

#### Single-Head Self-Attention:
- Uses ONE set of Q, K, V projections
- Computes attention ONCE
- Limited to one "representation subspace"

#### Multi-Head Self-Attention:
- Uses MULTIPLE (typically 8) parallel attention heads
- Each head has its own Q, K, V projections
- Each head learns different aspects of the sequence
- **Concatenate** all head outputs

**Formula**:
```
MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headₕ) × Wᵒ
where headᵢ = Attention(Q_i, K_i, V_i)
```

### 3.5 Advantages of Multi-Head Attention

| Aspect | Benefit |
|--------|---------|
| Representation Power | Each head focuses on different sequence aspects |
| Linguistic Knowledge | One head may capture syntax, another semantics |
| Robustness | Redundancy - if one head fails, others compensate |
| Parallel Processing | All heads compute simultaneously |

**Example**: For sentence "The dog jumped over the fence"
- Head 1: Captures subject-verb relationships
- Head 2: Captures object relationships  
- Head 3: Captures noun-adjective relationships
- Head 4-8: Capture other linguistic patterns

---

## 4. TRANSFORMERS ARCHITECTURE

### 4.1 Why Transformers?

**Problems with RNN/LSTM**:
- Sequential processing (cannot parallelize)
- Vanishing gradient problem for long sequences
- Slow training on large datasets

**Transformers Solution**:
- Process entire sequence in parallel
- No recurrence, based purely on attention
- Can handle very long sequences (up to 4096+ tokens)
- Faster training due to parallelization

### 4.2 Transformer Architecture Overview

**Main Components**:
1. **Encoder Stack** (6-12 layers typically)
2. **Decoder Stack** (6-12 layers typically)
3. **Positional Encoding** (since no sequential processing)
4. **Multi-Head Attention** (instead of RNN)
5. **Feed-Forward Networks** (dense layers in each position)

### 4.3 Encoder Layer Structure

Each encoder layer contains:

**1. Multi-Head Self-Attention**
```
Output = MultiHead(Q, K, V)
```

**2. Add & Normalize (Residual Connection)**
```
Z₁ = LayerNorm(X + MultiHead(Q,K,V))
```

**3. Feed-Forward Network**
```
Output = Dense(ReLU(Dense(Z₁)))
Typically: hidden_size → intermediate_size → hidden_size
Example: 512 → 2048 → 512
```

**4. Add & Normalize Again**
```
Z₂ = LayerNorm(Z₁ + FFN(Z₁))
```

### 4.4 Decoder Layer Structure

Similar to encoder, with additional **Cross-Attention** layer:

**1. Masked Multi-Head Self-Attention**
- Masked to prevent attending to future positions
- Ensures autoregressive generation

**2. Add & Normalize**

**3. Cross-Attention**
- Attends to encoder output
- Query from decoder, Key/Value from encoder
```
Output = MultiHead(Q_decoder, K_encoder, V_encoder)
```

**4. Add & Normalize**

**5. Feed-Forward Network**

**6. Add & Normalize**

### 4.5 Positional Encoding

Since transformers have no sequential processing, need to inject position information:

**Formula**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Where:
- pos = position in sequence
- i = dimension index
- d = embedding dimension

### 4.6 Advantages of Transformers

| Advantage | Impact |
|-----------|--------|
| Parallelization | Can process all positions simultaneously |
| Long-range dependencies | Attention can connect distant words directly |
| Faster training | Reduced training time (weeks → days) |
| Scalability | Can handle million+ token sequences |
| Transfer learning | Pre-trained models transferable to many tasks |

---

## 5. BERT (BIDIRECTIONAL ENCODER REPRESENTATIONS FROM TRANSFORMERS)

### 5.1 Definition & Overview
**BERT** is a pre-trained language model using bidirectional transformers. It reads text in BOTH directions (left-to-right AND right-to-left simultaneously), capturing full context.

### 5.2 Key Innovation: Bidirectionality

**Unidirectional (GPT)**:
- Reads: "The cat sat on the ___"
- Only uses left context
- Limited understanding

**Bidirectional (BERT)**:
- Reads: "The ___ sat on the mat"
- Uses both left AND right context
- Understands "cat" should go in blank because:
  - Left context: "The" (needs noun)
  - Right context: "sat on the mat" (fits with "cat")

### 5.3 BERT Architecture

**Configuration**:
- **BERT-Base**: 12 encoder layers, 768 hidden units, 12 attention heads
- **BERT-Large**: 24 encoder layers, 1024 hidden units, 16 attention heads

**No Decoder**: BERT is encoder-only (designed for understanding, not generation)

### 5.4 BERT Training Objectives

#### 1. Masked Language Modeling (MLM)
**Concept**: Mask 15% of input tokens and predict them

**Process**:
- Randomly select 15% of tokens
- 80% replaced with [MASK], 10% with random word, 10% kept unchanged
- Model learns to predict original tokens
- Loss only computed on masked positions

**Example**:
```
Original: "The quick brown fox jumps"
Masked: "The [MASK] brown [MASK] jumps"
Model predicts: "quick" and "fox"
```

#### 2. Next Sentence Prediction (NSP)
**Concept**: Predict if second sentence follows first

**Process**:
- Training pairs: 50% IsNext, 50% NotNext
- Model learns relationships between sentences
- Binary classification task

**Example**:
```
Sentence A: "The man went to the store."
Sentence B: "He bought a gallon of milk."
Label: IsNext (probability should be high)
```

### 5.5 BERT Input Processing

**Three Components**:

1. **Token Embedding**: Word vector representation
2. **Segment Embedding**: 0 for sentence A, 1 for sentence B
3. **Position Embedding**: Position in sequence

**Combined Input**:
```
Final embedding = Token_emb + Segment_emb + Position_emb
```

### 5.6 BERT Applications

| Task | Method | Example |
|------|--------|---------|
| Text Classification | Add [CLS] token, use its final representation | Sentiment analysis |
| Named Entity Recognition | Use each token's output representation | Extract person names |
| Question Answering | Two representations for answer span start/end | SQuAD dataset |
| Similarity | Compute cosine similarity of sentence embeddings | Paraphrase detection |

---

## 6. GPT (GENERATIVE PRE-TRAINED TRANSFORMER)

### 6.1 Definition & Overview
**GPT** is an autoregressive language model using decoder-only transformer architecture. It predicts the next word in a sequence based on all previous words.

### 6.2 Key Characteristics

**Autoregressive**:
- Generates text one token at a time
- Each prediction conditions on all previous predictions
- Formula: P(x₁, x₂, ..., xₙ) = Π P(xᵢ|x₁...xᵢ₋₁)

**Decoder-Only**:
- Only uses transformer decoder
- Masked self-attention (cannot attend to future positions)
- Optimized for text generation

**Unidirectional**:
- Only reads left-to-right (unlike BERT)
- Cannot use future context
- Matches autoregressive training objective

### 6.3 Training Process

**Objective**: Predict next word
```
Input: "The cat sat on the"
Target: "mat"
Model learns: P(mat | The cat sat on the)
```

**Scale**:
- GPT-2: 1.5 billion parameters
- GPT-3: 175 billion parameters
- Trained on 300+ billion tokens

### 6.4 GPT Applications

| Application | How It Works |
|-------------|------------|
| Text Generation | Start with prompt, iteratively predict next word |
| Translation | Provide source sentence + language tag |
| Summarization | Provide text + "TL;DR:" to trigger summary |
| Code Generation | Trained on code repositories, generates functions |
| Question Answering | Provide context + question, generate answer |

### 6.5 BERT vs GPT Comparison

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Direction | Bidirectional | Unidirectional |
| Attention Masking | No masking | Causal masking |
| Training | MLM + NSP | Language modeling |
| Best For | Understanding/Classification | Generation |
| Context | Can use future words | Only past words |

---

## 7. LAMDA (LANGUAGE MODEL FOR DIALOGUE APPLICATIONS)

### 7.1 Definition & Overview
**LaMDA** is Google's conversational AI model specifically designed for open-ended dialogue. It combines large-scale language modeling with dialogue-specific fine-tuning.

### 7.2 Architecture
- **Base**: Decoder-only transformer (similar to GPT)
- **Scale**: 137 billion parameters
- **Training Data**: Includes dialogue-specific data

### 7.3 Two-Stage Training

**Stage 1: Pre-training**
- Trained on vast public web text + dialogue data
- Trillions of words
- Objective: Next word prediction
- General language knowledge acquired

**Stage 2: Fine-tuning**
- Trained on high-quality dialogue data
- Human raters evaluate responses on:
  - **Sensibleness**: Does it make sense?
  - **Specificity**: Is it tailored to the query?
  - **Interestingness**: Is it engaging?
- Model learns to maximize these metrics

### 7.4 Groundedness in LaMDA
**Problem**: Language models hallucinate (make up facts)

**Solution**: Groundedness metric
- Measures if response can be verified in external sources
- LaMDA consults search engines for factual claims
- Improves accuracy and trustworthiness

### 7.5 LaMDA vs GPT-3

| Aspect | LaMDA | GPT-3 |
|--------|-------|-------|
| Purpose | Dialogue specialist | General text generation |
| Training Focus | Conversation quality | Broad task competency |
| Metrics | SSI (sensibleness, specificity, interestingness) | Task performance |
| Design | Specialized for chat | Versatile across tasks |
| Accuracy | Higher for conversations | More flexible |

### 7.6 LaMDA Applications
- Conversational AI assistants
- Virtual companions
- Customer service chatbots
- Tutoring systems
- Creative writing partners

---

## 8. PARADIGM SHIFT IN NLP WITH MODERN LANGUAGE MODELS

### 8.1 Evolution of NLP Approaches

**Traditional NLP (Pre-2012)**:
- Hand-crafted features
- Task-specific models
- Limited context understanding
- Separate models for each task

**Deep Learning Era (2012-2017)**:
- RNN/LSTM-based
- Learned representations
- Better context capturing
- Still task-specific

**Transformer Era (2017-Present)**:
- Attention-based, no recurrence
- Transfer learning dominant
- Few-shot learning possible
- Large pre-trained models

### 8.2 Key Paradigm Shifts

**1. From Hand-Crafted to Learned Features**
- OLD: Engineers manually design features
- NEW: Models automatically learn representations

**2. From Task-Specific to General-Purpose Models**
- OLD: Different model for each task
- NEW: Single pre-trained model for multiple tasks

**3. From Supervised to Self-Supervised Learning**
- OLD: Requires large labeled datasets
- NEW: Pre-training on unlabeled text, fine-tune with little data

**4. From Small Models to Very Large Models**
- OLD: Millions of parameters
- NEW: Billions to hundreds of billions of parameters
- Scaling laws: "bigger is often better"

**5. From Engineered Systems to End-to-End Learning**
- OLD: Pipeline of separate components
- NEW: Single neural network handles everything

### 8.3 Scale and Emergence

**Scaling Laws**:
- As model size increases → performance improves
- GPT-2 (1.5B params) shows some capabilities
- GPT-3 (175B params) shows many emergent abilities

**Emergent Abilities** (appear suddenly at scale):
- Few-shot learning without fine-tuning
- Chain-of-thought reasoning
- Code generation
- Cross-lingual understanding
- Mathematical reasoning

### 8.4 Impact on NLP Tasks

| Task | Traditional | Modern |
|------|-----------|--------|
| Sentiment Analysis | Training classifiers on labeled data | Use pre-trained model with prompting |
| Machine Translation | Seq2Seq with attention | Transformer fine-tuning |
| NER | CRF models + features | BERT fine-tuning |
| QA | Information retrieval | End-to-end neural models |
| Summarization | Template-based | Large language model generation |

---

## 9. NLP IN AI APPLICATIONS

### 9.1 Conversational AI

**Definition**: AI systems that interact with users through natural language.

**Components**:
1. **Natural Language Understanding (NLU)**
   - Intent recognition
   - Entity extraction
   - Context understanding

2. **Dialogue Management**
   - Conversation state tracking
   - Response selection
   - Context maintenance

3. **Natural Language Generation (NLG)**
   - Response formulation
   - Ensuring naturalness
   - Maintaining consistency

**Examples**:
- ChatGPT
- Google Assistant
- Amazon Alexa
- Microsoft Cortana

### 9.2 Machine Translation

**Modern Approach**: Seq2Seq with attention or Transformers

**Process**:
1. Input text tokenization
2. Encoder processes source language
3. Decoder generates target language
4. Attention focuses on relevant source words

**Challenges**:
- Word sense ambiguity
- Idioms and colloquialisms
- Named entity handling
- Context preservation

### 9.3 Question Answering

**Types**:
- **Extractive**: Extract answer span from given context
- **Generative**: Generate answer from knowledge

**Methods**:
- Retrieval-based: Find relevant documents, extract answer
- Reading comprehension: Model reads document and finds answer
- Knowledge-based: Query external knowledge base

### 9.4 Information Extraction

**Tasks**:
- Named Entity Recognition (NER)
- Relation extraction
- Event extraction
- Argument extraction

**Applications**:
- Resume parsing
- Document analysis
- Fact extraction
- Knowledge base construction

### 9.5 Sentiment Analysis

**Task**: Determine emotional tone of text

**Approaches**:
- Lexicon-based: Count positive/negative words
- Machine learning: Train classifier on labeled data
- Deep learning: Use embeddings and neural networks

**Applications**:
- Customer feedback analysis
- Social media monitoring
- Brand reputation management
- Product review analysis

### 9.6 Text Summarization

**Types**:
- **Abstractive**: Generate new summary (more natural)
- **Extractive**: Select key sentences (simpler)

**Methods**:
- Neural abstractive: Seq2Seq/Transformer-based
- Extractive: Ranking and selection
- Hybrid: Extraction then compression

**Applications**:
- News summarization
- Document summarization
- Meeting minutes
- Email summarization

### 9.7 Semantic Search

**Traditional Search**: Keyword matching
**Semantic Search**: Understanding meaning

**Process**:
- Convert query to embedding
- Convert documents to embeddings
- Find nearest documents in embedding space
- Return semantically similar documents

**Advantages**:
- Handles synonyms
- Understands context
- Better relevance
- Tolerates typos

---

# PART 2: KEY CONCEPTS & DEFINITIONS

## Important Terminology

**Tokenization**: Breaking text into words or characters
**Embedding**: Converting tokens to continuous vectors
**Context Vector**: Compressed representation of input sequence
**Attention Weights**: Normalized importance scores
**Self-Attention**: Each element attends to all elements in same sequence
**Bidirectional**: Reading text in both directions
**Autoregressive**: Predicting next token based on previous tokens
**Masked Language Modeling**: Predicting masked tokens in input
**Fine-tuning**: Training pre-trained model on specific task
**Transfer Learning**: Using knowledge from one task for another
**Pre-training**: Training on large unlabeled corpus
**Few-shot Learning**: Learning from few examples
**Emergent Abilities**: Capabilities appearing at model scale

---

# PART 3: PREDICTED EXAM QUESTIONS WITH COMPLETE ANSWERS

## Question 1: SEQ2SEQ Model Architecture (HIGH PROBABILITY: 95%)

### Question:
**"Explain the neural machine translation using Seq2Seq model architecture. Describe the role of encoder and decoder components."**

### Complete Answer:

**Introduction (1 mark)**
Sequence-to-Sequence (Seq2Seq) model is a neural network architecture that transforms an input sequence into an output sequence of variable length. It is the foundation for modern neural machine translation systems used in services like Google Translate.

**Encoder Component (2 marks)**
The encoder is typically a bidirectional LSTM or GRU that:
- Processes the entire input sequence word by word
- Converts each word to an embedding vector
- At each time step, updates its hidden state based on current input and previous hidden state
- Outputs a single context vector (final hidden state) that compresses all information from the input
- Mathematical representation: hₜ = LSTM(xₜ, hₜ₋₁)
- Context vector C = hₙ where n is sequence length

**Example**: For "go away":
- Time 1: Process 'go' → hidden state h₁
- Time 2: Process 'away' → hidden state h₂ (contains meaning of entire phrase)
- Context vector C = h₂

**Decoder Component (2 marks)**
The decoder is also an RNN (LSTM/GRU) that:
- Takes the context vector as initial hidden state
- Generates output sequence one token at a time
- At each time step, generates probability distribution over vocabulary
- Uses previously generated word (or ground truth during training) as input
- Continues until end-of-sequence token is generated
- Mathematical representation: P(yₜ|C, y₁, ..., yₜ₋₁) = softmax(Wd × sₜ)

**Example**: Generating "geh weg":
- Start with context C and [START] token
- Time 1: P(geh) - model predicts 'geh'
- Time 2: P(weg|geh) - model predicts 'weg' 
- Time 3: Model predicts [STOP]

**Overall Process (1 mark)**
1. Input tokenization and embedding
2. Encoder processes entire input → context vector
3. Context vector transferred to decoder
4. Decoder generates output word by word
5. Loss computed using cross-entropy for each output word
6. Backpropagation through time updates all weights

**Loss Function**:
Total Loss = Σ CrossEntropy(predicted_tᵢ, actual_tᵢ)

**Limitations Mentioned (if asked)**:
- Fixed context vector becomes bottleneck for long sequences
- Information loss for long inputs (BLEU score drops for 50+ word sentences)
- Sequential processing prevents parallelization

---

## Question 2: Attention Mechanism in Seq2Seq (HIGH PROBABILITY: 85%)

### Question:
**"Explain the attention mechanism in Seq2Seq model. How does attention overcome the limitations of basic Seq2Seq for machine translation?"**

### Complete Answer:

**Problem with Basic Seq2Seq (1 mark)**
The basic Seq2Seq model uses only one context vector to represent the entire input sequence. This creates a bottleneck:
- Long sequences lose information during compression
- Decoder cannot focus on relevant input words
- Performance degrades significantly for sequences > 20 words
- BLEU scores drop when translating longer sentences

**Example**: For long sentence, single context vector may forget important words at the beginning

**Attention Mechanism Concept (1.5 marks)**
Attention mechanism allows decoder to "look at" ALL encoder hidden states at each decoding step, rather than just the final context vector.

**Workflow**:
- Encoder produces hidden states for each input position: h₁, h₂, ..., hₘ
- For each decoder step j:
  - Compute alignment score eᵢⱼ between decoder state sⱼ and each encoder state hᵢ
  - Convert scores to attention weights using softmax: αᵢⱼ
  - Compute weighted context vector: cⱼ = Σ αᵢⱼ × hᵢ
  - Use context vector for output generation

**Attention Score Calculation (2 marks)**

**Step 1: Alignment Score**
```
eᵢⱼ = attention_function(sⱼ, hᵢ)
```

Common functions:
- **Additive (Bahdanau)**: eᵢⱼ = vᵀ tanh(Wₛsⱼ + Wₕhᵢ)
- **Multiplicative (Luong)**: eᵢⱼ = sⱼᵀ hᵢ  
- **Scaled Dot-Product**: eᵢⱼ = (sⱼᵀ hᵢ) / √d

**Step 2: Softmax Normalization**
```
αᵢⱼ = exp(eᵢⱼ) / Σₖ exp(eₖⱼ)
```
Ensures: 0 ≤ αᵢⱼ ≤ 1 and Σᵢ αᵢⱼ = 1

**Step 3: Context Vector**
```
cⱼ = Σᵢ αᵢⱼ × hᵢ
```
Weighted sum of all encoder hidden states.

**Step 4: Generate Output**
```
Combine cⱼ with decoder state sⱼ
c̃ⱼ = tanh(W[cⱼ ; sⱼ])
yⱼ = softmax(Wₒ c̃ⱼ)
```

**Example: "The dog bit the cat"→ German**

When generating German word #2, attention weights might be:
```
[0.03, 0.75, 0.10, 0.02, 0.10]
(on "the") (on "dog") (on "bit") (on "the") (on "cat")
```
75% focus on "dog" because it's the subject and most relevant for translating the next German word.

**Advantages of Attention (1.5 marks)**

| Advantage | Explanation | Evidence |
|-----------|-------------|----------|
| Handles long sequences | No info loss for 50+ word sentences | BLEU scores remain high |
| Interpretability | Can visualize which inputs are important | Show attention weights heatmap |
| No bottleneck | All encoder states accessible | Better gradient flow |
| Alignment learning | Model learns to align source-target words | Works across language pairs |

**Computational Trade-off**
- Seq2Seq basic: O(m + t) complexity
- Seq2Seq + Attention: O(m × t) complexity
- Worth the cost due to quality improvement

---

## Question 3: Attention Mechanisms - Single Head vs Multi-Head (HIGH PROBABILITY: 85%)

### Question:
**"Distinguish between single-head attention and multi-head attention. Explain with an example why multi-head attention is superior for neural machine translation."**

### Complete Answer:

**Single-Head Self-Attention (2 marks)**

**Definition**: Uses ONE set of Query, Key, Value projections to compute attention.

**Process**:
```
Q = Wq × X
K = Wk × X  
V = Wv × X

Scores = (Q × Kᵀ) / √d

Attention_weights = softmax(Scores)

Output = Attention_weights × V
```

Where X is input sequence embedding matrix.

**Limitations**:
- Single representation subspace
- Cannot simultaneously attend to different aspects
- Model representation power limited
- May miss important linguistic patterns

**Example**: For "The bank manager approved my loan"
- Single head must choose: focus on "bank" (financial) or "manager" (person)?
- Cannot simultaneously capture both interpretations

**Multi-Head Self-Attention (2 marks)**

**Definition**: Uses MULTIPLE (typically 8) parallel attention heads, each with independent Q, K, V projections.

**Process**:
```
For each head h (h = 1 to 8):
    Q_h = Wq^h × X
    K_h = Wk^h × X
    V_h = Wv^h × X
    
    Attention_h = softmax((Q_h × K_h^T) / √d) × V_h

Concatenate all heads:
Output = Concat(Attention_1, Attention_2, ..., Attention_8)

Final_output = Wᵒ × Output
```

**Advantages Over Single-Head** (2 marks)

| Aspect | Single-Head | Multi-Head | Benefit |
|--------|------------|-----------|---------|
| Representation | 1 subspace | 8 subspaces | Richer representations |
| Linguistic patterns | Limited | Multiple patterns | Better context capture |
| Robustness | Sensitive to initialization | Redundant heads | More stable |
| Parallelization | No | Yes | Faster computation |

**Detailed Comparison**:

1. **Representation Power**
   - Single-head: 768-dim attention space
   - Multi-head: 8 × 96-dim spaces = more diverse representations
   - Can capture syntax, semantics, and other aspects simultaneously

2. **Linguistic Knowledge**
   - Head 1: May learn subject-verb relationships
   - Head 2: May learn object relationships
   - Head 3: May learn noun-adjective relationships
   - Head 4-8: May learn long-range dependencies, tense agreement, etc.

3. **Example: "The bank approved my loan"**
   ```
   Head 1 (Syntax): Focuses on "approved" → "bank"
   Head 2 (Semantics): Focuses on "bank" → "loan"
   Head 3 (Agreement): Focuses on article-noun agreement
   Head 4 (Long-range): Focuses on "The" → "bank"
   ...
   Concatenated: Rich representation capturing all aspects
   ```

4. **Robustness**
   - If one head makes mistake, others compensate
   - Redundancy improves model stability
   - Better generalization to new data

**Original Transformer Settings**:
- 8 heads in original "Attention is All You Need" paper
- Shown empirically to work well
- Each head: 768/8 = 96 dimensions

**Why Multi-Head for Machine Translation** (1 mark):
Translation requires understanding:
- Grammatical structure (head 1)
- Semantic relationships (head 2)
- Word order differences (head 3)
- Long-range dependencies (head 4)
- Anaphora resolution (head 5)
- Idiom handling (head 6-8)

Single head cannot capture all simultaneously.

---

## Question 4: BERT Model Architecture and Applications (PROBABILITY: 80%)

### Question:
**"Explain the architecture of BERT model. Discuss its training objectives and applications in NLP tasks."**

### Complete Answer:

**Introduction (0.5 marks)**
BERT stands for Bidirectional Encoder Representations from Transformers. It is a pre-trained deep bidirectional transformer encoder that learns contextualized word representations.

**Architecture (1.5 marks)**

**Structural Details**:
- **Encoder-Only**: Only uses transformer encoder stack (12-24 layers)
- **No Decoder**: Unlike original transformer, no generation component
- **Bidirectional**: Reads text left-to-right AND right-to-left simultaneously

**Variants**:
- **BERT-Base**: 12 layers, 768 hidden units, 12 attention heads, 110M parameters
- **BERT-Large**: 24 layers, 1024 hidden units, 16 attention heads, 340M parameters

**Key Innovation: Bidirectionality**
- Traditional LMs: Left-to-right only (predict next word)
- BERT: Can see entire context around each word
- Example: For "The cat sat on the ___"
  - BERT sees: "The" "cat" "sat" "on" "the" → can predict "mat" better
  - Because it sees word "sat" (past tense) ahead of time

**Input Representation (1 mark)**

BERT combines three types of embeddings:

```
Final_embedding = Token_embedding + Segment_embedding + Position_embedding
```

1. **Token Embedding**: Word vector from vocabulary
2. **Segment Embedding**: 0 for sentence A, 1 for sentence B
   - Allows model to distinguish between two input sentences
3. **Position Embedding**: Learned embeddings for each position
   - Unlike transformers (sinusoidal positions), BERT learns these

**Special Tokens**:
- [CLS]: Classification token at beginning, used for classification tasks
- [SEP]: Separator between sentences
- [PAD]: Padding token
- [UNK]: Unknown word token
- [MASK]: Token to mask for MLM task

**Training Objectives (2 marks)**

**1. Masked Language Modeling (MLM)**

**Concept**: Randomly mask 15% of tokens and predict them

**Process**:
- Select 15% of tokens randomly
- 80% → replace with [MASK]
- 10% → replace with random token
- 10% → keep unchanged

**Purpose**: Forces model to learn context-dependent representations

**Example**:
```
Original: "The quick brown fox jumps over the lazy dog"
Masked: "The quick [MASK] fox [MASK] over the lazy dog"
Model predicts: "brown" and "jumps"
```

**Why 80-10-10 split**?
- 80% masking: Learn to predict masked tokens
- 10% random: Robustness - force model not to rely on specific tokens
- 10% unchanged: Bias towards original data distribution

**Loss**: Only computed on masked positions
```
Loss_MLM = CrossEntropy(predicted_brown, actual_brown) + 
           CrossEntropy(predicted_jumps, actual_jumps)
```

**2. Next Sentence Prediction (NSP)**

**Concept**: Predict if sentence B follows sentence A

**Process**:
- 50% of pairs: IsNext (correct sequence)
- 50% of pairs: NotNext (random sentence)
- Binary classification task

**Example**:
```
Sentence A: "The man went to the store."
Sentence B: "He bought a gallon of milk."
Label: IsNext (probability should be high)

---

Sentence A: "The man went to the store."
Sentence B: "Cats are fluffy animals."
Label: NotNext (probability should be low)
```

**Purpose**: Learn sentence-level relationships (useful for tasks like question answering)

**Applications (2 marks)**

**1. Text Classification**
- **Method**: Add linear classifier on top of [CLS] token
- **Example**: Sentiment analysis - classify movie reviews as positive/negative
- **Process**: 
  - Feed text through BERT
  - Use [CLS] token representation
  - Pass through linear layer
  - Output: probability distribution over classes

**2. Named Entity Recognition (NER)**
- **Method**: Use output of each token, add classification layer
- **Example**: "John Smith works at Google"
  - Classify "John" as PERSON
  - Classify "Smith" as PERSON
  - Classify "Google" as ORG
- **Process**:
  - Each token gets its own output representation
  - Linear classifier for each token
  - Sequence labeling task

**3. Question Answering**
- **Method**: Predict start and end positions of answer span
- **Example**: Context: "Paris is the capital of France"
  Question: "What is the capital of France?"
  Answer: Start=5, End=6 (word "Paris")
- **Process**:
  - Concatenate context and question
  - Use BERT to encode
  - Linear layers predict start position and end position
  - Extract span from context

**4. Semantic Similarity**
- **Method**: Encode two sentences, compute cosine similarity of [CLS] representations
- **Example**: Paraphrase detection
  - "The cat is on the mat"
  - "A cat is sitting on a mat"
  - High similarity score
- **Applications**: Duplicate detection, plagiarism detection

**5. Information Extraction**
- Extract relationships between entities
- Attribute extraction

**Fine-tuning Process (0.5 marks)**
1. Load pre-trained BERT weights
2. Add task-specific layer(s)
3. Fine-tune on labeled task data
4. Typically requires: 2-4 epochs, learning rate 2e-5 to 5e-5
5. Much faster than training from scratch

---

## Question 5: Transformers Architecture - Complete Explanation (PROBABILITY: 80%)

### Question:
**"Explain the complete architecture of Transformer model. Discuss how it overcomes limitations of RNNs. Differentiate between encoder and decoder layers."**

### Complete Answer:

**Introduction (1 mark)**
The Transformer is a neural network architecture based entirely on attention mechanisms, introduced in "Attention Is All You Need" paper (Vaswani et al., 2017). It eliminated the need for recurrence, enabling parallel processing and better handling of long-range dependencies.

**Why Transformers? (1 mark)**

**Limitations of RNNs**:
1. **Sequential Processing**: Cannot process words in parallel → slow training
2. **Vanishing Gradient**: Information from early words lost for long sequences
3. **Limited Parallelization**: Each time step depends on previous → cannot batch
4. **Poor Long-Range Dependencies**: Attention mechanism helps, but still inferior

**Transformer Advantages**:
- **Full Parallelization**: Process all positions simultaneously
- **Direct Long-Range Connections**: Attention connects any two positions directly
- **Efficient**: Can train on TPUs/GPUs effectively
- **Scalable**: Handles sequences up to 4096+ tokens

**Overall Architecture (1 mark)**

```
Input
  ↓
Embedding + Positional Encoding
  ↓
┌─────────────────────────────────┐
│  Encoder Stack (6-12 layers)    │
│  Each layer:                     │
│  - Multi-Head Self-Attention    │
│  - Feed-Forward Network         │
│  - Residual Connections + LN    │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  Decoder Stack (6-12 layers)    │
│  Each layer:                     │
│  - Masked Self-Attention        │
│  - Cross-Attention              │
│  - Feed-Forward Network         │
│  - Residual Connections + LN    │
└─────────────────────────────────┘
  ↓
Linear Layer + Softmax
  ↓
Output
```

**Input Processing (0.5 marks)**

**1. Embedding**:
- Convert each word to vector: vocabulary size → embedding dim (typically 512)
- Example: "hello" → [0.2, -0.5, 0.1, ..., 0.3] (512-dimensional)

**2. Positional Encoding**:
- Add position information (transformers have no sequential processing)
- **Formula**:
  ```
  PE(pos, 2i) = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
  ```
- Added to embeddings: Final_input = Embedding + PositionalEncoding

**Encoder Layer Structure (1.5 marks)**

Each encoder layer contains two sub-layers:

**Sub-layer 1: Multi-Head Self-Attention**
```
Q = Wq X,  K = Wk X,  V = Wv X

Attention(Q, K, V) = softmax(QK^T / √d) V

MultiHead(X) = Concat(head_1, ..., head_h) W^o
```

- Projects input to Query, Key, Value
- Computes attention scores: how much each position attends to each other position
- Output: contextualized representation of each word

**Sub-layer 2: Feed-Forward Network**
```
FFN(X) = max(0, XW_1 + b_1) W_2 + b_2
```

- Two dense layers with ReLU activation
- Typical: hidden_dim → 2048 → hidden_dim
- Applied position-wise (same network for each position)

**Add & Normalize (Residual Connections)**:
```
Output_1 = LayerNorm(X + MultiHeadAttention(X))
Output_2 = LayerNorm(Output_1 + FFN(Output_1))
```

- Residual connections: X + F(X) (helps with gradient flow)
- Layer normalization: stabilizes training

**Decoder Layer Structure (1.5 marks)**

Similar to encoder, but with THREE sub-layers:

**Sub-layer 1: Masked Multi-Head Self-Attention**
- Self-attention but with MASKING
- Each position can only attend to earlier positions (prevents attending to future)
- Ensures autoregressive generation

**Masking Matrix**:
```
       pos_0  pos_1  pos_2  pos_3
pos_0  [  1      0      0      0  ]
pos_1  [  1      1      0      0  ]
pos_2  [  1      1      1      0  ]
pos_3  [  1      1      1      1  ]
```

When computing attention for position 2:
- Can attend to positions 0, 1, 2
- Cannot attend to position 3 (future)

**Sub-layer 2: Cross-Attention**
```
Q = Wq X_decoder
K = Wk X_encoder
V = Wv X_encoder

Attention(Q, K, V) - queries from decoder, keys/values from encoder
```

- Decoder attends to encoder outputs
- Allows decoder to focus on relevant parts of input

**Sub-layer 3: Feed-Forward Network**
- Same as encoder

**Add & Normalize for each sub-layer**

**Encoder vs Decoder Comparison (1 mark)**

| Aspect | Encoder | Decoder |
|--------|---------|---------|
| Self-Attention | Unmasked (full) | Masked (future positions blocked) |
| Cross-Attention | No | Yes (attends to encoder) |
| Purpose | Understand input | Generate output |
| Can use future context | Yes | No (causal) |
| Processing | Parallel | Sequential (during inference) |

**Complete Forward Pass Example (1 mark)**

**Task**: Translate "Hello" from English to French

```
1. Embedding "hello" → vector
2. Add positional encoding → [0.5, -0.2, 0.3, ..., 0.1]

3. ENCODER:
   - Layer 1: Self-Attention on "hello" + FFN
   - Layer 2: Self-Attention on layer1_output + FFN
   - ... (repeat 6 times)
   - Output: Encoder_representation_hello

4. DECODER (generating "Bonjour"):
   - Start with [START] token
   
   - Layer 1:
     * Masked-Attention on [START]
     * Cross-Attention: [START] attends to encoder_representation
     * FFN
   
   - Output: Probability distribution over French vocabulary
   - Sample/argmax: Generate "Bonjour" (or more tokens)

5. Linear layer maps last decoder output to vocabulary logits
6. Softmax: P(word | context)
7. Continue generating until [END] token
```

**Advantages Over RNNs** (1 mark)

| RNN | Transformer |
|-----|-------------|
| Sequential: day 1, day 2, day 3... | Parallel: process all days simultaneously |
| Gradient vanishes for long sequences | Direct connections → stable gradients |
| Difficult to parallelize training | Highly parallelizable → faster training |
| Attention helps but still limited | Native attention mechanism for all |
| Days to weeks for training | Hours to days |

**Computational Complexity**:
- RNN: O(n) sequential steps × O(d²) per step = O(nd²)
- Transformer: O(n²d) self-attention + O(nd²) FFN = O(n²d)
- But can parallelize: faster wall-clock time

---

## Question 6: Conversational AI and Applications (PROBABILITY: 75%)

### Question:
**"Discuss how NLP is applied in Conversational AI. Explain the key components and challenges in building chatbots."**

### Complete Answer:

**Definition and Overview (1 mark)**
Conversational AI refers to AI systems designed to communicate with humans in natural language through text or voice. These systems understand user intent and generate contextually appropriate responses.

**Key Components (1.5 marks)**

**1. Natural Language Understanding (NLU)**
- **Purpose**: Understand what user wants
- **Tasks**:
  - Intent recognition: Identify user goal (e.g., "book flight")
  - Entity extraction: Extract key information (e.g., "New York", "tomorrow")
  - Context understanding: Maintain conversation context
  
- **Example**: "Book me a flight to New York tomorrow"
  - Intent: BOOK_FLIGHT
  - Entities: Destination=NewYork, Date=tomorrow
  - Context: Maybe previous message about budget

**2. Dialogue Management**
- **Purpose**: Manage conversation flow
- **Tasks**:
  - State tracking: Remember what was discussed
  - Response selection: Choose appropriate response
  - Handling follow-ups: Maintain continuity
  
- **Example Conversation Flow**:
  ```
  User: "I want to book a flight"
  Bot: "Where to?" [State: flight booking started]
  User: "New York"
  Bot: "When?" [State: destination confirmed]
  User: "Tomorrow"
  Bot: "Let me search..." [State: searching]
  Bot: "Found 3 flights. Price $300-500" [State: results shown]
  ```

**3. Natural Language Generation (NLG)**
- **Purpose**: Generate human-like responses
- **Tasks**:
  - Response formulation: Create appropriate response
  - Ensure naturalness: Avoid robotic language
  - Maintain consistency: Coherent conversation style

- **Example**:
  - Bad: "FLIGHT_FOUND. PRICE_300_DOLLARS. DEPARTURE_08_00"
  - Good: "Great! I found a perfect flight for you. It's $300 and departs at 8:00 AM."

**Architecture Types (1 mark)**

**1. Retrieval-Based Chatbots**
- Pre-defined response templates
- Selects best matching response from database
- **Process**:
  1. Parse user query
  2. Find matching intent
  3. Retrieve appropriate response
  4. Return response

- **Advantages**: Predictable, safe, fast
- **Disadvantages**: Limited flexibility, cannot generate novel responses

**2. Generative Chatbots**
- Generate responses from scratch using neural networks
- **Process**:
  1. Encode user input using encoder
  2. Decoder generates appropriate response word-by-word
  3. Use Seq2Seq, Transformer, or large language models

- **Advantages**: More flexible, can generate novel responses
- **Disadvantages**: May generate nonsensical responses, hallucinate facts

**3. Hybrid Chatbots**
- Combine retrieval and generation
- Use retrieval for common queries, generation for others
- Better reliability and flexibility

**NLP Techniques Used (1 mark)**

**1. Intent Classification**
- Train classifier: input text → intent class
- Use BERT, RNNs, or other classifiers
- Example classes: BOOK_FLIGHT, CANCEL_ORDER, CHECK_STATUS

**2. Named Entity Recognition (NER)**
- Extract entities: destination, date, price range
- Use sequence labeling (BiLSTM-CRF, BERT-NER)

**3. Semantic Understanding**
- Use word embeddings and contextual representations
- Understand meaning despite different phrasings
- "Book me a flight" vs "I'd like to fly to New York"

**4. Response Generation**
- Seq2Seq: Encoder-decoder model
- Transformers: More sophisticated, better context
- Large Language Models (GPT): Few-shot generation

**Challenges (1.5 marks)**

**1. Ambiguity and Context**
- Problem: Same phrase can mean different things
- Example: "Book a table" (restaurant) vs "Book a flight" (travel)
- Solution: Maintain context, ask clarifying questions

**2. Out-of-Domain Queries**
- Problem: User asks about something bot wasn't trained for
- Example: Chatbot trained for flights gets asked about hotels
- Solution: Graceful fallback, escalation to human

**3. Entity Linking**
- Problem: Identifying what entity refers to
- Example: "How much is it?" - what is "it"?
- Solution: Keep track of conversation state, context

**4. Coherence and Consistency**
- Problem: Response contradicts earlier conversation
- Example: "Book flight tomorrow" then later "your flight is next week"
- Solution: Maintain state machine, validate responses

**5. Factual Accuracy**
- Problem: LLMs hallucinate (make up facts)
- Example: Making up flight prices that don't exist
- Solution: Query databases, verify information

**6. Handling Negation and Conditionals**
- Problem: "Not interested in expensive flights"
- Chatbot needs to understand negation (NOT expensive)
- Solution: Sophisticated parsing or fine-tuned models

**7. Multi-Turn Dialogue**
- Problem: Understanding reference to previous turns
- Example: "The one mentioned earlier" - which one?
- Solution: Dialogue history tracking, context windows

**Real-World Applications (1 mark)**

**1. Customer Service Chatbots**
- Answer FAQs
- Handle complaints
- Process orders
- Example: Amazon's Alexa, Google Assistant

**2. Personal Assistants**
- Schedule meetings
- Set reminders
- Answer general questions
- Example: Siri, Cortana

**3. Specialized Domain Bots**
- Medical diagnosis assistance
- Legal advice
- Financial planning
- Example: Healthcare chatbots, banking chatbots

**4. Social/Entertainment Bots**
- Creative writing assistance
- Gaming companions
- Tutoring systems
- Example: ChatGPT, Character.AI

**Modern Approach: Large Language Models (0.5 marks)**

Recent advances use large pre-trained models:
- **GPT-3/GPT-4**: Few-shot in-context learning
- **LaMDA**: Specialized for conversation
- **PaLM**: Google's latest model

**Advantages**:
- Minimal fine-tuning required
- Few-shot learning capability
- More natural conversations

**Challenges**:
- Factual accuracy issues
- Bias in responses
- Privacy concerns with large models
- Computational cost

---

# PART 4: QUICK REVISION SHEET

## One-Liners for Each Topic

**Seq2Seq**: Encoder processes input to context vector, decoder generates output using context.

**Attention**: Allows decoder to focus on ALL encoder states instead of just context vector.

**Self-Attention**: Each word attends to all words in same sequence; Q, K, V projections.

**Multi-Head Attention**: Multiple parallel attention heads capture different linguistic aspects.

**Transformers**: All-attention architecture without recurrence; parallel processing possible.

**Encoder Layers**: Self-attention + FFN + residual connections; no masking.

**Decoder Layers**: Masked self-attention + cross-attention + FFN + residual connections.

**BERT**: Bidirectional encoder; trained with MLM (predict masked words) and NSP (next sentence).

**GPT**: Unidirectional decoder; autoregressive generation; predicts next word.

**LaMDA**: Dialogue-specialized model; two-stage training (pre-training + fine-tuning).

**Conversational AI**: Combines NLU + dialogue management + NLG.

## Key Formulas

**Attention Score**: eᵢⱼ = sⱼᵀ hᵢ (dot product)

**Softmax Attention Weights**: αᵢⱼ = exp(eᵢⱼ) / Σₖ exp(eₖⱼ)

**Context Vector**: cⱼ = Σᵢ αᵢⱼ × hᵢ

**Scaled Dot-Product**: Attention = softmax(QKᵀ / √d) V

**Multi-Head**: MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) Wᵒ

**Positional Encoding**: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...)

## Common Exam Topics

| Topic | Probability | Key Points |
|-------|-----------|-----------|
| Seq2Seq Machine Translation | 90-95% | Encoder-decoder, context vector, limitations |
| Attention Mechanism | 85-90% | Alignment, softmax, context computation |
| Single vs Multi-Head | 75-85% | Multiple heads capture diverse aspects |
| BERT Architecture | 80% | Bidirectional, MLM, NSP, applications |
| Transformer Model | 80% | Parallel processing, self-attention, positional encoding |
| Seq2Seq + Attention | 75% | Fixes bottleneck, allows focus on relevant input |
| NLP Applications | 75% | Conversational AI, translation, QA |

## Common Mistakes to Avoid

❌ Confusing Seq2Seq without attention with attention mechanism
✓ Clearly differentiate: basic uses context vector; with attention uses weighted sum of all states

❌ Saying Transformers use RNNs
✓ Transformers use ONLY attention; no RNNs involved

❌ Claiming BERT can generate text
✓ BERT is encoder-only; GPT is decoder-only (generates text)

❌ Mixing up bidirectional (BERT) and unidirectional (GPT)
✓ BERT reads both directions; GPT reads left-to-right only

❌ Saying attention has O(m+t) complexity
✓ Attention has O(m×t) complexity (m=input length, t=output length)

❌ Confusing positional encoding with position embeddings
✓ Transformers use sinusoidal PE; BERT uses learned position embeddings

## Important Applications to Remember

1. **Machine Translation**: Seq2Seq, Transformer
2. **Text Classification**: BERT, GPT
3. **Named Entity Recognition**: BERT fine-tuning
4. **Question Answering**: BERT, GPT
5. **Text Summarization**: Seq2Seq, Transformer
6. **Chatbots**: LaMDA, GPT, hybrid models
7. **Sentiment Analysis**: BERT classification
8. **Semantic Search**: BERT embeddings

---

# PART 5: COMPARISON TABLES FOR QUICK REFERENCE

## Architecture Comparison

| Feature | Seq2Seq | Transformer | BERT | GPT |
|---------|---------|-------------|------|-----|
| Components | Encoder-Decoder | Encoder-Decoder | Encoder-Only | Decoder-Only |
| Recurrence | Yes (RNN) | No | No | No |
| Parallelization | Limited | Full | Full | Full |
| Input Processing | Sequential | Parallel | Parallel | Parallel |
| Attention | Basic | Multi-head self | Multi-head self | Masked self |
| Training Task | NMT | Language modeling | MLM + NSP | Language modeling |
| Bidirectional | No | Yes | Yes | No (causal) |
| Best For | Generation/Translation | Both | Understanding | Generation |

## Attention Types

| Type | Formula | Use Case | Complexity |
|------|---------|----------|-----------|
| Additive | vᵀ tanh(Ws sⱼ + Wh hᵢ) | Original attention | O(m×t×d) |
| Multiplicative | sⱼᵀ hᵢ | Luong attention | O(m×t×d) |
| Scaled Dot-Product | (sⱼᵀ hᵢ)/√d | Transformers | O(m×t×d) |
| Self-Attention | (QKᵀ/√d)V | Same sequence | O(n²d) |

---

End of Comprehensive Unit 5 Guide
