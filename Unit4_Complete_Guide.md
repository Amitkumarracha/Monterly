# UNIT 4: RECURRENT NEURAL NETWORKS (RNN)
## Comprehensive Study Guide with Complete Answers & Revision Sheet

---

## TABLE OF CONTENTS
1. Detailed Topic Explanations with Definitions
2. Complete Predicted Questions with Answers
3. Quick Revision Sheet
4. Visual Diagrams & Flowcharts
5. Comparison Tables

---

# PART 1: DETAILED TOPIC EXPLANATIONS

## 1. RECURRENT NEURAL NETWORKS (RNN) - FUNDAMENTALS

### 1.1 Definition & Overview

**Recurrent Neural Network (RNN)** is a type of artificial neural network designed to process sequential data by maintaining an internal state (hidden state/memory) that captures information from previous inputs. Unlike feedforward neural networks that process inputs independently, RNNs use the output from previous time steps as input to current time steps.

**Key Innovation**: The ability to share weights across time steps while maintaining memory of past inputs.

### 1.2 Architecture Components

#### A. **Three Weight Matrices**

**W_x (Input-to-Hidden weight)**: 
- Dimensions: input_size × hidden_size
- Transforms current input to hidden space
- Example: 100-dimensional word embedding → 512-dimensional hidden state

**W_h (Hidden-to-Hidden weight)**:
- Dimensions: hidden_size × hidden_size
- Maintains recurrent connection
- Captures temporal dependencies across time steps
- **Critical for RNN functioning**: Creates memory of past states

**W_y (Hidden-to-Output weight)**:
- Dimensions: hidden_size × output_size
- Transforms hidden state to output space
- Example: 512-dimensional hidden state → vocabulary (50,000 dimensions)

#### B. **Hidden State (Memory)**

**Definition**: A vector that captures information from all previous inputs in the sequence.

**Recurrence Formula**:
```
h_t = tanh(W_x × x_t + W_h × h_{t-1} + b_h)
```

Where:
- h_t = hidden state at time t
- x_t = input at time t
- h_{t-1} = hidden state from previous time step
- tanh = activation function (ranges -1 to 1)
- b_h = bias for hidden layer

**Key Points**:
- h_0 = 0 or random initialization
- Each time step uses SAME weights W_x, W_h, W_y
- Creates temporal connections through h_t → h_{t+1} → h_{t+2}...

#### C. **Output at Each Time Step**

```
y_t = softmax(W_y × h_t + b_y)
```

For language modeling: outputs probability distribution over vocabulary

**Example**: For text generation
- If vocab = {the, cat, sat, ...} (10,000 words)
- y_t = [0.8, 0.05, 0.03, ...] (probabilities for each word)
- Most likely word: "the" with probability 0.8

### 1.3 Forward Propagation (Unfolding in Time)

**Concept**: RNN can be "unfolded" to show the same network repeated across time steps.

**Example: Processing sequence "I love cats"**

```
Time t=0: x_0 = embedding("I")
           h_0 = 0 (initial state)
           h_1 = tanh(W_x × x_0 + W_h × h_0 + b_h)
           y_1 = softmax(W_y × h_1 + b_y)
           Output: probabilities for next word

Time t=1: x_1 = embedding("love")
           h_1 = previous hidden state
           h_2 = tanh(W_x × x_1 + W_h × h_1 + b_h)
           y_2 = softmax(W_y × h_2 + b_y)
           Output: probabilities for next word

Time t=2: x_2 = embedding("cats")
           h_2 = previous hidden state
           h_3 = tanh(W_x × x_2 + W_h × h_2 + b_h)
           y_3 = softmax(W_y × h_3 + b_y)
           Output: probabilities for next word
```

**Key Observation**: Information from "I" (t=0) flows through h_1 → h_2 → h_3, influencing all future predictions.

### 1.4 Applications of RNN

| Task | Use Case | Example |
|------|----------|---------|
| Language Modeling | Predict next word | "The cat ___ on the mat" → predicts "sat" |
| Machine Translation | Sequence-to-sequence | English → German translation |
| Speech Recognition | Audio sequences to text | Audio waveform → text transcription |
| Text Generation | Generate new text | Input: prompt → Output: generated story |
| Sentiment Analysis | Classify sequence | Movie review text → positive/negative |
| Named Entity Recognition | Tag entities in sequence | "John lives in Paris" → PERSON, LOCATION |
| Video Analysis | Temporal analysis | Frame sequence → action recognition |

---

## 2. N-GRAM LANGUAGE MODELS IN CONTEXT OF RNN

### 2.1 Classical N-gram Model

**Definition**: N-gram is a contiguous sequence of N items (typically words) from a text.

#### N-gram Types:

**Unigram (1-gram)**:
- Single word: "The", "cat", "sat"
- P(w) = frequency of word / total words
- No context used

**Bigram (2-gram)**:
- Two consecutive words: "The cat", "cat sat", "sat on"
- P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
- Uses one previous word as context

**Trigram (3-gram)**:
- Three consecutive words: "The cat sat", "cat sat on"
- P(w_i | w_{i-2}, w_{i-1}) = count(w_{i-2}, w_{i-1}, w_i) / count(w_{i-2}, w_{i-1})
- Uses two previous words as context

#### Example: Probability Calculation

**Sentence**: "The cat sat on the mat"

**Bigram Probabilities**:
```
P(cat|The) = count(The, cat) / count(The)
P(sat|cat) = count(cat, sat) / count(cat)
P(on|sat) = count(sat, on) / count(sat)
```

From corpus: If "The" appears 1000 times and "The cat" appears 800 times:
P(cat|The) = 800/1000 = 0.8

### 2.2 Limitations of N-gram Models

1. **Limited Context**: 
   - N-gram can only look back N-1 words
   - Cannot capture long-range dependencies
   - Example: "The dog which was running in the park ___ barked"
     - Can't connect "dog" from many words back without very high N

2. **Sparsity Problem**:
   - For large N, most N-grams never appear in training data
   - P(unseen n-gram) = 0 (probability zero!)
   - Model cannot generalize to new combinations

3. **Exponential Vocabulary**:
   - Bigrams: 10,000² = 100 million possibilities
   - Trigrams: 10,000³ = 1 trillion possibilities
   - Memory and computation become infeasible for large N

4. **No Semantic Understanding**:
   - N-grams are purely statistical
   - Cannot understand word meaning or relationships

### 2.3 RNN Language Models vs N-gram Models

#### How RNN Improves Language Modeling:

**Theoretically Unbounded Context**:
```
RNN with sequence: "The dog which was running in the park barked"

Time steps:
t=1: h_1 encodes "The"
t=2: h_2 encodes "The dog"
t=3: h_3 encodes "The dog which"
...
t=9: h_9 encodes full sequence before "barked"

Hidden state h_9 can theoretically access all previous information!
```

**Single Model**:
- One RNN model handles all sequence lengths
- Same weights W_x, W_h, W_y shared across all time steps
- No need for separate N-gram models

**Continuous Representations**:
- Uses dense word embeddings (100-500 dimensions)
- Captures semantic similarities: "cat" and "dog" embeddings similar
- N-grams: discrete symbols, no semantic connection

#### Probability Calculation:

**N-gram**: 
```
P(sequence) = P(w_1) × P(w_2|w_1) × P(w_3|w_1,w_2) × ...
```

**RNN**:
```
h_t = f(W_x × x_t + W_h × h_{t-1})
P(w_t | h_{t-1}) = softmax(W_y × h_t)
P(sequence) = ∏_t P(w_t | h_{t-1})
```

Both compute sequence probability, but RNN uses continuous hidden states!

### 2.4 N-gram Smoothing Techniques (Relevant to RNN)

**Problem**: Unseen N-grams have probability 0.

**Solutions**:

1. **Add-one (Laplace) Smoothing**:
   - P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)
   - V = vocabulary size
   - Adds small probability to unseen n-grams

2. **Backoff**:
   - If trigram unseen, use bigram: P(w_i | w_{i-2}, w_{i-1}) ≈ P(w_i | w_{i-1})
   - If bigram unseen, use unigram: P(w_i | w_{i-1}) ≈ P(w_i)

3. **Interpolation**:
   - P(w_i | w_{i-2}, w_{i-1}) = λ_3 × P_trigram + λ_2 × P_bigram + λ_1 × P_unigram
   - λ values: 0.2, 0.3, 0.5 (weighted combination)

**RNN Advantage**: LSTM/GRU naturally learns to smooth probabilities through continuous hidden states!

---

## 3. BACKPROPAGATION THROUGH TIME (BPTT)

### 3.1 Basic Concept

**Definition**: Backpropagation Through Time (BPTT) is the algorithm for training RNNs. It extends standard backpropagation to sequential data by:
1. Unfolding RNN across time steps
2. Computing gradients for each time step
3. Summing gradients across all time steps

### 3.2 BPTT Algorithm Steps

#### Step 1: Forward Propagation (Unfolding)
```
For t = 1 to T (sequence length):
    h_t = tanh(W_x × x_t + W_h × h_{t-1} + b_h)
    y_t = softmax(W_y × h_t + b_y)
    loss_t = CrossEntropy(y_t, target_t)

Total Loss = Σ loss_t (sum across all time steps)
```

#### Step 2: Backward Propagation Through Time
```
For t = T down to 1:
    δy_t = y_t - target_t  (output layer error)
    
    ∂Loss/∂W_y = Σ_t δy_t × h_t^T
    ∂Loss/∂b_y = Σ_t δy_t
    
    δh_t = (W_y^T × δy_t) + δh_{t+1}  (backprop error through time)
    δh_t = δh_t ⊙ (1 - h_t²)  (tanh derivative)
    
    ∂Loss/∂W_h = Σ_t δh_t × h_{t-1}^T
    ∂Loss/∂W_x = Σ_t δh_t × x_t^T
    ∂Loss/∂b_h = Σ_t δh_t
```

#### Step 3: Weight Update
```
W_new = W_old - learning_rate × ∂Loss/∂W
```

### 3.3 Unfolding in Time Example

**Sequence**: "cat sat"

**Unfolded Network**:
```
t=1 (input: "cat")          t=2 (input: "sat")
    x_1 (embedding)             x_2 (embedding)
      ↓                           ↓
    [RNN cell] → h_1 → [RNN cell] → h_2
      ↓                           ↓
    [Output] → y_1 (pred: "sat") [Output] → y_2 (pred: "on")
      ↓                           ↓
    Loss_1 (compare with target)  Loss_2
```

**Backward Pass**:
```
Error from Loss_2 flows back: Loss_2 → h_2 → h_1
                                    ↓
                            Updates W_h using error from both t=2 AND t=1
                            
This is different from feedforward!
In feedforward, error only flows backward through layers.
In RNN, error flows backward through TIME (previous time steps).
```

### 3.4 BPTT Complexity

**Computational Cost**:
- Must process entire sequence (T time steps)
- Gradients accumulate across all T steps
- Memory required: O(T × hidden_size)

**Truncated BPTT**:
- Only backprop for last k steps (k << T)
- Reduces computation but loses long-range dependencies
- Trade-off: speed vs. learning capability

---

## 4. VANISHING GRADIENTS PROBLEM

### 4.1 Problem Definition

**Vanishing Gradient Problem**: The gradients during backpropagation become exponentially smaller as they propagate backward through many time steps, eventually approaching zero. This prevents early time steps from being learned.

### 4.2 Mathematical Explanation

#### Chain Rule in RNN:

```
∂Loss/∂h_j = (∂Loss/∂h_T) × (∂h_T/∂h_{T-1}) × ... × (∂h_{j+1}/∂h_j)
           = (∂Loss/∂h_T) × ∏_{t=j+1}^{T} (∂h_t/∂h_{t-1})
```

Where the gradient involves:
```
∂h_t/∂h_{t-1} = W_h^T × diag(1 - h_t²)  [tanh derivative]
```

#### Why Gradient Vanishes:

If we use sigmoid activation: max(σ'(x)) = 0.25

Gradient product:
```
∂Loss/∂h_j ≈ Product of (something < 0.25) repeated T-j times
           ≈ 0.25^(T-j)
           
For T - j = 20: 0.25^20 ≈ 10^{-12} (essentially zero!)
```

**Key Insight**: Multiplying many small gradients (< 1) together creates exponential decay.

### 4.3 Consequences

1. **Long-Range Dependencies Not Learned**:
   - Example: "The bank manager approved my ___ application"
   - Subject "bank manager" is far from "application"
   - Gradient from position 10 back to position 2 vanishes
   - RNN cannot learn this relationship

2. **Early Time Steps Poorly Updated**:
   - Weights connected to early inputs get almost zero gradient update
   - Model cannot learn from the beginning of sequences
   - Cannot remember information from early in sequence

3. **Model Performance**:
   - Perplexity (prediction error) worsens for longer sequences
   - Models struggle with sequences > 10-15 words
   - BLEU scores (translation quality) drop significantly

#### Mathematical Visualization:

```
Gradient Magnitude Over Time Steps (Vanilla RNN)

Gradient Value
    ^
    |
1.0 |●
    |
0.8 | ●
    |
0.5 |   ●
    |
0.1 |       ●
    |
    |           ●●●●●●●● (vanishes to ~0)
    |________________________________→ Time steps back
    0    5     10   15    20
```

### 4.4 Vanishing vs Exploding Gradients

| Gradient Problem | Cause | Effect | Solution |
|-----------------|-------|--------|----------|
| **Vanishing** | Gradient < 1 multiplied repeatedly | Gradient → 0, no learning | LSTM, GRU, Residual connections |
| **Exploding** | Gradient > 1 multiplied repeatedly | Gradient → ∞, NaN/inf | Gradient clipping |

**Exploding Gradient Example**:
```
If ∂h_t/∂h_{t-1} ≈ 1.5 (> 1):
∂Loss/∂h_j ≈ 1.5^(T-j)

For T - j = 20: 1.5^20 ≈ 3 × 10^15 (huge!)

During weight update: w_new = w_old - lr × 3×10^15
                    w_new becomes unreasonably large
                    Can cause NaN (Not a Number) in computation
```

**Solution for Exploding**: Gradient Clipping
```
if ||gradient|| > threshold:
    gradient = gradient × (threshold / ||gradient||)
```

---

## 5. LSTM (LONG SHORT-TERM MEMORY)

### 5.1 Innovation & Motivation

**Problem LSTM Solves**: Vanishing gradient problem of vanilla RNN.

**Key Insight**: Instead of single hidden state, maintain:
- **Cell State** (long-term memory)
- **Hidden State** (short-term output)

Use gates to selectively add/remove information.

### 5.2 LSTM Architecture

#### A. **Four Gates & One Cell**

**1. Forget Gate (f_t)**
```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)

Where σ = sigmoid (output 0-1)
- 0 means "completely forget"
- 1 means "completely retain"
```

**Purpose**: Decide what to discard from cell state.
- Example: In language modeling, "I" (singular) → forget number/gender info when encountering plural noun

**2. Input Gate (i_t)**
```
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)

Candidate values: C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)
```

**Purpose**: Decide what new information to add.
- i_t = which candidates to add
- C̃_t = what values to add

**3. Cell State Update**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
      ↑            ↑  ↑      ↑
      forget old   +  input  new
      info            gate   candidates
```

**Purpose**: Long-term memory update (central to LSTM).

⊙ = element-wise multiplication (Hadamard product)

**4. Output Gate (o_t)**
```
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)

h_t = o_t ⊙ tanh(C_t)
```

**Purpose**: Decide what to output from current cell state.

### 5.3 LSTM Advantages Over Vanilla RNN

#### 1. **Vanishing Gradient Mitigation**

**Forward Flow**:
```
Cell state update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

Gradient flow: ∂C_t/∂C_{t-1} = f_t
                              ≈ 0.5-0.8 (sigmoid output)
                              
NOT raised to power T-j!
Can be close to 1 for many steps!
```

**Why?** Addition operation (not multiplication chain):
```
∂C_t/∂C_{t-1} = f_t        (single multiply, not chain of multiplies)
```

#### 2. **Long-Range Dependencies**

**Example**: "The bank manager who was hired last year ___ approved my application"

- LSTM can preserve information about "manager" through 15+ words
- Cell state acts as "conveyor belt" carrying information
- Forget gate selectively removes irrelevant info
- Input gate adds when needed

#### 3. **Memory Mechanism**

- Cell state: long-term memory (stable over time)
- Hidden state: short-term output (used for predictions)
- Separation allows both stable storage and flexible output

### 5.4 LSTM Mathematics Summary

```
Forward Propagation:
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)          [forget gate]
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)          [input gate]
C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)       [candidate]
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t              [cell state]
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)          [output gate]
h_t = o_t ⊙ tanh(C_t)                         [hidden state]

Output:
y_t = softmax(W_y × h_t + b_y)

Total Parameters:
4 gate matrices × (prev_hidden + input)_dim × hidden_dim
≈ 4 × hidden_dim × (hidden_dim + input_dim)
```

---

## 6. GRU (GATED RECURRENT UNIT)

### 6.1 Overview & Motivation

**GRU**: Simplified variant of LSTM with fewer gates.
- Fewer parameters = faster training
- Comparable performance on many tasks
- Easier to understand and implement

### 6.2 GRU Architecture

#### Two Gates:

**1. Reset Gate (r_t)**
```
r_t = σ(W_r × [h_{t-1}, x_t] + b_r)

Candidate: h̃_t = tanh(W_h × [r_t ⊙ h_{t-1}, x_t] + b_h)
```

**Purpose**: 
- How much of previous hidden state to forget
- r_t = 0: completely reset (ignore h_{t-1})
- r_t = 1: completely keep (use h_{t-1})

**2. Update Gate (z_t)**
```
z_t = σ(W_z × [h_{t-1}, x_t] + b_z)

h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
      ↑         ↑              ↑
      new   *   contribution + old * contribution
```

**Purpose**:
- How much of new candidate vs. previous hidden state to use
- z_t = 0: use new candidate fully
- z_t = 1: keep previous hidden state

### 6.3 GRU vs LSTM Comparison

| Feature | LSTM | GRU |
|---------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (reset, update) |
| **Memory** | Cell state + hidden state | Single hidden state |
| **Parameters** | ~4 × (hidden × (input + hidden)) | ~3 × (hidden × (input + hidden)) |
| **Computation** | Slower (more gates) | Faster (fewer gates) |
| **Gradient Flow** | Through cell state | Through single state |
| **Long Sequences** | Excellent | Very good |
| **Complex Tasks** | Better | Comparable |

### 6.4 Performance Comparison

**Research Findings**:
- **Low-complexity sequences**: GRU performs as well as LSTM
- **High-complexity sequences**: LSTM slightly better (more modeling capacity)
- **Speed**: GRU ~10-15% faster training
- **Memory**: GRU uses ~25% less memory

**Recommendation**:
- Start with GRU (simpler, faster)
- Use LSTM if GRU insufficient performance
- For very long sequences (>500 steps): prefer LSTM

### 6.5 GRU Equations

```
r_t = σ(W_r × [h_{t-1}, x_t] + b_r)
z_t = σ(W_z × [h_{t-1}, x_t] + b_z)
h̃_t = tanh(W_h × [r_t ⊙ h_{t-1}, x_t] + b_h)
h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
```

---

# PART 2: PREDICTED EXAM QUESTIONS FOR UNIT 4

## QUESTION 1: RNN Architecture & Language Models (PROBABILITY: 90%)

### Question:
"Explain the Recurrent Neural Network (RNN) architecture. How does RNN maintain memory and model sequences? Discuss how RNN language models differ from N-gram models."

### Complete Answer:

**Introduction (1 mark)**
Recurrent Neural Networks (RNNs) are neural networks specifically designed for processing sequential data. Unlike feedforward networks that process inputs independently, RNNs maintain an internal hidden state that acts as memory, allowing them to capture temporal dependencies and patterns in sequences.

**RNN Architecture (1.5 marks)**

**Key Components**:

1. **Input Layer**: 
   - Receives input at each time step: x_t
   - Converted to embeddings: typically 100-500 dimensions
   - Example: word → 256-dimensional vector

2. **Hidden Layer (Memory)**:
   - Maintains hidden state: h_t
   - Recurrent formula: h_t = tanh(W_x × x_t + W_h × h_{t-1} + b_h)
   - Three weight matrices:
     * W_x: input-to-hidden (input_dim × hidden_dim)
     * W_h: hidden-to-hidden (hidden_dim × hidden_dim) - recurrent connection
     * b_h: bias

3. **Output Layer**:
   - Generates output: y_t = softmax(W_y × h_t + b_y)
   - W_y: hidden-to-output (hidden_dim × output_dim)
   - b_y: bias

**How RNN Maintains Memory (1.5 marks)**

**Sequential Processing**:
```
Time 1: Input: "I"          → h_1 encodes "I"
Time 2: Input: "love"       → h_2 encodes "I love"
Time 3: Input: "reading"    → h_3 encodes "I love reading"
```

**Hidden State Propagation**:
- h_{t-1} → h_t → h_{t+1} creates temporal connections
- Information from time 1 flows through h_1 → h_2 → h_3
- Early context can influence all future predictions
- Single weight matrix W_h shared across all time steps

**Example: Predicting "books"**
```
h_3 (hidden state after "I love reading")
contains information about:
- Subject: "I" (who is doing action)
- Verb: "love" (what action)
- Object: "reading" (what being loved)

Used to predict next word: probably "books" since "love reading books" is common
```

**Unfolding Representation**:
```
x_1→[RNN]→h_1→[RNN]→h_2→[RNN]→h_3
      ↓        ↓        ↓
     y_1      y_2      y_3
```

**RNN vs N-gram Models (2 marks)**

**N-gram Model Approach**:
- "The cat" (bigram) uses only previous 1 word
- "The cat sat" (trigram) uses previous 2 words
- Fixed window size N
- Formula: P(w_t | w_{t-N+1}, ..., w_{t-1})

**Limitations of N-gram**:
1. Cannot look back beyond N words
2. Sparsity: higher N → more unseen combinations
3. Discrete symbols: no semantic similarity
4. Example: "The dog which was running in the park ___ barked"
   - Trigram can't connect "dog" (far back) to verb

**RNN Language Model**:
- Uses entire history: theoretically unbounded context
- Continuous hidden states: semantic relationships preserved
- Single model: handles all sequence lengths
- Formula: P(w_t | h_{t-1}) where h_{t-1} = f(x_1, x_2, ..., x_{t-1})

**Advantages of RNN**:
1. **Long-range dependencies**: Information flows through h_t
2. **Parameter sharing**: Same W_x, W_h, W_y across all time steps
3. **Semantic understanding**: Embeddings capture meaning
4. **Flexible length**: Handles variable-length sequences

**Comparison Table** (1 mark):

| Aspect | N-gram | RNN |
|--------|--------|-----|
| Context | Fixed (N-1 words) | Theoretically unbounded |
| Memory | Frequency counts | Hidden state vector |
| Parameters | O(V^N) grows exponentially | O(hidden² + hidden×vocab) fixed |
| Sequence Length | Limited by N | Flexible |
| Semantic Info | None (discrete) | Yes (embeddings) |
| Performance | Good for short sequences | Better for longer sequences |

---

## QUESTION 2: Backpropagation Through Time (BPTT) (PROBABILITY: 85%)

### Question:
"Explain Backpropagation Through Time (BPTT) algorithm. How does BPTT differ from standard backpropagation in feedforward networks? Draw the computational graph."

### Complete Answer:

**Introduction (0.5 marks)**
Backpropagation Through Time (BPTT) is the algorithm for training RNNs. It extends standard backpropagation by propagating errors not only backward through layers but also backward through time steps.

**Key Difference from Standard Backpropagation (1 mark)**

**Feedforward Networks**:
```
Error propagates through layers:
Input → Hidden1 → Hidden2 → Output
                ↑
         Errors flow backward through layers only
```

**RNN with BPTT**:
```
t=1: x_1 →[RNN]→ h_1 → y_1
           ↓  ↑      ↓
t=2: x_2 →[RNN]→ h_2 → y_2
           ↓  ↑      ↓
t=3: x_3 →[RNN]→ h_3 → y_3

Errors flow BOTH backward through layers AND backward through time
```

**Critical Difference**:
- Feedforward: errors flow one direction (backward through layers)
- RNN: errors flow backward through TIME as well (h_3 → h_2 → h_1)

**BPTT Algorithm Steps (2 marks)**

**Step 1: Forward Propagation (Unfold)**
```
For t = 1 to T:
    h_t = tanh(W_x × x_t + W_h × h_{t-1} + b_h)
    y_t = softmax(W_y × h_t + b_y)
    loss_t = CrossEntropy(y_t, target_t)

Total Loss = Σ loss_t
```

**Step 2: Backward Pass (Truncated BPTT)**
```
For t = T down to 1:
    
    // Error at output
    δy_t = y_t - target_t
    
    // Gradient for output layer
    ∂Loss/∂W_y += δy_t × h_t^T
    ∂Loss/∂b_y += δy_t
    
    // Error propagates to hidden state
    δh_t = (W_y^T × δy_t) + δh_{t+1}
    
    // Apply tanh derivative: (1 - h_t²)
    δh_t = δh_t ⊙ (1 - h_t²)
    
    // Gradients for recurrent weights
    ∂Loss/∂W_h += δh_t × h_{t-1}^T
    ∂Loss/∂W_x += δh_t × x_t^T
    ∂Loss/∂b_h += δh_t
```

**Step 3: Weight Update**
```
W_new = W_old - learning_rate × ∂Loss/∂W
```

**Key Equations (1.5 marks)**

**Chain Rule for Gradient through Time**:
```
∂Loss/∂h_j = (∂Loss/∂h_T) × (∂h_T/∂h_{T-1}) × ... × (∂h_{j+1}/∂h_j)

Product of gradients from t=j+1 to T
```

**Hidden State Gradient Dependency**:
```
Error from time T propagates back:
Loss_T affects h_T
h_T is function of h_{T-1}, so affects gradient of h_{T-1}
h_{T-1} is function of h_{T-2}, so affects gradient of h_{T-2}
...

Chain of dependencies through time
```

**Unfolding and Weight Sharing**:
- Same W_h used at ALL time steps
- Gradient for W_h: sum of gradients from ALL time steps
```
∂Loss/∂W_h = Σ_{t=1}^T ∂loss_t/∂W_h
           = ∂loss_1/∂W_h + ∂loss_2/∂W_h + ... + ∂loss_T/∂W_h
```

**Truncated BPTT vs Full BPTT (0.5 marks)**

**Full BPTT**:
- Backprop through all T time steps
- Captures long-range dependencies
- High computational cost: O(T × hidden_size²)

**Truncated BPTT**:
- Only backprop through last k steps (e.g., k=50)
- Faster training
- Loses dependencies beyond k steps

**Computational Graph Example** (1 mark):

```
For sequence "cat sat on":

Forward:
x_1(cat) → [RNN] → h_1 → [Dense] → y_1(pred: sat)
                    ↓
x_2(sat) → [RNN] → h_2 → [Dense] → y_2(pred: on)
                    ↓
x_3(on) → [RNN] → h_3 → [Dense] → y_3(pred: mat)

Backward (BPTT):
Loss_3 ↓
    δy_3 ← y_3 - target_3
        ↓
    δh_3 ← W_y^T × δy_3 (+ error from h_4 if exists)
        ↓
    ∂Loss/∂W_h += δh_3 × h_2^T
    
Error propagates to h_2:
    δh_2 ← (W_y^T × δy_2) + (W_h^T × δh_3)
        ↓
    ∂Loss/∂W_h += δh_2 × h_1^T

Error propagates to h_1:
    δh_1 ← (W_y^T × δy_1) + (W_h^T × δh_2)
        ↓
    ∂Loss/∂W_h += δh_1 × h_0^T
    
Final: ∂Loss/∂W_h combines ALL time steps!
```

**Advantages of BPTT** (0.5 marks):
- Captures temporal dependencies
- Enables learning from long sequences
- Efficient gradient calculation
- Foundation for training LSTM, GRU

---

## QUESTION 3: Vanishing Gradient Problem (PROBABILITY: 85%)

### Question:
"Explain the vanishing gradient problem in RNNs. Why does it occur? What are its consequences? How can it be addressed?"

### Complete Answer:

**Definition (0.5 marks)**
The vanishing gradient problem occurs when gradients during backpropagation become exponentially smaller as they propagate backward through many time steps in an RNN, making it difficult or impossible for the network to learn long-range dependencies.

**Why It Occurs (1.5 marks)**

**Mathematical Foundation**:

Gradient through time for weight W_h:
```
∂Loss/∂h_j = Σ_t (∂Loss/∂y_t) × (∂y_t/∂h_t) × (∂h_t/∂h_{t-1}) × ... × (∂h_{j+1}/∂h_j)
           = Σ_t (...) × ∏_{i=j+1}^t (∂h_i/∂h_{i-1})
```

Each term ∂h_i/∂h_{i-1} involves:
```
∂h_i/∂h_{i-1} = W_h^T × diag(1 - h_i²)  [tanh derivative]
```

**Problem**:
1. Tanh derivative: max(1 - tanh(x)²) = 1, typical values ~0.1-0.25
2. Sigmoid would give: max(σ'(x)) = 0.25

Product over many steps:
```
∏_{i=j+1}^t (∂h_i/∂h_{i-1}) ≈ (0.25)^{t-j}

For t - j = 20: (0.25)^20 ≈ 10^{-12}  (essentially zero!)
```

**Visual Representation**:
```
Gradient flow through 20 time steps:

Time 1  Time 5      Time 10    Time 15  Time 20
 |G_0|  |G_1| = |G_0| × 0.25   |G_2| = |G_1| × 0.25    ... |G_20| ≈ 0
 
 1.0      0.25        0.06        0.015   0.0000000001
          ↓           ↓           ↓
        Decays exponentially!
```

**Eigenvalue Analysis**:

The recurrent weight matrix W_h has eigenvalues. If largest eigenvalue < 1:
```
||∏ ∂h_i/∂h_{i-1}|| ≤ (λ_max)^{t-j}

With λ_max < 1: gradient shrinks exponentially
```

**Consequences (1 mark)**

**1. Cannot Learn Long-Range Dependencies**:

Example: "The bank manager who was hired last year and trained extensively by the senior staff ___ approved my application"

The word "bank" is 20+ positions before "approved". 

- RNN cannot connect subject "bank" to verb "approved"
- Gradient from position 25 to position 3 ≈ 0
- Weights related to early tokens don't update
- Model forgets beginning of sentence

**2. Poor Sequence Modeling**:

For sequence "I love reading books about psychology"
```
t=1: h_1 captures "I"
     Gradient for w_h(t=1): ∂Loss/∂w_h(t=1) ≈ 10^-10
     
Update: w_h = w_h - lr × 10^{-10}
        w_h essentially unchanged!
        
Model cannot learn this example properly
```

**3. Language Modeling Performance**:

```
Perplexity (prediction error) vs Sequence Length:

Perplexity
    ^
 40 |●
    | ●●
 30 | ●●●
    |●●●●●●  Vanilla RNN
 20 |●●●●●●●●●●●●
 10 |
    |_____________________________→
    0   10   20   30   40   50
              Sequence Length
              
Performance degrades significantly for longer sequences
```

**Solutions (2 marks)**

**Solution 1: LSTM (Long Short-Term Memory)**

Mitigates through different update mechanism:
```
Cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

Gradient through cell state: ∂C_t/∂C_{t-1} = f_t

Key difference: ADDITION allows gradient of 1 × f_t
                (f_t ≈ 0.5-0.9 from sigmoid)
                
Instead of MULTIPLICATION of many small terms!

∂C_t/∂C_{t-20} = ∏_{i=1}^{20} f_i ≈ 0.7^20 (still reasonable!)
                 NOT 0.25^20 (10^-12)
```

**Solution 2: GRU (Gated Recurrent Unit)**

Similar concept with fewer gates, but same principle:
```
Update gate z_t controls information flow
h_t = (1-z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}

Addition allows better gradient flow
```

**Solution 3: Gradient Clipping**

For vanilla RNNs when gradients explode:
```
if ||gradient|| > threshold:
    gradient = gradient × (threshold / ||gradient||)
    
Prevents gradient explosion but doesn't solve vanishing
```

**Solution 4: Residual Connections**

```
h_t = f(h_{t-1}) + h_{t-1}

Gradient can flow directly: identity function
∂h_t/∂h_{t-1} contains identity term (doesn't vanish)

Example: ∂h_10/∂h_0 has additive paths
(not purely multiplicative paths)
```

**Solution 5: Different Activation Functions**

ReLU instead of tanh:
```
ReLU'(x) = 1 (for x > 0)
NOT 0.25 like sigmoid

Gradient doesn't shrink as fast
But can have other problems (exploding gradients more likely)
```

**Summary Table** (0.5 marks):

| Aspect | Vanilla RNN | LSTM/GRU | Residual | Clipping |
|--------|-------------|----------|----------|----------|
| Vanishing | ✗ severe | ✓ solved | ✓ mitigated | ✗ doesn't help |
| Exploding | ✓ can happen | ✓ possible | - | ✓ prevents |
| Complexity | Simple | Higher | Moderate | Low |
| Long sequences | Poor | Excellent | Good | Limited help |

---

## QUESTION 4: LSTM Architecture & Gates (PROBABILITY: 85%)

### Question:
"Explain the architecture of LSTM (Long Short-Term Memory) in detail. Describe the roles of forget gate, input gate, and output gate. How does LSTM overcome the vanishing gradient problem?"

### Complete Answer:

**Introduction (0.5 marks)**
LSTM (Long Short-Term Memory) is a variant of RNN designed to capture long-range dependencies in sequences. It addresses the vanishing gradient problem through a sophisticated gating mechanism and maintains two states: cell state (long-term memory) and hidden state (short-term output).

**LSTM Components (2 marks)**

**1. Cell State (C_t) - Long-Term Memory**:
```
Purpose: Store information over many time steps
Dimensions: (batch_size, hidden_dim)
Analogy: "Conveyor belt" carrying information through time
```

**2. Hidden State (h_t) - Short-Term Output**:
```
Purpose: Output for current time step
Dimensions: (batch_size, hidden_dim)
Used to: generate predictions, feed to next layer
```

**3. Forget Gate (f_t)**:
```
Formula:
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)

where σ = sigmoid function (output range: 0 to 1)
      W_f: learned weights for forget gate
      [h_{t-1}, x_t]: concatenation of previous hidden state and current input
      b_f: bias

Output: Vector of values between 0 and 1
```

**Role**: Decides what information to discard from cell state.
- f_t = 0: "completely forget"
- f_t = 1: "completely retain"
- f_t ≈ 0.5: "partial retention"

**Example - Language Modeling**:
```
Sentence: "I is hungry" (grammatically incorrect but illustrative)

At word "I" (singular):
- Store number info: singular, gender: unknown
- Cell state includes: {number: singular, ...}

At word "is" (singular verb):
- Confirm: number is singular

At word "we" (plural):
- Forget gate should output ~0 for number feature
- Forget gate: 0 × singular = 0 (forget old number info)
- Input gate: add new number info (plural)
```

**4. Input Gate (i_t) & Candidate (C̃_t)**:
```
Input gate:
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)

Candidate values:
C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)

Role:
i_t determines which candidate values to add
C̃_t provides candidate values to add
```

**Purpose**: Decide what new information to add to cell state.
- When i_t = 0: don't add anything
- When i_t = 1: add candidate fully
- tanh output: -1 to 1 range, provides rich candidate values

**Example**:
```
At word "apple":
- New information: fruit, color: red?, round?
- Input gate: high value (yes, add this info)
- Candidate: [fruit: yes, color: ??, shape: round]
- Addition: cell_state += input_gate * candidate
```

**5. Cell State Update**:
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
      ↑            ↑  ↑      ↑
     forget  *  old   +  input * new
     gate    state      gate   candidates

⊙ = element-wise multiplication (Hadamard product)
```

**Graphically**:
```
         f_t
         ↓
    C_{t-1} —→ [×] —→
                        ↓
                    [+] ← C_t (output)
                        ↑
                   [×] ←—
                   ↑ i_t
              C̃_t
```

**6. Output Gate (o_t)**:
```
Formula:
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)

Output:
h_t = o_t ⊙ tanh(C_t)
```

**Role**: Decide what information from cell state to output as hidden state.
- o_t = 0: don't output anything
- o_t = 1: output everything
- tanh(C_t): scale cell state to -1 to 1 range

**Example**:
```
Cell state: [hungry: yes, sentiment: positive]

At current time, we might NOT want to output sentiment
- Output gate learns: output only content feature, not sentiment
- o_t ≈ [0.8, 0.1] (high for content, low for sentiment)
- h_t ← combines cell state and gate, outputs content-focused representation
```

**How LSTM Overcomes Vanishing Gradient (1.5 marks)**

**Key Insight: Additive Update**

Vanilla RNN:
```
h_t = f(h_{t-1})  [composition]
∂h_t/∂h_{t-1} = f'(h_{t-1})  [product chain]

Over many steps: (f'(...))^T ≈ 0.25^T (vanishes!)
```

LSTM:
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t  [addition]
∂C_t/∂C_{t-1} = f_t  [NOT a product chain!]

Key: Forget gate output ≈ 0.5-0.9 (sigmoid)
Over 20 steps: 0.7^20 ≈ 0.0008 (much better than 0.25^20 ≈ 10^-12!)
```

**Mathematical Comparison**:

```
Gradient flow through time:

Vanilla RNN:
∂C_20/∂C_0 = ∂C_20/∂C_19 × ∂C_19/∂C_18 × ... × ∂C_1/∂C_0
           ≈ 0.25^20 ≈ 10^{-12}  (vanishes!)

LSTM:
∂C_20/∂C_0 = f_20 × f_19 × ... × f_1  (independent multiplications)
           ≈ 0.7^20 ≈ 0.0008  (survives!)
```

**Why Addition Helps**:

Recall chain rule: ∂(A + B)/∂x = ∂A/∂x + ∂B/∂x

In LSTM: C_t has additive path C_{t-1}
```
∂C_t/∂C_{t-1} includes:
1. Direct multiplicative path: f_t
2. Implicit additive contribution

This prevents pure exponential decay
```

**Gradient Stability Over Time**:

```
Gradient Magnitude

Vanilla RNN:        LSTM:
    ↓               ↓
1.0 |●              1.0 |●
    |  ●                |  ●
0.5 |    ●              0.5 |    ●
    |       ●               |      ●●●●●●  (stays reasonable!)
0   |...→0 ↓→∞ (unstable)   0 |________→ (stable)
    0  10  20            0  10  20
```

**LSTM Sequence of Events** (0.5 marks):

For input sequence "The cat sat":

```
t=1: word="The", x_1=embedding("The")
    - Forget: f_1 = σ(...)  What to forget?
    - Input: i_1 = σ(...), C̃_1 = tanh(...)  What to add?
    - C_1 = f_1 ⊙ 0 + i_1 ⊙ C̃_1  (first step, C_0=0)
    - Output: o_1 = σ(...)
    - h_1 = o_1 ⊙ tanh(C_1)

t=2: word="cat", x_2=embedding("cat")
    - Forget: f_2 = σ([h_1, x_2])  Forget what about "The"?
    - Input: i_2 = σ([h_1, x_2])  Add info about "cat"?
    - C_2 = f_2 ⊙ C_1 + i_2 ⊙ C̃_2  (retain some old, add new)
    - Output: o_2 = σ([h_1, x_2])
    - h_2 = o_2 ⊙ tanh(C_2)

t=3: word="sat", x_3=embedding("sat")
    - Similar process
    - C_3 can still "remember" information from C_1 through f gates!
```

---

## QUESTION 5: GRU vs LSTM (PROBABILITY: 75%)

### Question:
"Compare and contrast GRU (Gated Recurrent Unit) and LSTM architectures. What are the advantages and disadvantages of each? When would you choose one over the other?"

### Complete Answer:

**Introduction (0.5 marks)**
Both GRU and LSTM are gating mechanisms designed to solve the vanishing gradient problem in RNNs. While LSTM maintains separate cell and hidden states with three gates, GRU uses a single state with two gates. Both achieve similar performance with trade-offs in complexity and efficiency.

**GRU Architecture (1 mark)**

**1. Reset Gate (r_t)**:
```
r_t = σ(W_r × [h_{t-1}, x_t] + b_r)

Purpose: Decide what percentage of previous hidden state to use
- r_t = 0: ignore h_{t-1} completely
- r_t = 1: use h_{t-1} fully
```

**2. Candidate Hidden State (h̃_t)**:
```
h̃_t = tanh(W_h × [r_t ⊙ h_{t-1}, x_t] + b_h)

Note: Uses r_t ⊙ h_{t-1}, not h_{t-1}
This is where reset gate "resets" previous information
```

**3. Update Gate (z_t)**:
```
z_t = σ(W_z × [h_{t-1}, x_t] + b_z)

Purpose: Decide how much new candidate to use vs. keep old state
- z_t = 0: use new candidate fully
- z_t = 1: keep previous hidden state
```

**4. Hidden State Update**:
```
h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
      ↑         ↑              ↑
      weight    new        weight old
      new       candidate   state
```

**Comparison Table (1.5 marks)**

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (reset, update) |
| **States** | Cell state + hidden state | Single hidden state |
| **Parameters** | ~4× hidden×(hidden+input) | ~3× hidden×(hidden+input) |
| **Computation** | Slower (more gates) | Faster |
| **Memory** | More (separate cell) | Less |
| **Gradient Flow** | Through cell state | Through single state |
| **Complexity** | More complex | Simpler |
| **On Simple Sequences** | Good | Comparable/better |
| **On Complex Sequences** | Slightly better | Slightly worse |
| **Training Time** | Longer | Shorter (~10-15% faster) |

**Advantages & Disadvantages (2 marks)**

**LSTM Advantages**:
1. More modeling capacity (3 gates vs 2)
2. Separate cell state for long-term memory
3. Better for very complex/long-range dependencies
4. More fine-grained control over information flow
5. Proven on many standard benchmarks

**LSTM Disadvantages**:
1. More parameters (3× vs 2× for GRU)
2. Slower training (more computations)
3. More memory required during training/inference
4. Harder to interpret (4 gates per unit)
5. Overkill for simple tasks

**GRU Advantages**:
1. Fewer parameters (simpler model)
2. Faster training (10-15% speedup)
3. Lower memory footprint
4. Easier to implement and understand
5. Performs comparably to LSTM on many tasks

**GRU Disadvantages**:
1. Less modeling capacity (fewer gates)
2. Single hidden state might be insufficient for very long dependencies
3. Cannot have separate long-term/short-term memory
4. May underperform on highly complex tasks
5. Less commonly used in production

**When to Choose What (1 mark)**

**Choose GRU if**:
- Computational resources are limited
- Training time is a concern
- Working with relatively small datasets
- Sequences are relatively short/simple
- Need to deploy on edge devices
- Initial experimentation needed

**Choose LSTM if**:
- Ample computational resources available
- Targeting maximum performance
- Tasks involve very long-range dependencies (>100 steps)
- Complex patterns in data
- Standard benchmarks/published baselines use LSTM
- Production system requires reliability

**Research Findings (1 mark)**:

From empirical studies:
```
1. On low-complexity sequences:
   GRU and LSTM perform similarly
   GRU can even outperform LSTM slightly

2. On high-complexity sequences:
   LSTM typically better (more capacity)
   Performance difference: 1-3%

3. Training efficiency:
   GRU: 10-15% faster
   Memory: GRU uses ~25% less

4. Recommendation by researchers:
   "Start with GRU for simplicity
   Switch to LSTM if needed for performance"
```

**Practical Example** (0.5 marks):

**Task 1: Sentiment analysis on movie reviews (short texts)**
```
Review: "Good movie, great actors" (5-10 words)
GRU is sufficient: captures positive sentiment
Choice: GRU ✓
```

**Task 2: Machine translation (long sentences)**
```
Source: "The dog which was running in the park 
         and was barking at the cat sat down"
(30+ words, complex grammar)

Needs to understand:
- Long-range subject-verb agreement
- Nested clauses
- Complex dependencies

Choice: LSTM ✓
```

---

# PART 3: QUICK REVISION SHEET FOR UNIT 4

---

## UNIT 4 QUICK REVISION SHEET
## Recurrent Neural Networks (RNN)

---

### 1. RNN - ONE-LINER DEFINITIONS

**RNN**: Neural network with recurrent connections that maintains hidden state (memory) to process sequential data.

**Hidden State**: h_t = tanh(W_x×x_t + W_h×h_{t-1} + b_h) - carries information from all previous inputs.

**Unfolding**: Unwrapping RNN across time steps to visualize how same weights repeat across time.

**BPTT**: Backpropagation Through Time - training algorithm that propagates errors backward through time.

**Vanishing Gradient**: Gradients become exponentially smaller (~0.25^T) as backprop goes through time, preventing learning.

**LSTM**: Long Short-Term Memory - uses 4 gates and cell state to prevent vanishing gradients.

**GRU**: Gated Recurrent Unit - simplified LSTM with 2 gates instead of 3.

---

### 2. KEY FORMULAS

```
RNN Hidden State:
h_t = tanh(W_x × x_t + W_h × h_{t-1} + b_h)

RNN Output:
y_t = softmax(W_y × h_t + b_y)

Cell State Update (LSTM):
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

LSTM Forget Gate:
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)

LSTM Input Gate:
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)

LSTM Output Gate:
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)

GRU Reset Gate:
r_t = σ(W_r × [h_{t-1}, x_t] + b_r)

GRU Update Gate:
z_t = σ(W_z × [h_{t-1}, x_t] + b_z)
h_t = (1-z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}

N-gram Probability:
P(w_n | w_{n-N+1}...w_{n-1})

Gradient Through Time:
∂Loss/∂h_j = Product of ∂h_t/∂h_{t-1} for t=j+1 to T
```

---

### 3. COMPARISON TABLE - ARCHITECTURES

| Feature | Vanilla RNN | LSTM | GRU |
|---------|------------|------|-----|
| Hidden State | Single | Cell+Hidden | Single |
| Gates | 0 | 3 | 2 |
| Vanishing Gradient | ✗ Yes | ✓ No | ✓ No |
| Long Dependencies | ✗ Poor | ✓ Excellent | ✓ Good |
| Parameters | Least | Most | Medium |
| Speed | Fastest | Slowest | Fast |
| Memory | Least | Most | Medium |
| Performance (Simple) | OK | Good | Better |
| Performance (Complex) | Bad | Best | Good |

---

### 4. LSTM vs GRU QUICK GUIDE

**LSTM**:
- 3 gates: forget (what to forget), input (what to add), output (what to output)
- 2 states: cell state (long-term memory), hidden state (short-term output)
- Better: complex tasks, long sequences, maximum performance
- Slower: more parameters, more computation

**GRU**:
- 2 gates: reset (which part of h_{t-1} to use), update (how much new vs old)
- 1 state: hidden state (combined short+long term)
- Better: efficiency, speed, resource-constrained settings
- Simpler: easier to implement, understand, debug

**Decision**:
- Start with GRU (simpler, faster)
- Use LSTM if GRU insufficient

---

### 5. VANISHING GRADIENT - KEY POINTS

**Problem**: 
- Gradient multiplied by f'(x) ≈ 0.25 (sigmoid) repeatedly
- Product: 0.25^20 ≈ 10^{-12} (vanishes!)
- Weight updates for early time steps ≈ 0
- Cannot learn long-range dependencies

**Consequence**:
- "The dog which was running ... barked" → can't connect "dog" and "barked"
- Perplexity increases for longer sequences
- Model forgets beginning of sequences

**Solutions**:
1. LSTM/GRU (4 alternatives to multiplicative chain)
2. Gradient Clipping (prevent explosion, not vanishing)
3. Residual Connections (allow gradient flow)
4. Different Activations (ReLU: f'=1)

---

### 6. N-GRAM LANGUAGE MODELS

**Definition**: Sequence of N consecutive items (words/characters).

**Types**:
- Unigram (1-gram): "cat", "dog", "sat"
- Bigram (2-gram): "the cat", "cat sat"
- Trigram (3-gram): "the cat sat"

**Limitation vs RNN**:

| Aspect | N-gram | RNN |
|--------|--------|-----|
| Context | Fixed (N-1 words) | Theoretically unbounded |
| Sparsity | High (vocabulary^N) | None (fixed parameters) |
| Semantics | None (discrete) | Yes (embeddings) |
| Long sequences | Fails | Succeeds |

**Why RNN Better**:
- Theoretically unbounded context through h_t → h_{t+1}
- Continuous embeddings capture meaning
- Same parameters for any sequence length
- Natural handling of long-range dependencies (if using LSTM/GRU)

---

### 7. BPTT ALGORITHM - SIMPLIFIED

```
1. FORWARD PASS:
   For each time step t:
       h_t = f(x_t, h_{t-1})
       y_t = softmax(W_y × h_t)
       loss_t = CE(y_t, target_t)

2. BACKWARD PASS (from T to 1):
   Error propagates BACKWARD THROUGH TIME
   δh_t comes from:
   - Error at y_t: (W_y^T × δy_t)
   - Error from future: (W_h^T × δh_{t+1})
   
   These combine: δh_t = (W_y^T × δy_t) + (W_h^T × δh_{t+1})

3. GRADIENT ACCUMULATION:
   ∂Loss/∂W_h = Σ_t (δh_t × h_{t-1}^T)
   ∂Loss/∂W_x = Σ_t (δh_t × x_t^T)
   ∂Loss/∂W_y = Σ_t (δy_t × h_t^T)

4. WEIGHT UPDATE:
   W_new = W_old - learning_rate × ∂Loss/∂W
```

**Key Difference from Feedforward**:
- Feedforward: error flows backward through layers only
- RNN: error flows backward through LAYERS AND TIME

---

### 8. LSTM GATE ROLES - MNEMONICS

**Forget Gate** (f_t):
- "Should I forget this info?"
- Output: 0 = forget, 1 = keep
- Example: New pronoun → forget old gender info

**Input Gate** (i_t) + Candidate (C̃_t):
- "What new info should I add?"
- i_t: which candidate values
- C̃_t: what values to add
- Example: See new noun → add gender/number info

**Cell State** (C_t):
- "Long-term memory"
- Updated: C_t = forget×old + input×new
- Flows through time with minimal gradient loss

**Output Gate** (o_t):
- "What info should I output?"
- Output: 0 = hide, 1 = show
- Example: Don't output intermediate calculations to next layer

---

### 9. COMMON EXAM PATTERNS - HIGH PROBABILITY

| Topic | Probability | Mark |
|-------|-----------|------|
| RNN Architecture + Language Models | 90% | 6 |
| BPTT Algorithm | 85% | 5-6 |
| Vanishing Gradient Problem | 85% | 6 |
| LSTM Architecture | 85% | 5-6 |
| GRU vs LSTM | 75% | 5-6 |
| N-gram Models vs RNN | 70% | 4-5 |
| Backpropagation Comparison | 65% | 4 |

---

### 10. MISTAKES TO AVOID

❌ Claiming RNN weight matrices are different at each time step
✓ Same W_x, W_h, W_y shared across ALL time steps

❌ Saying LSTM has 2 hidden states
✓ LSTM has cell state (long-term) + hidden state (short-term output)

❌ BPTT is same as backpropagation
✓ BPTT includes temporal backpropagation through previous time steps

❌ Vanishing gradient can be solved by gradient clipping
✓ Clipping prevents EXPLOSION; LSTM/GRU solve VANISHING

❌ LSTM always better than GRU
✓ GRU performs comparably, faster, on many tasks

❌ N-gram can capture arbitrarily long dependencies
✓ N-gram limited to N-1 word context; sparsity problem

❌ RNN hidden state has no limit on information
✓ Despite theoretical unbounded context, gradient issues limit practical range

---

### 11. QUICK STUDY TIPS

**Day 1: Understand RNN Basics**
- Master h_t = f(x_t, h_{t-1}) formula
- Visualize unfolding across time
- Understand weight sharing

**Day 2: Learn About Problems**
- Vanishing gradient: 0.25^T becomes tiny
- Why N-grams insufficient
- What BPTT does differently

**Day 3: Master LSTM/GRU**
- Forget gate: what to forget
- Input gate + candidate: what to add
- Output gate: what to output
- Cell state: protected memory

**Day 4: Practice Questions**
- Full RNN architecture explanation
- LSTM gate functions
- Vanishing gradient causes/solutions
- BPTT algorithm steps

**Day 5: Last Revision**
- Review all formulas
- Comparison tables
- One-liners for each topic

---

### 12. FORMULA SHEET - QUICK REFERENCE

```
BASIC RNN:
─────────────────────────────────
h_t = tanh(W_x·x_t + W_h·h_{t-1} + b_h)
y_t = softmax(W_y·h_t + b_y)


LSTM (4 Gates):
─────────────────────────────────
Forget:  f_t = σ(W_f·[h_{t-1},x_t] + b_f)
Input:   i_t = σ(W_i·[h_{t-1},x_t] + b_i)
         C̃_t = tanh(W_c·[h_{t-1},x_t] + b_c)
Cell:    C_t = f_t⊙C_{t-1} + i_t⊙C̃_t
Output:  o_t = σ(W_o·[h_{t-1},x_t] + b_o)
         h_t = o_t⊙tanh(C_t)


GRU (2 Gates):
─────────────────────────────────
Reset:   r_t = σ(W_r·[h_{t-1},x_t] + b_r)
Update:  z_t = σ(W_z·[h_{t-1},x_t] + b_z)
Cand:    h̃_t = tanh(W_h·[r_t⊙h_{t-1},x_t] + b_h)
Hidden:  h_t = (1-z_t)⊙h̃_t + z_t⊙h_{t-1}


BPTT Gradient:
─────────────────────────────────
∂Loss/∂h_j = ∂Loss/∂h_T × ∏(∂h_t/∂h_{t-1}) for t=j to T
                         t=j+1


N-gram Probability:
─────────────────────────────────
P(w_n | w_{n-N+1},...,w_{n-1}) = Count(w_{n-N+1}...w_n) / Count(w_{n-N+1}...w_{n-1})
```

---

**End of Unit 4 Quick Revision Sheet**

