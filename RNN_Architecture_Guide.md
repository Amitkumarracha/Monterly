# RNN MODEL ARCHITECTURE
## Complete Guide with All Components & Variations

---

## TABLE OF CONTENTS
1. Basic RNN Architecture
2. Components & Parameters
3. RNN Variants & Architectures
4. Advanced Architectures
5. Mathematical Formulations
6. Practical Implementations
7. Architecture Comparison

---

# 1. BASIC RNN ARCHITECTURE

## 1.1 Definition

**Recurrent Neural Network (RNN)**: A neural network architecture with recurrent connections that process sequential data by maintaining an internal hidden state that captures temporal information from previous inputs.

**Key Characteristics**:
- Processes sequences one element at a time
- Maintains hidden state as memory
- Shares weights across time steps
- Can handle variable-length sequences

## 1.2 Core Components

### A. Input Layer
```
Purpose: Receive sequential input data
Dimensions: (batch_size, sequence_length, input_dim)

Example: 
- Text sequence: ["I", "love", "reading"]
- Each word converted to embedding: (3, 1, 300)
  - 3 words
  - 1 token per step
  - 300-dimensional embedding
```

### B. Embedding Layer (Optional)
```
Purpose: Convert discrete tokens to continuous vectors
Dimensions: 
- Input: (batch_size, sequence_length) - indices
- Output: (batch_size, sequence_length, embedding_dim)

Example:
Word index 5 → Embedding vector [0.2, -0.5, 0.8, ...]
Word index 10 → Embedding vector [0.1, 0.3, -0.2, ...]
```

### C. RNN Cell (Core)
```
Purpose: Process input and maintain hidden state
Formula: h_t = activation(W_x · x_t + W_h · h_{t-1} + b_h)

Parameters:
- W_x: Input-to-hidden weight matrix
  Dimensions: (input_dim, hidden_dim)
  
- W_h: Hidden-to-hidden weight matrix (recurrent)
  Dimensions: (hidden_dim, hidden_dim)
  
- b_h: Hidden bias
  Dimensions: (hidden_dim,)

Activation Function:
- tanh: f(x) = (e^x - e^-x) / (e^x + e^-x)
  Output range: [-1, 1]
  Used for internal representations
  
- ReLU: f(x) = max(0, x)
  Output range: [0, ∞)
  Used in modern architectures

Hidden State:
- h_t: Hidden state at time t
  Dimensions: (batch_size, hidden_dim)
  Contains compressed information about all previous inputs
```

### D. Output Layer
```
Purpose: Generate predictions from hidden state
Formula: y_t = W_y · h_t + b_y
        output_prob = softmax(y_t)

Parameters:
- W_y: Hidden-to-output weight matrix
  Dimensions: (hidden_dim, output_dim)
  
- b_y: Output bias
  Dimensions: (output_dim,)

Output Types:
1. Classification: softmax(y_t) → probability distribution
2. Regression: y_t directly → continuous value
3. Sequence: One output per time step (Many-to-Many)
```

## 1.3 Information Flow

```
TIME STEP t:

Input: x_t (current element)
       ↓
   [Embedding] (if needed)
       ↓
   h_{t-1} (previous hidden state) ──────────┐
       ↑                                      │
       │                                      ▼
    [RNN CELL]←─────────────────────────[×] (multiply)
       │                                      ▲
       │                         W_h · h_{t-1}
       │
       ├─ [+] (add inputs and previous state)
       │   ↑
       │   ├─ W_x · x_t
       │   ├─ W_h · h_{t-1}
       │   └─ b_h
       │
       ├─ [tanh] (activation)
       │
       ▼
    h_t (new hidden state)
       │
       ├─→ Passed to next time step
       │
       └─→ [Output Layer]
            │
            ├─ W_y · h_t + b_y
            │
            ▼
         y_t (output/prediction)
```

## 1.4 Parameter Sizes

```
For RNN processing sequence of embedding dimension D_e
with hidden dimension D_h and output dimension D_o:

Parameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input-to-Hidden: W_x
  Shape: (D_e, D_h)
  Parameters: D_e × D_h

Hidden-to-Hidden: W_h
  Shape: (D_h, D_h)
  Parameters: D_h × D_h (recurrent connections)

Hidden Bias: b_h
  Shape: (D_h,)
  Parameters: D_h

Hidden-to-Output: W_y
  Shape: (D_h, D_o)
  Parameters: D_h × D_o

Output Bias: b_y
  Shape: (D_o,)
  Parameters: D_o

Total Parameters: D_e × D_h + D_h² + D_h + D_h × D_o + D_o


Example (Language Model):
────────────────────────────
D_e = 300 (word embedding)
D_h = 512 (hidden dimension)
D_o = 50,000 (vocabulary size)

Parameters:
- W_x: 300 × 512 = 153,600
- W_h: 512 × 512 = 262,144 (dominates!)
- b_h: 512
- W_y: 512 × 50,000 = 25,600,000 (massive!)
- b_y: 50,000

Total: ~26 Million parameters
```

---

# 2. RNN ARCHITECTURES BY SEQUENCE MAPPING

## 2.1 Sequence-to-Sequence (Many-to-Many)

```
Purpose: Process input sequence, produce output sequence of same length

Example: Named Entity Recognition
Input:  "John loves Paris"  (3 words)
Output: "PERSON VERB LOCATION"  (3 tags)

Architecture:
───────────────────────────────────────

     x_1 ────→ [RNN] ──→ h_1 ──→ [Output] ──→ y_1
                 ↑                (tag for word 1)
     x_2 ────→ [RNN] ──→ h_2 ──→ [Output] ──→ y_2
                 ↑                (tag for word 2)
     x_3 ────→ [RNN] ──→ h_3 ──→ [Output] ──→ y_3
                                (tag for word 3)

h_1 flows through recurrent connection
Information preserved: x_1 → x_2 → x_3 context maintained
```

## 2.2 Sequence-to-Vector (Many-to-One)

```
Purpose: Process sequence, produce single output

Example: Sentiment Analysis
Input:  "This movie is absolutely amazing!"  (6 words)
Output: "Positive"  (single label)

Architecture:
───────────────────────────────────────

     x_1 ────→ [RNN] ──→ h_1
                 ↑
     x_2 ────→ [RNN] ──→ h_2
                 ↑
     x_3 ────→ [RNN] ──→ h_3
                 ↑
     x_4 ────→ [RNN] ──→ h_4
                 ↑
     x_5 ────→ [RNN] ──→ h_5
                 ↑
     x_6 ────→ [RNN] ──→ h_6
                           │
                           └─→ [Output] ──→ y_final
                               (use h_6 only)
```

## 2.3 Vector-to-Sequence (One-to-Many)

```
Purpose: Process single input, produce output sequence

Example: Image Captioning
Input:  Image (single)
Output: "A dog is running in the park"  (variable length)

Architecture:
───────────────────────────────────────

Image ──→ [Encoder] ──→ h_0 (context vector)
                          │
                          ├─→ [RNN] ──→ h_1 ──→ [Output] ──→ y_1 ("A")
                          │                           ↑
                          ├─→ [RNN] ──→ h_2 ──→ [Output] ──→ y_2 ("dog")
                          │                           ↑
                          ├─→ [RNN] ──→ h_3 ──→ [Output] ──→ y_3 ("is")
                          │                           ↑
                          └─→ [RNN] ──→ h_4 ──→ [Output] ──→ y_4 ("running")
                                              ...
```

## 2.4 Sequence-to-Sequence (Encoder-Decoder)

```
Purpose: Transform one sequence to another (translation)

Example: Machine Translation
Input:  "Hello how are you"  (English)
Output: "Hola cómo estás"  (Spanish)

Architecture:
───────────────────────────────────────

ENCODER:
     x_1 ────→ [RNN] ──→ h_1
                 ↑
     x_2 ────→ [RNN] ──→ h_2
                 ↑
     x_3 ────→ [RNN] ──→ h_3
                 ↑
     x_4 ────→ [RNN] ──→ h_4
                           │
                      Context Vector
                           │
                           ▼
DECODER:                h_4 ──→ [RNN] ──→ h'_1 ──→ [Output] ──→ y_1 ("Hola")
                                   ↑
                              [START] token
                                   
                           h'_1 ──→ [RNN] ──→ h'_2 ──→ [Output] ──→ y_2 ("cómo")
                                   ↑
                              y_1 ("Hola")
                                   
                           h'_2 ──→ [RNN] ──→ h'_3 ──→ [Output] ──→ y_3 ("estás")
                                   ↑
                              y_2 ("cómo")
```

---

# 3. RNN CELL VARIANTS

## 3.1 Vanilla RNN Cell

```
Mathematical Definition:
───────────────────────────

h_t = tanh(W_x · x_t + W_h · h_{t-1} + b_h)

y_t = softmax(W_y · h_t + b_y)


Advantages:
───────────
- Simple
- Fast computation
- Easy to implement

Disadvantages:
───────────────
- Vanishing gradient problem
- Cannot capture long-range dependencies
- Unstable gradients
```

## 3.2 LSTM (Long Short-Term Memory)

```
Mathematical Definition:
───────────────────────────

Input Gate:
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

Forget Gate:
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

Candidate Cell State:
    C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

Cell State Update:
    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

Output Gate:
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

Hidden State:
    h_t = o_t ⊙ tanh(C_t)

Output:
    y_t = softmax(W_y · h_t + b_y)


Architecture Diagram:
─────────────────────

    ┌────[Forget Gate]────┐
    │                     │
    │    f_t = σ(...)    │
    │                     ▼
C_{t-1} ────→ [×] ────→ ┐
                        ├─→ [+] ────→ C_t
               ┌─→ [×] ◄─┘
               │
          i_t ⊙ C̃_t

    ├─ Input: i_t = σ(...)
    ├─ Candidate: C̃_t = tanh(...)
    └─ Output Gate: o_t = σ(...)
                    h_t = o_t ⊙ tanh(C_t)


Advantages:
───────────
- Solves vanishing gradient
- Captures long-range dependencies
- Separate memory (cell state) and output (hidden state)

Disadvantages:
───────────────
- More parameters
- Slower training
- Complex to understand
```

## 3.3 GRU (Gated Recurrent Unit)

```
Mathematical Definition:
───────────────────────────

Reset Gate:
    r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

Update Gate:
    z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

Candidate Hidden State:
    h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

Hidden State Update:
    h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}

Output:
    y_t = softmax(W_y · h_t + b_y)


Architecture Diagram:
─────────────────────

    ┌─────[Reset Gate]─────┐
    │   r_t = σ(...)       │
    │                      ▼
h_{t-1} ────→ [×] ──────→ [RNN Cell]
               r_t ⊙ h_{t-1}
                            │
                            ├─→ h̃_t = tanh(...)
                            │
    ┌──────[Update Gate]───┤
    │  z_t = σ(...)       │
    │                     ▼
    │ (1-z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}
    │                     │
    └─────────────────────→ h_t


Advantages:
───────────
- Simpler than LSTM (2 gates vs 3)
- Fewer parameters
- Faster training (~10-15%)
- Comparable performance

Disadvantages:
───────────────
- Less modeling capacity than LSTM
- May underperform on very complex tasks
```

---

# 4. ADVANCED ARCHITECTURES

## 4.1 Bidirectional RNN

```
Purpose: Process sequences in both directions
Use Case: NER, POS tagging, machine translation

Architecture:
────────────────

Forward Direction:
    x_1 ──→ [RNN] ──→ h_f1
    x_2 ──→ [RNN] ──→ h_f2
    x_3 ──→ [RNN] ──→ h_f3

Backward Direction:
    x_1 ◄── [RNN] ◄── h_b1
    x_2 ◄── [RNN] ◄── h_b2
    x_3 ◄── [RNN] ◄── h_b3

Concatenation:
    o_1 = [h_f1; h_b1]  (concatenate forward and backward)
    o_2 = [h_f2; h_b2]
    o_3 = [h_f3; h_b3]

Output Dimension: 2 × hidden_dim

Information Flow:
    Each position has context from BOTH directions
    Position 2 knows about [word1, word2] and [word2, word3]
    Better context understanding


Implementation:
────────────────
class BidirectionalRNN:
    def __init__(self, input_dim, hidden_dim):
        self.forward_rnn = RNN(input_dim, hidden_dim)
        self.backward_rnn = RNN(input_dim, hidden_dim)
    
    def forward(self, x_sequence):
        # Forward pass
        forward_outputs = self.forward_rnn(x_sequence)
        
        # Backward pass
        backward_outputs = self.backward_rnn(reverse(x_sequence))
        
        # Concatenate
        bidirectional_output = concatenate([forward_outputs, reverse(backward_outputs)])
        return bidirectional_output
```

## 4.2 Stacked/Deep RNN

```
Purpose: Increase model capacity with multiple layers
Use Case: Complex tasks, large datasets

Architecture:
────────────────

Layer 1:
    x_1 ──→ [RNN] ──→ h¹_1
    x_2 ──→ [RNN] ──→ h¹_2
    x_3 ──→ [RNN] ──→ h¹_3

Layer 2:
    h¹_1 ──→ [RNN] ──→ h²_1
    h¹_2 ──→ [RNN] ──→ h²_2
    h¹_3 ──→ [RNN] ──→ h²_3

Layer 3:
    h²_1 ──→ [RNN] ──→ h³_1
    h²_2 ──→ [RNN] ──→ h³_2
    h²_3 ──→ [RNN] ──→ h³_3

Output:
    y_1 ← h³_1
    y_2 ← h³_2
    y_3 ← h³_3

Parameters per Layer:
    Layer 1: input_dim × hidden_dim + hidden_dim²
    Layer 2: hidden_dim² + hidden_dim²  (input to layer 2 is hidden_dim)
    Layer 3: hidden_dim² + hidden_dim²

Total Parameters: Increases with depth


Advantages:
───────────
- Higher representational power
- Better performance on complex tasks
- Can capture hierarchical patterns

Disadvantages:
───────────────
- More training time
- More parameters (overfitting risk)
- Gradient flow more difficult (needs careful initialization)
- Vanishing gradient more severe
```

## 4.3 Attention-Enhanced RNN

```
Purpose: Focus on relevant input positions
Use Case: Machine translation, summarization

Architecture:
────────────────

ENCODER:
    Input ──→ [RNN] ──→ h_1, h_2, h_3, h_4

DECODER with Attention:
    At step t:
    
    1. Hidden State: s_t from decoder
    
    2. Compute Attention Scores:
        e_i = score(s_t, h_i)
        For each encoder position i
    
    3. Softmax:
        α_i = softmax(e_i)
        Attention weights (0 to 1)
    
    4. Context Vector:
        c_t = Σ α_i · h_i
        Weighted sum of encoder states
    
    5. Generate Output:
        y_t = decoder(s_t, c_t)

Information Flow:
    Decoder doesn't just use final h_4
    Instead uses weighted combination of ALL encoder states
    Attention weights decide which positions are important


Example (Translation):
──────────────────────
Translating "The dog bit the cat"

When generating German word for "dog":
    Attention looks at all English words
    High attention on "dog" position
    Lower attention on "the", "cat"
    
    Context includes mostly "dog" info
    Generates correct German translation
```

---

# 5. MATHEMATICAL FORMULATIONS

## 5.1 Forward Propagation

```
For time step t = 1 to T:

Hidden State Update:
    h_t = tanh(W_x · x_t + W_h · h_{t-1} + b_h)

Output Generation:
    logits_t = W_y · h_t + b_y

Probability Distribution:
    y_t = softmax(logits_t)

Loss Calculation:
    L_t = CrossEntropy(y_t, target_t)

Total Loss:
    L = (1/T) · Σ_{t=1}^T L_t


Matrix Operations (Vectorized):
──────────────────────────────────

For batch of size B:

x_t: (B, input_dim)
h_{t-1}: (B, hidden_dim)
W_x: (input_dim, hidden_dim)
W_h: (hidden_dim, hidden_dim)
b_h: (hidden_dim,)

h_t = tanh(x_t @ W_x + h_{t-1} @ W_h + b_h)
      └─────┬─────┘   └──────┬──────┘
      (B, hidden_dim) (B, hidden_dim)
```

## 5.2 Backpropagation Through Time

```
Loss Gradient Computation:
──────────────────────────

For weight matrix W_h (shared across time steps):

∂L/∂W_h = Σ_{t=1}^T ∂L_t/∂W_h
        = Σ_{t=1}^T (∂L_t/∂y_t) · (∂y_t/∂h_t) · (∂h_t/∂W_h)

For h_t dependency through time:
    ∂h_t/∂h_{t-1} involves W_h
    
    ∂h_T/∂h_1 = ∂h_T/∂h_{T-1} · ∂h_{T-1}/∂h_{T-2} · ... · ∂h_2/∂h_1
               = Product of T-1 gradient terms
               
               Each term ≈ 0.25 (sigmoid derivative)
               
               Product: 0.25^(T-1)
               
               For T=20: 0.25^19 ≈ 10^-11 (VANISHES!)


Chain Rule Decomposition:
─────────────────────────

∂L/∂h_j = Σ_{t=j}^T (∂L/∂L_t) · Π_{i=j}^{t-1} (∂h_{i+1}/∂h_i)

Later time steps: Full gradient
Earlier time steps: Product reduces gradient
```

## 5.3 Gradient Flow Analysis

```
Vanishing Gradient Phenomenon:
──────────────────────────────

∂h_t/∂h_{t-1} = W_h^T · diag(1 - h_t²)

For tanh activation:
    max(1 - tanh(x)²) = 1  (but typical ≈ 0.1 - 0.25)

Product over time:
    ∂h_T/∂h_0 = ∏_{t=1}^T (W_h^T · diag(1 - h_t²))
               ≤ ∏_{t=1}^T 0.25
               = 0.25^T
               
For T = 100: 0.25^100 ≈ 10^-60 (essentially ZERO!)

Impact:
    Early time steps get NO gradient updates
    Cannot learn long-range dependencies
    Model fails on long sequences


Solution: LSTM/GRU
─────────────────

Cell state addition in LSTM:
    ∂C_t/∂C_{t-1} = f_t (forget gate output)
    
f_t typically ≈ 0.5 to 0.9 (sigmoid output)

Product over time:
    ∂C_T/∂C_0 = ∏_{t=1}^T f_t
               ≈ 0.7^T
               
For T = 100: 0.7^100 ≈ 3×10^-11 (survives better!)
```

---

# 6. PRACTICAL IMPLEMENTATION DETAILS

## 6.1 Initialization Strategies

```
Weight Initialization:
─────────────────────

1. Xavier/Glorot Initialization (Recommended for RNN)
    W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]
    
    Balances variance between forward and backward passes
    Prevents saturation of activations

2. He Initialization
    W ~ N(0, √(2/n_in))
    
    For ReLU activations
    Slightly different scaling

3. Random Orthogonal (LSTM Cells)
    W_h initialized as random orthogonal matrix
    
    Preserves gradient magnitude during multiplication
    Helps with vanishing gradient

Bias Initialization:
    b = 0  (usually)
    
    Exception: LSTM forget gate bias
    b_f = 1 (initialize to remember)
    Helps LSTM learn to preserve information initially

Hidden State Initialization:
    h_0 = 0 (most common)
    h_0 ~ N(0, small_variance)  (for better generalization)

```

## 6.2 Training Techniques

```
Gradient Clipping:
──────────────────

if ||∇W|| > threshold:
    ∇W = ∇W × (threshold / ||∇W||)

Prevents exploding gradients
Scales gradient to maximum allowed magnitude
Typical threshold: 1.0 to 5.0


Truncated BPTT:
────────────────

Only backprop through last k time steps (k << T)

Benefits:
    Faster training
    Lower memory usage
    
Trade-offs:
    Loses very long-range dependencies
    May miss important patterns

Typical k: 50-100 for language modeling


Layer Normalization:
────────────────────

Normalize activations of hidden state
h_t_norm = (h_t - mean(h_t)) / std(h_t)

Benefits:
    Stabilizes training
    Allows higher learning rates
    Reduces internal covariate shift
    
Applied per layer/time step
```

## 6.3 Hyperparameter Tuning

```
Key Hyperparameters:
────────────────────

1. Hidden Dimension (D_h)
    Smaller: Faster, less memory, lower capacity
    Larger: More parameters, better fit, slower
    Typical: 256, 512, 1024
    
    Rule of thumb: 2-3 × input_dim

2. Learning Rate (α)
    Too high: Training unstable, divergence
    Too low: Slow convergence, local minima
    Typical: 0.001 to 0.01 for Adam
            0.0001 to 0.001 for SGD

3. Batch Size
    Small (32-64): Noisy gradients, better generalization
    Large (256-512): Stable gradients, faster processing
    Typical: 32 or 64

4. Sequence Length
    Short: Less context, underfitting
    Long: More context, vanishing gradient
    Typical: 20-100 for language modeling

5. Dropout Rate
    0.0: No regularization
    0.5: Heavy regularization
    Typical: 0.2-0.4 for RNN

6. Embedding Dimension
    Typical: 100-500 for language tasks
    Affects initial input dimensionality
```

---

# 7. ARCHITECTURE COMPARISON

## 7.1 Vanilla RNN vs LSTM vs GRU

```
Architecture Comparison:
────────────────────────────────────────────────────────

Feature              Vanilla RNN    LSTM          GRU
─────────────────────────────────────────────────────────
Hidden States        1              2 (C+h)       1
Gates                0              3+1 candidate 2
Parameters           Least          Most          Medium
Computation Speed    Fastest        Slowest       Fast
Memory Required      Least          Most          Medium
Vanishing Gradient   ✗ Yes          ✓ Solved      ✓ Solved
Long Dependencies    ✗ Fails        ✓ Excellent   ✓ Good
Training Stability   Low            High          High
Interpretability     Easy           Complex       Medium
Practical Use        Educational    Production    Production

Typical Parameter Counts:
─────────────────────────────

For hidden_dim = 512, embedding_dim = 300, vocab_size = 50,000:

Vanilla RNN:
    W_x: 300×512 = 153,600
    W_h: 512×512 = 262,144
    W_y: 512×50,000 = 25,600,000
    Total: ~26 Million

LSTM:
    4 weight sets (forget, input, output, candidate)
    W_x: 4×300×512 = 614,400
    W_h: 4×512×512 = 1,048,576
    W_y: 512×50,000 = 25,600,000
    Total: ~27 Million

GRU:
    3 weight sets (reset, update, candidate)
    W_x: 3×300×512 = 460,800
    W_h: 3×512×512 = 786,432
    W_y: 512×50,000 = 25,600,000
    Total: ~26.8 Million
```

## 7.2 When to Use Each Architecture

```
Choose VANILLA RNN:
──────────────────
✓ Educational purposes
✓ Simple, short sequences
✓ Resource severely constrained
✗ NOT recommended for production


Choose LSTM:
────────────
✓ Long-range dependencies critical
✓ Complex sequences (20+ steps)
✓ Maximum performance priority
✓ Established benchmarks
Examples: Machine translation, complex language modeling

When to use LSTM:
    - Translating long sentences
    - Writing multi-paragraph text
    - Complex NLP tasks
    - Have computational resources


Choose GRU:
───────────
✓ Good balance of performance and efficiency
✓ Limited computational budget
✓ Fast training required
✓ Moderate sequence lengths
Examples: Short text classification, lightweight models

When to use GRU:
    - Sentiment analysis (short reviews)
    - Prototyping/rapid development
    - Mobile/embedded deployment
    - Edge devices


Choose Bidirectional:
──────────────────────
✓ Full sequence available (not streaming)
✓ NER, POS tagging
✓ Need context from both directions
✗ NOT for real-time/streaming
✗ NOT for language generation


Choose Stacked:
────────────────
✓ High model capacity needed
✓ Large datasets available
✓ Complex patterns
✗ Computationally intensive
✗ Overfitting with small data


Choose Attention:
──────────────────
✓ Long sequences with varying importance
✓ Interpretability important
✓ Machine translation
✓ Sequence summarization
```

---

## 7.3 Architecture Selection Guide

```
Decision Tree:
──────────────

START
  │
  ├─ Sequence modeling needed? 
  │    NO → Use feedforward/CNN
  │    YES ↓
  │
  ├─ Sequence length < 30 steps?
  │    YES → GRU might be sufficient
  │    NO ↓
  │
  ├─ Complex dependencies?
  │    NO → GRU ✓
  │    YES ↓
  │
  ├─ Computational budget?
  │    LOW → GRU ✓
  │    NORMAL → LSTM ✓
  │    HIGH → LSTM + Attention ✓
  │
  ├─ Need bidirectional context?
  │    YES → Bidirectional LSTM ✓
  │    NO ↓
  │
  ├─ Real-time/streaming?
  │    YES → Unidirectional LSTM
  │    NO ↓
  │
  └─ Very complex task?
       YES → Stacked Bidirectional LSTM
       NO → Single Layer Unidirectional LSTM


Practical Recommendation:
─────────────────────────

1. Start with: GRU (simplest, fastest)
2. If insufficient: Switch to LSTM (more capacity)
3. If needed: Add bidirectional (if offline)
4. If still insufficient: Stack layers + add attention
```

---

# END OF RNN ARCHITECTURE GUIDE

**This comprehensive guide covers:**
- All basic RNN components
- Complete mathematical formulations
- Advanced architectures (bidirectional, stacked, attention)
- Implementation details
- Hyperparameter guidance
- Architecture comparison and selection

