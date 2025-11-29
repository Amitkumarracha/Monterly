# UNIT 4: VISUAL DIAGRAMS & FLOWCHARTS
## Recurrent Neural Networks Illustrated

---

## 1. BASIC RNN ARCHITECTURE

```
                    ROLLED (Compact View)
                    
                    INPUT: "cat sat on"
                            |
                            ▼
                        [Embedding]
                      x_1, x_2, x_3
                            |
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │  RNN    │     │  RNN    │     │  RNN    │
        │ Cell    │────→│ Cell    │────→│ Cell    │
        │ (W_h)   │     │ (W_h)   │     │ (W_h)   │
        └─────────┘     └─────────┘     └─────────┘
            ▼               ▼               ▼
        [Output]        [Output]        [Output]
            │               │               │
            ▼               ▼               ▼
         "sat"           "on"           "the"  (predictions)


            UNROLLED (Expanded Through Time)

t=1: Input: embedding("cat")
     x_1 ──→ [RNN] ─────────────────────────────→ [RNN] ─────────────────────→ [RNN]
             h_t = tanh(W_x·x_1 + W_h·0 + b_h)        │                        │
             h_1 ────────────────────────────→ h_1     │
                                              ▼        │
     y_1 ← [Prediction from h_1]            h_2 ◄─────┘
                                            (h_1 used here!)
     
t=2: Input: embedding("sat")
             h_1 ──────────→ [RNN]
             x_2 ──────────→ 
             h_t = tanh(W_x·x_2 + W_h·h_1 + b_h)
             h_2
             ▼
     y_2 ← [Prediction from h_2]

t=3: Input: embedding("on")
             h_2 ──────────→ [RNN]
             x_3 ──────────→
             h_3 = tanh(W_x·x_3 + W_h·h_2 + b_h)
             ▼
     y_3 ← [Prediction from h_3]

KEY: Same RNN cell (W_x, W_h, W_y) used at ALL time steps!
     Hidden state flows: h_1 → h_2 → h_3 (carries memory)
```

---

## 2. HIDDEN STATE FLOW - HOW MEMORY WORKS

```
SEQUENCE: "I love dogs because they are loyal"

h_0 = [0, 0, 0, ...]  (initial: no information)
  │
  ▼ (input: "I")
h_1 = [0.3, -0.5, 0.2, ...]  (encodes: subject "I")
  │
  ▼ (input: "love")
h_2 = [0.4, -0.3, 0.1, ...]  (encodes: "I love")
  │
  ▼ (input: "dogs")
h_3 = [0.2, 0.4, -0.1, ...]  (encodes: "I love dogs")
  │
  ▼ (input: "because")
h_4 = [-0.1, 0.6, 0.3, ...]  (encodes: "I love dogs because")
  │
  ▼ (input: "they")
h_5 = [0.5, 0.2, 0.0, ...]  (encodes: "I love dogs because they")
  │
  ▼ (input: "are")
h_6 = [0.1, 0.3, 0.4, ...]  (encodes: "I love dogs because they are")
  │
  ▼ (input: "loyal")
h_7 = [0.2, 0.4, 0.5, ...]  (encodes: FULL SEQUENCE)
         │
         ├─→ USED FOR NEXT PREDICTION
         └─→ Contains all previous context!

INTERPRETATION:
h_7 doesn't literally store text, but it stores PATTERNS and CONTEXT
that represent the meaning of the entire sequence
```

---

## 3. BACKPROPAGATION THROUGH TIME (BPTT)

```
FORWARD PASS:
─────────────────────────────────────────────────
x_1 → [RNN] → h_1 → [Dense] → y_1 → Loss_1
               ↓
x_2 → [RNN] → h_2 → [Dense] → y_2 → Loss_2
               ↓
x_3 → [RNN] → h_3 → [Dense] → y_3 → Loss_3

Total Loss = Loss_1 + Loss_2 + Loss_3


BACKWARD PASS (BPTT):
─────────────────────────────────────────────────

Error flows backward THROUGH TIME:

Step 3: From Loss_3
        δy_3 = y_3 - target_3
        δh_3 = W_y^T · δy_3
        
        ∂Loss/∂W_h += δh_3 · h_2^T  (gradient for W_h from step 3)
        ∂Loss/∂W_y += δy_3 · h_3^T

Step 2: From Loss_2 + BACKPROP from step 3
        δy_2 = y_2 - target_2
        δh_2 = W_y^T · δy_2 + W_h^T · δh_3  ← Error from FUTURE step!
               │────────────────┤   │────────────────┤
               Error at t=2        Error backprop from t=3
        
        ∂Loss/∂W_h += δh_2 · h_1^T  (gradient for W_h from step 2)
        ∂Loss/∂W_y += δy_2 · h_2^T

Step 1: From Loss_1 + BACKPROP from step 2
        δy_1 = y_1 - target_1
        δh_1 = W_y^T · δy_1 + W_h^T · δh_2  ← Error from TWO future steps!
        
        ∂Loss/∂W_h += δh_1 · h_0^T  (gradient for W_h from step 1)
        ∂Loss/∂W_y += δy_1 · h_1^T

FINAL GRADIENT:
∂Loss/∂W_h = ∂Loss/∂W_h(t=1) + ∂Loss/∂W_h(t=2) + ∂Loss/∂W_h(t=3)
           = Sum of gradients from ALL time steps!

KEY INSIGHT:
Weight W_h gets updated using information from all 3 time steps
This is DIFFERENT from feedforward where each layer gets independent gradient
```

---

## 4. VANISHING GRADIENT VISUALIZATION

```
GRADIENT MAGNITUDE OVER BACKPROP TIME:

Vanilla RNN (Vanishing):
═════════════════════════════════════════════════
Gradient Value
    ^
    |  ●  Gradient at time t=1
    |  │  (large, near start)
 1.0 |  │
    |  ●  0.25
    |  │  (×0.25 each step back)
 0.5 |  ●  0.0625
    |  │  (×0.25 again)
    |  ●  0.0156  (t=1 from t=4)
 0.1 |  │  
    |  ●  0.0039  (t=1 from t=5)
    |  │
    |  ●●●●●●  (essentially ZERO at t=1 from t=20+)
    |__│__│_│_│_│_│_│_│_____________→ time steps back
    0  1 2 3 4 5 6... 20
    
    Pattern: Each step back multiplies by ~0.25
    Result: 0.25^20 ≈ 10^-12 (essentially zero!)


LSTM (Gradient Flows Better):
═════════════════════════════════════════════════
Gradient Value
    ^
    |  ●  Starting gradient
 1.0 |  │
    |  ●  0.7  (forget gate ≈ 0.7)
    |  │
 0.5 |  ●  0.49
    |  │  (×0.7 again)
    |  ●  0.34
 0.1 |  │  (×0.7 again)
    |  ●  0.24
    |  │  (×0.7 again)
    |  ●●●●●●  (still reasonable at t=1 from t=20!)
    |__│_│_│_│_│_│_│_│_____________→ time steps back
    0  1 2 3 4 5 6... 20
    
    Pattern: 0.7^20 ≈ 0.0008 (stays reasonable!)
    
    WHY BETTER? Addition in cell state update:
    ∂C_t/∂C_{t-1} = f_t (not multiplicative chain!)
```

---

## 5. LSTM ARCHITECTURE - DETAILED DIAGRAM

```
LSTM CELL STRUCTURE:
═════════════════════════════════════════════════════════════

Input at time t:
    h_{t-1} (previous hidden state)
    x_t     (current input)

    ┌─ Forget Gate ─────────────────────────────────┐
    │  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)         │
    │  Output: 0 = forget, 1 = remember            │
    └────────┬──────────────────────────────────────┘
             │
             ▼
         [×] ← Cell_{t-1} (multiply: forget)
             │
             ├─ Input Gate + Candidate ──────────────┐
             │  i_t = σ(W_i · [h_{t-1}, x_t] + b_i) │
             │  C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
             │  Output gate: 0 = discard, 1 = add   │
             └────────┬──────────────────────────────┘
                      │
                      ▼
                  [×] ← (multiply: add new)
                      │
                      ▼
                    [+] ← Cell_t (add old + new)
                      │
    ┌─ Output Gate ────┼────────────────────────────┐
    │  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        │
    │  Output: 0 = hide, 1 = show                 │
    └────────┬────────┘                            │
             │         Cell_t (long-term memory)   │
             │         │                           │
             │         ▼                           │
             │     [tanh] ← scale to [-1, 1]      │
             │         │                           │
             │         ▼                           │
             └─→ [×] ← (multiply: filter output)
                     │
                     ▼
                 h_t (hidden state output for this step)


GATE PURPOSES (MNEMONICS):
────────────────────────────────
[Forget]  →  "What old information to throw away?"
[Input]   →  "What new information to add?"
[Candidate] → "What are the candidate values?"
[Cell]    →  "Updated long-term memory"
[Output]  →  "What information to output?"
```

---

## 6. GRU ARCHITECTURE - SIMPLIFIED DIAGRAM

```
GRU CELL STRUCTURE (Simpler than LSTM):
═════════════════════════════════════════════════════════════

Input: h_{t-1}, x_t

    ┌─ Reset Gate ──────────────────────────────────┐
    │  r_t = σ(W_r · [h_{t-1}, x_t] + b_r)        │
    │  Output: 0 = ignore h_{t-1}, 1 = use it     │
    └────────┬────────────────────────────────────┘
             │
             ▼
    h_{t-1} [×] r_t  ← filter which part of h_{t-1} to use
             │
             ├─ Candidate Hidden State ────────────┐
             │  h̃_t = tanh(W_h · [r_t⊙h_{t-1}, x_t] + b_h)
             │  New proposed hidden state          │
             └────────┬───────────────────────────┘
                      │
    ┌─ Update Gate ────┼────────────────────────────┐
    │  z_t = σ(W_z · [h_{t-1}, x_t] + b_z)       │
    │  Output: 0 = use new, 1 = keep old         │
    └────────┬────────┘                          │
             │      h̃_t (new candidate)         │
             │      │                             │
             │      ├─ [×] (1-z_t) ← new weight
             │      │   │
             │      │   ▼
             │      ├─→ [+]
             │          │
             │          ├─ [×] z_t ← old weight
             │          │   │
             │     h_{t-1} [×] z_t
             │          │   │
             │          └─→ [+]
             │
             ▼
        h_t = (1-z_t)⊙h̃_t + z_t⊙h_{t-1}


COMPARISON:
────────────────────────────────
GRU:
- 1 hidden state (combines short + long term)
- 2 gates (reset, update)
- Simpler, faster training

LSTM:
- 2 states (cell for long-term, hidden for short-term output)
- 3 gates + candidate (forget, input+candidate, output)
- More powerful, more parameters
```

---

## 7. WEIGHT SHARING - WHY RNN IS PARAMETER EFFICIENT

```
FEEDFORWARD NETWORK (Different weights at each layer):
════════════════════════════════════════════════════════

Input Layer 1  → Dense Layer 1 (W₁) → Dense Layer 2 (W₂) → Dense Layer 3 (W₃)
Total Parameters: W₁ + W₂ + W₃ = 3 different weight matrices


RNN (SAME weights shared across time steps):
════════════════════════════════════════════════════════

x_1 → [RNN with W] → h_1
       (SAME W)

x_2 → [RNN with W] → h_2
       (SAME W)

x_3 → [RNN with W] → h_3
       (SAME W)

Total Parameters: Just W (SAME across all time steps!)


ADVANTAGE:
─────────────────────────────────────────────────────
Feedforward (for sequence of 100 words):
- 100 different weight matrices for 100 positions
- HUGE number of parameters
- Overfitting risk
- Cannot handle variable length sequences

RNN:
- ONE weight matrix (W_h, W_x, W_y)
- Works for ANY sequence length
- Fewer parameters
- Generalizes better
- Can process sequences of 10 or 1000 words!
```

---

## 8. N-GRAM VS RNN CONTEXT WINDOW

```
SENTENCE: "The dog which was running in the park barked loudly"
           1   2    3     4   5       6  7   8     9      10

N-GRAM MODELS (Fixed Context Window):
════════════════════════════════════════════════════════════

Bigram (2-gram):
Uses only 1 previous word:
- Position 9 ("barked"): context is "park" only
- Cannot see "dog" (7 positions back)
- P(barked | park) ≈ 0 (probably never saw this pair!)
- LOSES INFORMATION!

Trigram (3-gram):
Uses only 2 previous words:
- Position 9 ("barked"): context is ["in", "the", "park"]
- Still cannot see "dog"
- LIMITED WINDOW

10-gram:
- Position 9: context is ["The", "dog", "which", ... "park"]
- CAN NOW SEE "dog"!
- BUT: Vocabulary^10 combinations = INFEASIBLE (sparsity)


RNN (Theoretically Unbounded Context):
════════════════════════════════════════════════════════════

Time 1: h_1 = f("The") = some vector containing "The" info
        h_1 flows through network
        
Time 2: h_2 = f("dog", h_1)
        h_2 contains information about BOTH "The" and "dog"
        h_2 flows through network
        
Time 3: h_3 = f("which", h_2)
        h_3 contains info about "The", "dog", "which"
        
... (continues)

Time 9: h_9 = f("barked", h_8)
        h_9 potentially contains information about ALL previous words!
        Including "dog" from position 2
        
P(barked | h_8) where h_8 contains context from positions 1-8

ADVANTAGE:
- Context grows naturally (h_t contains all previous words)
- No need for huge N (no sparsity)
- Works for any sequence length
- Captures long-range dependencies (if using LSTM/GRU)


VISUALIZATION:
───────────────────────────────────────────────────────────

Context Window Size:

N-gram:
Context: ═══│X X X│=== (fixed window, misses "dog")
             ▲ ▲ ▲ ▲
          park the in dog
             position 9

RNN:
Context: ═══════════════════════│X X X X X X X X│ (grows!)
         The dog which was running in the park barked
         └──────────────────────────────────────────┘
         ALL previous context available to RNN at position 9!
```

---

## 9. TRAINING PROCESS - RNN vs LSTM

```
VANILLA RNN TRAINING:
════════════════════════════════════════════════════════════

Epoch 1:
  Forward: Process "The cat sat on"
  Loss: 2.5
  Backward: Gradients propagate
  ∂Loss/∂W_h at time 1: ≈ 10^{-12} (vanishing!)
  Update: W_h_new ≈ W_h_old (essentially NO change)
  
Epoch 2:
  Still cannot learn connection between "The" and "sat"
  Model STUCK on this pattern

Result: Cannot learn long sequences


LSTM TRAINING:
════════════════════════════════════════════════════════════

Epoch 1:
  Forward: Process "The cat sat on"
  Loss: 2.5
  Backward: Gradients propagate through cell state
  ∂Loss/∂W_h at time 1: ≈ 10^{-3} (MUCH BETTER!)
  Update: W_h_new = W_h_old - lr × 10^{-3} (meaningful change!)
  
Epoch 2:
  Learns connection between "The" and "sat"
  Model improving
  
Result: Successfully learns long-range dependencies
```

---

## 10. LSTM GATE BEHAVIOR - EXAMPLE

```
LANGUAGE TASK: Predicting next word
Sequence: "I am a student. He is ___"

Time 1-4: Process "I am a student"
          h_1, h_2, h_3, h_4, C_4 = [number:SG, gender:unk, ...]

Time 5: Process "He"
        Forget gate detects: NEW SUBJECT "He"
        f_5 ≈ [0.2, 0.1, ...] ← low for old number/gender (FORGET)
        
        Input gate detects: "He" is singular male
        i_5 ≈ [0.9, 0.8, ...]  ← high (ADD new info)
        C̃_5 ≈ [1, 1, ...] (new values: SG, M)
        
        C_5 = f_5 ⊙ C_4 + i_5 ⊙ C̃_5
            = [0.2×SG_old + 0.9×SG_new, ...]
            = [mostly new values] ← UPDATED!

Time 6: Process "is" (verb, singular)
        h_6 carries: gender=M, number=SG
        
Time 7: Predict next word
        Output gate: h_6 shows gender + number info
        Model predicts: likely singular verb or singular noun
        Good predictions: "a student", "a teacher", "happy", etc.


WHAT GATES DID:
───────────────
1. Detected change of subject (I → He)
2. Forgot old gender/number
3. Added new gender/number
4. Used this to make correct prediction

WITHOUT GATES (vanilla RNN):
─────────────────────────────
Would accumulate ALL information in h_t
h_t ≈ "I student. He" (mixed subject info)
Cannot disambiguate
Wrong predictions: "am", "a" (from subject I), not "is", "a" (from subject He)
```

---

## 11. DECISION TREE - WHICH MODEL TO USE

```
CHOOSING A SEQUENTIAL MODEL:
════════════════════════════════════════════════════════════

START: Do you have SEQUENTIAL data?
    │
    YES── Do you need to model LONG-RANGE dependencies (>50 words)?
    │        │
    │        YES── Do you have AMPLE computational resources?
    │        │        │
    │        │        YES── Use LSTM ✓
    │        │        │      (Maximum capacity)
    │        │        │
    │        │        NO── Use GRU ✓
    │        │             (Efficient, comparable performance)
    │        │
    │        NO── Do you have LIMITED resources?
    │              │
    │              YES── Use GRU ✓
    │              │     (Fast, efficient)
    │              │
    │              NO── Use LSTM or GRU ✓
    │                   (Either works fine)
    │
    NO── Use feedforward neural network (not RNN)
         (No sequence processing needed)


SPECIFIC RECOMMENDATIONS:
─────────────────────────
Task: Machine Translation (30-50 words)
→ LSTM ✓ (long dependencies critical)

Task: Sentiment analysis on short reviews (5-20 words)
→ GRU ✓ (short enough for GRU, faster)

Task: Time series forecasting (sequence length 100+)
→ LSTM ✓ (very long sequences)

Task: Named Entity Recognition (tagging each word)
→ GRU ✓ (mostly local dependencies)

Task: Anomaly detection in sensor data (real-time)
→ GRU ✓ (resource-constrained, needs speed)
```

---

## 12. COMPARISON FLOWCHART

```
CHOOSE YOUR RNN:
════════════════════════════════════════════════════════════

                    ┌─── Vanilla RNN
                    │    (Educational only)
                    │    ✗ Vanishing gradients
                    │    ✗ Poor long sequences
                    │
RNN TYPES ─────────┤
                    │
                    ├─── LSTM (Complex)
                    │    ✓ Solves vanishing gradients
                    │    ✓ Excellent long sequences
                    │    ✗ Slower training
                    │    ✗ More parameters
                    │    → Use for: maximum performance
                    │
                    └─── GRU (Simplified LSTM)
                         ✓ Solves vanishing gradients
                         ✓ Fast training (~10-15% faster)
                         ✓ Fewer parameters
                         ✓ Good on long sequences
                         → Use for: efficiency/prototyping


DETAILED COMPARISON:
──────────────────────────────────────────────────────────

Vanilla RNN:  1 Hidden State  →  LSTM:   2 States      →  GRU:  1 State
              0 Gates               3 Gates               2 Gates
              Fast (simple)         Slow (complex)        Medium (balanced)
              ✗ Vanishing          ✓ Solved              ✓ Solved
              ✗ Fails on long      ✓ Excellent           ✓ Good
```

---

**End of Unit 4 Visual Diagrams & Flowcharts**

