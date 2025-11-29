# UNIT 4: QUICK REVISION SHEET
## Recurrent Neural Networks (RNN) - Compact Study Guide

---

## 1. RNN - QUICK DEFINITIONS

**RNN (Recurrent Neural Network)**
- Neural network with loops/recurrent connections
- Maintains hidden state (memory) from previous time steps
- Processes sequential data word-by-word or frame-by-frame
- Same weights shared across all time steps

**Hidden State (h_t)**
- Internal memory of the RNN
- Formula: h_t = tanh(W_x·x_t + W_h·h_{t-1})
- Carries information from ALL previous inputs
- Updated at each time step

**Unfolding/Unrolling**
- Unwrapping RNN across time to show same network repeated
- Visualization technique
- Enables understanding BPTT algorithm

**Weight Sharing**
- Same W_x, W_h, W_y used at ALL time steps
- Not different weights at each layer
- Makes RNN parameter-efficient

---

## 2. CORE CONCEPTS AT A GLANCE

### RNN Applications
| Task | Example | Use |
|------|---------|-----|
| Language Modeling | "The cat ___ on the" → "sat" | Next word prediction |
| Machine Translation | English → German | Seq2Seq |
| Speech Recognition | Audio waveform → text | Acoustic modeling |
| Sentiment Analysis | Movie review → positive/negative | Classification |
| NER | "John lives in Paris" → PERSON, LOCATION | Tagging |

### N-gram Model Comparison

**Unigram**: Single word probability P(w)
- "cat" appears 1000/100000 times → P(cat) = 0.01

**Bigram**: Two-word probability P(w_i | w_{i-1})
- "the cat" / "the" → P(cat|the) = 0.8

**Trigram**: Three-word probability P(w_i | w_{i-2}, w_{i-1})
- "the big cat" / "the big" → P(cat|the,big) = 0.95

**N-gram Limitation**: Only looks back N-1 words (fixed window)

**RNN Advantage**: Theoretically unbounded context through hidden state

---

## 3. BACKPROPAGATION THROUGH TIME (BPTT) - SUMMARY

**What**: Training algorithm for RNNs that propagates errors backward through time

**How**:
1. Forward: Process entire sequence, compute outputs and losses
2. Backward: Error flows back through time (T → 1)
3. Accumulate: Gradients sum across all time steps
4. Update: Weights updated using accumulated gradients

**Key Difference**:
- Feedforward: Error flows backward through layers only
- RNN: Error flows backward through LAYERS AND TIME

**Truncated BPTT**:
- Only backprop through last k steps (e.g., k=50)
- Faster but loses very long-range dependencies
- Trade-off: speed vs learning capability

---

## 4. VANISHING GRADIENT PROBLEM - CONCISE EXPLANATION

**What**: Gradients become exponentially smaller (→ 0) as backprop goes backward through time

**Why**: 
```
Chain rule: multiply gradient by derivative repeatedly
Derivative of sigmoid: max value = 0.25
Result: 0.25 × 0.25 × 0.25 × ... = 0.25^T
For T=20: 0.25^20 ≈ 10^-12 (essentially zero!)
```

**Consequence**:
- Weights for early time steps don't update (gradient ≈ 0)
- Cannot learn long-range dependencies
- "The dog which was running ... barked" → can't connect "dog" and "barked"

**Solutions**:
| Solution | How | Pros | Cons |
|----------|-----|------|------|
| LSTM | Cell state + 3 gates | Solved problem | More complexity |
| GRU | 2 gates, simpler | Solved, faster | Slightly less powerful |
| Gradient Clipping | Cap gradient magnitude | Prevents explosion | Doesn't fix vanishing |
| Residual Connections | Allow direct paths | Better gradient flow | Architectural change |

---

## 5. LSTM (LONG SHORT-TERM MEMORY) - QUICK VERSION

### Architecture
**4 Gates & 1 Cell State**:

1. **Forget Gate** (f_t): What to remove from cell state
   ```
   f_t = σ(W_f·[h_{t-1},x_t] + b_f)
   0 = forget, 1 = keep
   ```

2. **Input Gate** (i_t) + Candidate (C̃_t): What to add
   ```
   i_t = σ(W_i·[h_{t-1},x_t] + b_i)
   C̃_t = tanh(W_c·[h_{t-1},x_t] + b_c)
   ```

3. **Cell State** (C_t): Long-term memory
   ```
   C_t = f_t⊙C_{t-1} + i_t⊙C̃_t
   (forget old + add new)
   ```

4. **Output Gate** (o_t): What to output
   ```
   o_t = σ(W_o·[h_{t-1},x_t] + b_o)
   h_t = o_t⊙tanh(C_t)
   ```

### Why LSTM Works
- **Addition** instead of **multiplication** in cell state update
- Gradient through cell: ∂C_t/∂C_{t-1} = f_t ≈ 0.7-0.9 (stays reasonable!)
- NOT 0.25^T (vanishes)
- Information preserved through time with minimal gradient loss

### LSTM Gate Roles (Mnemonics)
- **Forget**: "Should I forget this?" → 0=yes, 1=no
- **Input**: "What new info to add?" → which candidates, what values
- **Output**: "What to output?" → 0=hide, 1=show
- **Cell**: "Long-term storage" → update with forget+input

---

## 6. GRU (GATED RECURRENT UNIT) - SIMPLIFIED LSTM

### Architecture
**2 Gates & 1 Hidden State**:

1. **Reset Gate** (r_t): How much of h_{t-1} to use
   ```
   r_t = σ(W_r·[h_{t-1},x_t] + b_r)
   ```

2. **Update Gate** (z_t): New vs old balance
   ```
   z_t = σ(W_z·[h_{t-1},x_t] + b_z)
   ```

3. **Candidate** (h̃_t): Proposed new state
   ```
   h̃_t = tanh(W_h·[r_t⊙h_{t-1},x_t] + b_h)
   ```

4. **Hidden State Update**:
   ```
   h_t = (1-z_t)⊙h̃_t + z_t⊙h_{t-1}
   (new weight × new) + (old weight × old)
   ```

### GRU vs LSTM

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 | 2 |
| States | Cell + Hidden | Hidden only |
| Parameters | More (~4x) | Fewer (~3x) |
| Speed | Slower | Faster (~10-15%) |
| Vanishing Gradient | Solved ✓ | Solved ✓ |
| Complexity | Higher | Lower |
| Best For | Complex tasks | Efficiency |

**When to Use**:
- **GRU**: Limited resources, simple sequences, prototyping
- **LSTM**: Maximum performance, complex sequences, production

---

## 7. KEY FORMULAS - FORMULA SHEET

### Basic RNN
```
h_t = tanh(W_x·x_t + W_h·h_{t-1} + b_h)
y_t = softmax(W_y·h_t + b_y)
```

### LSTM (4 Gates)
```
f_t = σ(W_f·[h_{t-1},x_t] + b_f)                    [Forget]
i_t = σ(W_i·[h_{t-1},x_t] + b_i)                    [Input]
C̃_t = tanh(W_c·[h_{t-1},x_t] + b_c)                [Candidate]
C_t = f_t⊙C_{t-1} + i_t⊙C̃_t                         [Cell State]
o_t = σ(W_o·[h_{t-1},x_t] + b_o)                    [Output]
h_t = o_t⊙tanh(C_t)                                 [Hidden]
```

### GRU (2 Gates)
```
r_t = σ(W_r·[h_{t-1},x_t] + b_r)                    [Reset]
z_t = σ(W_z·[h_{t-1},x_t] + b_z)                    [Update]
h̃_t = tanh(W_h·[r_t⊙h_{t-1},x_t] + b_h)            [Candidate]
h_t = (1-z_t)⊙h̃_t + z_t⊙h_{t-1}                     [Hidden]
```

### BPTT Gradient
```
∂Loss/∂h_j = Σ_t (∂Loss/∂y_t × ∂y_t/∂h_t × Π(∂h_i/∂h_{i-1}))
             Product from t down to j
```

### N-gram Probability
```
P(w_n | w_{n-N+1},...,w_{n-1}) = Count(w_{n-N+1}...w_n) / Count(w_{n-N+1}...w_{n-1})
```

---

## 8. COMMON EXAM QUESTIONS - PREDICTED

| Rank | Topic | Probability | Marks | Key Focus |
|------|-------|-----------|-------|-----------|
| 1 | RNN Architecture + Language Models | 90% | 6 | Hidden state, unfolding, N-gram vs RNN |
| 2 | BPTT Algorithm | 85% | 5-6 | Forward, backward, weight sharing |
| 3 | Vanishing Gradient | 85% | 6 | Why occurs, consequences, solutions |
| 4 | LSTM Architecture | 85% | 5-6 | All 4 gates, cell state, why it works |
| 5 | GRU vs LSTM | 75% | 5-6 | Comparison, when to use each |
| 6 | N-gram Models | 70% | 4-5 | Limitations vs RNN |
| 7 | Backprop Comparison | 65% | 4 | Feedforward vs BPTT differences |

---

## 9. COMPARISON TABLE - QUICK REFERENCE

### RNN Architectures Comparison

| Feature | Vanilla RNN | LSTM | GRU |
|---------|-------------|------|-----|
| **Hidden States** | 1 | 2 (cell + hidden) | 1 |
| **Gates** | 0 | 3 | 2 |
| **Vanishing Gradient** | ✗ Severe | ✓ Solved | ✓ Solved |
| **Long Dependencies** | ✗ Fails | ✓ Excellent | ✓ Good |
| **Parameters** | Least | Most | Medium |
| **Computation** | Fastest | Slowest | Fast |
| **Training Time** | Fast | Slow | Fast |
| **On Simple Sequences** | OK | Good | Better |
| **On Complex Sequences** | Bad | Best | Good |
| **Recommended Use** | Educational | Production | Efficient |

### Gate Functions Summary

| Gate | Formula | Output | Role | Example |
|------|---------|--------|------|---------|
| **Forget (LSTM)** | σ(W_f·[h,x]) | 0-1 | What to discard | Forget number/gender |
| **Input (LSTM)** | σ(W_i·[h,x]) | 0-1 | Which new info | Add word importance |
| **Output (LSTM)** | σ(W_o·[h,x]) | 0-1 | What to output | Show processed info |
| **Reset (GRU)** | σ(W_r·[h,x]) | 0-1 | Which h to use | Proportion of h_{t-1} |
| **Update (GRU)** | σ(W_z·[h,x]) | 0-1 | New vs old balance | How much new state |

---

## 10. MISTAKES TO AVOID - COMMON ERRORS

❌ "RNN has different weights at each time step"
✓ CORRECT: Same weights W_x, W_h, W_y shared across all time steps

❌ "LSTM has 2 hidden states and 2 cell states"
✓ CORRECT: LSTM has 1 cell state (long-term) + 1 hidden state (short-term)

❌ "BPTT is just normal backpropagation for sequences"
✓ CORRECT: BPTT includes temporal backprop through previous time steps (extra)

❌ "Gradient clipping solves the vanishing gradient problem"
✓ CORRECT: Clipping prevents EXPLOSION; LSTM/GRU solve VANISHING

❌ "LSTM always better than GRU"
✓ CORRECT: GRU often comparable, faster, on many real tasks

❌ "N-gram models can capture any length dependency with large N"
✓ CORRECT: N-gram has sparsity; vocabulary^N becomes infeasible

❌ "RNN can remember arbitrary length sequences perfectly"
✓ CORRECT: Even LSTM/GRU have practical limits (~200-500 steps effective range)

❌ "Sigmoid has derivative = 1"
✓ CORRECT: Sigmoid max derivative = 0.25 (the source of vanishing gradient)

---

## 11. STUDY SCHEDULE - 5-DAY PREP

### Day 1: FUNDAMENTALS
- [ ] Understand RNN basic formula: h_t = tanh(W_x·x_t + W_h·h_{t-1})
- [ ] Master weight sharing concept
- [ ] Visualize unfolding in time
- [ ] Read: RNN architecture section

### Day 2: PROBLEMS & SOLUTIONS
- [ ] Learn vanishing gradient: 0.25^20 ≈ 10^-12
- [ ] Understand why N-grams insufficient
- [ ] Compare N-gram vs RNN language models
- [ ] Study: Vanishing gradient problem section

### Day 3: LSTM DEEP DIVE
- [ ] Master 4 LSTM gates: forget, input, output, cell state
- [ ] Understand why LSTM solves vanishing gradient
- [ ] Practice: h_t = f_t⊙C_{t-1} + i_t⊙C̃_t formula
- [ ] Study: LSTM section

### Day 4: GRU & BPTT
- [ ] Learn GRU 2 gates: reset, update
- [ ] LSTM vs GRU comparison
- [ ] Understand BPTT algorithm steps
- [ ] Study: GRU section + BPTT section

### Day 5: EXAM PREP
- [ ] Practice writing full answers to questions 1-5
- [ ] Do timed mock exam (~12 min per 6-mark question)
- [ ] Review comparison tables
- [ ] Go through mistakes to avoid

---

## 12. LAST-MINUTE REVIEW - 5 MINS

### MUST REMEMBER
1. **RNN Formula**: h_t = tanh(W_x·x_t + W_h·h_{t-1})
2. **Weight Sharing**: Same weights across ALL time steps
3. **Vanishing**: 0.25^T becomes tiny (T = time steps)
4. **LSTM Solution**: Addition (cell state) instead of multiplication chain
5. **LSTM Gates**: Forget (what to remove), Input (what to add), Output (what to show)
6. **GRU**: 2 gates (reset, update), simpler than LSTM
7. **N-gram**: Limited context, sparsity problem
8. **BPTT**: Backprop through time, error flows backward in time

---

## 13. QUICK REFERENCE - GATE EQUATIONS

### LSTM
```
Forget:  f_t = σ(Wf·[h_{t-1},x_t])   → 0=forget, 1=keep
Input:   i_t = σ(Wi·[h_{t-1},x_t])   → which candidates
         C̃_t = tanh(Wc·[h_{t-1},x_t]) → what values
Cell:    C_t = f_t⊙C_{t-1} + i_t⊙C̃_t → update memory
Output:  o_t = σ(Wo·[h_{t-1},x_t])   → 0=hide, 1=show
         h_t = o_t⊙tanh(C_t)         → output
```

### GRU
```
Reset:   r_t = σ(Wr·[h_{t-1},x_t])           → which h to use
Update:  z_t = σ(Wz·[h_{t-1},x_t])           → balance new/old
Cand:    h̃_t = tanh(Wh·[r_t⊙h_{t-1},x_t])    → new state
Hidden:  h_t = (1-z_t)⊙h̃_t + z_t⊙h_{t-1}     → final hidden
```

---

## 14. EXAM TIPS

**When Answering RNN Questions**:
1. Always mention hidden state and weight sharing
2. Draw unfolding diagram if space permits
3. Use concrete examples (e.g., "cat sat on")
4. Explain why LSTM better than vanilla RNN
5. Mention vanishing gradient naturally when discussing training

**Time Management**:
- 6-mark question: 12 minutes
- Spend 2 mins planning
- 8 mins writing
- 2 mins review

**Answer Structure**:
1. Definition (0.5 marks) - 1-2 lines
2. Architecture (1-2 marks) - detailed explanation + formula
3. Example (1 mark) - concrete illustration
4. Comparison/Applications (1-2 marks) - show depth

---

**End of Unit 4 Quick Revision Sheet**
**Total Preparation Time: 5-6 hours**
**Good Luck on Your Exam! 🎓**
