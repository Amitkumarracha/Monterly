# UNIT 1: INTRODUCTION TO LANGUAGE & NLP
## Comprehensive Study Guide with Complete Answers & Revision Sheet

---

## TABLE OF CONTENTS
1. Evolution of Natural Language
2. Introduction to NLP & Need of NLP
3. Applications of NLP
4. Phases of NLP
5. Data Preprocessing Techniques
6. Named Entity Recognition
7. Predicted Exam Questions with Answers
8. Quick Revision Sheet

---

# PART 1: DETAILED EXPLANATIONS

## 1. EVOLUTION OF NATURAL LANGUAGE

### 1.1 Definition & Overview

**Natural Language**: Language used by human beings for communication, such as English, Hindi, Spanish, Chinese, etc. It is a system of communication that evolved naturally among people.

**Characteristics of Natural Language**:
- Complex and ambiguous
- Context-dependent meaning
- Evolved over time and continues to evolve
- Has grammar, syntax, and semantics
- Speakers can understand novel combinations
- Contains idioms, metaphors, and cultural references

### 1.2 Historical Evolution of Language

**Stage 1: Pre-Written Language (50,000 BCE - 3100 BCE)**
- Oral communication only
- Gestural and symbolic communication
- Development of first symbols and pictograms
- No standardization
- Information passed through oral tradition

**Stage 2: Written Language (3100 BCE - 1440 CE)**
- Cuneiform script (3200 BCE, Sumerian)
- Egyptian hieroglyphics (3100 BCE)
- Chinese characters (1200 BCE)
- Alphabetic writing systems (Phoenician, Greek, Latin)
- Standardization began
- Better information preservation
- Example: Rosetta Stone helped decode Egyptian hieroglyphics

**Stage 3: Printed Language (1440 CE - 1900s)**
- Printing press invented by Gutenberg (1440)
- Mass production of written materials
- Standardization of spelling and grammar
- Increased literacy rates
- Formation of dictionaries and grammar rules
- Development of formal language standards

**Stage 4: Digital Language (1900s - Present)**
- Telephone (1876) - voice transmission
- Telegraph (1844) - digital encoding of language
- Television (1920s) - visual communication
- Internet (1990s) - text, voice, video communication
- Social media - new language forms (hashtags, emojis, abbreviations)
- Emergence of programming languages

**Stage 5: AI-Era Language (2010s - Present)**
- Machine understanding of language
- Neural networks processing text
- Voice assistants (Alexa, Siri)
- Real-time translation
- Chatbots and conversational AI
- Large language models (GPT, BERT)

### 1.3 Linguistic Aspects of Language Evolution

**Phonological Evolution**: Changes in sounds/pronunciation
- Old English: "hus" → Modern English: "house"
- Latin: "caballus" → English: "horse" → Spanish: "caballo"

**Morphological Evolution**: Changes in word structure
- Old English: "singeth" → Modern English: "sings"
- Loss of inflectional endings
- Simplification of grammar

**Lexical Evolution**: Changes in vocabulary
- New words added (smartphone, internet, COVID-19)
- Old words disappear or change meaning (gay: happy → homosexual)
- Borrowing from other languages (pizza from Italian, restaurant from French)

**Semantic Evolution**: Changes in meaning
- Awful: originally "full of awe" (positive) → now negative
- Nice: originally "foolish" → now "pleasant"

### 1.4 From Natural to Computational Language

**Challenges in Moving to Computation**:

1. **Ambiguity**
   - Lexical ambiguity: "bank" (financial institution vs. river side)
   - Syntactic ambiguity: "I saw the man with the telescope"
   - Semantic ambiguity: "Time flies" (noun + verb vs. idiom)

2. **Context Dependency**
   - Same words have different meanings in different contexts
   - Pronouns need antecedent resolution
   - Metaphors and idioms

3. **Variability**
   - Same meaning expressed in multiple ways
   - Dialects and regional variations
   - Informal vs. formal language

---

## 2. INTRODUCTION TO NLP & NEED OF NLP

### 2.1 Definition of NLP

**Natural Language Processing (NLP)**: Field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language in a meaningful and useful way.

**Core Goal**: Bridge the gap between human communication and computer understanding.

**Key Formula of NLP**:
```
NLP = Computational Linguistics + Machine Learning + AI + Linguistics
      
Where:
- Computational Linguistics: Rules for language processing
- Machine Learning: Learning patterns from data
- AI: Intelligence to make decisions
- Linguistics: Understanding language structure
```

### 2.2 Why Do We Need NLP?

#### Problem 1: Human-Computer Communication Gap
- Computers understand binary and structured data
- Humans communicate in natural, unstructured language
- NLP bridges this gap

**Example**: 
```
User: "What's the weather like tomorrow?"
Computer without NLP: ??? (cannot understand)
Computer with NLP: 
  1. Understands: Query about weather
  2. Extracts: Time = tomorrow, Topic = weather
  3. Retrieves: Weather data for next day
  4. Responds: "It will be sunny, 25°C"
```

#### Problem 2: Information Explosion
- Billions of documents created daily
- Humans cannot manually process all information
- Need automated analysis and extraction

**Statistics**:
- 2.5 quintillion bytes of data created daily (2024)
- 500+ million tweets per day
- Millions of reviews, articles, emails daily

#### Problem 3: Unlocking Unstructured Data
- 80-90% of data is unstructured (text, voice, images)
- Valuable insights hidden in this data
- Need methods to extract and analyze

**Example**:
```
Customer Review: "The product is great but delivery was slow"
Structured Data: Rating = ???

NLP extracts:
- Product sentiment: POSITIVE
- Delivery sentiment: NEGATIVE
- Issues: Delivery time
- Strengths: Product quality
```

#### Problem 4: Accessibility and Inclusion
- Enable communication for people with disabilities
- Speech-to-text for deaf individuals
- Text-to-speech for blind individuals
- Real-time translation for language barriers

#### Problem 5: Scale and Efficiency
- Automate tasks that would take humans months
- Enable personalization at scale
- Reduce human error

### 2.3 Scope and Importance of NLP

```
NLP Impact Areas:
─────────────────
Scope:
├─ Text Processing (written language)
├─ Speech Processing (spoken language)
├─ Sentiment Understanding
├─ Knowledge Extraction
├─ Language Generation
└─ Dialogue Systems

Importance:
├─ Business: Market insights, customer service
├─ Healthcare: Medical record analysis, diagnosis
├─ Education: Personalized learning
├─ Government: Policy analysis, public opinion
├─ Research: Scientific paper analysis
└─ Daily Life: Translation, search, recommendations
```

---

## 3. APPLICATIONS OF NLP

### 3.1 Major NLP Applications

#### A. **Sentiment Analysis / Opinion Mining**

**Definition**: Process of identifying, extracting, and analyzing opinions, sentiments, and emotions from text.

**How It Works**:
1. Extract opinion-bearing sentences
2. Determine sentiment polarity (positive/negative/neutral)
3. Assign sentiment strength/score
4. Aggregate results

**Applications**:
- Product reviews analysis (Amazon, Flipkart)
- Social media monitoring (track brand reputation)
- Customer feedback analysis
- Political opinion tracking
- Movie/book ratings analysis

**Example**:
```
Review: "The phone has great camera but battery life is poor"
Opinion 1: Camera → POSITIVE
Opinion 2: Battery life → NEGATIVE
Overall: MIXED sentiment

Sentiment Score: -0.3 (slightly negative)
```

**Business Impact**:
- Companies adjust products based on feedback
- Improve customer service
- Detect brand crises early
- Identify improvement areas

#### B. **Machine Translation**

**Definition**: Automatic translation of text or speech from one language to another.

**Types**:
1. Rule-based MT: Uses linguistic rules
2. Statistical MT: Uses probability models
3. Neural MT: Uses deep learning (modern approach)

**Examples**:
- Google Translate
- Microsoft Translator
- Baidu Translate

**Challenges**:
- Preserving meaning across languages
- Handling idioms and cultural references
- Word order differences
- Context-dependent translation

**Impact**: 
- Breaking language barriers
- Enabling global communication
- Supporting multilingual businesses

#### C. **Speech Recognition / Speech-to-Text**

**Definition**: Converting spoken language to written text.

**Applications**:
- Voice assistants (Alexa, Google Assistant, Siri)
- Dictation software
- Automated transcription
- Voice commands in cars

**Technologies**:
- Acoustic models (understand sound)
- Language models (understand word sequences)
- Deep learning (end-to-end learning)

#### D. **Information Extraction**

**Definition**: Automatically extracting structured information from unstructured text.

**Types of Extraction**:
1. Named Entity Recognition (NER): People, places, organizations
2. Relation Extraction: Relationships between entities
3. Event Extraction: What happened, when, where

**Example**:
```
Text: "Apple CEO Tim Cook announced earnings of $25 billion in Q3 2024"

Extracted Information:
- Company: Apple
- Person: Tim Cook, Title: CEO
- Metric: Earnings = $25 billion
- Time: Q3 2024
```

**Applications**:
- Resume parsing
- Biomedical literature analysis
- News analysis
- Legal document processing

#### E. **Dialogue Systems / Chatbots**

**Definition**: Systems designed to simulate conversation with users.

**Types**:
1. **Task-oriented Dialogue**: Help users accomplish specific goals
   - Booking flights, restaurants
   - Customer service bots
   - Information retrieval

2. **Open-domain Dialogue**: General conversation
   - Conversational companions
   - Social chatbots
   - Entertainment

**Evolution**:
- ELIZA (1966): Pattern matching based
- ALICE (1995): Keyword matching
- Modern Chatbots: Deep learning based (GPT, BERT)

**Applications**:
- Customer service automation (reduce wait time, 24/7 availability)
- Technical support
- Mental health support (companions)
- Virtual assistants
- Language learning partners

#### F. **Question Answering Systems**

**Definition**: Systems that answer questions asked in natural language.

**Approaches**:
1. Retrieval-based: Find relevant documents, extract answers
2. Knowledge-based: Query structured knowledge bases
3. Generative: Generate answers from context

**Examples**:
- Google Search
- Alexa answers
- ChatGPT Q&A
- IBM Watson

#### G. **Text Summarization**

**Definition**: Automatically generating concise summary of text.

**Types**:
1. Extractive: Select key sentences from original
2. Abstractive: Generate new sentences capturing meaning

**Applications**:
- News summarization
- Long document condensation
- Email summarization
- Research paper summarization

#### H. **Named Entity Recognition (NER)**

**Definition**: Identifying and classifying named entities in text.

**Entity Types**:
- Person: John, Mary, Obama
- Organization: Apple, Google, UN
- Location: Paris, USA, Amazon
- Date: January 1, 2024
- Time: 3:30 PM
- Money: $100, €50
- Percentage: 50%, 25%

**Applications**:
- News analysis: Extract who, where, what
- Resume parsing: Identify skills, companies
- Biomedical: Extract drug names, diseases

### 3.2 Discourse Analysis

**Definition**: Study of language use beyond individual sentences, examining how discourse is organized and structured.

**Key Components**:
- **Anaphora**: Reference to previous mentions
  ```
  "John went to the store. He bought milk."
  "He" refers to "John"
  ```

- **Coreference Resolution**: Identifying when different expressions refer to same entity
  ```
  "The president"
  "Obama"
  "He"
  All refer to same person
  ```

- **Discourse Relations**: How sentences connect logically
  ```
  "It was raining. Therefore, she took an umbrella."
  Relation: CAUSE-EFFECT
  ```

**Applications**:
- Understanding document structure
- Improving translation quality
- Question answering requiring context
- Summarization requiring coherence

### 3.3 Dialogue Analysis

**Definition**: Analysis of conversations, including turn-taking, speech acts, and dialogue states.

**Key Aspects**:
1. **Turn-taking**: Who speaks when
2. **Speech Acts**: Intention behind utterance
   - Request: "Can you pass the salt?"
   - Statement: "The sun is hot"
   - Question: "What's your name?"
   - Command: "Sit down"

3. **Dialogue State**: What has been discussed, what's been agreed

4. **Dialogue Acts**: Different types of contributions
   - Opening/Closing
   - Clarification requests
   - Confirmations
   - Disagreements

**Applications**:
- Building better chatbots
- Understanding conversations
- Dialogue-based information retrieval
- Conversation mining for insights

---

## 4. PHASES OF NLP

### 4.1 NLP Processing Phases (Pipeline)

NLP typically follows a structured pipeline of processing phases:

```
Raw Text Input
     │
     ▼
┌─────────────────────────┐
│ 1. TEXT ACQUISITION     │ ← Raw data collection
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 2. TEXT PREPROCESSING   │ ← Cleaning & preparation
│   - Tokenization        │
│   - Normalization       │
│   - Stop word removal   │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 3. FEATURE EXTRACTION   │ ← Convert to usable features
│   - Embeddings          │
│   - Bag of words        │
│   - TF-IDF              │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 4. PARSING              │ ← Analyze structure
│   - Lexical analysis    │
│   - Syntactic analysis  │
│   - Semantic analysis   │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 5. MODEL TRAINING       │ ← Train on data
│   - Supervised learning │
│   - Unsupervised        │
│   - Deep learning       │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 6. MODEL EVALUATION     │ ← Test performance
│   - Accuracy            │
│   - Precision/Recall    │
│   - F1-score            │
└─────────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ 7. DEPLOYMENT           │ ← Use in production
└─────────────────────────┘
```

### 4.2 Phase 1: Lexical & Morphological Analysis

**Definition**: Understanding words and their internal structure.

**Components**:
1. **Tokenization**: Split text into tokens (words/subwords)
2. **Morphological Analysis**: Analyzing word structure
   - Prefix: un-, re-, pre-
   - Root: run, walk, talk
   - Suffix: -ing, -tion, -ness

**Example**:
```
Word: "playing"
Morphological Analysis:
- Prefix: (none)
- Root: play
- Suffix: -ing (present participle)
- Meaning: Currently performing action
```

### 4.3 Phase 2: Syntactic Analysis (Parsing)

**Definition**: Analyzing sentence structure and grammatical relationships.

**Two Main Approaches**:

1. **Constituency Parsing**: Breaking into noun phrases, verb phrases
   ```
   Sentence: "The quick brown fox jumps"
   
   Parse Tree:
           S (Sentence)
          / \
         NP  VP
        /|   |
       DT A  A   N   V
       |  |  |   |   |
      The quick brown fox jumps
   
   NP = Noun Phrase
   DT = Determiner
   A = Adjective
   N = Noun
   V = Verb
   ```

2. **Dependency Parsing**: Relationships between words
   ```
   Sentence: "The quick brown fox jumps"
   
   Dependency Relations:
   - "fox" is head, "the", "quick", "brown" are dependents
   - "jumps" has "fox" as dependent
   - Relation: Subject-Verb relationship
   ```

### 4.4 Phase 3: Semantic Analysis

**Definition**: Understanding meaning of words and sentences.

**Components**:
1. **Word Sense Disambiguation**: Multiple meanings
   ```
   "Bank" can mean:
   - Financial institution
   - River side
   Need context to disambiguate
   ```

2. **Entity Recognition**: Identify named entities

3. **Relationship Extraction**: Find relations between entities

4. **Semantic Role Labeling**: Who did what to whom
   ```
   "John gave Mary a book"
   - Agent: John (who did it)
   - Action: gave
   - Recipient: Mary (to whom)
   - Object: book (what)
   ```

### 4.5 Phase 4: Pragmatic Analysis

**Definition**: Understanding meaning based on context and speaker intent.

**Components**:
1. **Speech Acts**: What the speaker intends
2. **Implicatures**: Implied meanings
3. **Reference Resolution**: Who/what does pronoun refer to

**Example**:
```
"Can you pass the salt?"
- Literal: Question about ability
- Pragmatic: Request to pass salt
- Context matters for interpretation
```

---

## 5. DATA PREPROCESSING TECHNIQUES

### 5.1 Overview

**Definition**: Process of cleaning and preparing raw text data for analysis.

**Why Important**:
- Raw text is messy, contains noise
- Preprocessing improves model performance
- Reduces irrelevant information
- Standardizes data format

### 5.2 Tokenization

**Definition**: Breaking text into individual tokens (words, subwords, or characters).

**Importance**: Fundamental step in NLP pipeline

**Types**:

#### A. **Sentence Tokenization**
```
Text: "Hello world. This is NLP."
Output: ["Hello world.", "This is NLP."]

Challenges:
- "Dr. Smith" - period is abbreviation, not sentence end
- "U.S.A." - multiple periods
- Question marks, exclamation marks
```

#### B. **Word Tokenization**
```
Text: "I don't like it."
Output: ["I", "don't", "like", "it", "."]

OR (handling contractions):
Output: ["I", "do", "n't", "like", "it", "."]

OR (keeping punctuation attached):
Output: ["I", "don't", "like", "it."]
```

#### C. **Subword Tokenization**
```
Word: "unbelievable"
Subword tokens: ["un", "##believe", "##able"]

Advantages:
- Handles rare words
- Reduces vocabulary size
- Better for morphologically rich languages
```

#### D. **Character Tokenization**
```
Word: "hello"
Character tokens: ["h", "e", "l", "l", "o"]

Uses:
- Character-level models
- Handling misspellings
- Languages without clear word boundaries (Chinese)
```

### 5.3 Normalization

**Definition**: Converting text to standard form for consistent processing.

**Techniques**:

#### A. **Lowercasing**
```
Original: "Hello World NLP"
Normalized: "hello world nlp"

Advantage: Treat "Hello" and "hello" as same word
Disadvantage: Lose information about proper nouns
```

#### B. **Removing Punctuation**
```
Original: "Hello, how are you?"
Normalized: "Hello how are you"

Methods:
- Remove all punctuation
- Keep important ones (apostrophes for contractions)
```

#### C. **Removing Numbers/Digits**
```
Original: "The meeting is on 2024-01-15 at 3:30 PM"
Normalized: "The meeting is on at PM"

When to use:
- Classification tasks (numbers add noise)
- When numbers not relevant
```

#### D. **Handling Whitespace**
```
Original: "Hello    world  NLP"  (multiple spaces)
Normalized: "Hello world NLP"

Fixes:
- Multiple spaces
- Tabs
- Line breaks
- Leading/trailing spaces
```

#### E. **Accent Removal / Diacritics**
```
Original: "café, naïve, résumé"
Normalized: "cafe, naive, resume"

For:
- French, Spanish, German text
- When accents not meaningful
```

### 5.4 Stemming

**Definition**: Reducing words to root form by removing suffixes and prefixes.

**Algorithm**: Rule-based approach

**Examples**:
```
running → run
running, runs, runner → run
playing, play, played → play
connection, connecting, connected → connect

Porter Stemmer (most common):
word "ponies" → "poni"
word "caresses" → "caress"
```

**Disadvantages**:
- Crude/oversimplified
- Can produce non-words
- Loses semantic information

```
Example issues:
"universe" → "univers" (not a real word)
"studies" → "studi" (not a real word)
```

### 5.5 Lemmatization

**Definition**: Reducing words to dictionary base form (lemma) using morphological analysis.

**Better than Stemming**:
- Uses vocabulary and morphological analysis
- Produces valid words
- Context-aware

**Examples**:
```
running, runs, ran → run (verb form matters)
studies, study, studied → study
better, best, good → good
```

**Process**:
1. Part-of-speech tagging
2. Dictionary lookup
3. Morphological rules applied based on POS

**Comparison**:
```
Word: "easily"

Stemming: "easili" (non-word)
Lemmatization: "easy" (real word)

Word: "writing"

Stemming: "write" (might work)
Lemmatization: "write" (correct)
```

### 5.6 Stop Words Removal

**Definition**: Removing common words that carry little semantic meaning.

**Common Stop Words**:
- Articles: "a", "an", "the"
- Pronouns: "I", "you", "he", "she"
- Prepositions: "in", "on", "at", "to"
- Conjunctions: "and", "or", "but"
- Verbs: "is", "are", "be"

**Example**:
```
Original: "The quick brown fox jumps over the lazy dog"
After removal: "quick brown fox jumps lazy dog"

Advantages:
- Reduces noise
- Faster processing
- Focus on meaningful words

Disadvantages:
- Lose context in some cases
- "not bad" becomes "bad" (lose negation)
- "a lot" becomes "lot" (lose meaning)
```

**When NOT to remove**:
- Sentiment analysis ("not good" changes meaning)
- Machine translation (article systems matter)
- Dependency parsing (prepositions are important)

### 5.7 Word Embeddings

**Definition**: Converting words into numerical vectors capturing semantic meaning.

**Types**:

#### A. **One-Hot Encoding** (Older)
```
Vocabulary: {cat, dog, mouse}
cat = [1, 0, 0]
dog = [0, 1, 0]
mouse = [0, 0, 1]

Problems:
- High dimensionality
- No semantic relationship
- Sparse vectors
```

#### B. **Word2Vec** (Modern, Popular)
```
cat = [0.25, -0.18, 0.93, 0.34, ...]
dog = [0.24, -0.17, 0.92, 0.35, ...]

Properties:
- Dense vectors (300-500 dims usually)
- Similar words have similar vectors
- Can do word arithmetic:
  king - man + woman ≈ queen
```

#### C. **GloVe**
```
Combines context window and matrix factorization
Captures both local and global context
```

#### D. **Contextual Embeddings (BERT, GPT)**
```
Same word has different embedding based on context
"bank" in "river bank" vs "bank account" → different vectors
More accurate, captures meaning variation
```

---

## 6. NAMED ENTITY RECOGNITION (NER)

### 6.1 Definition

**Named Entity Recognition (NER)**: Process of identifying and classifying named entities in text into predefined categories.

**Purpose**: Extract meaningful entities that carry important information.

### 6.2 Entity Types & Categories

| Entity Type | Definition | Examples |
|-------------|-----------|----------|
| **PERSON** | Individual names | John, Mary, Einstein, Obama |
| **ORGANIZATION** | Companies, institutions, groups | Apple, Google, United Nations, Harvard |
| **LOCATION** | Geographical places | Paris, USA, Amazon River, Tokyo |
| **DATE** | Temporal references | January 1, 2024, Q3 2024 |
| **TIME** | Time references | 3:30 PM, noon, midnight |
| **MONEY** | Monetary values | $100, €50, ¥1000 |
| **PERCENTAGE** | Percentages | 50%, 25% |
| **PRODUCT** | Products/brands | iPhone, Windows 10 |
| **EVENT** | Named events | World War II, Olympics 2024 |
| **FACILITY** | Buildings, infrastructure | Empire State Building, Statue of Liberty |
| **LANGUAGE** | Language names | English, Mandarin, Hindi |
| **GPE** | Geopolitical entities | USA, France, Canada |

### 6.3 NER Approaches

#### A. **Rule-Based Approach**

**Method**: Use handcrafted rules and patterns

**Components**:
1. **Regular Expressions**: Pattern matching
   ```
   Pattern for email: \w+@\w+\.\w+
   Pattern for phone: \d{3}-\d{3}-\d{4}
   ```

2. **Gazetteers**: Lists of known entities
   ```
   City gazetteer: ["Paris", "London", "Tokyo", ...]
   Company gazetteer: ["Apple", "Google", "Microsoft", ...]
   ```

3. **Linguistic Patterns**: Using POS tags and syntax
   ```
   Person pattern: (Title) (First Name) (Last Name)
   Example: Mr. John Smith → PERSON
   ```

**Advantages**:
- Interpretable
- Works well for specific patterns
- No labeled data needed

**Disadvantages**:
- Hard to scale
- Misses new entities
- Requires domain expertise
- Not adaptable

#### B. **Machine Learning Approach**

**Method**: Train classifier on annotated data

**Algorithms**:
1. **Hidden Markov Models (HMM)**
   - Probabilistic model
   - Considers word sequences
   
2. **Conditional Random Fields (CRF)**
   - More powerful than HMM
   - Can model complex features
   - Standard for sequence labeling

3. **Support Vector Machines (SVM)**
   - Classification algorithm
   - Works with engineered features

**Process**:
```
1. Feature Engineering
   - Word shape: "Xxx" (first letter capital)
   - POS tags: noun, verb, etc.
   - Prefixes/suffixes
   - Dictionary membership
   - Word embeddings

2. Training
   - Labeled data: "John | PERSON | works | at | Apple | ORG"
   - Model learns patterns

3. Prediction
   - Unseen text: "Mary works at Google"
   - Model predicts: "Mary | PERSON | Google | ORG"
```

**Advantages**:
- More accurate than rules
- Can learn complex patterns
- Handles variations

**Disadvantages**:
- Requires labeled data (expensive)
- Feature engineering needed
- Worse for new domains

#### C. **Deep Learning Approach**

**Method**: Neural networks learn representations

**Architectures**:

1. **BiLSTM (Bidirectional LSTM)**
   - Processes text forward and backward
   - Captures context from both directions
   - Output: Entity tag for each word

2. **BiLSTM-CRF**
   - Combines BiLSTM (learns patterns) with CRF (ensures valid tag sequences)
   - Better than BiLSTM alone
   - Most popular neural approach

3. **Transformer Models**
   - **BERT**: Pre-trained bidirectional model
   - **SpaCy**: Modern transformer pipelines
   - Better accuracy, handles nuances

**Advantages**:
- High accuracy (90%+ F1 score)
- No feature engineering
- Pre-trained models available
- Handles complex cases

**Disadvantages**:
- Computationally expensive
- Requires large labeled datasets
- Harder to interpret

### 6.4 NER Evaluation Metrics

**Metrics**:

| Metric | Definition | Formula |
|--------|-----------|---------|
| **Precision** | Of predicted entities, how many are correct? | TP/(TP+FP) |
| **Recall** | Of actual entities, how many found? | TP/(TP+FN) |
| **F1-Score** | Harmonic mean of precision & recall | 2×(P×R)/(P+R) |
| **Accuracy** | Overall correctness | (TP+TN)/(Total) |

Where:
- TP = True Positives (correct entities found)
- FP = False Positives (incorrect predictions)
- FN = False Negatives (missed entities)
- TN = True Negatives

**Example**:
```
Text: "John works at Apple in New York"

Predicted: John (PERSON), Apple (ORG), New York (LOCATION)
Actual: John (PERSON), Apple (ORG), New York (LOCATION)

Results:
- TP = 3 (all correct)
- FP = 0 (no false predictions)
- FN = 0 (no missed entities)

Precision = 3/3 = 1.0
Recall = 3/3 = 1.0
F1 = 1.0 (perfect!)
```

### 6.5 NER Applications

1. **Information Extraction**
   - Extract people, companies, locations from news
   - Build knowledge bases

2. **Resume Parsing**
   - Extract skills, experiences, companies
   - Automate hiring process

3. **Biomedical NER**
   - Extract disease names, drug names
   - Analyze medical literature

4. **Question Answering**
   - Identify what type of answer needed
   - "Who is Einstein?" → needs PERSON entity

5. **Recommendation Systems**
   - Identify entities users care about
   - Personalize recommendations

---

# PART 2: PREDICTED EXAM QUESTIONS WITH COMPLETE ANSWERS

## QUESTION 1: EVOLUTION OF LANGUAGE & NLP INTRODUCTION (PROBABILITY: 90%)

### Question:
**"Explain the evolution of natural language from pre-written language to the AI era. Why is NLP needed? Discuss the challenges in processing natural language."**

### Complete Answer:

**Evolution of Natural Language (2 marks)**

Natural language has evolved through several stages over millennia, each reflecting technological and societal changes:

**Stage 1: Pre-Written Language (50,000-3,100 BCE)**
- Only oral communication existed
- Information was transmitted through spoken word and memorization
- Gestures and symbols used for basic communication
- No standardization; dialects varied
- Limited information retention and spread

**Stage 2: Written Language (3,100 BCE-1440 CE)**
- Cuneiform (Sumerian, 3200 BCE) and Egyptian hieroglyphics (3100 BCE) first writing systems
- Later alphabetic systems: Phoenician, Greek, Latin
- Information could be preserved permanently
- Standardization began through formal writing rules
- Spread of knowledge became possible

**Stage 3: Printed Language (1440-1900)**
- Gutenberg's printing press (1440) revolutionized information distribution
- Mass production of books
- Grammar and spelling standardization
- Rise of dictionaries (Johnson's Dictionary 1755, Oxford English Dictionary)
- Literacy rates increased

**Stage 4: Digital Language (1900-Present)**
- Telegraph (1844): First digital communication
- Telephone (1876): Voice transmission
- Television (1920s): Audiovisual communication
- Internet (1990s): Global text, voice, video communication
- Social media (2000s): New language forms (hashtags, emojis, abbreviations)
- SMS/WhatsApp created new writing conventions

**Stage 5: AI-Era Language (2010-Present)**
- Machine understanding of language began
- Voice assistants (Alexa, Siri, Google Assistant)
- Real-time translation technology
- Chatbots and conversational AI
- Large language models (BERT, GPT-2, GPT-3, ChatGPT)
- Multimodal models understanding text, images, audio

**Why NLP Is Needed (1.5 marks)**

**Problem 1: Communication Gap**
- Computers understand binary and structured data
- Humans naturally use unstructured, ambiguous language
- Gap exists between human communication and computer understanding
- Solution: NLP acts as bridge

**Problem 2: Data Explosion**
- 2.5 quintillion bytes of data created daily
- 500 million tweets, 300 million emails daily
- Humans cannot manually process this volume
- NLP enables automated processing and analysis

**Problem 3: Extracting Value from Unstructured Data**
- 80-90% of data is unstructured (text, voice)
- Valuable insights hidden in customer reviews, social media, documents
- Example: Customer says "Product great but delivery slow" → NLP extracts both positive and negative aspects

**Problem 4: Accessibility & Inclusion**
- Speech-to-text for deaf individuals
- Text-to-speech for blind individuals
- Real-time translation overcomes language barriers
- Enables communication for all

**Problem 5: Scale and Efficiency**
- Tasks taking humans weeks completed in seconds
- Personalization at massive scale (millions of users)
- Cost reduction: Automation reduces human effort

**Challenges in Processing Natural Language (1.5 marks)**

**Challenge 1: Ambiguity**
- **Lexical Ambiguity**: Words with multiple meanings
  Example: "bank" (financial institution vs. river side)
  
- **Syntactic Ambiguity**: Multiple ways to parse sentence
  Example: "I saw the man with the telescope"
  (Did I use telescope to see? Or did man have telescope?)
  
- **Semantic Ambiguity**: Different interpretations
  Example: "Time flies" (saying time passes quickly OR command to measure flies)

**Challenge 2: Context Dependency**
- Same words mean different things in different contexts
- Pronouns need antecedent resolution
- "The bank manager approved the loan. She was experienced"
  → "She" refers to manager, not bank

**Challenge 3: Variability**
- Same meaning expressed in infinite ways
- Dialects and regional variations
- Formal vs. informal language
- "The rapid brown fox leaps" vs. "The quick brown fox jumps"
  → Same meaning, different words

**Challenge 4: Idioms and Metaphors**
- "It's raining cats and dogs" ≠ literally raining animals
- "Break the ice" ≠ physically break ice
- Computers must understand figurative language

**Challenge 5: Cultural and Linguistic Knowledge**
- "The White House announced..." ≠ physical white house
- "Apple released iOS 18" ≠ fruit
- World knowledge needed

---

## QUESTION 2: NLP APPLICATIONS & PHASES (PROBABILITY: 85%)

### Question:
**"Describe various applications of NLP. Explain the different phases of NLP processing pipeline. Give examples for each application."**

### Complete Answer:

**NLP Applications (2.5 marks)**

**1. Sentiment Analysis / Opinion Mining**
- **Definition**: Identifying sentiment (positive, negative, neutral) from text
- **Example**: Amazon reviews
  ```
  Review: "Phone has amazing camera but terrible battery"
  → Camera: POSITIVE, Battery: NEGATIVE
  → Overall: MIXED
  ```
- **Business Use**: Adjust products based on feedback, detect crises, improve customer service
- **Scale**: Analyze millions of reviews daily

**2. Machine Translation**
- **Definition**: Translating text from one language to another
- **Examples**: Google Translate, Microsoft Translator
- **Challenge**: Preserving meaning, handling idioms
- **Impact**: Breaking language barriers, enabling global communication

**3. Dialogue Systems & Chatbots**
- **Definition**: Systems simulating conversation
- **Examples**: Alexa, Google Assistant, ChatGPT
- **Evolution**: ELIZA (1966) → ALICE → Modern AI chatbots
- **Application**: Customer service (24/7, reduce costs), personal assistants, language learning

**4. Information Extraction**
- **Definition**: Automatically extracting structured information from text
- **Example**: "Apple CEO Tim Cook announced $25B earnings in Q3 2024"
  → Extract: Company=Apple, Person=Tim Cook, Amount=$25B, Period=Q3 2024
- **Use**: Resume parsing, news analysis, biomedical literature mining

**5. Question Answering Systems**
- **Definition**: Answering questions asked in natural language
- **Examples**: Google Search, ChatGPT, IBM Watson
- **Methods**: 
  - Retrieval-based: Find relevant documents, extract answer
  - Generative: Generate answer from knowledge

**6. Text Summarization**
- **Definition**: Generating concise summary of longer text
- **Types**: 
  - Extractive: Select key sentences
  - Abstractive: Generate new sentences
- **Application**: News summaries, document condensation, email summaries

**7. Named Entity Recognition**
- **Definition**: Identifying and classifying named entities
- **Example**: "John from Apple loves Paris"
  → John (PERSON), Apple (ORGANIZATION), Paris (LOCATION)
- **Use**: Information extraction, question answering, biomedical analysis

**8. Discourse & Dialogue Analysis**
- **Definition**: Understanding language use beyond sentences
- **Components**: Anaphora (reference resolution), discourse relations
- **Example**: "The president announced a policy. He said it will help economy."
  → "He" refers to president (anaphora resolution)

**NLP Processing Phases (1.5 marks)**

**Phase 1: Text Acquisition**
- Collect raw text from sources (websites, emails, documents, social media)
- No processing yet, just gathering

**Phase 2: Text Preprocessing**
- **Tokenization**: Split into words/sentences
- **Normalization**: Lowercase, remove punctuation
- **Cleaning**: Remove irrelevant data
- **Stop word removal**: Remove "the", "a", "and"
- **Stemming/Lemmatization**: Reduce to root form

**Phase 3: Feature Extraction**
- Convert text to numerical features for models
- Methods: Bag of Words, TF-IDF, Word2Vec, embeddings
- Create feature vectors that capture meaning

**Phase 4: Parsing/Syntactic Analysis**
- Analyze sentence structure
- POS tagging: Assign noun, verb, adjective to each word
- Dependency parsing: Find grammatical relationships

**Phase 5: Semantic Analysis**
- Understand meaning
- Entity recognition
- Relationship extraction
- Word sense disambiguation

**Phase 6: Model Training**
- Train machine learning models on data
- Learn patterns from examples
- Optimize parameters

**Phase 7: Evaluation & Deployment**
- Test model performance
- Deploy to production
- Monitor and update

---

## QUESTION 3: PREPROCESSING TECHNIQUES (PROBABILITY: 85%)

### Question:
**"Explain the important data preprocessing techniques in NLP: tokenization, normalization, stemming, and lemmatization. Compare stemming and lemmatization with examples."**

### Complete Answer:

**Tokenization (1 mark)**

**Definition**: Process of breaking text into smaller units called tokens.

**Types**:
1. **Sentence Tokenization**
   ```
   Input: "Hello world. This is NLP."
   Output: ["Hello world.", "This is NLP."]
   ```
   
2. **Word Tokenization**
   ```
   Input: "I don't like it."
   Output: ["I", "do", "n't", "like", "it", "."]
   ```
   
3. **Subword Tokenization** (modern, used in BERT)
   ```
   Input: "unbelievable"
   Output: ["un", "##believe", "##able"]
   Advantage: Handles rare words, reduces vocabulary
   ```

**Importance**: Fundamental step; all other steps depend on tokenization

**Normalization (1 mark)**

**Definition**: Converting text to standard form for consistency.

**Techniques**:

1. **Lowercasing**
   - "Hello World" → "hello world"
   - Treat "Hello" and "hello" as same word
   
2. **Removing Punctuation**
   - "Hello, how are you?" → "Hello how are you"
   
3. **Removing Numbers**
   - "Meeting on 2024-01-15 at 3:30" → "Meeting on at"
   
4. **Handling Whitespace**
   - "Hello    world" (multiple spaces) → "Hello world"
   
5. **Accent Removal**
   - "café" → "cafe"

**Purpose**: Standardize text for consistent processing

**Stemming (1.5 marks)**

**Definition**: Reduce words to root form by removing suffixes/prefixes.

**Method**: Rule-based approach (no vocabulary needed)

**Algorithm**: Porter Stemmer (most popular)
- Remove common suffixes
- "-ing", "-tion", "-ness", "-ment", etc.

**Examples**:
```
running, runs, runner → run
playing, played, plays → play
connection, connected, connecting → connect
ponies → poni
```

**Characteristics**:
- Fast (simple rules)
- Can produce non-words
- Crude/aggressive

**When to use**:
- Search engines (speed important, exact forms less important)
- Information retrieval

**Lemmatization (1.5 marks)**

**Definition**: Reduce words to dictionary form (lemma) using morphological analysis and vocabulary.

**Method**: Dictionary lookup + morphological analysis

**Process**:
1. Identify POS tag (noun, verb, adjective)
2. Look up in dictionary
3. Apply morphological rules based on POS

**Examples**:
```
running, runs, ran → run
studies, study, studied → study
better, best, good → good
easily → easy
```

**Characteristics**:
- Accurate (produces real words)
- Slower (needs dictionary lookup)
- Context-aware

**When to use**:
- Machine translation
- Chatbots (need real words)
- Question answering

**Stemming vs Lemmatization Comparison (1 mark)**

| Aspect | Stemming | Lemmatization |
|--------|----------|----------------|
| **Output** | May be non-word "poni", "easili" | Always real word "pony", "easy" |
| **Speed** | Fast (rule-based) | Slower (dictionary lookup) |
| **Accuracy** | Lower | Higher |
| **Information** | Loses semantic info | Preserves meaning |
| **Use Cases** | Search (speed critical) | Translation, chatbots |
| **Example** | "running" → "run" | "running" → "run" |
| **Example** | "ponies" → "poni" | "ponies" → "pony" |

**Practical Example**:
```
Sentence: "The students studied and playing games"

Tokenized: ["The", "students", "studied", "and", "playing", "games"]

After Stemming:
["The", "student", "studi", "and", "play", "game"]
Problem: "studi" is not a real word

After Lemmatization:
["The", "student", "study", "and", "playing", "game"]
Better: All are real words

Better still (with POS tagging):
["The", "student", "study", "and", "play", "game"]
All verbs properly lemmatized to base form
```

---

## QUESTION 4: NAMED ENTITY RECOGNITION (PROBABILITY: 80%)

### Question:
**"Explain Named Entity Recognition (NER). Describe different approaches to NER: rule-based, machine learning, and deep learning. Compare their advantages and disadvantages."**

### Complete Answer:

**NER Definition & Importance (1 mark)**

**NER**: Process of identifying and classifying named entities in text into predefined categories.

**Entity Types**:
- **PERSON**: John, Mary, Einstein
- **ORGANIZATION**: Apple, Google, UN
- **LOCATION**: Paris, USA, Tokyo
- **DATE**: January 1, 2024
- **MONEY**: $100, €50
- **Other**: TIME, PERCENTAGE, PRODUCT, EVENT

**Example**:
```
Text: "Tim Cook, CEO of Apple, lives in New York"
Entities:
- Tim Cook (PERSON)
- Apple (ORGANIZATION)
- New York (LOCATION)
- CEO (TITLE/ORGANIZATION)
```

**Rule-Based Approach (1.5 marks)**

**Method**: Use handcrafted rules, patterns, and lists

**Components**:

1. **Regular Expressions**
   ```
   Email: \w+@\w+\.\w+
   Phone: \d{3}-\d{3}-\d{4}
   Date: \d{4}-\d{2}-\d{2}
   ```

2. **Gazetteers** (lists of known entities)
   ```
   City gazetteer: {Paris, London, Tokyo, ...}
   Company gazetteer: {Apple, Google, Microsoft, ...}
   Person titles: {Mr., Dr., Prof., ...}
   ```

3. **Linguistic Patterns**
   ```
   Person pattern: (Title) (First Name) (Last Name)
   Example: "Mr. John Smith" → PERSON
   Location pattern: (City) + (Country)
   Example: "Paris, France" → LOCATION
   ```

**Advantages**:
- Interpretable and transparent
- Works well for specific, structured patterns
- No labeled training data needed
- Fast to implement for simple cases

**Disadvantages**:
- Difficult to scale
- Cannot handle variations (new entity names)
- Time-consuming rule creation
- Domain-specific (rules don't transfer)
- Misses new entities

**Machine Learning Approach (1.5 marks)**

**Method**: Train classifier on annotated labeled data

**Common Algorithms**:
1. **Hidden Markov Models (HMM)**
2. **Conditional Random Fields (CRF)**
3. **Support Vector Machines (SVM)**

**Process**:
```
Step 1: Feature Engineering
  - Word shape: "Xxx" (capitalized), "xxx" (lowercase)
  - POS tags from parser
  - Word prefixes/suffixes
  - Dictionary/gazetteer membership
  - Word embeddings
  
Step 2: Training
  Labeled data:
  "John | PERSON | works | O | at | O | Apple | ORGANIZATION"
  Model learns: Capitalized words often PERSON, company names ORG
  
Step 3: Testing
  Input: "Mary works at Google"
  Output: "Mary | PERSON | Google | ORGANIZATION"
```

**Advantages**:
- More accurate than rules
- Learns complex patterns automatically
- Handles variations
- Can adapt with new data

**Disadvantages**:
- Requires labeled data (expensive, time-consuming)
- Extensive feature engineering needed
- Difficult to debug when wrong
- Performance drops on new domains

**Deep Learning Approach (1.5 marks)**

**Method**: Neural networks learn representations automatically

**Architecture**: BiLSTM-CRF (most popular neural NER)

**Components**:
1. **BiLSTM** (Bidirectional LSTM)
   - Processes text forward and backward
   - Captures context from both directions
   - Learns representations automatically (no feature engineering)

2. **CRF Layer** (Conditional Random Fields)
   - Models dependencies between output tags
   - Ensures valid tag sequences
   - Example: "PERSON LOCATION PERSON" is valid, "PERSON PERSON LOCATION PERSON" might violate sequence rules

**Process**:
```
Input: "John works at Apple"
       ↓
Embedding: Dense vectors for each word
       ↓
BiLSTM Forward: Process left-to-right
BiLSTM Backward: Process right-to-left
       ↓
Concatenate: Get context from both directions
       ↓
CRF Layer: Decode best tag sequence
       ↓
Output: "John(PERSON) works(O) at(O) Apple(ORG)"
```

**Advanced Models**:
- **BERT**: Pre-trained transformer models achieving 90%+ F1 scores
- **SpaCy**: Modern transformer pipelines
- **Fine-tuned large language models**

**Advantages**:
- Highest accuracy (90-95% F1 scores)
- No manual feature engineering
- Pre-trained models available
- Handles complex linguistic patterns

**Disadvantages**:
- Requires large labeled datasets
- Computationally expensive
- Harder to interpret (black box)
- Requires significant computational resources

**Comparison Table** (1 mark)

| Aspect | Rule-Based | Machine Learning | Deep Learning |
|--------|-----------|-------------------|----------------|
| **Accuracy** | 60-70% | 75-85% | 90-95% |
| **Data Required** | None | 1000s of labeled examples | 10000s+ labeled examples |
| **Speed** | Fast | Fast | Slower |
| **Interpretability** | High | Medium | Low |
| **Scalability** | Poor | Good | Excellent |
| **Domain Adaptation** | Difficult | Medium | Can fine-tune |
| **Best For** | Simple, structured patterns | Moderate tasks | Complex, nuanced tasks |

---

# PART 3: QUICK REVISION SHEET FOR UNIT 1

---

## UNIT 1 QUICK REVISION SHEET
## Introduction to Language & NLP

---

### 1. EVOLUTION OF LANGUAGE - KEY DATES

```
Pre-Written (50k-3100 BCE)   → Oral only, no records
Written (3100 BCE-1440 CE)   → Cuneiform, hieroglyphics, alphabets
Printed (1440-1900)          → Gutenberg press, standardization
Digital (1900-Present)       → Telegraph, telephone, internet
AI-Era (2010-Present)        → Machine understanding, LLMs
```

**Key Insight**: From oral → written → printed → digital → AI
Each stage increased information spread, permanence, and accessibility

---

### 2. WHY NLP? - 5 REASONS

1. **Communication Gap**: Computers don't understand natural language naturally
2. **Data Explosion**: Can't manually process billions of texts daily
3. **Hidden Value**: 80-90% data is unstructured text
4. **Accessibility**: Enable communication for all (blind, deaf, language barriers)
5. **Efficiency**: Automate tasks taking humans weeks in seconds

---

### 3. MAJOR NLP APPLICATIONS - QUICK SUMMARY

| Application | What It Does | Example |
|-------------|-------------|---------|
| **Sentiment Analysis** | Identifies positive/negative opinions | Amazon reviews: 4.5/5 stars |
| **Machine Translation** | Translates between languages | Google Translate: English→Spanish |
| **Chatbots** | Conversational AI | ChatGPT, Alexa |
| **Speech Recognition** | Converts voice to text | Siri dictation |
| **Information Extraction** | Extracts structured data | Resume: John, 5 years experience |
| **Text Summarization** | Creates concise summary | Condense long articles |
| **NER** | Identifies named entities | John (PERSON) at Apple (ORG) |
| **Question Answering** | Answers questions | "Who is Einstein?" |

---

### 4. NLP PIPELINE - 7 PHASES

```
1. Text Acquisition    → Collect raw text
2. Preprocessing       → Clean, tokenize, normalize
3. Feature Extraction  → Convert to numerical features
4. Parsing             → Analyze structure
5. Semantic Analysis   → Understand meaning
6. Model Training      → Learn patterns
7. Evaluation/Deploy   → Test and use
```

---

### 5. PREPROCESSING TECHNIQUES - QUICK REFERENCE

| Technique | What It Does | Example |
|-----------|-------------|---------|
| **Tokenization** | Split into words | "hello world" → ["hello", "world"] |
| **Normalization** | Convert to standard form | "Hello" + "hello" → same |
| **Lowercasing** | Convert to lowercase | "HELLO" → "hello" |
| **Stemming** | Remove suffixes | "running" → "run" (might be non-word) |
| **Lemmatization** | Dictionary form | "running" → "run" (always real word) |
| **Stop Words** | Remove common words | "the", "and", "a" removed |
| **Embeddings** | Convert to vectors | "cat" → [0.2, -0.5, 0.8, ...] |

**Key Difference**: 
- Stemming: "ponies" → "poni" ❌ (non-word)
- Lemmatization: "ponies" → "pony" ✓ (real word)

---

### 6. CHALLENGES IN NLP - 5 MAIN

1. **Ambiguity**: "bank" = financial OR river side
2. **Context**: "not bad" ≠ "bad"
3. **Variability**: Same meaning, different words
4. **Idioms**: "Raining cats and dogs" ≠ literal
5. **World Knowledge**: "Apple released iOS" ≠ fruit

---

### 7. NAMED ENTITY RECOGNITION (NER)

**Definition**: Identify & classify named entities

**Entity Types**:
- PERSON: John, Mary
- ORGANIZATION: Apple, Google
- LOCATION: Paris, USA
- DATE: January 1, 2024
- MONEY: $100, €50

**Three Approaches**:

| Approach | Method | Accuracy | Speed |
|----------|--------|----------|-------|
| Rule-Based | Patterns + lists | 60-70% | Fast |
| ML | Train on labeled data | 75-85% | Fast |
| Deep Learning | Neural networks | 90-95% | Slower |

**Best NER**: BiLSTM-CRF (combines neural network with sequence modeling)

---

### 8. ONE-LINER SUMMARIES

- **NLP**: Making computers understand human language
- **Tokenization**: Breaking text into words
- **Normalization**: Making text standard
- **Stemming**: Crude root extraction (non-words OK)
- **Lemmatization**: Dictionary-based root extraction
- **Stop Words**: Common words with little meaning
- **Embeddings**: Words as vectors capturing meaning
- **NER**: Identifying important named entities
- **Sentiment**: Determining emotional tone
- **Dialogue**: Conversational systems

---

### 9. COMMON EXAM PATTERNS - PREDICTED QUESTIONS

| Rank | Topic | Probability |
|------|-------|-----------|
| 1 | Evolution + NLP need + challenges | 90% |
| 2 | NLP applications (sentiment, MT, chatbot) | 85% |
| 3 | Preprocessing techniques | 85% |
| 4 | NER approaches + comparison | 80% |
| 5 | NLP phases/pipeline | 75% |
| 6 | Tokenization vs normalization | 70% |

---

### 10. COMPARISON TABLE - STEMMING vs LEMMATIZATION

```
Word: "running"
Stemming: "run"
Lemmatization: "run"
(Both same here)

Word: "ponies"
Stemming: "poni" (non-word) ❌
Lemmatization: "pony" (real word) ✓

Word: "studies" (verb)
Stemming: "studi" (non-word)
Lemmatization: "study" (real word)

Decision: When unsure, use LEMMATIZATION
- More accurate
- Produces real words
- Only slower when speed critical
```

---

### 11. APPLICATIONS QUICK REFERENCE

**For Each Application, Know**:
1. What it does (definition)
2. Why it's important
3. One real-world example
4. Main challenge

---

### 12. KEY FORMULAS / DEFINITIONS TO MEMORIZE

1. **NLP = Computational Linguistics + ML + AI + Linguistics**
2. **Tokenization**: Text → Tokens
3. **Normalization**: Text → Standard Form
4. **Stemming**: Word → Root (may be non-word)
5. **Lemmatization**: Word → Dictionary Form
6. **NER**: Text → Named Entities + Classification
7. **Sentiment Analysis**: Text → Polarity (positive/negative/neutral)

---

### 13. EXAMPLE QUESTIONS - PRACTICE ANSWERS

**Q: What's difference between stemming and lemmatization?**
A: Stemming removes suffixes (fast, crude), lemmatization uses dictionary (accurate, slow). Stemming: "ponies"→"poni", Lemmatization: "ponies"→"pony"

**Q: Why is NLP important?**
A: 5 reasons: (1) Communication gap, (2) Data explosion, (3) Hidden value, (4) Accessibility, (5) Efficiency

**Q: Name 3 NLP applications**
A: (1) Sentiment analysis (Amazon reviews), (2) Machine translation (Google Translate), (3) Chatbots (ChatGPT)

**Q: What's tokenization?**
A: Breaking text into tokens. "I love NLP" → ["I", "love", "NLP"]

**Q: What are stop words?**
A: Common words (the, a, and) with little meaning. Removed during preprocessing

---

### 14. LAST-MINUTE TIPS FOR EXAM

✓ Always mention 5 reasons for NLP need
✓ For applications: definition + example + advantage
✓ For preprocessing: know all 7 techniques
✓ For NER: Know 3 approaches and their pros/cons
✓ Stemming: non-words OK, fast. Lemmatization: real words, slow
✓ Evolution has 5 stages (pre-written to AI-era)
✓ Pipeline has 7 phases
✓ For comparisons: always make table/chart

---

**End of Unit 1 Quick Revision Sheet**

