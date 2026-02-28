# llm_symbol_lab_test

## Tipping Points in Fact-Grounded LLMs (Tool-Grounded vs. Direct Injection)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-brightgreen)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-lightgrey)
![Gephi](https://img.shields.io/badge/Gephi-Network%20Visualization-purple)

---

### 🔗 Quick Links (Repo Files)
- 📓 Notebook: [`final.ipynb`](./final.ipynb)
- 🖥️ Presentation slides: [`Network Superheroes Final Presentation.pdf`](./Network%20Superheroes%20Final%20Presentation.pdf)
- 🧾 Notebook slides / detailed methods: [`Network Superheroes Final Notebook.pdf`](./Network%20Superheroes%20Final%20Notebook.pdf)
- 🧩 Visual replication assets (JSON + Gephi): [`data_and_gephi/`](./data_and_gephi/)

---

### 🧭 Project Overview
Modern LLM applications often rely on **external context** (search/RAG/tools) to reduce hallucinations.  
However, if that external source is **biased or fallible**, model outputs can shift dramatically.

This project investigates **tipping points**—the point where model behavior **flips** as biased context increases—and compares two grounding strategies:

- **Tool-grounded context (mock tool calls):** simulates production-style tool augmentation
- **Direct prompt injection:** context provided directly in the user prompt

We quantify response drift using **embedding similarity scoring** and analyze structure using **tipping curves + network visualizations**.

---

### 🎯 Research Questions
1. **How does tipping behavior change** as biased external context increases?
2. How do different **fact-grounding techniques** (direct prompt injection vs. tool-based context delivery) **modulate tipping dynamics**?
3. Are **article bias scores** closely linked to **LLM output bias**, and do these relationships show structured patterns (e.g., clustering) in a network view?

---

### 🏗 Experimental Framework

#### 1) Bias-Controlled External Context (Synthetic Articles)
To precisely control context bias, we generate journalist-style synthetic articles across an **ordered bias scale**:
- Standardized prompt + consistent tone/length for comparability  
- Post-processing to remove prompt artifacts / meta-instructions  
- Articles stored as structured inputs for downstream experiments  

This design keeps bias strength as a clean experimental variable while avoiding time-intensive manual collection.

---

#### 2) Grounding Strategies Compared (Delivery Mechanisms)
We compare two ways of providing external context:

**A. Tool-grounded context (mock tool calls)**  
- Articles are provided as results from a **mock search tool**  
- Simulates production-style tool-augmented workflows  

**B. Direct prompt injection**  
- Articles are provided directly in the user message as context  

This isolates the effect of the delivery mechanism while holding article content constant.

---

#### 3) Embedding-Based Drift Scoring
We quantify output drift via sentence embeddings + cosine similarity:

- Candidate sentences represent binary decision points (e.g., “I like Windows” vs. “I like MacOS”)  
- Encode candidates and model outputs using **SentenceTransformers `all-MiniLM-L6-v2`**  
- Compute a single relative bias score:

**score = sim(output, Windows_candidates) − sim(output, MacOS_candidates)**

Interpretation:
- **Positive** → Windows-leaning  
- **Negative** → MacOS-leaning  
- **Near zero** → neutral/ambiguous  

To improve scoring relevance, we score the **first sentence** of each model output (paired with prompting to encourage short, template-aligned responses).

---

#### 4) Sampling Design (Clusters + Repeated Runs)
- Experiments sample **clusters of similar articles** to reduce confounds from conflicting sources  
- Runs vary by:
  - number of injected articles (1, 2, 3)  
  - injection strategy (tool-grounded vs direct)  
- Repeated runs under identical settings capture stochastic variability.

---

#### 5) Network Analysis & Visualization (NetworkX + Gephi)
We visualize article–outcome dynamics as a graph:

- **Nodes:** articles + bucketed outcome scores (e.g., 0.05 increments)  
- **Edges:** an article connects to an outcome bucket when it appears in an observation yielding that outcome  

We compare structure against a random baseline and use Gephi to explore bias-aligned clustering.

---

### 📈 Key Findings

#### 1) Non-linear Tipping as Bias Increases
As article bias shifts, model preferences **flip** (MacOS → Windows), showing **non-linear tipping behavior** rather than smooth linear change.

#### 2) Tool-grounded Pipelines Amplify Bias More Strongly
Tipping dynamics differ significantly by grounding strategy:
- Tool-grounded context shows a **steeper tipping curve** centered near neutrality  
- Direct injection produces a noticeably **weaker** response shift  

This indicates tool-augmented pipelines can increase sensitivity to biased sources.

#### 3) Bias-aligned Clustering in Network Structure
Network visualizations show:
- distinct MacOS-leaning vs Windows-leaning clusters  
- dense connections within clusters, sparse connections across opposing biases  
- structure stronger than expected under random edge assignment.

---

### 🧪 Model Configuration
- **Model:** Qwen 3 (0.6B)  
- Prompting encourages one-sentence outputs to align with candidate scoring templates  
- Reasoning traces disabled to reduce noise and simplify scoring/parsing.

---

### 📊 Technical Stack
- **LLM / Inference:** HuggingFace Transformers  
- **Scoring:** SentenceTransformers (`all-MiniLM-L6-v2`), cosine similarity  
- **Data modeling:** Python dataclasses for structured experiment inputs/outputs  
- **Graphs:** NetworkX (construction + metrics)  
- **Visualization:** Gephi (final network exploration + presentation visuals)

---

### 💡 Professional Impact
This project supports reliability evaluation for tool-augmented LLM workflows (e.g., RAG/search):
- provides a reproducible pattern for measuring **bias sensitivity** and **tipping thresholds**  
- shows that context delivery (tools vs direct injection) can materially change robustness  
- helps frame risk discussions for production systems when external knowledge sources are biased or fallible.
