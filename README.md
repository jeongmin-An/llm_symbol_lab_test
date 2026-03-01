# llm_symbol_lab_test

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-brightgreen)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-lightgrey)
![Gephi](https://img.shields.io/badge/Gephi-Network%20Visualization-purple)

---

## 🔗 Quick Links (Repo Files)
- 📓 Notebook: [`final.ipynb`](./final.ipynb)
- 🖥️ Presentation slides: [`Network Superheroes Final Presentation.pdf`](./Network%20Superheroes%20Final%20Presentation.pdf)
- 🧾 Notebook slides / detailed methods: [`Network Superheroes Final Notebook.pdf`](./Network%20Superheroes%20Final%20Notebook.pdf)
- 🧩 Visual replication assets (JSON + Gephi): [`data_and_gephi/`](./data_and_gephi/)

---

## 🧭 Project Overview
Modern LLM applications often rely on **external context** (search/RAG/tools) to improve factuality.
However, if that external source is **biased or fallible**, model outputs can shift dramatically.

This project investigates **tipping points**—the point where model behavior **flips** as biased context increases—and compares two grounding strategies:

- **Tool-grounded context (mock tool calls):** simulates production-style tool augmentation
- **Direct prompt injection:** context provided directly in the user prompt

We quantify response drift using **embedding similarity scoring** and analyze structure using **tipping curves + network visualizations**.

---

## 🎯 Research Questions
1. How does model behavior **change (and tip/flip)** as biased external context increases?
2. How do **grounding mechanisms** (direct prompt injection vs. tool-grounded delivery) **affect tipping dynamics**?
3. How strongly do **article bias scores** align with **LLM output bias**, and do we observe **clustered structure** in a network view?

---

## 🏗 Experimental Framework

### 1) Bias-Controlled External Context (Synthetic Articles)
- Generate journalist-style synthetic articles along an **ordered bias scale** using standardized prompts
- Post-process to remove prompt artifacts / meta-instructions
- Store articles as structured inputs for repeatable experiments

### 2) Grounding Strategies Compared (Delivery Mechanisms)
- **Tool-grounded context (mock tool calls):** articles delivered as simulated tool/search results (production-like)
- **Direct prompt injection:** articles inserted directly into the user prompt  
> The content is held constant to isolate the impact of the delivery mechanism. 

### 3) Embedding-Based Drift Scoring
- Encode outputs and paired binary candidates using **SentenceTransformers `all-MiniLM-L6-v2`**
- Compute a relative bias score:

**score = sim(output, Windows_candidates) − sim(output, MacOS_candidates)**

Interpretation:
- **score > 0** → Windows-leaning  
- **score < 0** → MacOS-leaning  
- **score ≈ 0** → neutral/ambiguous  
> Score the **first sentence** only to reduce noise and keep outputs consistent with the scoring templates.

### 4) Sampling Design (Repeated Runs)
- Vary:
  - number of injected articles (**1 / 2 / 3**)
  - delivery mechanism (**tool-grounded vs direct**)
- Repeat runs per setting to capture stochastic variability

### 5) Network Analysis & Visualization (NetworkX + Gephi)
- Build a bipartite graph: **articles ↔ outcome score buckets** (e.g., 0.05 increments)
- Compare structure against a random baseline and explore clustering in **Gephi**


---

## 📈 Key Findings
- **Non-linear tipping:** As article bias shifts, model preferences **flip** (MacOS → Windows), showing **non-linear tipping behavior** rather than smooth linear change.
- **Tool-grounded context amplifies bias:** steeper tipping curves than direct injection  
- **Network structure is not random:** bias-aligned clusters emerge (dense within-bias, sparse cross-bias)

### Notes (brief)
1) **Tipping behavior:** shifts are abrupt rather than linear as bias grows.  
2) **Delivery matters:** tool-grounded runs show stronger sensitivity to biased context.  
3) **Network view:** outcomes cluster by bias direction more than a random baseline.

---

## 🧪 Model Configuration
- **Model:** Qwen 3 (0.6B)  
- Prompting encourages **one-sentence** outputs (consistent scoring)
- Reasoning traces disabled to reduce noise and simplify scoring/parsing.
  
---

## 📊 Technical Stack
- **LLM / Inference:** HuggingFace Transformers  
- **Scoring:** SentenceTransformers (`all-MiniLM-L6-v2`), cosine similarity  
- **Graphs:** NetworkX (construction + metrics)
- **Visualization:** Gephi (final network exploration + presentation visuals)

---

## 💡 Why This Matters
This work provides a reproducible evaluation pattern for tool-augmented LLM systems (e.g., RAG/search): 
- measures **bias sensitivity** and identifies **tipping thresholds**
- shows delivery mechanisms (tools vs direct) can materially change robustness
- supports risk discussions for production workflows when external sources are biased or unreliable


