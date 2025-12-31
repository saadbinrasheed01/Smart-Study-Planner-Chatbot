
# 💬 Smart Study Planner Chatbot (GA) — Computational Intelligence Semester Project

**Course:** Computational Intelligence  
**Instructor:** Hamas Ur Rehman (Visiting Faculty CSIT)  
**Student:** Saad Bin Rasheed And Hafsa Khanam — **22PWDSC0061 22Pwdsc0088 **    

A real-time **chatbot-style AI agent** that converses with the user and generates an optimized **weekly study timetable** using a **Genetic Algorithm (GA)** under constraints such as off-days, daily hour limits, and maximum consecutive hours per subject.

---

## ✅ Why This Project (CI Alignment)
Study planning is a **constrained optimization** problem with a huge search space. This project uses **Evolutionary Computation (Genetic Algorithm)** to search for better schedules using a fitness function and constraint penalties.

> ✅ This is **NOT a RAG-based** app. The core output is produced via **GA optimization + fitness design + constraint handling**. 

---

## ✨ Key Features
- 💬 **Real-time Chatbot UI** (single chat input, step-by-step interaction)
- 🧬 **Genetic Algorithm Optimization**
  - tournament selection, crossover, mutation, elitism
- 🎯 **Priority-aware targets**
  - subject **weight** (priority) + **minimum weekly hours**
- ⛔ Constraints supported:
  - Off-day(s) (e.g., Friday)
  - Max study hours per day
  - Max consecutive hours for the same subject
  - Soft balancing across days (avoid packing all study into 1–2 days)
- 📊 Outputs:
  - Weekly timetable table
  - Hours per subject chart
  - GA convergence plot (best/average fitness per generation)
- ⬇️ Export schedule as CSV

---

## ⭐ Unique Feature (Standout / Original Contribution)
### ✅ 1) Evaluation Layer: **GA vs Random Baseline**
To prove the CI method is genuinely optimizing, the system compares:
- **GA-generated schedule** vs a **Random Baseline schedule** under the **same constraints**
- Reports:
  - Target deviation (how close the schedule matches target hours)
  - Constraint violation counts
  - Improvement percentage

This makes the project more than a simple generator—it includes measurable evaluation.

### ✅ 2) Explainable Planning Summary
The agent produces a human-readable summary explaining:
- Why a subject received more hours (higher weight / minimum hours)
- Which constraints were enforced (off-days, daily max, consecutive max)
- Targets vs achieved hours per subject

---

## 🧠 Methodology (How It Works)

### Chromosome Representation
A timetable is encoded as a **7 × slots_per_day** matrix:
- Each cell is either a subject index or empty slot (`-1`)

### Target Allocation
Targets are computed by:
1) Allocating each subject’s minimum weekly hours  
2) Distributing remaining hours according to subject weights (priorities)

### Fitness Function (Maximize)
Fitness rewards closeness to targets and penalizes constraint violations:
- Off-day study penalty
- Daily max hours penalty
- Consecutive subject penalty
- Imbalance penalty (avoid packing study in few days)

### GA Operators
- **Selection:** tournament
- **Crossover:** day-level uniform
- **Mutation:** swap/change/toggle
- **Elitism:** keep top schedules

---

## 🧰 Tech Stack
- **Python**
- **Streamlit** (chat UI + visualization)
- **NumPy**, **Pandas**, **Matplotlib**

---

## 📁 Repository Structure
```bash
study-planner-ga-chatbot/
│── chat_app.py
│── ga_scheduler.py
│── requirements.txt
│── README.md
│
├── docs/
│   └── group_details.md
│
└── assets/
    └── screenshots/
        ├── chat_ui.png
        ├── timetable.png
        ├── convergence.png
        └── baseline_compare.png
