
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ga_scheduler import run_ga, DAYS

st.set_page_config(page_title="Study Planner Chatbot (GA)", layout="wide")

# ---------------------------
# Session State Init
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stage" not in st.session_state:
    st.session_state.stage = "ask_total_hours"

if "config" not in st.session_state:
    st.session_state.config = {
        "total_weekly_hours": None,
        "slots_per_day": None,
        "max_hours_per_day": None,
        "off_days": ["Fri"],
        "max_consecutive": 2,
        "subjects": []
    }

if "result" not in st.session_state:
    st.session_state.result = None  # (schedule_df, totals_df, info)

# ---------------------------
# Helpers
# ---------------------------
def bot_say(text):
    st.session_state.messages.append({"role": "assistant", "content": text})

def user_say(text):
    st.session_state.messages.append({"role": "user", "content": text})

def parse_int(text, lo=None, hi=None):
    digits = "".join([c for c in text if c.isdigit()])
    if digits == "":
        return None
    val = int(digits)
    if lo is not None and val < lo:
        return None
    if hi is not None and val > hi:
        return None
    return val

def parse_offdays(text):
    text = text.lower()
    selected = []
    for d in DAYS:
        if d.lower() in text:
            selected.append(d)
    # aliases
    if "friday" in text or "fri" in text:
        if "Fri" not in selected:
            selected.append("Fri")
    return selected if selected else None

def parse_subject_line(text):
    # Format: SubjectName, weight, min_hours
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        return None
    name = parts[0]
    try:
        weight = float(parts[1])
        min_h = float(parts[2])
        if weight <= 0 or min_h < 0:
            return None
        return {"name": name, "weight": weight, "min_hours": min_h}
    except:
        return None

# ---------------------------
# Header
# ---------------------------
st.title("💬 Smart Study Planner Chatbot (Genetic Algorithm)")
st.caption("Single input chat UI | Output appears only at the end")

# First greeting
if len(st.session_state.messages) == 0:
    bot_say(
        "Assalam o Alaikum! 😊 Main tumhara Study Planner Agent hoon.\n\n"
        "**Q1:** Total weekly study hours kitni honi chahiye? (e.g., 14)"
    )

# ---------------------------
# Chat messages (TOP)
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Single input (ONLY ONE)
# ---------------------------
prompt = st.chat_input("Reply yahan do...")

if prompt:
    user_say(prompt)
    cfg = st.session_state.config
    stage = st.session_state.stage

    if stage == "ask_total_hours":
        v = parse_int(prompt, lo=5, hi=60)
        if v is None:
            bot_say("Please 5–60 ke darmiyan number do. Example: 14")
        else:
            cfg["total_weekly_hours"] = v
            st.session_state.stage = "ask_slots"
            bot_say(f"✅ Total weekly hours: **{v}**\n\n**Q2:** Slots per day kitne? (1 slot = 1 hour). Example: 6")

    elif stage == "ask_slots":
        v = parse_int(prompt, lo=2, hi=12)
        if v is None:
            bot_say("Slots 2–12 ke darmiyan do. Example: 6")
        else:
            cfg["slots_per_day"] = v
            st.session_state.stage = "ask_max_per_day"
            bot_say(f"✅ Slots per day: **{v}**\n\n**Q3:** Max study hours/day kitni? Example: 2")

    elif stage == "ask_max_per_day":
        v = parse_int(prompt, lo=1, hi=cfg["slots_per_day"])
        if v is None:
            bot_say(f"Max hours 1–{cfg['slots_per_day']} ke darmiyan do.")
        else:
            cfg["max_hours_per_day"] = v
            st.session_state.stage = "ask_off_days"
            bot_say(f"✅ Max hours/day: **{v}**\n\n**Q4:** Off-day(s) kaun se? Example: Fri\nAvailable: {', '.join(DAYS)}")

    elif stage == "ask_off_days":
        v = parse_offdays(prompt)
        if v is None:
            bot_say("Off-day ka name do. Example: Fri")
        else:
            cfg["off_days"] = v
            st.session_state.stage = "ask_max_consecutive"
            bot_say(f"✅ Off-day(s): **{', '.join(v)}**\n\n**Q5:** Max consecutive hours same subject? Example: 2")

    elif stage == "ask_max_consecutive":
        v = parse_int(prompt, lo=1, hi=4)
        if v is None:
            bot_say("Number 1–4 ke darmiyan do. Example: 2")
        else:
            cfg["max_consecutive"] = v
            st.session_state.stage = "collect_subjects"
            bot_say(
                f"✅ Max consecutive: **{v}**\n\n"
                "Ab subjects add karo.\n\n"
                "Format: **SubjectName, weight, min_hours**\n"
                "Example: `Math, 3, 2`\n\n"
                "Jab complete ho jaye to type: **done**"
            )

    elif stage == "collect_subjects":
        if prompt.strip().lower() == "done":
            if len(cfg["subjects"]) < 2:
                bot_say("Kam az kam 2 subjects add karo phir done likho.")
            else:
                bot_say("✅ Great! Ab GA run kar raha hoon... ⏳")

                off_idx = [DAYS.index(d) for d in cfg["off_days"]]
                schedule_df, totals_df, info = run_ga(
                    subjects=cfg["subjects"],
                    total_weekly_hours=int(cfg["total_weekly_hours"]),
                    slots_per_day=int(cfg["slots_per_day"]),
                    max_hours_per_day=int(cfg["max_hours_per_day"]),
                    off_days=off_idx,
                    max_consecutive=int(cfg["max_consecutive"]),
                    pop_size=80,
                    generations=150,
                    mut_rate=0.2,
                    cx_prob=0.8,
                    elite_k=4,
                    seed=42,
                )
                st.session_state.result = (schedule_df, totals_df, info)
                st.session_state.stage = "finished"

                bot_say(f"✅ Done! Best fitness: **{info['best_fitness']:.2f}**\n\nScroll down to see results 👇")
        else:
            s = parse_subject_line(prompt)
            if s is None:
                bot_say("Format ghalat. Example: `Math, 3, 2`")
            else:
                cfg["subjects"].append(s)
                bot_say(f"✅ Added: **{s['name']}** (weight={s['weight']}, min={s['min_hours']})\n\nNext subject bhejo ya **done**.")

# ---------------------------
# RESULTS (LAST SECTION ONLY)
# ---------------------------
if st.session_state.stage == "finished" and st.session_state.result is not None:
    schedule_df, totals_df, info = st.session_state.result

    st.markdown("---")
    st.header("📌 Final Output (Schedule + Charts)")

    st.subheader("📅 Weekly Timetable")
    st.dataframe(schedule_df, use_container_width=True)

    st.download_button(
        "⬇️ Download Schedule CSV",
        data=schedule_df.to_csv().encode("utf-8"),
        file_name="study_schedule.csv",
        mime="text/csv",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏱️ Hours per Subject")
        if len(totals_df):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(totals_df["Subject"], totals_df["Hours"], color="#4C78A8")
            ax.set_ylabel("Hours")
            ax.set_title("Weekly Allocation")
            ax.tick_params(axis="x", rotation=35)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("No hours allocated.")

    with col2:
        st.subheader("📈 GA Convergence")
        hist = info["history"]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(hist["best"], label="Best")
        ax2.plot(hist["avg"], label="Average", alpha=0.7)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Fitness")
        ax2.set_title("Fitness Over Generations")
        ax2.legend()
        st.pyplot(fig2, clear_figure=True)

    st.success("✅ Output end me show ho raha hai. (Chat continues above)")
