import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ga_scheduler import run_ga, DAYS

st.set_page_config(page_title="Smart Study Planner (GA)", layout="wide")

st.title("📚 Smart Study Planner AI Agent (Genetic Algorithm)")
st.caption("Computational Intelligence project: GA optimization + constraints -> weekly timetable")

with st.sidebar:
    st.header("⚙️ Inputs")
    total_weekly_hours = st.slider("Total study hours per week", 5, 60, 14)
    slots_per_day = st.slider("Slots per day (each = 1 hour)", 2, 12, 6)
    max_hours_per_day = st.slider("Max study hours per day", 1, slots_per_day, 2)

    off_days_names = st.multiselect("Off-day(s)", options=DAYS, default=["Fri"])
    off_days = [DAYS.index(d) for d in off_days_names]

    max_consecutive = st.slider("Max consecutive hours of same subject", 1, 4, 2)

    st.subheader("GA Settings")
    pop_size = st.slider("Population size", 20, 200, 80, step=10)
    generations = st.slider("Generations", 30, 400, 120, step=10)
    mut_rate = st.slider("Mutation rate", 0.05, 0.6, 0.2, step=0.05)
    cx_prob = st.slider("Crossover probability", 0.2, 0.95, 0.8, step=0.05)
    elite_k = st.slider("Elites kept", 1, 10, 4)
    seed = st.number_input("Random seed", value=42)

st.subheader("🧾 Subjects")

if "subjects" not in st.session_state:
    st.session_state.subjects = [
        {"name": "Math", "weight": 3.0, "min_hours": 2.0},
        {"name": "Programming", "weight": 4.0, "min_hours": 3.0},
        {"name": "CI", "weight": 2.0, "min_hours": 2.0},
        {"name": "English", "weight": 1.0, "min_hours": 1.0},
    ]

subjects_df = pd.DataFrame(st.session_state.subjects)
edited = st.data_editor(subjects_df, num_rows="dynamic", use_container_width=True)
subjects = edited.to_dict(orient="records")

run_btn = st.button("🚀 Generate Schedule", type="primary")

if run_btn:
    with st.spinner("Running GA..."):
        schedule_df, totals_df, info = run_ga(
            subjects=subjects,
            total_weekly_hours=int(total_weekly_hours),
            slots_per_day=int(slots_per_day),
            max_hours_per_day=int(max_hours_per_day),
            off_days=off_days,
            max_consecutive=int(max_consecutive),
            pop_size=int(pop_size),
            generations=int(generations),
            cx_prob=float(cx_prob),
            mut_rate=float(mut_rate),
            elite_k=int(elite_k),
            seed=int(seed),
        )

    st.success(f"Done! Best fitness: {info['best_fitness']:.2f}")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("📅 Weekly Timetable")
        st.dataframe(schedule_df, use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            data=schedule_df.to_csv().encode("utf-8"),
            file_name="study_schedule.csv",
            mime="text/csv",
        )

    with col2:
        st.subheader("⏱️ Hours per Subject")
        if len(totals_df) == 0:
            st.warning("No study slots assigned. Try increasing total hours or generations.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(totals_df["Subject"], totals_df["Hours"], color="#4C78A8")
            ax.set_ylabel("Hours")
            ax.set_title("Weekly Allocation")
            ax.tick_params(axis='x', rotation=35)
            st.pyplot(fig, clear_figure=True)

    st.subheader("📈 GA Convergence")
    hist = info["history"]
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(hist["best"], label="Best")
    ax2.plot(hist["avg"], label="Average", alpha=0.7)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.set_title("Fitness over Generations")
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

    st.subheader("🎯 Targets vs Achieved")
    target_df = pd.DataFrame({
        "Subject": [s["name"] for s in subjects],
        "TargetHours": np.round(info["targets"], 2),
    })
    achieved = totals_df.rename(columns={"Hours": "AchievedHours"}) if len(totals_df) else pd.DataFrame({"Subject": [], "AchievedHours": []})
    merged = target_df.merge(achieved, on="Subject", how="left").fillna(0)
    st.dataframe(merged, use_container_width=True)

else:
    st.info("Set your subjects and click Generate Schedule.")
