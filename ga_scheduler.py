# GA-based weekly study scheduler (Computational Intelligence project)
# Chromosome: integer matrix A (7 x slots_per_day)
# A[d, t] = subject_index or -1 for empty

import random
import numpy as np
import pandas as pd

BLOCK_HOURS = 1
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_targets(subjects, total_weekly_hours):
    # Compute target hours per subject.
    # Strategy:
    #   1) allocate min_hours first
    #   2) distribute remaining hours by normalized weights
    min_sum = sum(float(s.get("min_hours", 0)) for s in subjects)

    if min_sum > total_weekly_hours:
        scale = total_weekly_hours / max(min_sum, 1e-9)
        mins = [float(s.get("min_hours", 0)) * scale for s in subjects]
        remaining = 0.0
    else:
        mins = [float(s.get("min_hours", 0)) for s in subjects]
        remaining = float(total_weekly_hours) - min_sum

    weights = np.array([float(s.get("weight", 1.0)) for s in subjects], dtype=float)
    weights = np.maximum(weights, 1e-9)
    weights = weights / weights.sum()

    targets = np.array(mins, dtype=float) + remaining * weights
    return targets


def random_chromosome(n_subjects, slots_per_day, max_hours_per_day, off_days=None):
    off_days = set(off_days or [])
    A = -1 * np.ones((7, slots_per_day), dtype=int)

    for d in range(7):
        if d in off_days:
            continue
        fill = random.randint(0, max_hours_per_day)
        slots = random.sample(range(slots_per_day), k=fill) if fill > 0 else []
        for t in slots:
            A[d, t] = random.randrange(n_subjects)
    return A


def crossover(a, b, p=0.5):
    # Day-level uniform crossover
    child = a.copy()
    for d in range(7):
        if random.random() < p:
            child[d, :] = b[d, :]
    return child


def mutate(A, n_subjects, max_hours_per_day, off_days=None, mut_rate=0.15):
    # Mutation: swap/change/toggle within a day
    off_days = set(off_days or [])
    A = A.copy()
    slots_per_day = A.shape[1]

    for d in range(7):
        if d in off_days:
            A[d, :] = -1
            continue

        if random.random() < mut_rate:
            op = random.choice(["swap", "change", "toggle"])

            if op == "swap":
                t1, t2 = random.sample(range(slots_per_day), 2)
                A[d, t1], A[d, t2] = A[d, t2], A[d, t1]

            elif op == "change":
                t = random.randrange(slots_per_day)
                if A[d, t] != -1:
                    A[d, t] = random.randrange(n_subjects)

            else:  # toggle
                t = random.randrange(slots_per_day)
                if A[d, t] == -1:
                    if (A[d, :] != -1).sum() < max_hours_per_day:
                        A[d, t] = random.randrange(n_subjects)
                else:
                    A[d, t] = -1

    return A


def fitness(A, targets, max_hours_per_day, off_days=None, max_consecutive=2,
            offday_penalty=10.0, daily_limit_penalty=5.0,
            consecutive_penalty=2.0, imbalance_penalty=0.2):
    # Higher fitness is better.
    off_days = set(off_days or [])
    n_subjects = len(targets)

    counts = np.zeros(n_subjects, dtype=float)
    for d in range(7):
        for t in range(A.shape[1]):
            s = A[d, t]
            if s >= 0:
                counts[s] += 1

    achieved = counts * BLOCK_HOURS
    diff = np.abs(achieved - targets)
    reward = -diff.sum()  # smaller diff => better

    pen = 0.0

    for d in off_days:
        pen += offday_penalty * float((A[d, :] != -1).sum())

    for d in range(7):
        day_hours = float((A[d, :] != -1).sum()) * BLOCK_HOURS
        if day_hours > max_hours_per_day:
            pen += daily_limit_penalty * (day_hours - max_hours_per_day)

    for d in range(7):
        consec = 1
        prev = A[d, 0]
        for t in range(1, A.shape[1]):
            cur = A[d, t]
            if cur != -1 and cur == prev:
                consec += 1
            else:
                consec = 1
            prev = cur
            if cur != -1 and consec > max_consecutive:
                pen += consecutive_penalty * (consec - max_consecutive)

    daily_study = np.array([(A[d, :] != -1).sum() for d in range(7)], dtype=float)
    if daily_study.sum() > 0:
        cv = daily_study.std() / (daily_study.mean() + 1e-9)
        pen += imbalance_penalty * cv * 10.0

    return reward - pen


def chromosome_to_dataframe(A, subjects):
    slots_per_day = A.shape[1]
    cols = [f"Slot {i+1}" for i in range(slots_per_day)]
    data = []
    for d in range(7):
        row = []
        for t in range(slots_per_day):
            s = A[d, t]
            row.append(subjects[s]["name"] if s >= 0 else "—")
        data.append(row)
    return pd.DataFrame(data, index=DAYS, columns=cols)


def compute_totals(schedule_df):
    flat = [x for x in schedule_df.values.flatten().tolist() if x != "—"]
    if not flat:
        return pd.DataFrame(columns=["Subject", "Hours"])
    ser = pd.Series(flat)
    counts = ser.value_counts().sort_index()
    totals = pd.DataFrame({"Subject": counts.index, "Hours": counts.values * BLOCK_HOURS})
    return totals.sort_values("Hours", ascending=False).reset_index(drop=True)


def run_ga(subjects, total_weekly_hours=14, slots_per_day=6, max_hours_per_day=2,
           off_days=None, max_consecutive=2, pop_size=80, generations=120,
           cx_prob=0.8, mut_rate=0.2, elite_k=4, seed=42):
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    n_subjects = len(subjects)
    targets = build_targets(subjects, total_weekly_hours)

    pop = [random_chromosome(n_subjects, slots_per_day, max_hours_per_day, off_days) for _ in range(pop_size)]
    best_hist, avg_hist = [], []

    def eval_pop(population):
        return np.array([
            fitness(ind, targets, max_hours_per_day, off_days, max_consecutive=max_consecutive)
            for ind in population
        ], dtype=float)

    for _g in range(generations):
        fits = eval_pop(pop)
        best_hist.append(float(fits.max()))
        avg_hist.append(float(fits.mean()))

        elite_idx = fits.argsort()[-elite_k:][::-1]
        elites = [pop[i] for i in elite_idx]

        def tournament(k=3):
            cand = random.sample(range(pop_size), k)
            best_i = max(cand, key=lambda i: fits[i])
            return pop[best_i]

        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            child = p1.copy()
            if random.random() < cx_prob:
                child = crossover(p1, p2)
            child = mutate(child, n_subjects, max_hours_per_day, off_days, mut_rate=mut_rate)
            new_pop.append(child)

        pop = new_pop

    fits = eval_pop(pop)
    best_idx = int(fits.argmax())
    best = pop[best_idx]

    schedule_df = chromosome_to_dataframe(best, subjects)
    totals_df = compute_totals(schedule_df)

    
    info = {
        "targets": targets,
        "best_fitness": float(fits[best_idx]),
        "history": {"best": best_hist, "avg": avg_hist},
        "best_chromosome": best,   # ✅ add this line
}


    return schedule_df, totals_df, info

def chromosome_subject_hours(A, n_subjects):
    """Return achieved hours per subject from chromosome A."""
    counts = np.zeros(n_subjects, dtype=float)
    for d in range(7):
        for t in range(A.shape[1]):
            s = A[d, t]
            if s >= 0:
                counts[s] += 1
    return counts * BLOCK_HOURS


def compute_target_deviation(A, targets):
    """Sum absolute deviation from targets (lower is better)."""
    achieved = chromosome_subject_hours(A, len(targets))
    return float(np.abs(achieved - targets).sum())


def compute_violations(A, max_hours_per_day, off_days=None, max_consecutive=2):
    """Count constraint violations for reporting (lower is better)."""
    off_days = set(off_days or [])
    violations = {
        "off_day_study_slots": 0,
        "daily_limit_excess_hours": 0.0,
        "consecutive_excess_slots": 0,
    }

    # Off-day study slots
    for d in off_days:
        violations["off_day_study_slots"] += int((A[d, :] != -1).sum())

    # Daily limit excess
    for d in range(7):
        day_hours = float((A[d, :] != -1).sum()) * BLOCK_HOURS
        if day_hours > max_hours_per_day:
            violations["daily_limit_excess_hours"] += (day_hours - max_hours_per_day)

    # Consecutive exceed
    for d in range(7):
        consec = 1
        prev = A[d, 0]
        for t in range(1, A.shape[1]):
            cur = A[d, t]
            if cur != -1 and cur == prev:
                consec += 1
            else:
                consec = 1
            prev = cur
            if cur != -1 and consec > max_consecutive:
                violations["consecutive_excess_slots"] += 1

    return violations


def generate_random_baseline(subjects, targets, slots_per_day, max_hours_per_day,
                            off_days=None, max_consecutive=2, seed=42):
    """
    Generate a random-but-feasible baseline schedule under same constraints.
    It tries to meet targets greedily but uses randomness to remain a baseline.
    Returns chromosome A (7 x slots_per_day).
    """
    random.seed(int(seed))
    np.random.seed(int(seed))

    n_subjects = len(subjects)
    off_days = set(off_days or [])

    A = -1 * np.ones((7, slots_per_day), dtype=int)

    # remaining hours per subject in blocks
    remaining = (targets / BLOCK_HOURS).copy()

    for d in range(7):
        if d in off_days:
            continue

        filled = 0
        prev = -999
        consec = 0

        # Random order of slots
        slot_order = list(range(slots_per_day))
        random.shuffle(slot_order)

        for t in slot_order:
            if filled >= max_hours_per_day:
                break

            # candidates: subjects with remaining > 0
            candidates = [i for i in range(n_subjects) if remaining[i] > 0.1]
            if not candidates:
                # if no remaining, choose any randomly
                candidates = list(range(n_subjects))

            # prefer higher remaining but randomize
            candidates.sort(key=lambda i: remaining[i], reverse=True)
            top_k = candidates[: max(2, min(4, len(candidates)))]
            s = random.choice(top_k)

            # consecutive rule check
            if s == prev:
                consec += 1
            else:
                consec = 1

            if consec > max_consecutive:
                # pick different subject
                alt = [x for x in candidates if x != prev]
                if alt:
                    s = random.choice(alt[: max(2, min(4, len(alt)))])
                    consec = 1
                else:
                    continue

            A[d, t] = s
            prev = s
            filled += 1
            remaining[s] -= 1

    return A
