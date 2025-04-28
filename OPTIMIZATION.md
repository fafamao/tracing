# Optimization Progress Log for [tracing]

**Last Updated:** 2024-04-28 HH:MM (Current Time: Monday, April 28, 2025 at 11:41:13 AM CST)

## 1. Overall Optimization Goals

* **Primary Goal:** [To boost speed of rendering]
* **Secondary Goals:** [To achieve higher FPS, ie 30 FPS]
* **Key Performance Indicators (KPIs) to Track:**
    * KPI 1: total execution time by chrono
    * KPI 2: profiling results from gprof
* **Target Platform/Environment (if specific):** [Ubuntu 22.04]

## 2. Baseline Performance

* **Date Measured:** 2025-03-18
* **Version/Commit:** [04b310fd4071e28bc4b211cc0bb0c1d5b7b25ee1]
* **Methodology:** [One time execution with profiling mode build and gprof]

### Attempt #1: [Refine [] operator in Vec3 class]

* **Date:** 2025-04-28
* **Time Started:** HH:MM
* **Time Ended:** HH:MM
* **Version/Commit:** [Link to Git commit hash or version tag of this change]
* **Engineer(s):** [fafamao]

* **Hypothesis/Goal:**
    * [solve the most time-consuming execution from gprof results, which is the [] operator]

* **Changes Made:**
    * [get rid of switch logic]

* **Metrics Tracked:**
    * [e.g., Average `getUserProfile()` execution time, DB query count for profiles, Cache hit/miss ratio, Overall request latency]

* **Test Environment & Methodology:**
    * [Ubuntu 22.04]

* **Result & Analysis:**
    * [] operator Was a major hotspot at 7.35% of self-time (0.15s). Now
        * 2.43      1.07     0.06 11384135     0.00     0.00  Vec3::operator[](unsigned long) const

---

### Attempt #Y: [Another Optimization]

* **Date:** YYYY-MM-DD
* ... (repeat structure)

---

## 4. Tools Used

* **Profiling:** [gprof]
