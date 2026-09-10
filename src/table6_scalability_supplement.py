# -*- coding: utf-8 -*-
"""
Supplementary M50/L170 benchmark for the UAV inspection full-path MILP and
route-intensified ALNS-Q-learning-SA algorithm.

Experimental scales:
    Medium: M50;
    Large : L170.

Experimental logic:
    - Gurobi is run once for both M50 and L170 as a time-limited full-path MILP
      benchmark (1,800 s per instance).
    - ALNS-QSA is independently run 10 times for each instance under a nominal
      600 s wall-clock limit per seed.
    - Results report Gurobi incumbent/lower bound/MIP gap together with ALNS
      best/mean/standard deviation/CV, feasibility, coverage, and runtime.
    - A positive "improvement over Gurobi UB" means that ALNS found a lower
      feasible objective than Gurobi within the stated time limit. It is not a
      proof of global optimality unless Gurobi status is OPTIMAL.
    - Gurobi node files, a soft memory limit, checkpointing, resume logic, and
      preservation of incumbent/bound information after memory interruption are
      enabled to avoid losing long-running results.
"""

from __future__ import annotations

# ============================================================
# 1. 参数配置区：所有可调参数集中在这里
# ============================================================

import os
import math
import time
import random
import itertools
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any, Iterable

import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except Exception:
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False


# ---------- 路径设置 ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "table6_scalability_supplement_M50_L170"))
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
INSTANCE_DIR = os.path.join(PROJECT_ROOT, "instances")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
GUROBI_NODEFILE_DIR = os.path.join(PROJECT_ROOT, "gurobi_nodefiles")
os.makedirs(GUROBI_NODEFILE_DIR, exist_ok=True)

RAW_CSV = os.path.join(RESULT_DIR, "raw_M50_L170_full_path_results.csv")
SUMMARY_CSV = os.path.join(RESULT_DIR, "summary_M50_L170_by_instance.csv")
SUMMARY_XLSX = os.path.join(RESULT_DIR, "summary_M50_L170_benchmark.xlsx")
SCALE_SUMMARY_CSV = os.path.join(RESULT_DIR, "summary_M50_L170_by_scale.csv")
PAPER_TABLE_CSV = os.path.join(RESULT_DIR, "paper_table_M50_L170.csv")
PAPER_TABLE_XLSX = os.path.join(RESULT_DIR, "paper_table_M50_L170.xlsx")


# ---------- 实验开关 ----------
RUN_ALNS_QSA = True
SAVE_INSTANCE_CSV = True

# Long-run protection. Existing successful rows in each instance checkpoint
# are loaded and skipped automatically when the script is restarted.
RESUME_FROM_CHECKPOINT = True
SKIP_COMPLETED_GUROBI = True
SKIP_COMPLETED_ALNS_SEEDS = True

# Gurobi运行开关：小规模用于精确性验证；中规模用于限时基准；
# 大规模默认关闭，避免长时间卡住。需要大规模Gurobi对比时改为 True。
RUN_GUROBI_SMALL = False
RUN_GUROBI_MEDIUM = True
RUN_GUROBI_LARGE = True

# ---------- 调试开关 ----------
# 当前版本顺序运行 M50 和 L170，每个实例ALNS独立运行10次。
# 若需要重新进入单实例调试，可把 DEBUG_MODE 改为 True，并设置 DEBUG_INSTANCE_IDS。
DEBUG_MODE = False
DEBUG_INSTANCE_IDS = {"M50", "L170"}
DEBUG_PRINT_SOLUTION_AUDIT = False

# 同时运行 medium 与 large；程序按 M50、L170 顺序执行。
ENABLED_SCALES = ["medium", "large"]


# ---------- 补跑实例规模：M50与L170 ----------
OPEN_PLATFORMS_FIXED = 4  # medium-scale default; each instance can override it
N_UAV_TYPES = 3
N_NO_FLY_ZONES = 2

INSTANCE_SPECS = [
    # M50 follows the medium-scale design used for M60–M80:
    # 12 candidate platforms, 4 opened platforms, and 6 hazardous sources.
    {"instance_id": "M50", "scale": "medium", "n_points": 50,
     "n_candidate_platforms": 12, "open_platforms_fixed": 4,
     "n_hazard_sources": 6, "seed": 20260950, "run_gurobi": True},

    # L170 extends the L150 design while holding the platform configuration
    # fixed at 6/18, so the added difficulty primarily reflects routing scale.
    {"instance_id": "L170", "scale": "large", "n_points": 170,
     "n_candidate_platforms": 18, "open_platforms_fixed": 6,
     "n_hazard_sources": 10, "seed": 20261070, "run_gurobi": True},
]


# ---------- 园区空间参数 ----------
AREA_WIDTH_KM = 8.0
AREA_HEIGHT_KM = 8.0
MIN_PLATFORM_HAZARD_DISTANCE_KM = 0.45
MIN_POINT_HAZARD_DISTANCE_KM = 0.25
MIN_POINT_PLATFORM_DISTANCE_KM = 0.15
NFZ_MIN_SIZE_KM = 0.70
NFZ_MAX_SIZE_KM = 1.15
NFZ_DETOUR_FACTOR = 0.45
MAX_POINT_GENERATION_TRIALS = 20000
MAX_PLATFORM_GENERATION_TRIALS = 20000


# ---------- 任务与覆盖参数 ----------
MIN_COVERAGE_RATE = 0.90
PLANNING_HORIZON_HOURS = 8.0
INSPECTION_FREQ_CHOICES = [1, 1, 1, 2, 2, 3]
SERVICE_TIME_RANGE_HOURS = (0.05, 0.10)
UNCOVERED_PENALTY = 1000.0


# ---------- 成本参数 ----------
ENERGY_COST_PER_KM = 1.00
PLATFORM_FIXED_COST_RANGE = (85.0, 125.0)
MAX_UAVS_PER_PLATFORM_TYPE = 4

# UAV type parameters. max_round_trip_km is the safe round-trip distance.
UAV_TYPE_DATA = {
    0: {
        "name": "light",
        "speed_kmph": 45.0,
        "max_round_trip_km": 7.0,
        "acquisition_cost": 18.0,
        "maintenance_cost": 2.0,
        "energy_multiplier": 0.85,
    },
    1: {
        "name": "medium",
        "speed_kmph": 60.0,
        "max_round_trip_km": 14.0,
        "acquisition_cost": 32.0,
        "maintenance_cost": 3.0,
        "energy_multiplier": 1.00,
    },
    2: {
        "name": "heavy",
        "speed_kmph": 72.0,
        "max_round_trip_km": 24.0,
        "acquisition_cost": 52.0,
        "maintenance_cost": 5.0,
        "energy_multiplier": 1.20,
    },
}


# ---------- Gurobi 参数 ----------
GUROBI_TIME_LIMIT_SMALL_SEC = 7200.0
GUROBI_TIME_LIMIT_MEDIUM_SEC = 1800.0
GUROBI_TIME_LIMIT_LARGE_SEC = 1800.0
GUROBI_MIP_GAP = 1e-6
GUROBI_THREADS = 4
GUROBI_OUTPUT_FLAG = 1
GUROBI_MIP_FOCUS = 1
GUROBI_HEURISTICS = 0.20
GUROBI_CUTS = 2

# Memory protection for the full-path MIP. For a 16-GB computer, 12 GB is a
# conservative starting value. Change this value in the parameter section if
# the machine has substantially less or more memory.
GUROBI_NODEFILE_START_GB = 0.5
GUROBI_SOFT_MEM_LIMIT_GB = 12.0

# ---------- Full-path Gurobi benchmark extensions ----------
# This richer MILP is much harder than the compact assignment model. Keep large-scale
# Gurobi off unless you deliberately want a long time-limited benchmark.
FULL_PATH_USE_FREQUENCY_EXPANSION = True
FULL_PATH_BETA_CONFIDENCE = 0.95
FULL_PATH_ENERGY_MEAN = 1.00
FULL_PATH_ENERGY_STD = 0.08
FULL_PATH_MAX_COMPLETE_TASKS = 70
FULL_PATH_NEAREST_ROUTE_NEIGHBORS = 14
FULL_PATH_BLOCK_DIRECT_ARCS_BETWEEN_SAME_POINT_OCCURRENCES = True
FULL_PATH_TIME_LIMIT_SMALL_SEC = 7200.0
FULL_PATH_TIME_LIMIT_MEDIUM_SEC = 1800.0
FULL_PATH_TIME_LIMIT_LARGE_SEC = 1800.0
FULL_PATH_EXPORT_SOLUTION = True

# 与论文模型一致的规划—路径联动约束：
# 论文中的上层变量 y_{ijk} 表示监测点 j 选择平台 i 和无人机类型 k；
# 下层每个频次任务 u（其中 pi(u)=j）必须服从同一个平台-机型组合。
# 代码中使用 point_assign[j,i,k] 实现该逻辑，避免与论文中表示无人机编号的 q 冲突。
ENFORCE_POINT_LEVEL_PLATFORM_TYPE_ASSIGNMENT = True

# 公平性约束：Gurobi 禁止同一监测点的两个频次任务直接相连，但允许它们
# 在同一架无人机路线中被其他监测点隔开后再次访问。ALNS采用相同规则：
# 仅禁止相邻重复访问，不再禁止同一点在同一路线中非相邻出现。
ALNS_BLOCK_ADJACENT_SAME_POINT_OCCURRENCES = True


# ---------- ALNS-Q-learning-SA 参数 ----------
ALNS_RUNS_PER_INSTANCE = 10
ALNS_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
ALNS_TIME_LIMIT_SMALL_SEC = 600.0
ALNS_TIME_LIMIT_MEDIUM_SEC = 600.0
ALNS_TIME_LIMIT_LARGE_SEC = 600.0
ALNS_MAX_ITER_SMALL = 5000
ALNS_MAX_ITER_MEDIUM = 12000
ALNS_MAX_ITER_LARGE = 20000
ALNS_INITIAL_TEMPERATURE = 100.0
ALNS_COOLING_RATE = 0.995
ALNS_MIN_TEMPERATURE = 1e-4
ALNS_EPSILON_INITIAL = 0.25
ALNS_EPSILON_FINAL = 0.03
QL_ALPHA = 0.20
QL_GAMMA = 0.80
DESTROY_RATIO_MIN = 0.15
DESTROY_RATIO_MAX = 0.35
BIG_PENALTY = 100000.0

# A reserve is removed from the nominal 600 s limit. The reserve absorbs the
# last route-evaluation call, CSV checkpointing, and Python scheduling delay.
# Consequently, the reported runtime should remain below or very close to 600 s.
ALNS_TIME_GUARD_MEDIUM_SEC = 35.0
ALNS_TIME_GUARD_LARGE_SEC = 55.0
ALNS_TIME_LIMIT_TOLERANCE_SEC = 5.0

# ---------- ALNS路线解码、覆盖优先与强化参数 ----------
# Medium/large instances use a deliberately compact multi-start design. The
# previous 250-combination initialization was the main source of 1000--3500 s
# overruns even though the nominal limit was 600 s.
INITIAL_RESTARTS_PER_PLATFORM_COMBINATION = 1
INITIAL_ROUTE_AWARE_TOP_COMBINATIONS = 2
INITIAL_PLATFORM_COMBINATION_CAP_MEDIUM = 10
INITIAL_PLATFORM_COMBINATION_CAP_LARGE = 3
ROUTE_FIRST_INITIAL_BUDGET_MEDIUM_SEC = 90.0
ROUTE_FIRST_INITIAL_BUDGET_LARGE_SEC = 220.0

# Coverage-first route construction and rescue. A monitoring point is committed
# only when all of its frequency-expanded visits can be inserted under one
# platform-UAV-type combination, preserving the manuscript-level linkage.
ENABLE_COVERAGE_FIRST_INITIALIZATION = True
ENABLE_COVERAGE_RESCUE = True
COVERAGE_FIRST_MAX_OPTIONS_MEDIUM = 8
COVERAGE_FIRST_MAX_OPTIONS_LARGE = 6
COVERAGE_RESCUE_INTERVAL = 40
COVERAGE_RESCUE_MAX_POINTS_MEDIUM = 10
COVERAGE_RESCUE_MAX_POINTS_LARGE = 24
COVERAGE_RESCUE_MIN_REMAINING_SEC = 20.0

# 路线解码时将启用一架新无人机的固定成本纳入插入增量，避免无必要地
# 将任务分散到多架无人机上。
DECODER_INCLUDE_UAV_ACTIVATION_COST = True

# 路线级强化：中大规模采用 first-improvement 和较小邻域，以确保严格限时。
ENABLE_ROUTE_LEVEL_INTENSIFICATION = True
ROUTE_INTENSIFICATION_INTERVAL = 80
ROUTE_INTENSIFICATION_ON_IMPROVING_CANDIDATE = False
ROUTE_INTENSIFICATION_MAX_ROUNDS = 1
FINAL_ROUTE_INTENSIFICATION_MAX_ROUNDS = 2

ENABLE_UAV_ELIMINATION = True
ENABLE_ROUTE_MERGE = True
ENABLE_INTER_ROUTE_RELOCATE = True
ENABLE_INTER_ROUTE_SWAP = True
ENABLE_ROUTE_TWO_OPT = True

LOCAL_SEARCH_MAX_MOVE_EVALUATIONS = 200
UAV_ELIMINATION_ORDER_TRIALS = 3
LOCAL_SEARCH_FIRST_IMPROVEMENT = True

# Pair reassignment is retained in the source for reproducibility but disabled
# for the official medium/large runs because its O(|J|^2) enumeration caused
# severe runtime overruns, especially for M120 and L300.
ENABLE_PAIR_REASSIGNMENT_POLISH = False
PAIR_REASSIGNMENT_INTERVAL = 200
PAIR_REASSIGNMENT_MAX_EVALUATIONS = 300
PAIR_REASSIGNMENT_MAX_PASSES = 1

# Feasibility-first scalarization used by SA/Q-learning while a solution is
# infeasible. Feasible solutions are always preferred to infeasible solutions.
FEASIBILITY_BASE_SCORE = 1.0e12
FEASIBILITY_COVERAGE_WEIGHT = 1.0e9
FEASIBILITY_OTHER_VIOLATION_WEIGHT = 1.0e7

# Active per-run hard deadline. Route construction reads this value to abort a
# long nested insertion loop safely. It is set only inside solve_alns_qsa().
_ACTIVE_ALNS_DEADLINE: Optional[float] = None


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    scale: str
    n_points: int
    n_candidate_platforms: int
    n_hazard_sources: int
    n_no_fly_zones: int
    n_uav_types: int
    open_platforms_fixed: int
    seed: int


@dataclass
class UAVInstance:
    spec: InstanceSpec
    points: Dict[int, Tuple[float, float]]
    platforms: Dict[int, Tuple[float, float]]
    hazards: Dict[int, Tuple[float, float]]
    no_fly_zones: Dict[int, Tuple[float, float, float, float]]
    platform_fixed_cost: Dict[int, float]
    inspection_freq: Dict[int, int]
    service_time: Dict[int, float]
    site_feasible: Dict[int, int]
    safe_distance: Dict[Tuple[int, int], float]
    feasible_assign: Dict[Tuple[int, int, int], int]
    path_accessible: Dict[Tuple[int, int], int]
    uav_data: Dict[int, Dict[str, float]]


@dataclass
class Solution:
    open_platforms: Tuple[int, ...]
    assignment: Dict[int, Optional[Tuple[int, int]]]
    obj: float
    fitness: float
    feasible: bool
    coverage: float
    n_uavs: Dict[Tuple[int, int], int]
    workloads: Dict[Tuple[int, int], float]
    violation: Dict[str, float]
    # Full-path routes are shared by Gurobi extraction and ALNS evaluation.
    # Key: (platform_id, uav_type, uav_copy); value: ordered point visits.
    # A point appears inspection_freq[j] times if all its frequency-expanded tasks are served.
    routes: Optional[Dict[Tuple[int, int, int], List[int]]] = None
    components: Optional[Dict[str, float]] = None


# ============================================================
# 3. 基础几何函数
# ============================================================

def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_rect(p: Tuple[float, float], rect: Tuple[float, float, float, float]) -> bool:
    x, y = p
    xmin, ymin, xmax, ymax = rect
    return xmin <= x <= xmax and ymin <= y <= ymax


def orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2


def on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) - 1e-12 <= b[0] <= max(a[0], c[0]) + 1e-12
        and min(a[1], c[1]) - 1e-12 <= b[1] <= max(a[1], c[1]) + 1e-12
    )


def segments_intersect(p1: Tuple[float, float], q1: Tuple[float, float],
                       p2: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def segment_intersects_rect(a: Tuple[float, float], b: Tuple[float, float],
                            rect: Tuple[float, float, float, float]) -> bool:
    if point_in_rect(a, rect) or point_in_rect(b, rect):
        return True
    xmin, ymin, xmax, ymax = rect
    edges = [
        ((xmin, ymin), (xmax, ymin)),
        ((xmax, ymin), (xmax, ymax)),
        ((xmax, ymax), (xmin, ymax)),
        ((xmin, ymax), (xmin, ymin)),
    ]
    return any(segments_intersect(a, b, e1, e2) for e1, e2 in edges)


def safe_detour_distance(a: Tuple[float, float], b: Tuple[float, float],
                         no_fly_zones: Dict[int, Tuple[float, float, float, float]]) -> Tuple[float, int]:
    base = euclidean(a, b)
    extra = 0.0
    crosses = 0
    for rect in no_fly_zones.values():
        if segment_intersects_rect(a, b, rect):
            xmin, ymin, xmax, ymax = rect
            extra += NFZ_DETOUR_FACTOR * ((xmax - xmin) + (ymax - ymin))
            crosses += 1
    return base + extra, 1


def random_point_outside_rects(rng: random.Random,
                               rects: Dict[int, Tuple[float, float, float, float]],
                               trials: int = 10000) -> Tuple[float, float]:
    for _ in range(trials):
        p = (rng.uniform(0.3, AREA_WIDTH_KM - 0.3), rng.uniform(0.3, AREA_HEIGHT_KM - 0.3))
        if not any(point_in_rect(p, rect) for rect in rects.values()):
            return p
    raise RuntimeError("Failed to sample a point outside no-fly zones.")


# ============================================================
# 4. 实例生成
# ============================================================

def build_instance(spec: InstanceSpec) -> UAVInstance:
    rng = random.Random(spec.seed)

    no_fly_zones: Dict[int, Tuple[float, float, float, float]] = {}
    for z in range(spec.n_no_fly_zones):
        w = rng.uniform(NFZ_MIN_SIZE_KM, NFZ_MAX_SIZE_KM)
        h = rng.uniform(NFZ_MIN_SIZE_KM, NFZ_MAX_SIZE_KM)
        xmin = rng.uniform(1.0, AREA_WIDTH_KM - w - 1.0)
        ymin = rng.uniform(1.0, AREA_HEIGHT_KM - h - 1.0)
        no_fly_zones[z] = (xmin, ymin, xmin + w, ymin + h)

    hazards: Dict[int, Tuple[float, float]] = {}
    for h in range(spec.n_hazard_sources):
        hazards[h] = random_point_outside_rects(rng, no_fly_zones)

    platforms: Dict[int, Tuple[float, float]] = {}
    platform_fixed_cost: Dict[int, float] = {}
    site_feasible: Dict[int, int] = {}

    trials = 0
    while len(platforms) < spec.n_candidate_platforms and trials < MAX_PLATFORM_GENERATION_TRIALS:
        trials += 1
        p = random_point_outside_rects(rng, no_fly_zones)
        if any(euclidean(p, q) < MIN_POINT_PLATFORM_DISTANCE_KM for q in platforms.values()):
            continue
        i = len(platforms)
        platforms[i] = p
        platform_fixed_cost[i] = rng.uniform(*PLATFORM_FIXED_COST_RANGE)
        hazard_ok = all(euclidean(p, hazards[h]) >= MIN_PLATFORM_HAZARD_DISTANCE_KM for h in hazards)
        nfz_ok = not any(point_in_rect(p, rect) for rect in no_fly_zones.values())
        site_feasible[i] = 1 if hazard_ok and nfz_ok else 0

    if len(platforms) < spec.n_candidate_platforms:
        raise RuntimeError("Failed to generate enough candidate platforms.")

    if sum(site_feasible.values()) < spec.open_platforms_fixed:
        # In the unlikely event of insufficient feasible sites, relax only the generated flags.
        sorted_platforms = sorted(platforms, key=lambda i: min(euclidean(platforms[i], hazards[h]) for h in hazards), reverse=True)
        for i in sorted_platforms[:spec.open_platforms_fixed]:
            site_feasible[i] = 1

    points: Dict[int, Tuple[float, float]] = {}
    inspection_freq: Dict[int, int] = {}
    service_time: Dict[int, float] = {}

    trials = 0
    while len(points) < spec.n_points and trials < MAX_POINT_GENERATION_TRIALS:
        trials += 1
        p = random_point_outside_rects(rng, no_fly_zones)
        if any(euclidean(p, hazards[h]) < MIN_POINT_HAZARD_DISTANCE_KM for h in hazards):
            continue
        j = len(points)
        points[j] = p
        inspection_freq[j] = rng.choice(INSPECTION_FREQ_CHOICES)
        service_time[j] = rng.uniform(*SERVICE_TIME_RANGE_HOURS)

    if len(points) < spec.n_points:
        raise RuntimeError("Failed to generate enough monitoring points.")

    safe_distance: Dict[Tuple[int, int], float] = {}
    path_accessible: Dict[Tuple[int, int], int] = {}
    feasible_assign: Dict[Tuple[int, int, int], int] = {}

    for i, pi in platforms.items():
        for j, pj in points.items():
            d, accessible = safe_detour_distance(pi, pj, no_fly_zones)
            safe_distance[(i, j)] = d
            path_accessible[(i, j)] = accessible
            for k, data in UAV_TYPE_DATA.items():
                feasible = (
                    site_feasible[i] == 1
                    and accessible == 1
                    and 2.0 * d <= data["max_round_trip_km"] + 1e-9
                )
                feasible_assign[(j, i, k)] = 1 if feasible else 0

    inst = UAVInstance(
        spec=spec,
        points=points,
        platforms=platforms,
        hazards=hazards,
        no_fly_zones=no_fly_zones,
        platform_fixed_cost=platform_fixed_cost,
        inspection_freq=inspection_freq,
        service_time=service_time,
        site_feasible=site_feasible,
        safe_distance=safe_distance,
        feasible_assign=feasible_assign,
        path_accessible=path_accessible,
        uav_data=UAV_TYPE_DATA,
    )

    if SAVE_INSTANCE_CSV:
        save_instance_csv(inst)

    return inst


def save_instance_csv(inst: UAVInstance) -> None:
    base_dir = os.path.join(INSTANCE_DIR, inst.spec.instance_id)
    os.makedirs(base_dir, exist_ok=True)

    pd.DataFrame([
        {"point_id": j, "x_km": p[0], "y_km": p[1],
         "freq": inst.inspection_freq[j], "service_time_h": inst.service_time[j]}
        for j, p in inst.points.items()
    ]).to_csv(os.path.join(base_dir, "monitoring_points.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"platform_id": i, "x_km": p[0], "y_km": p[1],
         "fixed_cost": inst.platform_fixed_cost[i], "site_feasible": inst.site_feasible[i]}
        for i, p in inst.platforms.items()
    ]).to_csv(os.path.join(base_dir, "candidate_platforms.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"hazard_id": h, "x_km": p[0], "y_km": p[1]}
        for h, p in inst.hazards.items()
    ]).to_csv(os.path.join(base_dir, "hazard_sources.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"nfz_id": z, "xmin_km": r[0], "ymin_km": r[1], "xmax_km": r[2], "ymax_km": r[3]}
        for z, r in inst.no_fly_zones.items()
    ]).to_csv(os.path.join(base_dir, "no_fly_zones.csv"), index=False, encoding="utf-8-sig")


# ============================================================
# 5. 共同目标函数与可行性评价
# ============================================================

def fleet_unit_cost(inst: UAVInstance, k: int) -> float:
    return inst.uav_data[k]["acquisition_cost"] + inst.uav_data[k]["maintenance_cost"]


def assignment_energy_cost(inst: UAVInstance, j: int, i: int, k: int) -> float:
    d = inst.safe_distance[(i, j)]
    freq = inst.inspection_freq[j]
    mult = inst.uav_data[k]["energy_multiplier"]
    return ENERGY_COST_PER_KM * mult * freq * 2.0 * d


def assignment_workload(inst: UAVInstance, j: int, i: int, k: int) -> float:
    d = inst.safe_distance[(i, j)]
    freq = inst.inspection_freq[j]
    speed = inst.uav_data[k]["speed_kmph"]
    return freq * (2.0 * d / speed + inst.service_time[j])


def required_uavs_for_workload(workload: float) -> int:
    """Return the required UAV count and safely reject non-finite workloads."""
    if not math.isfinite(workload):
        return MAX_UAVS_PER_PLATFORM_TYPE + 1
    if workload <= 1e-12:
        return 0
    return int(math.ceil(workload / PLANNING_HORIZON_HOURS - 1e-12))


def min_required_covered_points(inst: UAVInstance) -> int:
    return int(math.ceil(MIN_COVERAGE_RATE * len(inst.points) - 1e-9))


def route_distance_and_time(inst: UAVInstance,
                            platform_id: int,
                            uav_type: int,
                            route: List[int]) -> Tuple[float, float, float]:
    """Return (distance, travel_time, service_time) for platform -> route -> platform."""
    if _hard_deadline_reached():
        return float("inf"), float("inf"), float("inf")
    if not route:
        return 0.0, 0.0, 0.0
    dist = 0.0
    prev_point: Optional[int] = None
    for idx, j in enumerate(route):
        if idx % 16 == 0 and _hard_deadline_reached():
            return float("inf"), float("inf"), float("inf")
        if prev_point is None:
            dist += inst.safe_distance[(platform_id, j)]
        else:
            d, _ = safe_detour_distance(inst.points[prev_point], inst.points[j], inst.no_fly_zones)
            dist += d
        prev_point = j
    if prev_point is not None:
        dist += inst.safe_distance[(platform_id, prev_point)]
    speed = inst.uav_data[uav_type]["speed_kmph"]
    travel_time = dist / speed if speed > 0 else float("inf")
    service_time = sum(inst.service_time[j] for j in route)
    return dist, travel_time, service_time


def route_respects_same_point_arc_rule(route: List[int]) -> bool:
    """Match the full-path MILP rule that blocks direct arcs between two occurrences of the same point."""
    if not ALNS_BLOCK_ADJACENT_SAME_POINT_OCCURRENCES:
        return True
    return all(route[idx] != route[idx + 1] for idx in range(len(route) - 1))


def route_is_feasible(inst: UAVInstance,
                      platform_id: int,
                      uav_type: int,
                      route: List[int]) -> Tuple[bool, float, float, float]:
    """Check route feasibility under the same endurance, time, and same-point arc rules as the MILP."""
    if not route:
        return True, 0.0, 0.0, 0.0
    if not route_respects_same_point_arc_rule(route):
        return False, float("inf"), float("inf"), float("inf")
    dist, travel_time, service_time = route_distance_and_time(inst, platform_id, uav_type, route)
    mult = inst.uav_data[uav_type]["energy_multiplier"]
    robust_distance = robust_energy_factor() * mult * dist
    endurance = inst.uav_data[uav_type]["max_round_trip_km"]
    total_time = travel_time + service_time
    feasible = robust_distance <= endurance + 1e-9 and total_time <= PLANNING_HORIZON_HOURS + 1e-9
    return feasible, dist, robust_distance, total_time


def route_insertion_delta(inst: UAVInstance,
                          platform_id: int,
                          uav_type: int,
                          route: List[int],
                          point_id: int,
                          position: int) -> Tuple[bool, float, List[int]]:
    """Try inserting one visit and return feasibility, cost increase, and the new route."""
    if _hard_deadline_reached():
        return False, float("inf"), route
    old_ok, old_dist, _, _ = route_is_feasible(inst, platform_id, uav_type, route)
    if not old_ok:
        return False, float("inf"), route

    # Match the MILP exactly: only a direct same-point arc is forbidden.
    # The same point may reappear later in the route if another point separates
    # the two visits, which permits one UAV to execute multiple occurrences.
    if ALNS_BLOCK_ADJACENT_SAME_POINT_OCCURRENCES:
        left_same = position > 0 and route[position - 1] == point_id
        right_same = position < len(route) and route[position] == point_id
        if left_same or right_same:
            return False, float("inf"), route

    new_route = list(route)
    new_route.insert(position, point_id)
    new_ok, new_dist, _, _ = route_is_feasible(inst, platform_id, uav_type, new_route)
    if not new_ok:
        return False, float("inf"), route
    mult = inst.uav_data[uav_type]["energy_multiplier"]
    delta = ENERGY_COST_PER_KM * mult * (new_dist - old_dist)
    return True, delta, new_route


def construct_routes_from_assignment(inst: UAVInstance,
                                     open_platforms: Iterable[int],
                                     assignment: Dict[int, Optional[Tuple[int, int]]]) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Decode point-level assignments into explicit UAV-copy routes.

    Improvements over the earlier decoder:
        1) frequency occurrences are inserted by occurrence rounds rather than
           batching all occurrences of one point consecutively;
        2) opening an empty UAV route incurs the UAV fixed activation cost in
           the insertion score;
        3) the same monitoring point may reappear non-consecutively on one UAV
           route, consistent with the full-path MILP;
        4) partial frequency service is removed before evaluation.
    """
    open_tuple = tuple(sorted(set(open_platforms)))
    all_routes: Dict[Tuple[int, int, int], List[int]] = {
        (i, k, v): []
        for i in open_tuple
        for k in inst.uav_data
        for v in range(MAX_UAVS_PER_PLATFORM_TYPE)
    }

    valid_points: List[int] = []
    for j, val in assignment.items():
        if val is None:
            continue
        i, k = val
        if i in open_tuple and inst.feasible_assign.get((j, i, k), 0) == 1:
            valid_points.append(j)

    # Occurrence-round ordering makes it possible to place repeated visits to
    # the same point on one route with other points between them.
    max_freq = max((max(1, int(inst.inspection_freq[j])) for j in valid_points), default=0)
    visit_sequence: List[int] = []
    for occ in range(max_freq):
        round_points = [j for j in valid_points if max(1, int(inst.inspection_freq[j])) > occ]
        round_points.sort(
            key=lambda j: (
                -inst.safe_distance[assignment[j]],  # type: ignore[index]
                -inst.service_time[j],
                j,
            )
        )
        visit_sequence.extend(round_points)

    failed_points = set()
    inserted_count: Dict[int, int] = {j: 0 for j in valid_points}

    for j in visit_sequence:
        if _hard_deadline_reached():
            break
        if j in failed_points:
            continue
        i, k = assignment[j]  # type: ignore[misc]
        best_key: Optional[Tuple[int, int, int]] = None
        best_route: Optional[List[int]] = None
        best_score = float("inf")

        for v in range(MAX_UAVS_PER_PLATFORM_TYPE):
            if _hard_deadline_reached():
                break
            veh = (i, k, v)
            route = all_routes[veh]
            activation_cost = 0.0
            if DECODER_INCLUDE_UAV_ACTIVATION_COST and not route:
                activation_cost = fleet_unit_cost(inst, k)

            for pos in range(len(route) + 1):
                if _hard_deadline_reached():
                    break
                ok, energy_delta, new_route = route_insertion_delta(inst, i, k, route, j, pos)
                if not ok:
                    continue
                score = activation_cost + energy_delta
                # Deterministic tie-break: prefer an already-used UAV, then the
                # lower copy index, to improve fleet consolidation.
                tie = (1 if not route else 0, v, pos)
                if score < best_score - 1e-12:
                    best_score = score
                    best_key = veh
                    best_route = new_route
                    best_tie = tie
                elif abs(score - best_score) <= 1e-12 and best_key is not None:
                    if tie < best_tie:
                        best_key = veh
                        best_route = new_route
                        best_tie = tie

        if best_key is None or best_route is None:
            failed_points.add(j)
        else:
            all_routes[best_key] = best_route
            inserted_count[j] += 1

    # Remove partial service of any point so the evaluator sees either all
    # required occurrences or none.
    for j in valid_points:
        required = max(1, int(inst.inspection_freq[j]))
        if inserted_count.get(j, 0) != required:
            failed_points.add(j)

    if failed_points:
        for veh, route in list(all_routes.items()):
            all_routes[veh] = [j for j in route if j not in failed_points]

    return {veh: rt for veh, rt in all_routes.items() if rt}

def evaluate_full_path_routes(inst: UAVInstance,
                              open_platforms: Iterable[int],
                              routes: Dict[Tuple[int, int, int], List[int]]) -> Solution:
    """
    Common full-path objective and feasibility evaluator used by both Gurobi and ALNS.

    Objective = opened-platform fixed cost + used-UAV fixed cost + route-arc energy cost
              + uncovered monitoring-point penalty.

    This evaluator mirrors the full-path Gurobi model at route level, so ALNS can no longer
    report assignment-level costs that are below the Gurobi lower bound for the same model.
    """
    open_tuple = tuple(sorted(set(open_platforms)))
    open_set = set(open_tuple)

    platform_cost = sum(inst.platform_fixed_cost[i] for i in open_tuple)
    fleet_cost = 0.0
    route_energy_cost = 0.0
    route_time_violation = 0.0
    endurance_violation = 0.0
    invalid_route_count = 0.0
    invalid_assignment_count = 0.0

    point_visit_count: Dict[int, int] = {j: 0 for j in inst.points}
    point_platform_type_sets: Dict[int, set] = {j: set() for j in inst.points}
    assignment: Dict[int, Optional[Tuple[int, int]]] = {j: None for j in inst.points}
    n_uavs: Dict[Tuple[int, int], int] = {(i, k): 0 for i in inst.platforms for k in inst.uav_data}
    workloads: Dict[Tuple[int, int], float] = {(i, k): 0.0 for i in inst.platforms for k in inst.uav_data}

    for (i, k, v), route in routes.items():
        if not route:
            continue
        if i not in open_set or inst.site_feasible.get(i, 0) != 1 or k not in inst.uav_data or v < 0 or v >= MAX_UAVS_PER_PLATFORM_TYPE:
            invalid_route_count += 1.0
            continue

        local_invalid = False
        for j in route:
            if j not in inst.points or inst.feasible_assign.get((j, i, k), 0) != 1:
                local_invalid = True
                invalid_assignment_count += 1.0
        if local_invalid:
            continue

        ok, dist, robust_dist, total_time = route_is_feasible(inst, i, k, route)
        mult = inst.uav_data[k]["energy_multiplier"]
        route_energy_cost += ENERGY_COST_PER_KM * mult * dist
        fleet_cost += fleet_unit_cost(inst, k)
        n_uavs[(i, k)] += 1
        workloads[(i, k)] += total_time

        endurance = inst.uav_data[k]["max_round_trip_km"]
        if robust_dist > endurance + 1e-9:
            endurance_violation += robust_dist - endurance
        if total_time > PLANNING_HORIZON_HOURS + 1e-9:
            route_time_violation += total_time - PLANNING_HORIZON_HOURS
        if not ok:
            invalid_route_count += 1.0

        for j in route:
            point_visit_count[j] += 1
            point_platform_type_sets[j].add((i, k))
            if assignment[j] is None:
                assignment[j] = (i, k)

    covered_points = [j for j in inst.points if point_visit_count[j] >= max(1, int(inst.inspection_freq[j]))]
    covered = len(covered_points)
    uncovered = len(inst.points) - covered
    uncovered_penalty = UNCOVERED_PENALTY * uncovered

    base_obj = platform_cost + fleet_cost + route_energy_cost + uncovered_penalty

    wrong_open_count = abs(len(open_tuple) - inst.spec.open_platforms_fixed)
    infeasible_site_count = sum(1 for i in open_tuple if inst.site_feasible[i] != 1)
    coverage_shortage = max(0, min_required_covered_points(inst) - covered)
    duplicate_or_partial_count = 0.0
    for j in inst.points:
        freq = max(1, int(inst.inspection_freq[j]))
        # Partial service is not accepted as point coverage and is penalized to guide ALNS.
        if 0 < point_visit_count[j] < freq:
            duplicate_or_partial_count += (freq - point_visit_count[j])
        if point_visit_count[j] > freq:
            duplicate_or_partial_count += (point_visit_count[j] - freq)

    point_platform_type_inconsistency_count = float(
        sum(max(0, len(point_platform_type_sets[j]) - 1) for j in inst.points)
    )

    violation = {
        "wrong_open_count": float(wrong_open_count),
        "infeasible_site_count": float(infeasible_site_count),
        "coverage_shortage": float(coverage_shortage),
        "route_time_violation": float(route_time_violation),
        "endurance_violation": float(endurance_violation),
        "invalid_route_count": float(invalid_route_count),
        "invalid_assignment_count": float(invalid_assignment_count),
        "duplicate_or_partial_count": float(duplicate_or_partial_count),
        "point_platform_type_inconsistency_count": point_platform_type_inconsistency_count,
    }
    total_violation = sum(violation.values())
    feasible = total_violation <= 1e-9
    coverage = covered / len(inst.points) if inst.points else 0.0
    fitness = base_obj + BIG_PENALTY * total_violation

    components = {
        "platform_cost": float(platform_cost),
        "fleet_cost": float(fleet_cost),
        "route_energy_cost": float(route_energy_cost),
        "uncovered_penalty": float(uncovered_penalty),
        "covered_points": float(covered),
        "uncovered_points": float(uncovered),
    }

    # Only points that are fully served should remain assigned in the returned solution.
    clean_assignment: Dict[int, Optional[Tuple[int, int]]] = {}
    covered_set = set(covered_points)
    for j in inst.points:
        clean_assignment[j] = assignment[j] if j in covered_set else None

    return Solution(
        open_platforms=open_tuple,
        assignment=clean_assignment,
        obj=base_obj,
        fitness=fitness,
        feasible=feasible,
        coverage=coverage,
        n_uavs=n_uavs,
        workloads=workloads,
        violation=violation,
        routes=routes,
        components=components,
    )


def evaluate_solution(inst: UAVInstance,
                      open_platforms: Iterable[int],
                      assignment: Dict[int, Optional[Tuple[int, int]]]) -> Solution:
    """
    Unified ALNS evaluator.

    ALNS still manipulates point-level assignments for speed, but every candidate is converted
    into explicit full-path UAV-copy routes and then evaluated by evaluate_full_path_routes().
    This makes the reported ALNS objective directly comparable with the full-path Gurobi MILP.
    """
    routes = construct_routes_from_assignment(inst, open_platforms, assignment)
    return evaluate_full_path_routes(inst, open_platforms, routes)


# ============================================================
# 6. Gurobi full-path MILP benchmark
# ============================================================

def should_run_gurobi(scale: str) -> bool:
    if scale == "small":
        return RUN_GUROBI_SMALL
    if scale == "medium":
        return RUN_GUROBI_MEDIUM
    if scale == "large":
        return RUN_GUROBI_LARGE
    return False


def get_gurobi_time_limit(scale: str) -> float:
    # The full-path model is intentionally harder than the previous compact model.
    if scale == "small":
        return FULL_PATH_TIME_LIMIT_SMALL_SEC
    if scale == "medium":
        return FULL_PATH_TIME_LIMIT_MEDIUM_SEC
    return FULL_PATH_TIME_LIMIT_LARGE_SEC


def get_alns_time_limit(scale: str) -> float:
    if scale == "small":
        return ALNS_TIME_LIMIT_SMALL_SEC
    if scale == "medium":
        return ALNS_TIME_LIMIT_MEDIUM_SEC
    return ALNS_TIME_LIMIT_LARGE_SEC


def get_alns_max_iter(scale: str) -> int:
    if scale == "small":
        return ALNS_MAX_ITER_SMALL
    if scale == "medium":
        return ALNS_MAX_ITER_MEDIUM
    return ALNS_MAX_ITER_LARGE


def get_search_profile(inst: UAVInstance) -> Dict[str, Any]:
    """Scale-specific controls for strict timing and coverage-first search."""
    if inst.spec.scale == "large":
        return {
            "time_guard_sec": ALNS_TIME_GUARD_LARGE_SEC,
            "platform_combo_cap": INITIAL_PLATFORM_COMBINATION_CAP_LARGE,
            "route_first_budget_sec": ROUTE_FIRST_INITIAL_BUDGET_LARGE_SEC,
            "coverage_option_cap": COVERAGE_FIRST_MAX_OPTIONS_LARGE,
            "coverage_rescue_max_points": COVERAGE_RESCUE_MAX_POINTS_LARGE,
            "local_move_evaluations": 60,
            "initial_polish_passes": 0,
            "periodic_polish_interval": 0,
            "route_intensification_interval": 120,
            "route_intensification_rounds": 1,
            "final_intensification_rounds": 1,
            "enable_route_aware_repair": False,
            "enable_pair_polish": False,
        }

    # M120 receives a slightly more conservative profile than M80/M100.
    if inst.spec.instance_id == "M120":
        return {
            "time_guard_sec": 45.0,
            "platform_combo_cap": 6,
            "route_first_budget_sec": 130.0,
            "coverage_option_cap": 6,
            "coverage_rescue_max_points": 14,
            "local_move_evaluations": 90,
            "initial_polish_passes": 1,
            "periodic_polish_interval": 100,
            "route_intensification_interval": 100,
            "route_intensification_rounds": 1,
            "final_intensification_rounds": 1,
            "enable_route_aware_repair": False,
            "enable_pair_polish": False,
        }

    return {
        "time_guard_sec": ALNS_TIME_GUARD_MEDIUM_SEC,
        "platform_combo_cap": INITIAL_PLATFORM_COMBINATION_CAP_MEDIUM,
        "route_first_budget_sec": ROUTE_FIRST_INITIAL_BUDGET_MEDIUM_SEC,
        "coverage_option_cap": COVERAGE_FIRST_MAX_OPTIONS_MEDIUM,
        "coverage_rescue_max_points": COVERAGE_RESCUE_MAX_POINTS_MEDIUM,
        "local_move_evaluations": LOCAL_SEARCH_MAX_MOVE_EVALUATIONS,
        "initial_polish_passes": 1,
        "periodic_polish_interval": 75,
        "route_intensification_interval": ROUTE_INTENSIFICATION_INTERVAL,
        "route_intensification_rounds": ROUTE_INTENSIFICATION_MAX_ROUNDS,
        "final_intensification_rounds": FINAL_ROUTE_INTENSIFICATION_MAX_ROUNDS,
        "enable_route_aware_repair": False,
        "enable_pair_polish": False,
    }


def get_alns_time_guard(inst: UAVInstance) -> float:
    return float(get_search_profile(inst)["time_guard_sec"])


def remaining_time(deadline: Optional[float]) -> float:
    if deadline is None:
        return float("inf")
    return max(0.0, deadline - time.time())


def total_violation_value(sol: Solution) -> float:
    return float(sum(max(0.0, safe_float(v) or 0.0) for v in sol.violation.values()))


def coverage_shortage_value(sol: Solution) -> float:
    return float(max(0.0, safe_float(sol.violation.get("coverage_shortage")) or 0.0))


def other_violation_value(sol: Solution) -> float:
    return max(0.0, total_violation_value(sol) - coverage_shortage_value(sol))


def solution_priority_key(sol: Solution) -> Tuple[float, ...]:
    """Lexicographic feasibility-first ordering used throughout ALNS."""
    if sol.feasible:
        return (0.0, float(sol.obj))
    covered = float((sol.components or {}).get("covered_points", sol.coverage))
    return (
        1.0,
        coverage_shortage_value(sol),
        other_violation_value(sol),
        -covered,
        float(sol.fitness),
    )


def is_better_solution(candidate: Solution, incumbent: Optional[Solution]) -> bool:
    if incumbent is None:
        return True
    return solution_priority_key(candidate) < solution_priority_key(incumbent)


def search_score(sol: Solution) -> float:
    """Scalar score for SA acceptance; consistent with feasibility-first ordering."""
    if sol.feasible:
        return float(sol.obj)
    return (
        FEASIBILITY_BASE_SCORE
        + FEASIBILITY_COVERAGE_WEIGHT * coverage_shortage_value(sol)
        + FEASIBILITY_OTHER_VIOLATION_WEIGHT * other_violation_value(sol)
        + 1.0e5 * max(0.0, 1.0 - float(sol.coverage))
        + min(abs(float(sol.obj)), 1.0e8)
    )


def normal_quantile_approx(beta: float) -> float:
    """Return a small table-based standard-normal quantile for common confidence levels."""
    table = {
        0.80: 0.841621,
        0.85: 1.036433,
        0.90: 1.281552,
        0.95: 1.644854,
        0.975: 1.959964,
        0.99: 2.326348,
    }
    key = min(table, key=lambda x: abs(x - beta))
    return table[key]


def robust_energy_factor() -> float:
    """
    Deterministic equivalent for the chance-constrained energy coefficient.
    If unit energy use xi ~ N(mu, sigma^2), enforcing
        P(xi * distance <= endurance) >= beta
    is represented conservatively by
        (mu + z_beta * sigma) * distance <= endurance.
    """
    z_beta = normal_quantile_approx(FULL_PATH_BETA_CONFIDENCE)
    return FULL_PATH_ENERGY_MEAN + z_beta * FULL_PATH_ENERGY_STD


def build_frequency_expanded_tasks(inst: UAVInstance) -> Tuple[List[int], Dict[int, int], Dict[int, int]]:
    """
    Build inspection tasks from monitoring points.

    task_to_point[t] gives the original monitoring point of task t.
    task_occurrence[t] gives the within-point occurrence index.

    If FULL_PATH_USE_FREQUENCY_EXPANSION is True, a point with frequency f_j
    produces f_j inspection tasks. The coverage variable is still defined at
    monitoring-point level: if a point is covered, all its occurrence tasks must
    be served.
    """
    tasks: List[int] = []
    task_to_point: Dict[int, int] = {}
    task_occurrence: Dict[int, int] = {}
    t = 0
    for j in inst.points:
        freq = int(inst.inspection_freq[j]) if FULL_PATH_USE_FREQUENCY_EXPANSION else 1
        freq = max(1, freq)
        for r in range(freq):
            tasks.append(t)
            task_to_point[t] = j
            task_occurrence[t] = r
            t += 1
    return tasks, task_to_point, task_occurrence


def point_pair_distance_cache(inst: UAVInstance) -> Dict[Tuple[int, int], float]:
    """Safe detour distances between monitoring points, accounting for no-fly-zone intersections."""
    cache: Dict[Tuple[int, int], float] = {}
    for j in inst.points:
        for l in inst.points:
            if j == l:
                cache[(j, l)] = 0.0
            else:
                d, _ = safe_detour_distance(inst.points[j], inst.points[l], inst.no_fly_zones)
                cache[(j, l)] = d
    return cache


def export_full_path_solution(inst: UAVInstance,
                              cov: Dict[int, Any],
                              w: Dict[Tuple[int, int, int], Any],
                              y: Dict[Tuple[int, int, int, int], Any],
                              a: Dict[Tuple[int, int, int, int, int], Any],
                              tasks: List[int],
                              task_to_point: Dict[int, int],
                              point_assign: Optional[Dict[Tuple[int, int, int], Any]] = None) -> None:
    """Export a readable route-level solution for checking and later plotting."""
    if not FULL_PATH_EXPORT_SOLUTION:
        return
    base_dir = os.path.join(RESULT_DIR, "full_path_solution", inst.spec.instance_id)
    os.makedirs(base_dir, exist_ok=True)

    rows_cov = []
    for j in inst.points:
        rows_cov.append({
            "point_id": j,
            "covered": int(round(cov[j].X)) if cov[j].X is not None else 0,
            "freq": inst.inspection_freq[j],
            "x_km": inst.points[j][0],
            "y_km": inst.points[j][1],
        })
    pd.DataFrame(rows_cov).to_csv(os.path.join(base_dir, "covered_points.csv"), index=False, encoding="utf-8-sig")

    rows_task = []
    for (t, i, k, v), var in y.items():
        if var.X > 0.5:
            rows_task.append({
                "task_id": t,
                "point_id": task_to_point[t],
                "platform_id": i,
                "uav_type": k,
                "uav_copy": v,
            })
    pd.DataFrame(rows_task).to_csv(os.path.join(base_dir, "task_assignment.csv"), index=False, encoding="utf-8-sig")

    # Export the manuscript-level monitoring-point assignment y_{ijk}.
    # This file makes it easy to audit that all occurrences of point j share
    # one platform-UAV-type combination.
    if point_assign is not None:
        rows_point_assign = []
        for (j, i, k), var in point_assign.items():
            try:
                selected = var.X > 0.5
            except Exception:
                selected = False
            if selected:
                rows_point_assign.append({
                    "point_id": j,
                    "platform_id": i,
                    "uav_type": k,
                    "inspection_frequency": inst.inspection_freq[j],
                })
        pd.DataFrame(rows_point_assign).to_csv(
            os.path.join(base_dir, "point_platform_type_assignment.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    rows_arc = []
    for (i, k, v, u, t), var in a.items():
        if var.X > 0.5:
            rows_arc.append({
                "platform_id": i,
                "uav_type": k,
                "uav_copy": v,
                "from_node": u,
                "to_node": t,
                "from_point": "depot" if u == -1 else task_to_point[u],
                "to_point": "depot" if t == -1 else task_to_point[t],
            })
    pd.DataFrame(rows_arc).to_csv(os.path.join(base_dir, "route_arcs.csv"), index=False, encoding="utf-8-sig")


def extract_routes_from_gurobi_arcs(inst: UAVInstance,
                                    w: Dict[Tuple[int, int, int], Any],
                                    a: Dict[Tuple[int, int, int, int, int], Any],
                                    task_to_point: Dict[int, int]) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Extract ordered platform -> task -> ... -> platform routes from Gurobi arc variables.
    The returned route representation is identical to the one used by ALNS.
    """
    routes: Dict[Tuple[int, int, int], List[int]] = {}
    selected_successors: Dict[Tuple[int, int, int, int], int] = {}

    for (i, k, v, u, t), var in a.items():
        try:
            val = var.X
        except Exception:
            val = 0.0
        if val > 0.5:
            selected_successors[(i, k, v, u)] = t

    for (i, k, v), var in w.items():
        try:
            used = var.X > 0.5
        except Exception:
            used = False
        if not used:
            continue
        route: List[int] = []
        current = -1
        visited_tasks = set()
        guard = 0
        while True:
            guard += 1
            if guard > len(task_to_point) + 2:
                break
            nxt = selected_successors.get((i, k, v, current))
            if nxt is None or nxt == -1:
                break
            if nxt in visited_tasks:
                break
            visited_tasks.add(nxt)
            route.append(task_to_point[nxt])
            current = nxt
        if route:
            routes[(i, k, v)] = route
    return routes


def solve_gurobi_mip(inst: UAVInstance) -> Dict[str, Any]:
    """
    Full-path location-routing MILP benchmark.

    Compared with the compact benchmark, this model explicitly represents:
        1) opened platforms;
        2) heterogeneous UAV types and UAV copies;
        3) point-level coverage and frequency-expanded inspection tasks;
        4) route arcs from platform depot to tasks, between tasks, and back to depot;
        5) route continuity and subtour elimination through MTZ order variables;
        6) UAV flight-time and robust endurance constraints;
        7) no-fly-zone detour distances in depot-task and task-task arcs.

    This is closer to the location-routing structure described in the manuscript, but it is
    much harder than the compact platform-assignment MILP. For medium and large instances,
    use it as a time-limited benchmark rather than expecting proven optimality.
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available. Please install/configure Gurobi first.")

    model = gp.Model(f"UAV_full_path_{inst.spec.instance_id}")
    model.Params.TimeLimit = get_gurobi_time_limit(inst.spec.scale)
    model.Params.MIPGap = GUROBI_MIP_GAP
    model.Params.OutputFlag = GUROBI_OUTPUT_FLAG
    model.Params.Threads = GUROBI_THREADS
    model.Params.MIPFocus = GUROBI_MIP_FOCUS
    model.Params.Heuristics = GUROBI_HEURISTICS
    model.Params.Cuts = GUROBI_CUTS
    model.Params.NodefileStart = GUROBI_NODEFILE_START_GB
    model.Params.NodefileDir = GUROBI_NODEFILE_DIR
    model.Params.SoftMemLimit = GUROBI_SOFT_MEM_LIMIT_GB

    I = list(inst.platforms.keys())
    J = list(inst.points.keys())
    K = list(inst.uav_data.keys())
    V = list(range(MAX_UAVS_PER_PLATFORM_TYPE))

    tasks, task_to_point, task_occurrence = build_frequency_expanded_tasks(inst)
    T = list(tasks)
    n_tasks = len(T)
    point_dist = point_pair_distance_cache(inst)
    energy_factor = robust_energy_factor()

    print(f"[FullPathMIP] points={len(J)}, expanded_tasks={n_tasks}, platforms={len(I)}, "
          f"types={len(K)}, copies/type={len(V)}, beta={FULL_PATH_BETA_CONFIDENCE}")

    # Basic decision variables.
    x = model.addVars(I, vtype=GRB.BINARY, name="open_platform")
    cov = model.addVars(J, vtype=GRB.BINARY, name="covered_point")
    w = model.addVars(I, K, V, vtype=GRB.BINARY, name="use_uav")

    # Feasible task-vehicle assignments.
    y_keys: List[Tuple[int, int, int, int]] = []
    task_vehicle_keys: Dict[int, List[Tuple[int, int, int, int]]] = {t: [] for t in T}
    vehicle_tasks: Dict[Tuple[int, int, int], List[int]] = {(i, k, v): [] for i in I for k in K for v in V}

    for t in T:
        j = task_to_point[t]
        for i in I:
            for k in K:
                if inst.feasible_assign.get((j, i, k), 0) != 1:
                    continue
                for v in V:
                    key = (t, i, k, v)
                    y_keys.append(key)
                    task_vehicle_keys[t].append(key)
                    vehicle_tasks[(i, k, v)].append(t)

    y = model.addVars(y_keys, vtype=GRB.BINARY, name="assign_task")

    # Manuscript-level monitoring-point assignment variable.
    # point_assign[j,i,k] corresponds to y_{ijk} in the paper:
    # 1 if monitoring point j is assigned to platform i and UAV type k.
    # We intentionally do not call this variable q because q indexes UAV copies
    # in the manuscript's lower-level model.
    point_tasks_map: Dict[int, List[int]] = {
        j: [t for t in T if task_to_point[t] == j]
        for j in J
    }
    point_assign_keys: List[Tuple[int, int, int]] = [
        (j, i, k)
        for j in J
        for i in I
        for k in K
        if inst.feasible_assign.get((j, i, k), 0) == 1
    ]
    point_assign = model.addVars(
        point_assign_keys,
        vtype=GRB.BINARY,
        name="assign_point_platform_type",
    )

    # Route arc variables. Depot is represented by node -1.
    arc_keys: List[Tuple[int, int, int, int, int]] = []
    arc_dist: Dict[Tuple[int, int, int, int, int], float] = {}
    outgoing: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int, int]]] = {}
    incoming: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int, int]]] = {}
    task_task_arc_keys: List[Tuple[int, int, int, int, int]] = []

    for i in I:
        for k in K:
            for v in V:
                veh = (i, k, v)
                F = list(vehicle_tasks[veh])
                if not F:
                    continue

                # Depot-task and task-depot arcs.
                for t in F:
                    j = task_to_point[t]
                    d = inst.safe_distance[(i, j)]
                    key1 = (i, k, v, -1, t)
                    key2 = (i, k, v, t, -1)
                    arc_keys.extend([key1, key2])
                    arc_dist[key1] = d
                    arc_dist[key2] = d

                # Task-task arcs. Complete for small task sets; nearest-neighbor pruning for larger sets.
                complete_arcs = len(F) <= FULL_PATH_MAX_COMPLETE_TASKS
                for t in F:
                    jt = task_to_point[t]
                    candidates = []
                    for s in F:
                        if s == t:
                            continue
                        js = task_to_point[s]
                        if (FULL_PATH_BLOCK_DIRECT_ARCS_BETWEEN_SAME_POINT_OCCURRENCES
                                and jt == js):
                            continue
                        candidates.append((point_dist[(jt, js)], s))
                    candidates.sort(key=lambda item: item[0])
                    if not complete_arcs:
                        candidates = candidates[:FULL_PATH_NEAREST_ROUTE_NEIGHBORS]
                    for d, s in candidates:
                        key = (i, k, v, t, s)
                        arc_keys.append(key)
                        arc_dist[key] = d
                        task_task_arc_keys.append(key)

    a = model.addVars(arc_keys, vtype=GRB.BINARY, name="route_arc")
    order_keys = [(t, i, k, v) for (t, i, k, v) in y_keys]
    ordv = model.addVars(order_keys, vtype=GRB.CONTINUOUS, lb=0.0, ub=max(1, n_tasks), name="mtz_order")

    for key in arc_keys:
        i, k, v, u, t = key
        outgoing.setdefault((i, k, v, u), []).append(key)
        incoming.setdefault((i, k, v, t), []).append(key)

    # Platform opening and site feasibility.
    model.addConstr(gp.quicksum(x[i] for i in I) == inst.spec.open_platforms_fixed,
                    name="fixed_number_of_platforms")
    for i in I:
        model.addConstr(x[i] <= inst.site_feasible[i], name=f"site_feasibility[{i}]")

    # ------------------------------------------------------------------
    # Coverage and upper-/lower-level assignment consistency
    # ------------------------------------------------------------------
    # The manuscript first assigns each covered monitoring point j to exactly
    # one platform-UAV-type combination (i,k). Every frequency-expanded task
    # t with task_to_point[t] == j must then be executed by a UAV copy v from
    # that same selected combination. UAV-copy selection remains task-specific.
    for j in J:
        feasible_point_keys = [
            (j, i, k)
            for i in I
            for k in K
            if (j, i, k) in point_assign
        ]

        if ENFORCE_POINT_LEVEL_PLATFORM_TYPE_ASSIGNMENT:
            # Equivalent to the paper's point-level assignment equation:
            # sum_i sum_k y_{ijk} = z_j.
            model.addConstr(
                gp.quicksum(point_assign[key] for key in feasible_point_keys) == cov[j],
                name=f"point_assignment_exactly_one[{j}]",
            )

            for (jj, i, k) in feasible_point_keys:
                # Strengthening links to the upper-level platform/fleet decisions.
                model.addConstr(
                    point_assign[jj, i, k] <= x[i],
                    name=f"point_assignment_requires_open_platform[{jj},{i},{k}]",
                )
                model.addConstr(
                    point_assign[jj, i, k] <= gp.quicksum(w[i, k, v] for v in V),
                    name=f"point_assignment_requires_uav_type[{jj},{i},{k}]",
                )

            # Lower-level linking: for every occurrence task t of point j,
            # sum_v r_t^{ikv} = y_{ijk}. Thus all occurrences use the same
            # platform and UAV type, while different occurrences may use
            # different UAV copies.
            for t in point_tasks_map[j]:
                for (jj, i, k) in feasible_point_keys:
                    task_copy_keys = [
                        (t, i, k, v)
                        for v in V
                        if (t, i, k, v) in y
                    ]
                    model.addConstr(
                        gp.quicksum(y[key] for key in task_copy_keys)
                        == point_assign[jj, i, k],
                        name=f"task_inherits_point_assignment[{j},{t},{i},{k}]",
                    )
        else:
            # Legacy benchmark logic retained only as an explicit fallback.
            for t in point_tasks_map[j]:
                keys_t = task_vehicle_keys[t]
                model.addConstr(
                    gp.quicksum(y[key] for key in keys_t) == cov[j],
                    name=f"task_coverage_link[{j},{t}]",
                )

    model.addConstr(
        gp.quicksum(cov[j] for j in J) >= min_required_covered_points(inst),
        name="minimum_point_coverage",
    )

    # UAV-use and platform-use linking.
    for i in I:
        for k in K:
            for v in V:
                model.addConstr(w[i, k, v] <= x[i], name=f"uav_requires_open_platform[{i},{k},{v}]")

    for (t, i, k, v) in y_keys:
        model.addConstr(y[t, i, k, v] <= w[i, k, v], name=f"task_requires_used_uav[{t},{i},{k},{v}]")

    # Route continuity: each used UAV starts at its platform depot and returns to it once.
    for i in I:
        for k in K:
            for v in V:
                F = vehicle_tasks[(i, k, v)]
                start_arcs = outgoing.get((i, k, v, -1), [])
                end_arcs = incoming.get((i, k, v, -1), [])
                model.addConstr(gp.quicksum(a[key] for key in start_arcs) == w[i, k, v],
                                name=f"route_start[{i},{k},{v}]")
                model.addConstr(gp.quicksum(a[key] for key in end_arcs) == w[i, k, v],
                                name=f"route_end[{i},{k},{v}]")

                for t in F:
                    y_key = (t, i, k, v)
                    if y_key not in y:
                        continue
                    in_arcs = incoming.get((i, k, v, t), [])
                    out_arcs = outgoing.get((i, k, v, t), [])
                    model.addConstr(gp.quicksum(a[key] for key in in_arcs) == y[y_key],
                                    name=f"flow_in[{t},{i},{k},{v}]")
                    model.addConstr(gp.quicksum(a[key] for key in out_arcs) == y[y_key],
                                    name=f"flow_out[{t},{i},{k},{v}]")

    # MTZ subtour elimination and order activation.
    big_m = max(2, n_tasks + 1)
    for (t, i, k, v) in order_keys:
        model.addConstr(ordv[t, i, k, v] <= n_tasks * y[t, i, k, v],
                        name=f"order_upper[{t},{i},{k},{v}]")
        model.addConstr(ordv[t, i, k, v] >= y[t, i, k, v],
                        name=f"order_lower[{t},{i},{k},{v}]")

    for key in task_task_arc_keys:
        i, k, v, t, s = key
        if (t, i, k, v) in ordv and (s, i, k, v) in ordv:
            model.addConstr(ordv[t, i, k, v] + 1 <= ordv[s, i, k, v] + big_m * (1 - a[key]),
                            name=f"mtz[{i},{k},{v},{t},{s}]")

    # Flight-time and robust endurance constraints for each UAV route.
    for i in I:
        for k in K:
            speed = inst.uav_data[k]["speed_kmph"]
            endurance = inst.uav_data[k]["max_round_trip_km"]
            mult = inst.uav_data[k]["energy_multiplier"]
            for v in V:
                veh_arc_keys = [key for key in arc_keys if key[0] == i and key[1] == k and key[2] == v]
                veh_task_keys = [(t, i, k, v) for t in vehicle_tasks[(i, k, v)] if (t, i, k, v) in y]

                travel_time = gp.quicksum((arc_dist[key] / speed) * a[key] for key in veh_arc_keys)
                service_time = gp.quicksum(inst.service_time[task_to_point[t]] * y[t, i, k, v]
                                           for (t, _, _, _) in veh_task_keys)
                model.addConstr(travel_time + service_time <= PLANNING_HORIZON_HOURS * w[i, k, v],
                                name=f"route_time_capacity[{i},{k},{v}]")

                robust_energy = gp.quicksum(energy_factor * mult * arc_dist[key] * a[key]
                                            for key in veh_arc_keys)
                model.addConstr(robust_energy <= endurance * w[i, k, v],
                                name=f"robust_endurance[{i},{k},{v}]")

    # Objective: platform fixed cost + UAV use cost + route energy cost + uncovered-point penalty.
    platform_cost = gp.quicksum(inst.platform_fixed_cost[i] * x[i] for i in I)
    fleet_cost = gp.quicksum(fleet_unit_cost(inst, k) * w[i, k, v] for i in I for k in K for v in V)
    route_energy_cost = gp.quicksum(ENERGY_COST_PER_KM * inst.uav_data[k]["energy_multiplier"] * arc_dist[key] * a[key]
                                    for key in arc_keys for i, k, v, u, t in [key])
    uncovered_cost = gp.quicksum(UNCOVERED_PENALTY * (1 - cov[j]) for j in J)

    model.setObjective(platform_cost + fleet_cost + route_energy_cost + uncovered_cost, GRB.MINIMIZE)

    print(
        f"[FullPathMIP] point_assign_keys={len(point_assign_keys)}, "
        f"y_keys={len(y_keys)}, arc_keys={len(arc_keys)}, "
        f"task_task_arcs={len(task_task_arc_keys)}"
    )

    t0 = time.time()
    optimize_exception: Optional[Exception] = None
    try:
        model.optimize()
    except gp.GurobiError as exc:
        # Gurobi may still have a valid incumbent and bound when a hard memory
        # error is raised. Preserve those attributes instead of replacing the
        # whole row by ERROR.
        optimize_exception = exc
        print(f"[Gurobi warning] optimize terminated with {type(exc).__name__}: {exc}")
    runtime = time.time() - t0

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        getattr(GRB, "MEM_LIMIT", -99999): "MEM_LIMIT",
    }
    try:
        model_status = model.Status
    except Exception:
        model_status = None
    status = status_map.get(model_status, str(model_status))
    if optimize_exception is not None:
        error_code = getattr(optimize_exception, "errno", None)
        error_text = str(optimize_exception).lower()
        if error_code == 10001 or "out of memory" in error_text:
            status = "OUT_OF_MEMORY"
        elif status in ("None", "INTERRUPTED"):
            status = "INTERRUPTED_WITH_EXCEPTION"

    feasible = model.SolCount > 0
    obj = model.ObjVal if feasible else None
    try:
        lb = model.ObjBound
    except Exception:
        lb = None
    if lb is not None and (not math.isfinite(float(lb)) or (not feasible and float(lb) <= 1e-9)):
        # A zero/non-finite bound before completion of the root relaxation is
        # not informative and must not be reported as a valid lower bound.
        lb = None
    try:
        gap = model.MIPGap * 100.0 if feasible else None
    except Exception:
        gap = None

    open_platforms = []
    assignment: Dict[int, Optional[Tuple[int, int]]] = {j: None for j in J}
    n_uavs: Dict[Tuple[int, int], int] = {(i, k): 0 for i in I for k in K}
    coverage = None
    routes: Dict[Tuple[int, int, int], List[int]] = {}
    obj_recomputed = None
    obj_diff = None
    evaluator_feasible = None
    evaluator_violation = None
    point_assignment_consistency_ok = None
    point_assignment_inconsistencies: List[Dict[str, Any]] = []
    objective_for_report = obj

    if feasible:
        open_platforms = [i for i in I if x[i].X > 0.5]
        routes = extract_routes_from_gurobi_arcs(inst, w, a, task_to_point)
        eval_sol = evaluate_full_path_routes(inst, open_platforms, routes)
        assignment = eval_sol.assignment
        n_uavs = eval_sol.n_uavs
        coverage = eval_sol.coverage
        obj_recomputed = eval_sol.obj
        obj_diff = abs(float(obj) - float(obj_recomputed)) if obj is not None else None
        evaluator_feasible = eval_sol.feasible
        evaluator_violation = eval_sol.violation

        if DEBUG_PRINT_SOLUTION_AUDIT:
            print("[DEBUG_GUROBI_RECHECK]")
            print("model_obj        =", obj)
            print("recomputed_obj   =", obj_recomputed)
            print("obj_diff         =", obj_diff)
            print("evaluator_feasible =", evaluator_feasible)
            print("evaluator_violation =", evaluator_violation)
            print("components       =", eval_sol.components)
            print("route_count      =", len(routes))

        # Audit the manuscript-level linking logic after optimization.
        if ENFORCE_POINT_LEVEL_PLATFORM_TYPE_ASSIGNMENT:
            for j in J:
                selected_point_combos = [
                    (i, k)
                    for (jj, i, k), var in point_assign.items()
                    if jj == j and var.X > 0.5
                ]
                selected_task_combos = set()
                for t in point_tasks_map[j]:
                    for (tt, i, k, v), var in y.items():
                        if tt == t and var.X > 0.5:
                            selected_task_combos.add((i, k))
                expected_count = 1 if cov[j].X > 0.5 else 0
                if (len(selected_point_combos) != expected_count
                        or len(selected_task_combos) != expected_count
                        or (expected_count == 1
                            and selected_task_combos != set(selected_point_combos))):
                    point_assignment_inconsistencies.append({
                        "point_id": j,
                        "covered": int(cov[j].X > 0.5),
                        "point_level_combos": selected_point_combos,
                        "task_level_combos": sorted(selected_task_combos),
                    })
            point_assignment_consistency_ok = len(point_assignment_inconsistencies) == 0
        else:
            point_assignment_consistency_ok = True

        if DEBUG_PRINT_SOLUTION_AUDIT:
            print("point_assignment_consistency_ok =", point_assignment_consistency_ok)
            if point_assignment_inconsistencies:
                print("point_assignment_inconsistencies =", point_assignment_inconsistencies)

        # Use the model objective as the authoritative Gurobi UB, but keep the recomputed
        # full-path objective as a consistency audit. They should be numerically identical
        # except for minor tolerance if extraction is correct.
        objective_for_report = obj
        export_full_path_solution(
            inst, cov, w, y, a, tasks, task_to_point, point_assign=point_assign
        )

    return {
        "method": "Gurobi",
        "status": status,
        "obj": objective_for_report,
        "lb": lb,
        "gap_percent": gap,
        "time_sec": runtime,
        "feasible": feasible,
        "coverage": coverage,
        "open_platforms": open_platforms,
        "assignment": assignment,
        "n_uavs": n_uavs,
        "routes": routes,
        "obj_recomputed": obj_recomputed,
        "obj_diff": obj_diff,
        "evaluator_feasible": evaluator_feasible,
        "evaluator_violation": evaluator_violation,
        "point_assignment_consistency_ok": point_assignment_consistency_ok,
        "point_assignment_inconsistencies": point_assignment_inconsistencies,
        "model_vars": model.NumVars,
        "model_constrs": model.NumConstrs,
        "termination_note": str(optimize_exception) if optimize_exception is not None else None,
    }


# ============================================================
# 7. ALNS-Q-learning-SA heuristic benchmark
# ============================================================

def feasible_platforms(inst: UAVInstance) -> List[int]:
    return [i for i, ok in inst.site_feasible.items() if ok == 1]


def all_assignment_options(inst: UAVInstance, j: int, open_platforms: Iterable[int]) -> List[Tuple[int, int]]:
    opts = []
    for i in open_platforms:
        for k in inst.uav_data:
            if inst.feasible_assign.get((j, i, k), 0) == 1:
                opts.append((i, k))
    return opts


def marginal_assignment_cost(inst: UAVInstance,
                             j: int,
                             i: int,
                             k: int,
                             workloads: Dict[Tuple[int, int], float]) -> float:
    before_w = workloads.get((i, k), 0.0)
    after_w = before_w + assignment_workload(inst, j, i, k)
    before_n = required_uavs_for_workload(before_w)
    after_n = required_uavs_for_workload(after_w)
    fleet_inc = max(0, after_n - before_n) * fleet_unit_cost(inst, k)
    energy_inc = assignment_energy_cost(inst, j, i, k)
    return fleet_inc + energy_inc


def build_greedy_assignment(inst: UAVInstance,
                            open_platforms: Iterable[int],
                            rng: random.Random,
                            random_tie: bool = True) -> Solution:
    open_tuple = tuple(sorted(open_platforms))
    workloads: Dict[Tuple[int, int], float] = {(i, k): 0.0 for i in inst.platforms for k in inst.uav_data}
    assignment: Dict[int, Optional[Tuple[int, int]]] = {j: None for j in inst.points}

    # Sort harder points first: fewer feasible options first, then higher frequency.
    ordered_points = sorted(
        inst.points,
        key=lambda j: (len(all_assignment_options(inst, j, open_tuple)), -inst.inspection_freq[j], rng.random()),
    )

    for j in ordered_points:
        options = all_assignment_options(inst, j, open_tuple)
        best_opt = None
        best_score = float("inf")
        if random_tie:
            rng.shuffle(options)
        for i, k in options:
            w_new = workloads[(i, k)] + assignment_workload(inst, j, i, k)
            n_new = required_uavs_for_workload(w_new)
            if n_new > MAX_UAVS_PER_PLATFORM_TYPE:
                continue
            score = marginal_assignment_cost(inst, j, i, k, workloads)
            if score < best_score:
                best_score = score
                best_opt = (i, k)
        if best_opt is not None:
            assignment[j] = best_opt
            workloads[best_opt] += assignment_workload(inst, j, best_opt[0], best_opt[1])

    return evaluate_solution(inst, open_tuple, assignment)


def _platform_quality_key(inst: UAVInstance, platform_id: int) -> Tuple[float, float, float]:
    feasible_points = 0
    distance_sum = 0.0
    for j in inst.points:
        feasible_types = [k for k in inst.uav_data if inst.feasible_assign.get((j, platform_id, k), 0) == 1]
        if feasible_types:
            feasible_points += 1
            distance_sum += inst.safe_distance[(platform_id, j)]
    avg_distance = distance_sum / max(feasible_points, 1)
    return (-float(feasible_points), avg_distance, float(inst.platform_fixed_cost[platform_id]))


def candidate_platform_combinations(inst: UAVInstance,
                                    rng: random.Random,
                                    cap: int) -> List[Tuple[int, ...]]:
    """Generate a small, diverse set of promising platform combinations."""
    fp = feasible_platforms(inst)
    if len(fp) < inst.spec.open_platforms_fixed:
        fp = list(inst.platforms.keys())
    fp = sorted(fp, key=lambda i: _platform_quality_key(inst, i))
    q = inst.spec.open_platforms_fixed
    base = tuple(sorted(fp[:q]))
    combos: List[Tuple[int, ...]] = [base]
    seen = {base}

    # Perturb the deterministic base combination. This is much cheaper than
    # enumerating/sampling hundreds of combinations and is sufficient when
    # L300 opens 14 of 18 candidate platforms.
    attempts = 0
    while len(combos) < max(1, cap) and attempts < 200:
        attempts += 1
        selected = set(base)
        closed = [i for i in fp if i not in selected]
        if not closed:
            break
        n_swaps = 1 if len(combos) < max(2, cap // 2) else min(2, len(closed), len(selected))
        remove_items = rng.sample(list(selected), n_swaps)
        add_items = rng.sample(closed, n_swaps)
        selected.difference_update(remove_items)
        selected.update(add_items)
        combo = tuple(sorted(selected))
        if len(combo) == q and combo not in seen:
            seen.add(combo)
            combos.append(combo)
    return combos


def _point_option_order(inst: UAVInstance,
                        point_id: int,
                        open_platforms: Iterable[int]) -> List[Tuple[int, int]]:
    options = all_assignment_options(inst, point_id, open_platforms)
    options.sort(
        key=lambda opt: (
            assignment_energy_cost(inst, point_id, opt[0], opt[1])
            + fleet_unit_cost(inst, opt[1]),
            inst.safe_distance[(opt[0], point_id)],
            opt[1],
            opt[0],
        )
    )
    return options


def try_insert_full_point(inst: UAVInstance,
                          routes: Dict[Tuple[int, int, int], List[int]],
                          point_id: int,
                          option: Tuple[int, int],
                          deadline: Optional[float]) -> Tuple[bool, float, Dict[Tuple[int, int, int], List[int]]]:
    """Insert all occurrences of one point under one platform/type combination."""
    if _deadline_reached(deadline):
        return False, float("inf"), routes
    i, k = option
    trial_routes = {veh: list(rt) for veh, rt in routes.items()}

    # Remove a partial representation of this point before rebuilding it.
    for veh in list(trial_routes):
        cleaned = [j for j in trial_routes[veh] if j != point_id]
        if cleaned:
            trial_routes[veh] = cleaned
        else:
            trial_routes.pop(veh, None)

    total_delta = 0.0
    required = max(1, int(inst.inspection_freq[point_id]))
    for _occ in range(required):
        if _deadline_reached(deadline):
            return False, float("inf"), routes
        best_key: Optional[Tuple[int, int, int]] = None
        best_route: Optional[List[int]] = None
        best_score = float("inf")
        best_tie = (99, 99, 99)

        for v in range(MAX_UAVS_PER_PLATFORM_TYPE):
            if _deadline_reached(deadline):
                return False, float("inf"), routes
            veh = (i, k, v)
            route = trial_routes.get(veh, [])
            activation = fleet_unit_cost(inst, k) if (DECODER_INCLUDE_UAV_ACTIVATION_COST and not route) else 0.0
            for pos in range(len(route) + 1):
                if _deadline_reached(deadline):
                    return False, float("inf"), routes
                ok, delta, new_route = route_insertion_delta(inst, i, k, route, point_id, pos)
                if not ok:
                    continue
                score = activation + delta
                tie = (1 if not route else 0, v, pos)
                if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and tie < best_tie):
                    best_score = score
                    best_key = veh
                    best_route = new_route
                    best_tie = tie

        if best_key is None or best_route is None:
            return False, float("inf"), routes
        trial_routes[best_key] = best_route
        total_delta += best_score

    return True, total_delta, normalize_routes(trial_routes)


def build_route_first_coverage_solution(inst: UAVInstance,
                                        open_platforms: Tuple[int, ...],
                                        rng: random.Random,
                                        deadline: Optional[float]) -> Solution:
    """Construct explicit routes while prioritizing the 90% coverage constraint."""
    routes: Dict[Tuple[int, int, int], List[int]] = {}
    profile = get_search_profile(inst)
    option_cap = int(profile["coverage_option_cap"])

    points = list(inst.points.keys())
    rng.shuffle(points)
    points.sort(
        key=lambda j: (
            len(all_assignment_options(inst, j, open_platforms)),
            -inst.inspection_freq[j],
            min((inst.safe_distance[(i, j)] for i in open_platforms), default=float("inf")),
        )
    )

    inserted = 0
    target = min_required_covered_points(inst)
    for j in points:
        if _deadline_reached(deadline):
            break
        options = _point_option_order(inst, j, open_platforms)[:max(1, option_cap)]
        best_routes: Optional[Dict[Tuple[int, int, int], List[int]]] = None
        best_delta = float("inf")
        for opt in options:
            ok, delta, candidate_routes = try_insert_full_point(inst, routes, j, opt, deadline)
            if ok and delta < best_delta - 1e-12:
                best_delta = delta
                best_routes = candidate_routes
        if best_routes is not None:
            routes = best_routes
            inserted += 1

        # Once the mandatory coverage is attained, continue only while at least
        # 15% of the initialization budget remains; this preserves ALNS time.
        if inserted >= target and remaining_time(deadline) < 15.0:
            break

    return evaluate_route_dictionary(inst, open_platforms, routes)


def coverage_rescue_solution(inst: UAVInstance,
                             sol: Solution,
                             rng: random.Random,
                             deadline: Optional[float],
                             max_points: Optional[int] = None) -> Solution:
    """Greedily add complete points until the minimum coverage constraint is met."""
    if not ENABLE_COVERAGE_RESCUE or _deadline_reached(deadline):
        return sol
    if sol.coverage + 1e-12 >= MIN_COVERAGE_RATE and other_violation_value(sol) <= 1e-9:
        return sol

    routes = normalize_routes(sol.routes or {})
    current = evaluate_route_dictionary(inst, sol.open_platforms, routes)
    profile = get_search_profile(inst)
    option_cap = int(profile["coverage_option_cap"])
    point_cap = int(max_points if max_points is not None else profile["coverage_rescue_max_points"])

    uncovered = [j for j in inst.points if current.assignment.get(j) is None]
    rng.shuffle(uncovered)
    uncovered.sort(
        key=lambda j: (
            len(all_assignment_options(inst, j, current.open_platforms)),
            -inst.inspection_freq[j],
        )
    )

    added = 0
    for j in uncovered:
        if _deadline_reached(deadline) or added >= point_cap:
            break
        options = _point_option_order(inst, j, current.open_platforms)[:max(1, option_cap)]
        best_routes = None
        best_delta = float("inf")
        for opt in options:
            ok, delta, candidate_routes = try_insert_full_point(inst, routes, j, opt, deadline)
            if ok and delta < best_delta - 1e-12:
                best_delta = delta
                best_routes = candidate_routes
        if best_routes is None:
            continue
        routes = best_routes
        current = evaluate_route_dictionary(inst, current.open_platforms, routes)
        added += 1
        if current.feasible or (
            current.coverage + 1e-12 >= MIN_COVERAGE_RATE
            and other_violation_value(current) <= 1e-9
        ):
            break
    return current


def generate_initial_solution(inst: UAVInstance,
                              rng: random.Random,
                              deadline: Optional[float] = None) -> Solution:
    """Strict-time multi-start initialization with route-first coverage construction."""
    profile = get_search_profile(inst)
    combos = candidate_platform_combinations(inst, rng, int(profile["platform_combo_cap"]))
    phase_deadline = deadline
    if deadline is not None:
        phase_deadline = min(deadline, time.time() + float(profile["route_first_budget_sec"]))

    best: Optional[Solution] = None
    for combo in combos:
        if _deadline_reached(phase_deadline):
            break

        if ENABLE_COVERAGE_FIRST_INITIALIZATION:
            route_first = build_route_first_coverage_solution(inst, combo, rng, phase_deadline)
            if is_better_solution(route_first, best):
                best = route_first
            if route_first.feasible and inst.spec.scale == "large":
                # For L300, a feasible route-first solution is more valuable
                # than spending the whole initialization budget on extra starts.
                break

        if _deadline_reached(phase_deadline):
            break
        greedy = build_greedy_assignment(inst, combo, rng, random_tie=True)
        if is_better_solution(greedy, best):
            best = greedy

    if best is None:
        fp = feasible_platforms(inst)
        if len(fp) < inst.spec.open_platforms_fixed:
            fp = list(inst.platforms.keys())
        chosen = tuple(sorted(fp[:inst.spec.open_platforms_fixed]))
        empty_routes: Dict[Tuple[int, int, int], List[int]] = {}
        best = evaluate_route_dictionary(inst, chosen, empty_routes)

    if (not best.feasible
            and remaining_time(deadline) >= COVERAGE_RESCUE_MIN_REMAINING_SEC):
        rescued = coverage_rescue_solution(
            inst,
            best,
            rng,
            deadline,
            max_points=int(profile["coverage_rescue_max_points"]),
        )
        if is_better_solution(rescued, best):
            best = rescued
    return best


def copy_solution(sol: Solution) -> Tuple[Tuple[int, ...], Dict[int, Optional[Tuple[int, int]]]]:
    return tuple(sol.open_platforms), dict(sol.assignment)


def remove_random_tasks(inst: UAVInstance, sol: Solution, rng: random.Random) -> Tuple[Tuple[int, ...], Dict[int, Optional[Tuple[int, int]]]]:
    open_platforms, assignment = copy_solution(sol)
    assigned = [j for j, val in assignment.items() if val is not None]
    if not assigned:
        return open_platforms, assignment
    q = max(1, int(rng.uniform(DESTROY_RATIO_MIN, DESTROY_RATIO_MAX) * len(inst.points)))
    for j in rng.sample(assigned, min(q, len(assigned))):
        assignment[j] = None
    return open_platforms, assignment


def remove_worst_tasks(inst: UAVInstance, sol: Solution, rng: random.Random) -> Tuple[Tuple[int, ...], Dict[int, Optional[Tuple[int, int]]]]:
    open_platforms, assignment = copy_solution(sol)
    contributions = []
    for j, val in assignment.items():
        if val is None:
            continue
        i, k = val
        contributions.append((assignment_energy_cost(inst, j, i, k), j))
    if not contributions:
        return open_platforms, assignment
    contributions.sort(reverse=True)
    q = max(1, int(rng.uniform(DESTROY_RATIO_MIN, DESTROY_RATIO_MAX) * len(inst.points)))
    for _, j in contributions[:q]:
        assignment[j] = None
    return open_platforms, assignment


def remove_related_tasks(inst: UAVInstance, sol: Solution, rng: random.Random) -> Tuple[Tuple[int, ...], Dict[int, Optional[Tuple[int, int]]]]:
    open_platforms, assignment = copy_solution(sol)
    assigned = [j for j, val in assignment.items() if val is not None]
    if not assigned:
        return open_platforms, assignment
    center = rng.choice(assigned)
    q = max(1, int(rng.uniform(DESTROY_RATIO_MIN, DESTROY_RATIO_MAX) * len(inst.points)))
    ranked = sorted(assigned, key=lambda j: euclidean(inst.points[j], inst.points[center]))
    for j in ranked[:q]:
        assignment[j] = None
    return open_platforms, assignment



def remove_one_route_points(inst: UAVInstance,
                            sol: Solution,
                            rng: random.Random) -> Tuple[Tuple[int, ...], Dict[int, Optional[Tuple[int, int]]]]:
    """Remove all monitoring points represented on one UAV route."""
    open_platforms, assignment = copy_solution(sol)
    if not sol.routes:
        return open_platforms, assignment
    nonempty = [(veh, route) for veh, route in sol.routes.items() if route]
    if not nonempty:
        return open_platforms, assignment
    # Prefer short/expensive routes because eliminating them can save a UAV.
    nonempty.sort(key=lambda item: (len(item[1]), -fleet_unit_cost(inst, item[0][1])))
    candidate_pool = nonempty[:max(1, min(4, len(nonempty)))]
    _, route = rng.choice(candidate_pool)
    for j in set(route):
        assignment[j] = None
    return open_platforms, assignment


def remove_platform_type_group(inst: UAVInstance,
                               sol: Solution,
                               rng: random.Random) -> Tuple[Tuple[int, ...], Dict[int, Optional[Tuple[int, int]]]]:
    """Remove all points assigned to one active platform-UAV-type group."""
    open_platforms, assignment = copy_solution(sol)
    groups: Dict[Tuple[int, int], List[int]] = {}
    for j, val in assignment.items():
        if val is not None:
            groups.setdefault(val, []).append(j)
    if not groups:
        return open_platforms, assignment
    # Bias toward smaller groups to support vehicle/type elimination while retaining diversity.
    keys = sorted(groups, key=lambda g: (len(groups[g]), -fleet_unit_cost(inst, g[1])))
    chosen = rng.choice(keys[:max(1, min(4, len(keys)))])
    for j in groups[chosen]:
        assignment[j] = None
    return open_platforms, assignment


def swap_one_platform(inst: UAVInstance, sol: Solution, rng: random.Random) -> Tuple[Tuple[int, ...], Dict[int, Optional[Tuple[int, int]]]]:
    old_open, assignment = copy_solution(sol)
    fp = feasible_platforms(inst)
    closed = [i for i in fp if i not in old_open]
    if not old_open or not closed:
        return old_open, assignment
    remove_i = rng.choice(list(old_open))
    add_i = rng.choice(closed)
    new_open = tuple(sorted((set(old_open) - {remove_i}) | {add_i}))
    for j, val in list(assignment.items()):
        if val is not None and val[0] == remove_i:
            assignment[j] = None
        elif val is not None and val[0] not in new_open:
            assignment[j] = None
    return new_open, assignment


def repair_greedy(inst: UAVInstance,
                  open_platforms: Tuple[int, ...],
                  assignment: Dict[int, Optional[Tuple[int, int]]],
                  rng: random.Random) -> Solution:
    # Reconstruct workloads from existing assignments.
    temp_sol = evaluate_solution(inst, open_platforms, assignment)
    workloads = dict(temp_sol.workloads)
    repaired = dict(temp_sol.assignment)

    unassigned = [j for j, val in repaired.items() if val is None]
    rng.shuffle(unassigned)
    unassigned.sort(key=lambda j: (len(all_assignment_options(inst, j, open_platforms)), -inst.inspection_freq[j]))

    for j in unassigned:
        best_opt = None
        best_score = float("inf")
        options = all_assignment_options(inst, j, open_platforms)
        rng.shuffle(options)
        for i, k in options:
            w_new = workloads[(i, k)] + assignment_workload(inst, j, i, k)
            if required_uavs_for_workload(w_new) > MAX_UAVS_PER_PLATFORM_TYPE:
                continue
            score = marginal_assignment_cost(inst, j, i, k, workloads)
            if score < best_score:
                best_score = score
                best_opt = (i, k)
        if best_opt is not None:
            repaired[j] = best_opt
            workloads[best_opt] += assignment_workload(inst, j, best_opt[0], best_opt[1])

    return evaluate_solution(inst, open_platforms, repaired)


def repair_random(inst: UAVInstance,
                  open_platforms: Tuple[int, ...],
                  assignment: Dict[int, Optional[Tuple[int, int]]],
                  rng: random.Random) -> Solution:
    temp_sol = evaluate_solution(inst, open_platforms, assignment)
    workloads = dict(temp_sol.workloads)
    repaired = dict(temp_sol.assignment)

    unassigned = [j for j, val in repaired.items() if val is None]
    rng.shuffle(unassigned)

    for j in unassigned:
        options = all_assignment_options(inst, j, open_platforms)
        rng.shuffle(options)
        for i, k in options:
            w_new = workloads[(i, k)] + assignment_workload(inst, j, i, k)
            if required_uavs_for_workload(w_new) <= MAX_UAVS_PER_PLATFORM_TYPE:
                repaired[j] = (i, k)
                workloads[(i, k)] = w_new
                break

    return evaluate_solution(inst, open_platforms, repaired)


def repair_regret2(inst: UAVInstance,
                   open_platforms: Tuple[int, ...],
                   assignment: Dict[int, Optional[Tuple[int, int]]],
                   rng: random.Random) -> Solution:
    temp_sol = evaluate_solution(inst, open_platforms, assignment)
    workloads = dict(temp_sol.workloads)
    repaired = dict(temp_sol.assignment)

    while True:
        candidates = []
        for j, val in repaired.items():
            if val is not None:
                continue
            option_costs = []
            for i, k in all_assignment_options(inst, j, open_platforms):
                w_new = workloads[(i, k)] + assignment_workload(inst, j, i, k)
                if not math.isfinite(w_new):
                    continue
                if required_uavs_for_workload(w_new) > MAX_UAVS_PER_PLATFORM_TYPE:
                    continue
                option_costs.append((marginal_assignment_cost(inst, j, i, k, workloads), (i, k)))
            if not option_costs:
                continue
            option_costs.sort(key=lambda x: x[0])
            best_cost, best_opt = option_costs[0]
            second_cost = option_costs[1][0] if len(option_costs) > 1 else best_cost + 1000.0
            regret = second_cost - best_cost
            candidates.append((regret, -best_cost, j, best_opt))
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, _, j_sel, opt_sel = candidates[0]
        repaired[j_sel] = opt_sel
        workloads[opt_sel] += assignment_workload(inst, j_sel, opt_sel[0], opt_sel[1])

    return evaluate_solution(inst, open_platforms, repaired)


def repair_balanced(inst: UAVInstance,
                    open_platforms: Tuple[int, ...],
                    assignment: Dict[int, Optional[Tuple[int, int]]],
                    rng: random.Random) -> Solution:
    temp_sol = evaluate_solution(inst, open_platforms, assignment)
    workloads = dict(temp_sol.workloads)
    repaired = dict(temp_sol.assignment)

    unassigned = [j for j, val in repaired.items() if val is None]
    rng.shuffle(unassigned)

    for j in unassigned:
        best_opt = None
        best_score = float("inf")
        for i, k in all_assignment_options(inst, j, open_platforms):
            w_inc = assignment_workload(inst, j, i, k)
            w_new = workloads[(i, k)] + w_inc
            n_new = required_uavs_for_workload(w_new)
            if n_new > MAX_UAVS_PER_PLATFORM_TYPE:
                continue
            load_ratio = w_new / (MAX_UAVS_PER_PLATFORM_TYPE * PLANNING_HORIZON_HOURS)
            score = marginal_assignment_cost(inst, j, i, k, workloads) + 10.0 * load_ratio
            if score < best_score:
                best_score = score
                best_opt = (i, k)
        if best_opt is not None:
            repaired[j] = best_opt
            workloads[best_opt] += assignment_workload(inst, j, best_opt[0], best_opt[1])

    return evaluate_solution(inst, open_platforms, repaired)



def repair_route_aware(inst: UAVInstance,
                       open_platforms: Tuple[int, ...],
                       assignment: Dict[int, Optional[Tuple[int, int]]],
                       rng: random.Random) -> Solution:
    """
    Route-aware greedy repair.

    Each candidate platform-type assignment is evaluated through the explicit
    route decoder, so the repair accounts for UAV activation, route consolidation,
    endurance, and no-fly-zone detours rather than relying only on workload proxies.
    """
    profile = get_search_profile(inst)
    if not bool(profile.get("enable_route_aware_repair", False)):
        return repair_regret2(inst, open_platforms, assignment, rng)

    repaired = dict(assignment)
    current = evaluate_solution(inst, open_platforms, repaired)

    while True:
        unassigned = [j for j, val in current.assignment.items() if val is None]
        if not unassigned:
            break

        # Prioritize points with few feasible alternatives and high frequency.
        unassigned.sort(
            key=lambda j: (
                len(all_assignment_options(inst, j, open_platforms)),
                -inst.inspection_freq[j],
                rng.random(),
            )
        )

        best_global: Optional[Solution] = None
        best_point: Optional[int] = None
        best_opt: Optional[Tuple[int, int]] = None

        # Evaluate several difficult points and select the best full-route insertion.
        candidate_points = unassigned[:max(1, min(5, len(unassigned)))]
        for j in candidate_points:
            options = all_assignment_options(inst, j, open_platforms)
            rng.shuffle(options)
            for opt in options:
                trial_assignment = dict(current.assignment)
                trial_assignment[j] = opt
                trial = evaluate_solution(inst, open_platforms, trial_assignment)
                if best_global is None or trial.fitness < best_global.fitness - 1e-9:
                    best_global = trial
                    best_point = j
                    best_opt = opt

        if best_global is None or best_point is None or best_opt is None:
            break
        current = best_global

    return current


def pair_reassignment_polish(inst: UAVInstance,
                             sol: Solution,
                             rng: random.Random,
                             max_passes: int = PAIR_REASSIGNMENT_MAX_PASSES,
                             deadline: Optional[float] = None) -> Solution:
    """Jointly reassign two points, allowing coordinated platform/type changes."""
    profile = get_search_profile(inst)
    if (not ENABLE_PAIR_REASSIGNMENT_POLISH
            or not bool(profile.get("enable_pair_polish", False))
            or not sol.feasible):
        return sol

    best = sol
    evaluations = 0
    for _pass in range(max(1, max_passes)):
        if _deadline_reached(deadline):
            break
        improved = False
        pairs = list(itertools.combinations(list(inst.points.keys()), 2))
        rng.shuffle(pairs)

        for j1, j2 in pairs:
            if _deadline_reached(deadline) or evaluations >= PAIR_REASSIGNMENT_MAX_EVALUATIONS:
                break

            base_assignment = dict(best.assignment)
            base_assignment[j1] = None
            base_assignment[j2] = None
            options1 = all_assignment_options(inst, j1, best.open_platforms)
            options2 = all_assignment_options(inst, j2, best.open_platforms)
            rng.shuffle(options1)
            rng.shuffle(options2)

            pair_best: Optional[Solution] = None
            for opt1 in options1:
                for opt2 in options2:
                    if _deadline_reached(deadline) or evaluations >= PAIR_REASSIGNMENT_MAX_EVALUATIONS:
                        break
                    evaluations += 1
                    trial_assignment = dict(base_assignment)
                    trial_assignment[j1] = opt1
                    trial_assignment[j2] = opt2
                    trial = evaluate_solution(inst, best.open_platforms, trial_assignment)
                    if trial.feasible and trial.obj < best.obj - 1e-9:
                        if pair_best is None or trial.obj < pair_best.obj - 1e-9:
                            pair_best = trial
                if _deadline_reached(deadline) or evaluations >= PAIR_REASSIGNMENT_MAX_EVALUATIONS:
                    break

            if pair_best is not None:
                best = pair_best
                improved = True
                if LOCAL_SEARCH_FIRST_IMPROVEMENT:
                    break

        if not improved:
            break
    return best


def local_polish(inst: UAVInstance, sol: Solution, rng: random.Random, max_passes: int = 5,
                 deadline: Optional[float] = None) -> Solution:
    """Local improvement with an optional wall-clock deadline.

    The previous debugging run showed that the final polishing stage could make an
    ALNS run exceed the nominal 600 s cap. The deadline checks below keep the
    official small-scale runs much closer to the reported time limit.
    """
    best = sol
    profile = get_search_profile(inst)
    max_passes = min(max_passes, max(0, int(profile.get("initial_polish_passes", max_passes))))
    for _ in range(max_passes):
        if deadline is not None and time.time() >= deadline:
            break
        improved = False
        points = list(inst.points.keys())
        rng.shuffle(points)
        for j in points:
            if deadline is not None and time.time() >= deadline:
                break
            current_assignment = dict(best.assignment)
            current_assignment[j] = None
            base = evaluate_solution(inst, best.open_platforms, current_assignment)
            best_j_sol = best
            workloads = dict(base.workloads)
            for opt in all_assignment_options(inst, j, best.open_platforms):
                if deadline is not None and time.time() >= deadline:
                    break
                i, k = opt
                w_new = workloads[(i, k)] + assignment_workload(inst, j, i, k)
                if not math.isfinite(w_new):
                    continue
                if required_uavs_for_workload(w_new) > MAX_UAVS_PER_PLATFORM_TYPE:
                    continue
                trial_assign = dict(base.assignment)
                trial_assign[j] = opt
                trial_sol = evaluate_solution(inst, best.open_platforms, trial_assign)
                if trial_sol.fitness + 1e-9 < best_j_sol.fitness:
                    best_j_sol = trial_sol
            if best_j_sol.fitness + 1e-9 < best.fitness:
                best = best_j_sol
                improved = True
        if not improved:
            break
    return best



def normalize_routes(routes: Dict[Tuple[int, int, int], List[int]]) -> Dict[Tuple[int, int, int], List[int]]:
    return {veh: list(route) for veh, route in routes.items() if route}


def evaluate_route_dictionary(inst: UAVInstance,
                              open_platforms: Iterable[int],
                              routes: Dict[Tuple[int, int, int], List[int]]) -> Solution:
    return evaluate_full_path_routes(inst, open_platforms, normalize_routes(routes))


def _deadline_reached(deadline: Optional[float]) -> bool:
    effective = deadline if deadline is not None else _ACTIVE_ALNS_DEADLINE
    return effective is not None and time.time() >= effective


def _hard_deadline_reached() -> bool:
    return _ACTIVE_ALNS_DEADLINE is not None and time.time() >= _ACTIVE_ALNS_DEADLINE


def _candidate_route_keys_same_type(routes: Dict[Tuple[int, int, int], List[int]],
                                    source_key: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    i, k, _ = source_key
    return [veh for veh, route in routes.items() if route and veh != source_key and veh[0] == i and veh[1] == k]


def improve_by_uav_elimination(inst: UAVInstance,
                               sol: Solution,
                               rng: random.Random,
                               deadline: Optional[float] = None) -> Solution:
    """Try to empty one used UAV route by inserting all of its visits into other used UAVs of the same platform/type."""
    if not ENABLE_UAV_ELIMINATION or not sol.feasible or not sol.routes:
        return sol

    best = sol
    route_keys = list(sol.routes.keys())
    route_keys.sort(
        key=lambda veh: (
            len(sol.routes.get(veh, [])),
            -fleet_unit_cost(inst, veh[1]),
            veh,
        )
    )

    for source_key in route_keys:
        if _deadline_reached(deadline):
            break
        source_route = list(best.routes.get(source_key, [])) if best.routes else []
        if not source_route:
            continue

        base_routes = normalize_routes(best.routes or {})
        target_keys = _candidate_route_keys_same_type(base_routes, source_key)
        if not target_keys:
            continue

        orderings: List[List[int]] = [list(source_route), list(reversed(source_route))]
        for _ in range(max(0, UAV_ELIMINATION_ORDER_TRIALS - len(orderings))):
            seq = list(source_route)
            rng.shuffle(seq)
            orderings.append(seq)

        best_elimination: Optional[Solution] = None
        for visits in orderings:
            if _deadline_reached(deadline):
                break
            trial_routes = {veh: list(rt) for veh, rt in base_routes.items() if veh != source_key}
            success = True

            for point_id in visits:
                best_target = None
                best_new_route = None
                best_delta = float("inf")
                for target_key in target_keys:
                    route = trial_routes.get(target_key, [])
                    i, k, _ = target_key
                    for pos in range(len(route) + 1):
                        ok, delta, new_route = route_insertion_delta(inst, i, k, route, point_id, pos)
                        if ok and delta < best_delta - 1e-12:
                            best_delta = delta
                            best_target = target_key
                            best_new_route = new_route
                if best_target is None or best_new_route is None:
                    success = False
                    break
                trial_routes[best_target] = best_new_route

            if not success:
                continue
            candidate = evaluate_route_dictionary(inst, best.open_platforms, trial_routes)
            if candidate.feasible and candidate.obj < best.obj - 1e-9:
                if best_elimination is None or candidate.obj < best_elimination.obj - 1e-9:
                    best_elimination = candidate
                    if LOCAL_SEARCH_FIRST_IMPROVEMENT:
                        break

        if best_elimination is not None:
            return best_elimination

    return best


def improve_by_route_merge(inst: UAVInstance,
                           sol: Solution,
                           rng: random.Random,
                           deadline: Optional[float] = None) -> Solution:
    """Merge two routes of the same platform/type when a single feasible route is cheaper."""
    if not ENABLE_ROUTE_MERGE or not sol.feasible or not sol.routes:
        return sol

    best = sol
    keys = list(sol.routes.keys())
    pairs = [
        (a, b)
        for idx, a in enumerate(keys)
        for b in keys[idx + 1:]
        if a[0] == b[0] and a[1] == b[1]
    ]
    rng.shuffle(pairs)
    evaluations = 0

    for a_key, b_key in pairs:
        if _deadline_reached(deadline) or evaluations >= int(get_search_profile(inst)["local_move_evaluations"]):
            break
        ra = list(best.routes[a_key])  # type: ignore[index]
        rb = list(best.routes[b_key])  # type: ignore[index]
        i, k, _ = a_key

        sequences = [
            ra + rb,
            ra + list(reversed(rb)),
            list(reversed(ra)) + rb,
            list(reversed(ra)) + list(reversed(rb)),
            rb + ra,
            rb + list(reversed(ra)),
        ]

        for merged in sequences:
            evaluations += 1
            ok, _, _, _ = route_is_feasible(inst, i, k, merged)
            if not ok:
                continue
            trial_routes = {veh: list(rt) for veh, rt in (best.routes or {}).items() if veh not in {a_key, b_key}}
            trial_routes[a_key] = merged
            candidate = evaluate_route_dictionary(inst, best.open_platforms, trial_routes)
            if candidate.feasible and candidate.obj < best.obj - 1e-9:
                return candidate

    return best


def improve_by_inter_route_relocate(inst: UAVInstance,
                                    sol: Solution,
                                    rng: random.Random,
                                    deadline: Optional[float] = None) -> Solution:
    """Move one occurrence between routes of the same platform/type."""
    if not ENABLE_INTER_ROUTE_RELOCATE or not sol.feasible or not sol.routes:
        return sol

    best_candidate: Optional[Solution] = None
    routes = normalize_routes(sol.routes)
    source_keys = list(routes.keys())
    rng.shuffle(source_keys)
    evaluations = 0

    for source_key in source_keys:
        if _deadline_reached(deadline) or evaluations >= int(get_search_profile(inst)["local_move_evaluations"]):
            break
        source_route = routes[source_key]
        positions = list(range(len(source_route)))
        rng.shuffle(positions)

        for src_pos in positions:
            point_id = source_route[src_pos]
            reduced_source = source_route[:src_pos] + source_route[src_pos + 1:]
            if reduced_source:
                ok_src, _, _, _ = route_is_feasible(inst, source_key[0], source_key[1], reduced_source)
                if not ok_src:
                    continue

            target_keys = _candidate_route_keys_same_type(routes, source_key)
            rng.shuffle(target_keys)
            for target_key in target_keys:
                target_route = routes[target_key]
                for tgt_pos in range(len(target_route) + 1):
                    if _deadline_reached(deadline) or evaluations >= int(get_search_profile(inst)["local_move_evaluations"]):
                        break
                    evaluations += 1
                    ok_tgt, _, new_target = route_insertion_delta(
                        inst, target_key[0], target_key[1], target_route, point_id, tgt_pos
                    )
                    if not ok_tgt:
                        continue

                    trial_routes = {veh: list(rt) for veh, rt in routes.items()}
                    if reduced_source:
                        trial_routes[source_key] = reduced_source
                    else:
                        trial_routes.pop(source_key, None)
                    trial_routes[target_key] = new_target
                    candidate = evaluate_route_dictionary(inst, sol.open_platforms, trial_routes)
                    if candidate.feasible and candidate.obj < sol.obj - 1e-9:
                        if LOCAL_SEARCH_FIRST_IMPROVEMENT:
                            return candidate
                        if best_candidate is None or candidate.obj < best_candidate.obj - 1e-9:
                            best_candidate = candidate

    return best_candidate if best_candidate is not None else sol


def improve_by_inter_route_swap(inst: UAVInstance,
                                sol: Solution,
                                rng: random.Random,
                                deadline: Optional[float] = None) -> Solution:
    """Swap one occurrence between two routes of the same platform/type."""
    if not ENABLE_INTER_ROUTE_SWAP or not sol.feasible or not sol.routes:
        return sol

    routes = normalize_routes(sol.routes)
    keys = list(routes.keys())
    pairs = [
        (a, b)
        for idx, a in enumerate(keys)
        for b in keys[idx + 1:]
        if a[0] == b[0] and a[1] == b[1]
    ]
    rng.shuffle(pairs)
    best_candidate: Optional[Solution] = None
    evaluations = 0

    for a_key, b_key in pairs:
        if _deadline_reached(deadline) or evaluations >= int(get_search_profile(inst)["local_move_evaluations"]):
            break
        ra = routes[a_key]
        rb = routes[b_key]
        idx_a = list(range(len(ra)))
        idx_b = list(range(len(rb)))
        rng.shuffle(idx_a)
        rng.shuffle(idx_b)

        for pa in idx_a:
            for pb in idx_b:
                if _deadline_reached(deadline) or evaluations >= int(get_search_profile(inst)["local_move_evaluations"]):
                    break
                evaluations += 1
                if ra[pa] == rb[pb]:
                    continue
                new_a = list(ra)
                new_b = list(rb)
                new_a[pa], new_b[pb] = new_b[pb], new_a[pa]
                ok_a, _, _, _ = route_is_feasible(inst, a_key[0], a_key[1], new_a)
                ok_b, _, _, _ = route_is_feasible(inst, b_key[0], b_key[1], new_b)
                if not (ok_a and ok_b):
                    continue
                trial_routes = {veh: list(rt) for veh, rt in routes.items()}
                trial_routes[a_key] = new_a
                trial_routes[b_key] = new_b
                candidate = evaluate_route_dictionary(inst, sol.open_platforms, trial_routes)
                if candidate.feasible and candidate.obj < sol.obj - 1e-9:
                    if LOCAL_SEARCH_FIRST_IMPROVEMENT:
                        return candidate
                    if best_candidate is None or candidate.obj < best_candidate.obj - 1e-9:
                        best_candidate = candidate

    return best_candidate if best_candidate is not None else sol


def improve_by_two_opt(inst: UAVInstance,
                       sol: Solution,
                       rng: random.Random,
                       deadline: Optional[float] = None) -> Solution:
    """Apply 2-opt within individual routes."""
    if not ENABLE_ROUTE_TWO_OPT or not sol.feasible or not sol.routes:
        return sol

    routes = normalize_routes(sol.routes)
    keys = list(routes.keys())
    rng.shuffle(keys)
    best_candidate: Optional[Solution] = None
    evaluations = 0

    for veh in keys:
        if _deadline_reached(deadline) or evaluations >= int(get_search_profile(inst)["local_move_evaluations"]):
            break
        route = routes[veh]
        if len(route) < 3:
            continue
        pairs = [(a, b) for a in range(len(route) - 1) for b in range(a + 1, len(route))]
        rng.shuffle(pairs)
        for a, b in pairs:
            if _deadline_reached(deadline) or evaluations >= int(get_search_profile(inst)["local_move_evaluations"]):
                break
            evaluations += 1
            new_route = route[:a] + list(reversed(route[a:b + 1])) + route[b + 1:]
            if new_route == route:
                continue
            ok, _, _, _ = route_is_feasible(inst, veh[0], veh[1], new_route)
            if not ok:
                continue
            trial_routes = {key: list(rt) for key, rt in routes.items()}
            trial_routes[veh] = new_route
            candidate = evaluate_route_dictionary(inst, sol.open_platforms, trial_routes)
            if candidate.feasible and candidate.obj < sol.obj - 1e-9:
                if LOCAL_SEARCH_FIRST_IMPROVEMENT:
                    return candidate
                if best_candidate is None or candidate.obj < best_candidate.obj - 1e-9:
                    best_candidate = candidate

    return best_candidate if best_candidate is not None else sol


def route_level_intensification(inst: UAVInstance,
                                sol: Solution,
                                rng: random.Random,
                                max_rounds: int,
                                deadline: Optional[float] = None) -> Solution:
    """Repeatedly apply route-level operators until no further improvement is found."""
    if not ENABLE_ROUTE_LEVEL_INTENSIFICATION or not sol.feasible:
        return sol

    best = sol
    operators = [
        improve_by_uav_elimination,
        improve_by_route_merge,
        improve_by_inter_route_relocate,
        improve_by_inter_route_swap,
        improve_by_two_opt,
    ]

    for _round in range(max(1, max_rounds)):
        if _deadline_reached(deadline):
            break
        improved = False
        for operator in operators:
            if _deadline_reached(deadline):
                break
            candidate = operator(inst, best, rng, deadline)
            if candidate.feasible and candidate.obj < best.obj - 1e-9:
                best = candidate
                improved = True
        if not improved:
            break
    return best


DESTROY_OPERATORS = [
    ("random_remove", remove_random_tasks),
    ("worst_remove", remove_worst_tasks),
    ("related_remove", remove_related_tasks),
    ("route_remove", remove_one_route_points),
    ("platform_type_remove", remove_platform_type_group),
    ("platform_swap", swap_one_platform),
]

REPAIR_OPERATORS = [
    ("greedy", repair_greedy),
    ("random", repair_random),
    ("regret2", repair_regret2),
    ("balanced", repair_balanced),
    ("route_aware", repair_route_aware),
]


def get_state(iter_idx: int, max_iter: int, current: Solution, best: Solution) -> str:
    phase = "early" if iter_idx < 0.33 * max_iter else "middle" if iter_idx < 0.67 * max_iter else "late"
    quality = "feasible" if current.feasible else "infeasible"
    current_score = search_score(current)
    best_score = search_score(best)
    gap = (current_score - best_score) / max(abs(best_score), 1.0)
    level = "near" if gap < 0.02 else "far"
    return f"{phase}_{quality}_{level}"


def select_operator_pair(q_table: Dict[str, List[float]],
                         state: str,
                         rng: random.Random,
                         epsilon: float) -> int:
    n_pairs = len(DESTROY_OPERATORS) * len(REPAIR_OPERATORS)
    if state not in q_table:
        q_table[state] = [0.0 for _ in range(n_pairs)]
    if rng.random() < epsilon:
        return rng.randrange(n_pairs)
    values = q_table[state]
    max_val = max(values)
    best_indices = [idx for idx, val in enumerate(values) if abs(val - max_val) <= 1e-12]
    return rng.choice(best_indices)


def decode_pair(pair_idx: int):
    r_count = len(REPAIR_OPERATORS)
    d_idx = pair_idx // r_count
    r_idx = pair_idx % r_count
    return DESTROY_OPERATORS[d_idx], REPAIR_OPERATORS[r_idx]


def solve_alns_qsa(inst: UAVInstance, random_seed: int) -> Dict[str, Any]:
    global _ACTIVE_ALNS_DEADLINE

    rng = random.Random(random_seed)
    t_start = time.time()
    max_iter = get_alns_max_iter(inst.spec.scale)
    nominal_time_limit = get_alns_time_limit(inst.spec.scale)
    profile = get_search_profile(inst)
    guard_sec = min(float(profile["time_guard_sec"]), max(0.0, nominal_time_limit - 1.0))
    deadline = t_start + nominal_time_limit - guard_sec
    _ACTIVE_ALNS_DEADLINE = deadline

    current = generate_initial_solution(inst, rng, deadline=deadline)
    initial_passes = int(profile.get("initial_polish_passes", 0))
    if current.feasible and initial_passes > 0 and not _deadline_reached(deadline):
        current = local_polish(inst, current, rng, max_passes=initial_passes, deadline=deadline)

    if (not current.feasible
            and remaining_time(deadline) >= COVERAGE_RESCUE_MIN_REMAINING_SEC):
        rescued = coverage_rescue_solution(inst, current, rng, deadline)
        if is_better_solution(rescued, current):
            current = rescued

    if (ENABLE_ROUTE_LEVEL_INTENSIFICATION
            and current.feasible
            and remaining_time(deadline) >= COVERAGE_RESCUE_MIN_REMAINING_SEC):
        current = route_level_intensification(
            inst,
            current,
            rng,
            max_rounds=int(profile["route_intensification_rounds"]),
            deadline=deadline,
        )

    best: Optional[Solution] = current if current.feasible else None
    best_any = current
    q_table: Dict[str, List[float]] = {}
    temperature = ALNS_INITIAL_TEMPERATURE
    accepted_count = 0
    improved_count = 0
    last_score = search_score(current)
    it = 0

    for it in range(1, max_iter + 1):
        if _deadline_reached(deadline):
            break

        progress = it / max_iter
        epsilon = ALNS_EPSILON_INITIAL + (ALNS_EPSILON_FINAL - ALNS_EPSILON_INITIAL) * progress
        state = get_state(it, max_iter, current, best if best is not None else best_any)
        pair_idx = select_operator_pair(q_table, state, rng, epsilon)
        (_d_name, d_func), (_r_name, r_func) = decode_pair(pair_idx)

        open_platforms, partial_assignment = d_func(inst, current, rng)
        if _deadline_reached(deadline):
            break
        candidate = r_func(inst, open_platforms, partial_assignment, rng)

        # Coverage rescue is invoked only while infeasible and at a controlled
        # interval. It directly inserts complete frequency sets into routes.
        if (not candidate.feasible
                and ENABLE_COVERAGE_RESCUE
                and it % max(1, COVERAGE_RESCUE_INTERVAL) == 0
                and remaining_time(deadline) >= COVERAGE_RESCUE_MIN_REMAINING_SEC):
            rescued = coverage_rescue_solution(inst, candidate, rng, deadline)
            if is_better_solution(rescued, candidate):
                candidate = rescued

        polish_interval = int(profile.get("periodic_polish_interval", 0))
        if (candidate.feasible
                and polish_interval > 0
                and it % polish_interval == 0
                and remaining_time(deadline) >= COVERAGE_RESCUE_MIN_REMAINING_SEC):
            candidate = local_polish(inst, candidate, rng, max_passes=1, deadline=deadline)

        intensification_interval = int(profile["route_intensification_interval"])
        if (ENABLE_ROUTE_LEVEL_INTENSIFICATION
                and candidate.feasible
                and intensification_interval > 0
                and it % intensification_interval == 0
                and remaining_time(deadline) >= COVERAGE_RESCUE_MIN_REMAINING_SEC):
            candidate = route_level_intensification(
                inst,
                candidate,
                rng,
                max_rounds=int(profile["route_intensification_rounds"]),
                deadline=deadline,
            )

        current_score = search_score(current)
        candidate_score = search_score(candidate)
        delta = candidate_score - current_score
        if delta <= 0:
            accept = True
        else:
            scaled_delta = min(delta, 1.0e6)
            prob = math.exp(-scaled_delta / max(temperature, ALNS_MIN_TEMPERATURE))
            accept = rng.random() < prob

        reward = 0.0
        if accept:
            accepted_count += 1
            current = candidate
            reward += 0.5
            if candidate_score < last_score - 1e-9:
                reward += 1.0
            last_score = search_score(current)

        if is_better_solution(candidate, best_any):
            best_any = candidate
            reward += 2.0

        if candidate.feasible and (best is None or candidate.obj < best.obj - 1e-9):
            best = candidate
            improved_count += 1
            reward += 5.0

        next_state = get_state(it, max_iter, current, best if best is not None else best_any)
        if next_state not in q_table:
            q_table[next_state] = [0.0 for _ in range(len(DESTROY_OPERATORS) * len(REPAIR_OPERATORS))]
        old_q = q_table[state][pair_idx]
        next_best_q = max(q_table[next_state])
        q_table[state][pair_idx] = old_q + QL_ALPHA * (reward + QL_GAMMA * next_best_q - old_q)
        temperature = max(ALNS_MIN_TEMPERATURE, temperature * ALNS_COOLING_RATE)

    final_sol = best if best is not None else best_any

    # One final feasibility rescue is more important than cost polishing when
    # no feasible L300 solution has yet been found.
    if (not final_sol.feasible
            and remaining_time(deadline) >= COVERAGE_RESCUE_MIN_REMAINING_SEC):
        rescued = coverage_rescue_solution(
            inst,
            final_sol,
            rng,
            deadline,
            max_points=int(profile["coverage_rescue_max_points"]),
        )
        if is_better_solution(rescued, final_sol):
            final_sol = rescued
        if final_sol.feasible:
            best = final_sol

    if (final_sol.feasible
            and remaining_time(deadline) >= COVERAGE_RESCUE_MIN_REMAINING_SEC):
        final_sol = route_level_intensification(
            inst,
            final_sol,
            rng,
            max_rounds=int(profile["final_intensification_rounds"]),
            deadline=deadline,
        )
        if best is None or final_sol.obj < best.obj - 1e-9:
            best = final_sol

    final_sol = best if best is not None else final_sol
    _ACTIVE_ALNS_DEADLINE = None
    elapsed = time.time() - t_start
    time_limit_respected = elapsed <= nominal_time_limit + ALNS_TIME_LIMIT_TOLERANCE_SEC

    if DEBUG_PRINT_SOLUTION_AUDIT:
        print("[DEBUG_ALNS_RECHECK]")
        print("final_obj        =", final_sol.obj)
        print("final_fitness    =", final_sol.fitness)
        print("feasible         =", final_sol.feasible)
        print("coverage         =", final_sol.coverage)
        print("violation        =", final_sol.violation)
        print("components       =", final_sol.components)
        print("route_count      =", len(final_sol.routes or {}))
        print("nominal_limit    =", nominal_time_limit)
        print("internal_deadline=", nominal_time_limit - guard_sec)
        print("time_respected   =", time_limit_respected)

    return {
        "method": "ALNS-Q-learning-SA",
        "status": "OK" if final_sol.feasible else "NO_FEASIBLE_FINAL",
        "obj": final_sol.obj if final_sol.feasible else None,
        "fitness": final_sol.fitness,
        "time_sec": elapsed,
        "nominal_time_limit_sec": nominal_time_limit,
        "internal_search_budget_sec": nominal_time_limit - guard_sec,
        "time_limit_respected": time_limit_respected,
        "feasible": final_sol.feasible,
        "coverage": final_sol.coverage,
        "open_platforms": list(final_sol.open_platforms),
        "assignment": final_sol.assignment,
        "n_uavs": final_sol.n_uavs,
        "routes": final_sol.routes,
        "components": final_sol.components,
        "obj_recomputed": final_sol.obj,
        "obj_diff": 0.0,
        "accepted_count": accepted_count,
        "improved_count": improved_count,
        "iterations": it,
        "used_uav_count": len(final_sol.routes or {}),
        "platform_cost": (final_sol.components or {}).get("platform_cost"),
        "fleet_cost": (final_sol.components or {}).get("fleet_cost"),
        "route_energy_cost": (final_sol.components or {}).get("route_energy_cost"),
        "violation": final_sol.violation,
    }


# ============================================================
# 8. 结果统计
# ============================================================

def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def mean(values: List[float]) -> Optional[float]:
    clean = [safe_float(v) for v in values]
    clean = [v for v in clean if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def std(values: List[float]) -> Optional[float]:
    clean = [safe_float(v) for v in values]
    clean = [v for v in clean if v is not None]
    if len(clean) == 0:
        return None
    if len(clean) == 1:
        return 0.0
    avg = sum(clean) / len(clean)
    return math.sqrt(sum((v - avg) ** 2 for v in clean) / (len(clean) - 1))


def cv_percent(values: List[float]) -> Optional[float]:
    avg = mean(values)
    sd = std(values)
    if avg is None or sd is None or abs(avg) < 1e-12:
        return None
    return sd / abs(avg) * 100.0


def gap_to_ref_percent(obj: Optional[float], ref: Optional[float]) -> Optional[float]:
    obj = safe_float(obj)
    ref = safe_float(ref)
    if obj is None or ref is None or abs(ref) < 1e-12:
        return None
    return (obj - ref) / abs(ref) * 100.0


def gap_to_lb_percent(obj: Optional[float], lb: Optional[float]) -> Optional[float]:
    obj = safe_float(obj)
    lb = safe_float(lb)
    if obj is None or lb is None or abs(obj) < 1e-12:
        return None
    return (obj - lb) / abs(obj) * 100.0


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instance_id, g in raw_df.groupby("instance_id"):
        base = {
            "instance_id": instance_id,
            "scale": g["scale"].iloc[0],
            "n_points": int(g["n_points"].iloc[0]),
            "n_candidate_platforms": int(g["n_candidate_platforms"].iloc[0]),
            "open_platforms_fixed": int(g["open_platforms_fixed"].iloc[0]),
            "seed": int(g["seed"].iloc[0]),
        }

        gd = g[g["method"] == "Gurobi"]
        al = g[g["method"] == "ALNS-Q-learning-SA"]

        gurobi_status = None
        gurobi_obj = None
        gurobi_lb = None
        gurobi_gap = None
        gurobi_time = None
        gurobi_feasible = False
        gurobi_vars = None
        gurobi_constrs = None
        gurobi_obj_recomputed = None
        gurobi_obj_diff = None
        gurobi_evaluator_feasible = None
        gurobi_evaluator_violation = None

        if not gd.empty:
            row = gd.iloc[0]
            gurobi_status = row.get("status")
            gurobi_obj = safe_float(row.get("obj"))
            gurobi_lb = safe_float(row.get("lb"))
            gurobi_gap = safe_float(row.get("gap_percent"))
            gurobi_time = safe_float(row.get("time_sec"))
            gurobi_feasible = bool(row.get("feasible"))
            gurobi_vars = row.get("model_vars")
            gurobi_constrs = row.get("model_constrs")
            gurobi_obj_recomputed = safe_float(row.get("obj_recomputed"))
            gurobi_obj_diff = safe_float(row.get("obj_diff"))
            gurobi_evaluator_feasible = row.get("evaluator_feasible")
            gurobi_evaluator_violation = row.get("evaluator_violation")

        al_feas = al[al["feasible"] == True]
        al_objs = [safe_float(v) for v in al_feas["obj"].tolist()]
        al_objs = [v for v in al_objs if v is not None]
        al_completed = al[al["status"] != "ERROR"] if "status" in al.columns else al
        al_times = [safe_float(v) for v in al_completed["time_sec"].tolist()]
        al_times = [v for v in al_times if v is not None]
        al_covs = [safe_float(v) for v in al_completed["coverage"].tolist()]
        al_covs = [v for v in al_covs if v is not None]
        al_time_respected_values = [
            bool(v) for v in al_completed.get("time_limit_respected", pd.Series(dtype=bool)).tolist()
            if pd.notna(v)
        ]
        al_time_limit_respected_rate = (
            100.0 * sum(al_time_respected_values) / len(al_time_respected_values)
            if al_time_respected_values else None
        )

        al_best = min(al_objs) if al_objs else None
        al_mean = mean(al_objs)
        al_std = std(al_objs)
        al_cv = cv_percent(al_objs)
        al_time_mean = mean(al_times)
        al_time_min = min(al_times) if al_times else None
        al_cov_mean = mean(al_covs)
        al_feasible_rate = len(al_feas) / max(len(al), 1) * 100.0 if not al.empty else None

        exact_validation_qualified = (
            str(gurobi_status).upper() == "OPTIMAL"
            and gurobi_feasible
            and gurobi_obj is not None
        )
        if exact_validation_qualified:
            al_gap_to_opt = gap_to_ref_percent(al_best, gurobi_obj)
            al_gap_mean_to_opt = gap_to_ref_percent(al_mean, gurobi_obj)
        else:
            al_gap_to_opt = None
            al_gap_mean_to_opt = None

        al_gap_to_lb = gap_to_lb_percent(al_best, gurobi_lb)
        if gurobi_feasible and gurobi_obj is not None and al_best is not None:
            al_improvement_vs_gurobi = (gurobi_obj - al_best) / abs(gurobi_obj) * 100.0
        else:
            al_improvement_vs_gurobi = None

        # A negative ALNS-vs-LB gap is a diagnostic red flag for inconsistent objectives.
        # After the full-path evaluator alignment, this should normally be False.
        alns_below_gurobi_lb = (
            al_best is not None and gurobi_lb is not None and al_best < gurobi_lb - 1e-5
        )

        rows.append({
            **base,
            "gurobi_status": gurobi_status,
            "gurobi_obj": gurobi_obj,
            "gurobi_lb": gurobi_lb,
            "gurobi_gap_percent": gurobi_gap,
            "gurobi_time_sec": gurobi_time,
            "gurobi_model_vars": gurobi_vars,
            "gurobi_model_constrs": gurobi_constrs,
            "gurobi_obj_recomputed": gurobi_obj_recomputed,
            "gurobi_obj_diff": gurobi_obj_diff,
            "gurobi_evaluator_feasible": gurobi_evaluator_feasible,
            "gurobi_evaluator_violation": gurobi_evaluator_violation,
            "alns_best_obj": al_best,
            "alns_mean_obj": al_mean,
            "alns_std_obj": al_std,
            "alns_cv_percent": al_cv,
            "alns_feasible_runs": int(len(al_feas)),
            "alns_total_runs": int(len(al)),
            "exact_validation_qualified": bool(exact_validation_qualified),
            "alns_gap_best_to_gurobi_opt_percent": al_gap_to_opt,
            "alns_gap_mean_to_gurobi_opt_percent": al_gap_mean_to_opt,
            # Kept for backward compatibility with older table templates:
            "alns_gap_to_gurobi_opt_percent": al_gap_to_opt,
            "alns_gap_to_gurobi_lb_percent": al_gap_to_lb,
            "alns_improvement_vs_gurobi_ub_percent": al_improvement_vs_gurobi,
            "alns_below_gurobi_lb_flag": alns_below_gurobi_lb,
            "alns_mean_time_sec": al_time_mean,
            "alns_min_time_sec": al_time_min,
            "alns_mean_coverage": al_cov_mean,
            "alns_feasible_rate_percent": al_feasible_rate,
            "alns_time_limit_respected_rate_percent": al_time_limit_respected_rate,
        })

    return pd.DataFrame(rows)


def summarize_by_scale(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary_df.empty:
        return pd.DataFrame(rows)
    for scale, g in summary_df.groupby("scale"):
        rows.append({
            "scale": scale,
            "n_instances": len(g),
            "mean_gurobi_time_sec": mean(g["gurobi_time_sec"].tolist()),
            "mean_gurobi_gap_percent": mean(g["gurobi_gap_percent"].tolist()),
            "mean_alns_best_obj": mean(g["alns_best_obj"].tolist()),
            "mean_alns_mean_obj": mean(g["alns_mean_obj"].tolist()),
            "mean_alns_gap_best_to_gurobi_opt_percent": mean(g["alns_gap_best_to_gurobi_opt_percent"].tolist()) if "alns_gap_best_to_gurobi_opt_percent" in g else None,
            "mean_alns_gap_mean_to_gurobi_opt_percent": mean(g["alns_gap_mean_to_gurobi_opt_percent"].tolist()) if "alns_gap_mean_to_gurobi_opt_percent" in g else None,
            "mean_alns_gap_to_gurobi_opt_percent": mean(g["alns_gap_to_gurobi_opt_percent"].tolist()),
            "mean_alns_gap_to_gurobi_lb_percent": mean(g["alns_gap_to_gurobi_lb_percent"].tolist()),
            "mean_alns_time_sec": mean(g["alns_mean_time_sec"].tolist()),
            "mean_alns_feasible_rate_percent": mean(g["alns_feasible_rate_percent"].tolist()),
        })
    return pd.DataFrame(rows)



def fmt_float(value: Any, ndigits: int = 4) -> Any:
    """Return a rounded float for paper tables; keep missing values as an en dash."""
    v = safe_float(value)
    if v is None:
        return "–"
    return round(v, ndigits)


def build_exact_validation_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the table required for exact small-scale validation.

    Gap(min) and Gap(mean) are reported only when Gurobi proves OPTIMAL. If an
    instance is not solved to proven optimality, the gap cells are left as "–",
    and the qualification flag is set to "No". This is the key difference from
    time-limited incumbent comparison.
    """
    rows: List[Dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame(rows)

    ordered = summary_df.sort_values(["n_points", "instance_id"]).copy()
    for _, r in ordered.iterrows():
        qualified = bool(r.get("exact_validation_qualified", False))
        feasible_runs = r.get("alns_feasible_runs")
        total_runs = r.get("alns_total_runs")
        feasible_str = "–"
        if pd.notna(feasible_runs) and pd.notna(total_runs):
            feasible_str = f"{int(feasible_runs)}/{int(total_runs)}"

        rows.append({
            "Instance": r.get("instance_id"),
            "|J|": int(r.get("n_points")),
            "Variable": int(r.get("gurobi_model_vars")) if pd.notna(r.get("gurobi_model_vars")) else "–",
            "Constraint": int(r.get("gurobi_model_constrs")) if pd.notna(r.get("gurobi_model_constrs")) else "–",
            "Gurobi status": r.get("gurobi_status"),
            "Gurobi Obj.": fmt_float(r.get("gurobi_obj"), 4) if qualified else "–",
            "Gurobi time (s)": fmt_float(r.get("gurobi_time_sec"), 2),
            "Gurobi gap (%)": fmt_float(r.get("gurobi_gap_percent"), 4),
            "ALNS Obj. (min)": fmt_float(r.get("alns_best_obj"), 4),
            "ALNS Obj. (mean)": fmt_float(r.get("alns_mean_obj"), 4),
            "Gap (min) (%)": fmt_float(r.get("alns_gap_best_to_gurobi_opt_percent"), 4) if qualified else "–",
            "Gap (mean) (%)": fmt_float(r.get("alns_gap_mean_to_gurobi_opt_percent"), 4) if qualified else "–",
            "CPU (mean) (s)": fmt_float(r.get("alns_mean_time_sec"), 2),
            "CPU (min) (s)": fmt_float(r.get("alns_min_time_sec"), 2),
            "Feasible runs": feasible_str,
            "Exact validation": "Yes" if qualified else "No",
        })

    table = pd.DataFrame(rows)

    # Average row: objective values are not averaged for presentation, but
    # computational times and gaps are useful for the supervisor/reviewer standard.
    qualified_rows = summary_df[summary_df.get("exact_validation_qualified", False) == True] \
        if "exact_validation_qualified" in summary_df.columns else pd.DataFrame()
    avg_row = {
        "Instance": "Average",
        "|J|": "–",
        "Variable": "–",
        "Constraint": "–",
        "Gurobi status": "–",
        "Gurobi Obj.": "–",
        "Gurobi time (s)": fmt_float(mean(qualified_rows["gurobi_time_sec"].tolist()) if not qualified_rows.empty else None, 2),
        "Gurobi gap (%)": fmt_float(mean(qualified_rows["gurobi_gap_percent"].tolist()) if not qualified_rows.empty else None, 4),
        "ALNS Obj. (min)": "–",
        "ALNS Obj. (mean)": "–",
        "Gap (min) (%)": fmt_float(mean(qualified_rows["alns_gap_best_to_gurobi_opt_percent"].tolist()) if not qualified_rows.empty else None, 4),
        "Gap (mean) (%)": fmt_float(mean(qualified_rows["alns_gap_mean_to_gurobi_opt_percent"].tolist()) if not qualified_rows.empty else None, 4),
        "CPU (mean) (s)": fmt_float(mean(summary_df["alns_mean_time_sec"].tolist()) if "alns_mean_time_sec" in summary_df.columns else None, 2),
        "CPU (min) (s)": fmt_float(mean(summary_df["alns_min_time_sec"].tolist()) if "alns_min_time_sec" in summary_df.columns else None, 2),
        "Feasible runs": "–",
        "Exact validation": f"{len(qualified_rows)}/{len(summary_df)} qualified",
    }
    table = pd.concat([table, pd.DataFrame([avg_row])], ignore_index=True)

    return table




def build_medium_large_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build the publication table for the revised medium/large experiment."""
    rows: List[Dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame(rows)

    scale_order = {"medium": 0, "large": 1}
    ordered = summary_df.copy()
    ordered["_scale_order"] = ordered["scale"].map(scale_order).fillna(99)
    ordered = ordered.sort_values(["_scale_order", "n_points", "instance_id"])

    for _, r in ordered.iterrows():
        feasible_runs = int(r.get("alns_feasible_runs", 0))
        total_runs = int(r.get("alns_total_runs", 0))
        status = r.get("gurobi_status")
        status_text = str(status) if pd.notna(status) and status not in (None, "None") else "Not run"

        gurobi_ub = safe_float(r.get("gurobi_obj"))
        gurobi_lb = safe_float(r.get("gurobi_lb"))
        alns_best = safe_float(r.get("alns_best_obj"))
        improvement = None
        if gurobi_ub is not None and alns_best is not None and abs(gurobi_ub) > 1e-12:
            # Positive means ALNS obtained a lower feasible objective.
            improvement = (gurobi_ub - alns_best) / abs(gurobi_ub) * 100.0

        rows.append({
            "Scale": str(r.get("scale")).capitalize(),
            "Instance": r.get("instance_id"),
            "|J|": int(r.get("n_points")),
            "Open/total platforms": f"{int(r.get('open_platforms_fixed'))}/{int(r.get('n_candidate_platforms'))}",
            "Gurobi status": status_text,
            "Gurobi UB": fmt_float(gurobi_ub, 2),
            "Gurobi LB": fmt_float(gurobi_lb, 2),
            "Gurobi MIP gap (%)": fmt_float(r.get("gurobi_gap_percent"), 2),
            "Gurobi CPU (s)": fmt_float(r.get("gurobi_time_sec"), 2),
            "ALNS best": fmt_float(alns_best, 2),
            "ALNS mean": fmt_float(r.get("alns_mean_obj"), 2),
            "ALNS std.": fmt_float(r.get("alns_std_obj"), 2),
            "ALNS CV (%)": fmt_float(r.get("alns_cv_percent"), 2),
            "Mean coverage": fmt_float(r.get("alns_mean_coverage"), 3),
            "Feasible runs": f"{feasible_runs}/{total_runs}",
            "ALNS CPU mean (s)": fmt_float(r.get("alns_mean_time_sec"), 2),
            "Time-limit compliance (%)": fmt_float(r.get("alns_time_limit_respected_rate_percent"), 1),
            "Improvement over Gurobi UB (%)": fmt_float(improvement, 2),
        })

    return pd.DataFrame(rows)

# ============================================================
# 8.1 安全保存与逐次检查点
# ============================================================

def _timestamped_fallback_path(path: str) -> str:
    """Create a non-conflicting fallback filename beside the requested file."""
    root, ext = os.path.splitext(path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = f"{root}_{stamp}{ext}"
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{root}_{stamp}_{counter}{ext}"
        counter += 1
    return candidate


def safe_write_csv(df: pd.DataFrame,
                   path: str,
                   label: str,
                   index: bool = False) -> str:
    """
    Save a CSV without terminating the experiment when the target is occupied.

    If Excel/WPS/PyCharm locks the requested file, the data are written to a
    timestamped fallback file in the same directory and the actual path is
    returned.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_csv(path, index=index, encoding="utf-8-sig")
        print(f"[SAVE OK] {label}: {path}")
        return path
    except (PermissionError, OSError) as exc:
        fallback = _timestamped_fallback_path(path)
        df.to_csv(fallback, index=index, encoding="utf-8-sig")
        print(f"[SAVE WARNING] {label} could not be written to the requested path.")
        print(f"[SAVE WARNING] Reason: {type(exc).__name__}: {exc}")
        print(f"[SAVE WARNING] Data were saved instead to: {fallback}")
        return fallback


def safe_write_excel(path: str,
                     sheets: Dict[str, pd.DataFrame],
                     label: str) -> str:
    """Save a multi-sheet Excel workbook with a timestamped fallback on lock."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _write(target: str) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            for sheet_name, dataframe in sheets.items():
                dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    try:
        _write(path)
        print(f"[SAVE OK] {label}: {path}")
        return path
    except (PermissionError, OSError) as exc:
        fallback = _timestamped_fallback_path(path)
        _write(fallback)
        print(f"[SAVE WARNING] {label} could not be written to the requested path.")
        print(f"[SAVE WARNING] Reason: {type(exc).__name__}: {exc}")
        print(f"[SAVE WARNING] Workbook was saved instead to: {fallback}")
        return fallback


def save_instance_checkpoint(inst: UAVInstance,
                             rows: List[Dict[str, Any]]) -> str:
    """
    Save all completed rows for one instance immediately after each solver run.

    This protects completed Gurobi/ALNS runs even if a later run, final summary,
    or workbook export is interrupted.
    """
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{inst.spec.instance_id}_raw_checkpoint.csv",
    )
    return safe_write_csv(
        pd.DataFrame(rows),
        checkpoint_path,
        label=f"checkpoint {inst.spec.instance_id}",
        index=False,
    )


# ============================================================
# 9. 主控运行
# ============================================================

def make_raw_row(inst: UAVInstance, method_result: Dict[str, Any], run_seed: Optional[int]) -> Dict[str, Any]:
    spec = inst.spec
    return {
        "instance_id": spec.instance_id,
        "scale": spec.scale,
        "n_points": spec.n_points,
        "n_candidate_platforms": spec.n_candidate_platforms,
        "n_hazard_sources": spec.n_hazard_sources,
        "n_no_fly_zones": spec.n_no_fly_zones,
        "n_uav_types": spec.n_uav_types,
        "open_platforms_fixed": spec.open_platforms_fixed,
        "seed": spec.seed,
        "method": method_result.get("method"),
        "run_seed": run_seed,
        "status": method_result.get("status"),
        "obj": method_result.get("obj"),
        "lb": method_result.get("lb"),
        "gap_percent": method_result.get("gap_percent"),
        "time_sec": method_result.get("time_sec"),
        "nominal_time_limit_sec": method_result.get("nominal_time_limit_sec"),
        "internal_search_budget_sec": method_result.get("internal_search_budget_sec"),
        "time_limit_respected": method_result.get("time_limit_respected"),
        "feasible": method_result.get("feasible"),
        "coverage": method_result.get("coverage"),
        "open_platforms": str(method_result.get("open_platforms")),
        "total_uavs": sum(method_result.get("n_uavs", {}).values()) if isinstance(method_result.get("n_uavs"), dict) else None,
        "route_count": len(method_result.get("routes", {}) or {}),
        "obj_recomputed": method_result.get("obj_recomputed"),
        "obj_diff": method_result.get("obj_diff"),
        "evaluator_feasible": method_result.get("evaluator_feasible"),
        "evaluator_violation": str(method_result.get("evaluator_violation")),
        "components": str(method_result.get("components")),
        "model_vars": method_result.get("model_vars"),
        "model_constrs": method_result.get("model_constrs"),
        "iterations": method_result.get("iterations"),
        "accepted_count": method_result.get("accepted_count"),
        "improved_count": method_result.get("improved_count"),
        "violation": str(method_result.get("violation")),
        "termination_note": method_result.get("termination_note"),
    }



def load_instance_checkpoint(instance_id: str) -> List[Dict[str, Any]]:
    """Load previously completed rows for one instance."""
    if not RESUME_FROM_CHECKPOINT:
        return []
    path = os.path.join(CHECKPOINT_DIR, f"{instance_id}_raw_checkpoint.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        rows = df.to_dict("records")
        print(f"[RESUME] Loaded {len(rows)} rows from {path}")
        return rows
    except Exception as exc:
        print(f"[RESUME WARNING] Could not read {path}: {exc}")
        return []


def _row_is_completed_gurobi(row: Dict[str, Any]) -> bool:
    if str(row.get("method")) != "Gurobi":
        return False
    status = str(row.get("status", "")).upper()
    # A memory-limited run with a preserved incumbent/bound is a valid
    # time-limited benchmark and need not be repeated automatically.
    return status not in {"", "NONE", "ERROR", "INTERRUPTED_WITH_EXCEPTION"}


def _completed_alns_seeds(rows: List[Dict[str, Any]]) -> set:
    completed = set()
    for row in rows:
        if str(row.get("method")) != "ALNS-Q-learning-SA":
            continue
        if str(row.get("status", "")).upper() != "OK":
            continue
        seed = row.get("run_seed")
        try:
            if pd.notna(seed):
                completed.add(int(float(seed)))
        except Exception:
            pass
    return completed


def run_one_instance(spec_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    spec = InstanceSpec(
        instance_id=spec_dict["instance_id"],
        scale=spec_dict["scale"],
        n_points=int(spec_dict["n_points"]),
        n_candidate_platforms=int(spec_dict.get("n_candidate_platforms", 8)),
        n_hazard_sources=int(spec_dict.get("n_hazard_sources", 3)),
        n_no_fly_zones=N_NO_FLY_ZONES,
        n_uav_types=N_UAV_TYPES,
        open_platforms_fixed=int(spec_dict.get("open_platforms_fixed", OPEN_PLATFORMS_FIXED)),
        seed=int(spec_dict["seed"]),
    )

    print("\n" + "=" * 110)
    print(f"Running {spec.instance_id}: J={spec.n_points}, I={spec.n_candidate_platforms}, "
          f"fixed_open={spec.open_platforms_fixed}, seed={spec.seed}")
    print("=" * 110)

    inst = build_instance(spec)
    rows: List[Dict[str, Any]] = load_instance_checkpoint(spec.instance_id)

    def append_and_checkpoint(row: Dict[str, Any]) -> None:
        # Replace an existing row with the same method/run seed; this prevents
        # duplicate statistics when an incomplete run is repeated.
        method = str(row.get("method"))
        run_seed = row.get("run_seed")
        filtered = []
        for old in rows:
            same_method = str(old.get("method")) == method
            old_seed = old.get("run_seed")
            if pd.isna(old_seed) and pd.isna(run_seed):
                same_seed = True
            else:
                try:
                    same_seed = int(float(old_seed)) == int(float(run_seed))
                except Exception:
                    same_seed = str(old_seed) == str(run_seed)
            if not (same_method and same_seed):
                filtered.append(old)
        rows[:] = filtered
        rows.append(row)
        save_instance_checkpoint(inst, rows)

    run_gurobi_this_instance = bool(
        spec_dict.get("run_gurobi", should_run_gurobi(spec.scale))
    )
    existing_gurobi = next((r for r in rows if _row_is_completed_gurobi(r)), None)
    if run_gurobi_this_instance and SKIP_COMPLETED_GUROBI and existing_gurobi is not None:
        print(
            f"[RESUME] Skip Gurobi for {spec.instance_id}; "
            f"existing status={existing_gurobi.get('status')}, obj={existing_gurobi.get('obj')}."
        )
    elif run_gurobi_this_instance:
        try:
            print("[Gurobi] Start full-path location-routing MILP benchmark...")
            gr = solve_gurobi_mip(inst)
            print(f"[Gurobi] status={gr['status']}, obj={gr['obj']}, lb={gr['lb']}, "
                  f"gap={gr['gap_percent']}, time={gr['time_sec']:.2f}s")
            append_and_checkpoint(make_raw_row(inst, gr, run_seed=None))
        except Exception as exc:
            traceback.print_exc()
            append_and_checkpoint(make_raw_row(inst, {
                "method": "Gurobi",
                "status": "ERROR",
                "obj": None,
                "lb": None,
                "gap_percent": None,
                "time_sec": None,
                "feasible": False,
                "coverage": None,
                "open_platforms": None,
                "n_uavs": {},
                "violation": {"error": str(exc)},
            }, run_seed=None))

    if RUN_ALNS_QSA:
        completed_seeds = _completed_alns_seeds(rows) if SKIP_COMPLETED_ALNS_SEEDS else set()
        for idx, seed in enumerate(ALNS_SEEDS[:ALNS_RUNS_PER_INSTANCE], start=1):
            if seed in completed_seeds:
                existing = next(
                    (r for r in rows
                     if str(r.get("method")) == "ALNS-Q-learning-SA"
                     and int(float(r.get("run_seed"))) == seed
                     and str(r.get("status", "")).upper() == "OK"),
                    None,
                )
                print(
                    f"[RESUME] Skip ALNS run {idx}/{ALNS_RUNS_PER_INSTANCE}, seed={seed}; "
                    f"existing obj={existing.get('obj') if existing else None}."
                )
                continue
            try:
                print(f"[ALNS-QSA] Run {idx}/{ALNS_RUNS_PER_INSTANCE}, seed={seed}...")
                ar = solve_alns_qsa(inst, random_seed=seed)
                print(f"[ALNS-QSA] seed={seed}, status={ar['status']}, obj={ar['obj']}, "
                      f"coverage={ar['coverage']:.3f}, time={ar['time_sec']:.2f}s")
                append_and_checkpoint(make_raw_row(inst, ar, run_seed=seed))
            except Exception as exc:
                traceback.print_exc()
                append_and_checkpoint(make_raw_row(inst, {
                    "method": "ALNS-Q-learning-SA",
                    "status": "ERROR",
                    "obj": None,
                    "fitness": None,
                    "time_sec": None,
                    "feasible": False,
                    "coverage": None,
                    "open_platforms": None,
                    "n_uavs": {},
                    "violation": {"error": str(exc)},
                }, run_seed=seed))

    return rows


def main() -> None:
    print("Supplementary UAV benchmark: M50 and L170")
    print(f"Gurobi available: {GUROBI_AVAILABLE}")
    print(f"Output directory: {RESULT_DIR}")
    enabled_specs = [x for x in INSTANCE_SPECS if x["scale"] in ENABLED_SCALES]
    if DEBUG_MODE:
        enabled_specs = [x for x in enabled_specs if x["instance_id"] in DEBUG_INSTANCE_IDS]
    print(f"Debug mode: {DEBUG_MODE}, debug instances: {sorted(DEBUG_INSTANCE_IDS) if DEBUG_MODE else 'ALL'}")
    print(f"Enabled scales: {ENABLED_SCALES}")
    print(f"Instances: {[x['instance_id'] for x in enabled_specs]}")
    print(f"Run Gurobi defaults: small={RUN_GUROBI_SMALL}, medium={RUN_GUROBI_MEDIUM}, large={RUN_GUROBI_LARGE}")
    print("Per-instance Gurobi flags:", {x["instance_id"]: x.get("run_gurobi") for x in enabled_specs})
    print(f"Resume={RESUME_FROM_CHECKPOINT}; skip Gurobi={SKIP_COMPLETED_GUROBI}; skip ALNS seeds={SKIP_COMPLETED_ALNS_SEEDS}")
    print(f"ALNS nominal limit: medium={ALNS_TIME_LIMIT_MEDIUM_SEC}s, large={ALNS_TIME_LIMIT_LARGE_SEC}s")
    print(f"ALNS guards: medium={ALNS_TIME_GUARD_MEDIUM_SEC}s, large={ALNS_TIME_GUARD_LARGE_SEC}s")

    t0 = time.time()
    all_rows: List[Dict[str, Any]] = []

    raw_csv_written = RAW_CSV
    for spec_dict in enabled_specs:
        rows = run_one_instance(spec_dict)
        all_rows.extend(rows)
        # Aggregate checkpoint after every completed instance. If the standard
        # file is occupied, safe_write_csv automatically creates a timestamped
        # fallback instead of terminating the experiment.
        raw_csv_written = safe_write_csv(
            pd.DataFrame(all_rows),
            RAW_CSV,
            label="aggregate raw-results checkpoint",
            index=False,
        )

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize(raw_df)
    scale_summary_df = summarize_by_scale(summary_df)
    paper_table_df = build_medium_large_paper_table(summary_df)

    raw_csv_written = safe_write_csv(raw_df, RAW_CSV, "raw results", index=False)
    summary_csv_written = safe_write_csv(summary_df, SUMMARY_CSV, "instance summary", index=False)
    scale_csv_written = safe_write_csv(scale_summary_df, SCALE_SUMMARY_CSV, "scale summary", index=False)
    paper_csv_written = safe_write_csv(paper_table_df, PAPER_TABLE_CSV, "paper table CSV", index=False)

    summary_xlsx_written = safe_write_excel(
        SUMMARY_XLSX,
        {
            "raw_results": raw_df,
            "summary_by_instance": summary_df,
            "summary_by_scale": scale_summary_df,
            "paper_medium_large": paper_table_df,
        },
        label="summary workbook",
    )

    paper_xlsx_written = safe_write_excel(
        PAPER_TABLE_XLSX,
        {"paper_medium_large": paper_table_df},
        label="paper-table workbook",
    )

    print("\n" + "=" * 110)
    print("Benchmark finished.")
    print(f"Raw CSV:      {raw_csv_written}")
    print(f"Summary CSV:  {summary_csv_written}")
    print(f"Scale CSV:    {scale_csv_written}")
    print(f"Paper CSV:    {paper_csv_written}")
    print(f"Excel file:   {summary_xlsx_written}")
    print(f"Paper Excel:  {paper_xlsx_written}")
    print(f"Checkpoints:  {CHECKPOINT_DIR}")
    print(f"Total runtime: {time.time() - t0:.2f}s")
    print("=" * 110)

    print("\nSummary preview:")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        print("\nMedium/large paper-table preview:")
        print(paper_table_df.to_string(index=False))
        print("\n[REPORTING NOTE] Medium/large results are scalability and stability results.")
        print("Improvement over Gurobi UB is reported only when Gurobi has a feasible incumbent; it is not a global optimality gap.")
        if "alns_below_gurobi_lb_flag" in summary_df.columns:
            bad = summary_df[summary_df["alns_below_gurobi_lb_flag"] == True]
            if not bad.empty:
                print("\n[WARNING] ALNS objective is still below Gurobi lower bound in the following instances:")
                print(bad[["instance_id", "gurobi_lb", "alns_best_obj", "alns_below_gurobi_lb_flag"]].to_string(index=False))
            else:
                print("\n[CHECK PASSED] No ALNS objective is below the Gurobi lower bound in the current run.")


if __name__ == "__main__":
    main()
