# -*- coding: utf-8 -*-
"""Shared TQ-ALNS benchmark core.

This module contains the common instance generator, integrated objective,
route decoding, feasibility checks, Greedy/ALNS/ALNS-SA/TQ-ALNS methods,
Q-learning, simulated annealing, and problem-specific feasibility repair
mechanisms used by the manuscript's configuration and robustness experiments.

It intentionally contains no component-removal comparison experiment.
"""
from __future__ import print_function

import copy
import csv
import itertools
import math
import os
import random
import statistics
import time

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ============================================================================
# 0. PARAMETER CONFIGURATION AREA -- change parameters here only
# ============================================================================

# ------------------------ experiment control ------------------------
INSTANCE_SEED = 7
SEARCH_SEEDS = list(range(1001, 1011))
ROBUST_INSTANCE_SEEDS = list(range(1, 11))   # Benchmark: 10 independent random instances
ROBUST_SEARCH_SEEDS = list(range(1001, 1011))
HEURISTIC_ITERS = 600
RUN_MODE = "uav_configuration_10instances"
PRINT_EVERY = 100
SAVE_VISUALIZATION = False
SAVE_Q_TRACE = True

# ------------------------ benchmark configuration controls ------------------------
CONFIG_INSTANCE_SEEDS = list(range(1, 11))
CONFIG_SEARCH_SEEDS = list(range(1001, 1011))
CONFIG_RUNS = 10
CONFIG_SAVE_Q_TRACE = False
# In every run all four configurations use the SAME instance seed and search seed.

# ------------------------ medium instance ----------------------------
AREA_W_M = 10000.0
AREA_H_M = 10000.0
N_CANDIDATE_PLATFORMS = 15
N_MONITORING_POINTS = 50
N_HAZARDS = 3
N_NOFLY_ZONES = 3

# Exact opened-platform count: choose locations from all candidate sites; the count is fixed for scale control.
MIN_OPEN_PLATFORMS = 4
MAX_OPEN_PLATFORMS = 4
MIN_COVERAGE = 0.90
SAFE_DISTANCE_HAZARD_M = 500.0
PLANNING_HORIZON_H = 16.0

# UAV fleet size/type remains an optimization decision under the following common bounds.
# Each open platform has 1..MAX_UAVS_PER_PLATFORM homogeneous UAV copies.
MIN_UAVS_PER_OPEN_PLATFORM = 1
MAX_UAVS_PER_PLATFORM = 3
MAX_TOTAL_UAVS = 10

# ------------------------ energy uncertainty ------------------------
BETA = 0.95
Z_BETA = 1.645

# ------------------------ original capital prices -------------------
PLATFORM_PURCHASE_COST_WAN = 25.0
ENERGY_COST_YUAN_PER_KM = 0.80
UNCOVERED_PENALTY_WAN_PER_POINT = 1.50

# ------------------------ planning-horizon equivalent capital cost --
# CRF = r(1+r)^L / ((1+r)^L - 1)
# Daily equivalent = purchase_cost * CRF / OPERATING_DAYS_PER_YEAR
# These are calibration assumptions and should be documented/cited before submission.
USE_PLANNING_HORIZON_EQUIVALENT_CAPEX = True
DISCOUNT_RATE = 0.08
PLATFORM_LIFE_YEARS = 10.0
DRONE_LIFE_YEARS = 5.0
OPERATING_DAYS_PER_YEAR = 365.0

# ------------------------ lower-level route cost --------------------
# Previously, 0.02 wan/h appeared only as a proxy in insertion scoring.
# Here it is explicitly included in the final route objective.
ROUTE_TIME_COST_WAN_PER_H = 0.020
ROUTE_MAKESPAN_COST_WAN_PER_H = 0.010
CHARGING_POWER_W = 1000.0
NOFLY_DETOUR_FACTOR = 1.40

# Route decoding settings.
ROUTE_NEAREST_CANDIDATES = 12
MAX_SORTIES_PER_UAV = 20

# ------------------------ task heterogeneity ------------------------
# 0=light capability, 1=medium, 2=heavy.
# Replace these shares with your actual monitoring-task composition if available.
TASK_CAPABILITY_SHARES = (0.65, 0.25, 0.10)

# name, purchase cost (wan), annual maintenance (wan), speed km/h,
# battery Wh, unit energy mean Wh/km, std Wh/km, single-leg radius km
DRONE_SPECS = [
    ("Type A - Light Inspection UAV",       12.0, 0.0,  60.0,  800.0, 20.0, 4.0,  5.0),
    ("Type B - Medium Long-Endurance UAV", 26.0, 0.0,  80.0, 1200.0, 25.0, 5.0,  7.0),
    ("Type C - Heavy-Duty UAV",            48.0, 0.0, 100.0, 2000.0, 35.0, 7.0, 10.0),
]

# ------------------------ SA ----------------------------------------
SA_INITIAL_RATIO = 0.10
SA_MIN_T0 = 0.20
SA_COOLING = 0.995
SA_REHEAT_AFTER = 100
SA_REHEAT_RATIO = 0.60

# ------------------------ true Q-learning ---------------------------
Q_ALPHA = 0.15
Q_GAMMA = 0.90
Q_EPSILON_START = 0.20
Q_EPSILON_MIN = 0.04
Q_EPSILON_DECAY = 0.998
STATE_RECENT_WINDOW = 20
STATE_STAG_MILD = 30
STATE_STAG_SEVERE = 100
REWARD_REL_IMPROVEMENT_SCALE = 2000.0
REWARD_NEW_BEST_BONUS = 5.0
REWARD_ACCEPTED_WORSE = -0.10
REWARD_INFEASIBLE = -2.00

# ------------------------ TQ-specific intensification ---------------
# Count all time in runtime. This is a real algorithm component, not hidden post-processing.
TQ_USE_ELITE_INTENSIFICATION = True
TQ_INTENSIFY_ROUNDS = 3
TQ_TASK_MOVE_SAMPLES = 80
TQ_PLATFORM_SWAP_SAMPLES = 50
TQ_RESOURCE_DOWNSIZE_PASSES = 2
IMPROVEMENT_TOL = 1e-10

# ============================================================================
# 1. data structures
# ============================================================================

class DroneType(object):
    def __init__(self, name, purchase_wan, maintenance_wan_year,
                 speed_kmh, battery_Wh, energy_mu_Wh_km, energy_sigma_Wh_km,
                 radius_km):
        self.name = str(name)
        self.purchase_wan = float(purchase_wan)
        self.maintenance_wan_year = float(maintenance_wan_year)
        self.speed_kmh = float(speed_kmh)
        self.battery_Wh = float(battery_Wh)
        self.energy_mu_Wh_km = float(energy_mu_Wh_km)
        self.energy_sigma_Wh_km = float(energy_sigma_Wh_km)
        self.radius_km = float(radius_km)


class Route(object):
    def __init__(self, platform, uav_index, uav_type, seq=None,
                 distance_km=0.0, flight_h=0.0, service_h=0.0,
                 charge_h=0.0, energy_Wh=0.0):
        self.platform = platform
        self.uav_index = uav_index
        self.uav_type = uav_type
        self.seq = list(seq or [])
        self.distance_km = float(distance_km)
        self.flight_h = float(flight_h)
        self.service_h = float(service_h)
        self.charge_h = float(charge_h)
        self.energy_Wh = float(energy_Wh)


class Solution(object):
    def __init__(self):
        self.open_platform = {}            # i -> 0/1
        self.platform_type = {}            # i -> k
        self.platform_uav_count = {}       # i -> number of copies, 0 if closed
        self.assign_j = {}                 # j -> platform i or None
        self.routes = {}                   # i -> list[Route]
        self.covered = {}

        self.obj = 1e100
        self.feasible = False
        self.coverage_ratio = 0.0
        self.runtime_s = 0.0
        self.time_to_best_s = 0.0

        # objective decomposition
        self.platform_capex_wan = 0.0
        self.drone_capex_wan = 0.0
        self.route_energy_cost_wan = 0.0
        self.route_time_cost_wan = 0.0
        self.route_makespan_cost_wan = 0.0
        self.uncovered_penalty_wan = 0.0
        self.total_route_distance_km = 0.0
        self.total_route_time_h = 0.0
        self.route_makespan_h = 0.0
        self.total_uavs = 0
        self.unserved_visits = 0

        # Q-learning diagnostics
        self.q_states_visited = 0
        self.q_updates = 0


class Instance(object):
    pass


# ============================================================================
# 2. cost conversion and geometry
# ============================================================================

def capital_recovery_factor(rate, life_years):
    rate = float(rate)
    life_years = float(life_years)
    if life_years <= 0:
        raise ValueError("life_years must be positive")
    if abs(rate) < 1e-12:
        return 1.0 / life_years
    x = (1.0 + rate) ** life_years
    return rate * x / (x - 1.0)


def equivalent_planning_cost_wan(purchase_wan, annual_maintenance_wan,
                                 life_years):
    if not USE_PLANNING_HORIZON_EQUIVALENT_CAPEX:
        return float(purchase_wan) + float(annual_maintenance_wan) / OPERATING_DAYS_PER_YEAR
    crf = capital_recovery_factor(DISCOUNT_RATE, life_years)
    annualized = float(purchase_wan) * crf + float(annual_maintenance_wan)
    return annualized / OPERATING_DAYS_PER_YEAR


def _orient(ax, ay, bx, by, cx, cy):
    return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)


def _on_segment(ax, ay, bx, by, cx, cy):
    return (min(ax,bx)-1e-12 <= cx <= max(ax,bx)+1e-12 and
            min(ay,by)-1e-12 <= cy <= max(ay,by)+1e-12)


def segments_intersect(a, b, c, d):
    ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
    o1 = _orient(ax, ay, bx, by, cx, cy)
    o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay)
    o4 = _orient(cx, cy, dx, dy, bx, by)
    if abs(o1) < 1e-12 and _on_segment(ax,ay,bx,by,cx,cy): return True
    if abs(o2) < 1e-12 and _on_segment(ax,ay,bx,by,dx,dy): return True
    if abs(o3) < 1e-12 and _on_segment(cx,cy,dx,dy,ax,ay): return True
    if abs(o4) < 1e-12 and _on_segment(cx,cy,dx,dy,bx,by): return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def point_in_poly(p, poly):
    px, py = p
    inside = False
    n = len(poly)
    for t in range(n):
        x1,y1 = poly[t]
        x2,y2 = poly[(t+1) % n]
        if abs(_orient(x1,y1,x2,y2,px,py)) < 1e-12 and _on_segment(x1,y1,x2,y2,px,py):
            return True
        if (y1 > py) != (y2 > py):
            xint = x1 + (py-y1)*(x2-x1)/float(y2-y1)
            if xint > px:
                inside = not inside
    return inside


def segment_intersects_poly(a, b, poly):
    if point_in_poly(a, poly) or point_in_poly(b, poly):
        return True
    for t in range(len(poly)):
        if segments_intersect(a, b, poly[t], poly[(t+1) % len(poly)]):
            return True
    return False


def euclid_km(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1]) / 1000.0


def safe_distance_km(a, b, nofly):
    base = euclid_km(a, b)
    cross = sum(1 for poly in nofly if segment_intersects_poly(a, b, poly))
    return base * (NOFLY_DETOUR_FACTOR if cross > 0 else 1.0)


def _sample_outside(rng, nofly, hazards, min_hazard_m=0.0, max_tries=50000):
    for _ in range(max_tries):
        p = (rng.uniform(0, AREA_W_M), rng.uniform(0, AREA_H_M))
        if any(point_in_poly(p, poly) for poly in nofly):
            continue
        if min_hazard_m > 0 and any(math.hypot(p[0]-h[0], p[1]-h[1]) < min_hazard_m for h in hazards):
            continue
        return p
    raise RuntimeError("Unable to generate a feasible point")


# ============================================================================
# 3. instance generator -- one fixed instance shared by every algorithm
# ============================================================================

def build_instance(seed=INSTANCE_SEED):
    rng = random.Random(int(seed))
    inst = Instance()
    inst.I = N_CANDIDATE_PLATFORMS
    inst.J = N_MONITORING_POINTS
    inst.Hz = N_HAZARDS
    inst.Z = N_NOFLY_ZONES
    inst.MIN_OPEN = MIN_OPEN_PLATFORMS
    inst.MAX_OPEN = MAX_OPEN_PLATFORMS
    inst.min_cover_ratio = MIN_COVERAGE
    inst.horizon_h = PLANNING_HORIZON_H
    inst.beta = BETA
    inst.z_beta = Z_BETA

    # No-fly zones.
    inst.nofly = []
    for _ in range(inst.Z):
        cx = rng.uniform(1800.0, AREA_W_M-1800.0)
        cy = rng.uniform(1800.0, AREA_H_M-1800.0)
        w = rng.uniform(900.0, 1700.0)
        h = rng.uniform(900.0, 1700.0)
        inst.nofly.append([
            (cx-w/2, cy-h/2), (cx+w/2, cy-h/2),
            (cx+w/2, cy+h/2), (cx-w/2, cy+h/2)
        ])

    inst.hazards = []
    for _ in range(inst.Hz):
        inst.hazards.append(_sample_outside(rng, inst.nofly, [], 0.0))

    inst.platform_xy = []
    while len(inst.platform_xy) < inst.I:
        p = _sample_outside(rng, inst.nofly, inst.hazards, SAFE_DISTANCE_HAZARD_M)
        if all(math.hypot(p[0]-q[0], p[1]-q[1]) >= 250.0 for q in inst.platform_xy):
            inst.platform_xy.append(p)
    inst.platform_valid = [True] * inst.I
    inst.valid_platform_indices = list(range(inst.I))

    inst.point_xy = []
    while len(inst.point_xy) < inst.J:
        inst.point_xy.append(_sample_outside(rng, inst.nofly, inst.hazards, 120.0))

    inst.req_freq = [rng.randint(1,3) for _ in range(inst.J)]
    inst.inspect_time_h = [rng.uniform(0.08, 0.30) for _ in range(inst.J)]

    # Task capability requirement activates heterogeneous fleet decisions.
    p0, p1, p2 = TASK_CAPABILITY_SHARES
    if abs((p0+p1+p2) - 1.0) > 1e-9:
        raise ValueError("TASK_CAPABILITY_SHARES must sum to 1")
    inst.task_required_type = []
    for _ in range(inst.J):
        u = rng.random()
        if u < p0:
            inst.task_required_type.append(0)
        elif u < p0+p1:
            inst.task_required_type.append(1)
        else:
            inst.task_required_type.append(2)

    inst.drone_types = [DroneType(*x) for x in DRONE_SPECS]
    inst.K = len(inst.drone_types)

    inst.platform_equiv_wan = equivalent_planning_cost_wan(
        PLATFORM_PURCHASE_COST_WAN, 0.0, PLATFORM_LIFE_YEARS)
    inst.drone_equiv_wan = [
        equivalent_planning_cost_wan(dt.purchase_wan, dt.maintenance_wan_year, DRONE_LIFE_YEARS)
        for dt in inst.drone_types
    ]

    # Platform-point and point-point safe distances.
    inst.d_ip = [[0.0]*inst.J for _ in range(inst.I)]
    for i in range(inst.I):
        for j in range(inst.J):
            inst.d_ip[i][j] = safe_distance_km(inst.platform_xy[i], inst.point_xy[j], inst.nofly)

    inst.d_pp = [[0.0]*inst.J for _ in range(inst.J)]
    for j in range(inst.J):
        for l in range(j+1, inst.J):
            d = safe_distance_km(inst.point_xy[j], inst.point_xy[l], inst.nofly)
            inst.d_pp[j][l] = d
            inst.d_pp[l][j] = d

    return inst


# ============================================================================
# 4. solution helpers
# ============================================================================

def clone_solution(sol):
    return copy.deepcopy(sol)


def open_set(sol):
    return [i for i,v in sol.open_platform.items() if v == 1]


def total_uavs(sol):
    return sum(sol.platform_uav_count.get(i,0) for i in open_set(sol))


def _allowed_types(inst):
    return sorted(getattr(inst, "allowed_types", list(range(inst.K))))


def _default_type(inst):
    allowed = _allowed_types(inst)
    return allowed[0]


def _previous_allowed_type(inst, k):
    allowed = _allowed_types(inst)
    lower = [x for x in allowed if x < k]
    return max(lower) if lower else k


def min_type_for_point(inst, i, j):
    """Minimum ALLOWED UAV type feasible for point j from platform i."""
    d = inst.d_ip[i][j]
    task_req = inst.task_required_type[j]
    for k in _allowed_types(inst):
        if k < task_req:
            continue
        dt = inst.drone_types[k]
        if d > dt.radius_km + 1e-9:
            continue
        q = dt.energy_mu_Wh_km + inst.z_beta * dt.energy_sigma_Wh_km
        if q * (2.0*d) <= dt.battery_Wh + 1e-9:
            return k
    return None

def distance_between(inst, i, prev_j, next_j):
    if prev_j is None:
        return inst.d_ip[i][next_j]
    return inst.d_pp[prev_j][next_j]


def return_distance(inst, i, j):
    return inst.d_ip[i][j]


def _task_visits_for_platform(inst, sol, i):
    visits = []
    for j in range(inst.J):
        if sol.assign_j.get(j, None) == i:
            for rep in range(inst.req_freq[j]):
                visits.append((j, rep))
    return visits


# ============================================================================
# 5. real lower-level route decoder
# ============================================================================

def _sortie_can_append(inst, i, k, current_j, j,
                       sortie_distance, sortie_energy_Wh, sortie_service_h):
    dt = inst.drone_types[k]
    leg = distance_between(inst, i, current_j, j)
    back = return_distance(inst, i, j)
    q = dt.energy_mu_Wh_km + inst.z_beta * dt.energy_sigma_Wh_km
    new_energy_with_return = sortie_energy_Wh + q*leg + q*back
    if new_energy_with_return > dt.battery_Wh + 1e-9:
        return False
    # Single-leg practical range restriction.
    if leg > 2.0*dt.radius_km + 1e-9 or back > dt.radius_km + 1e-9:
        return False
    return True


def decode_platform_routes(inst, sol, i):
    """Deterministic greedy lower-level decoder for one platform.

    Returns dict with route metrics and unserved visits. Each UAV may perform multiple
    platform-to-platform sorties and recharge between sorties. Safe return is checked
    at every insertion using the beta-quantile unit-energy consumption.
    """
    if sol.open_platform.get(i,0) != 1:
        return dict(routes=[], unserved=[], distance_km=0.0, total_time_h=0.0,
                    makespan_h=0.0, energy_Wh=0.0, feasible=True)

    k = int(sol.platform_type.get(i,0))
    n_uav = max(1, int(sol.platform_uav_count.get(i,1)))
    dt = inst.drone_types[k]
    q = dt.energy_mu_Wh_km + inst.z_beta * dt.energy_sigma_Wh_km

    visits = _task_visits_for_platform(inst, sol, i)
    # Remove visits that are impossible for the selected type.
    remaining = []
    impossible = []
    for v in visits:
        j = v[0]
        if inst.task_required_type[j] > k or min_type_for_point(inst, i, j) is None or min_type_for_point(inst, i, j) > k:
            impossible.append(v)
        else:
            remaining.append(v)

    routes = []
    uav_work = [0.0 for _ in range(n_uav)]
    total_distance = 0.0
    total_energy = 0.0

    # Harder/farther visits are inserted first across UAVs, then nearest-neighbor within sortie.
    remaining.sort(key=lambda v: (inst.task_required_type[v[0]], inst.d_ip[i][v[0]]), reverse=True)

    for u in range(n_uav):
        sortie_count = 0
        while remaining and sortie_count < MAX_SORTIES_PER_UAV:
            if uav_work[u] >= inst.horizon_h - 1e-9:
                break
            sortie_count += 1
            seq = []
            current_j = None
            sortie_distance = 0.0
            sortie_energy = 0.0
            sortie_flight_h = 0.0
            sortie_service_h = 0.0

            while remaining:
                # Candidate list sorted by distance from current location.
                cand = sorted(
                    remaining,
                    key=lambda v: distance_between(inst, i, current_j, v[0])
                )[:ROUTE_NEAREST_CANDIDATES]

                chosen = None
                chosen_leg = None
                for v in cand:
                    j = v[0]
                    if not _sortie_can_append(inst, i, k, current_j, j,
                                              sortie_distance, sortie_energy, sortie_service_h):
                        continue
                    leg = distance_between(inst, i, current_j, j)
                    back = return_distance(inst, i, j)
                    trial_distance = sortie_distance + leg + back
                    trial_flight = trial_distance / dt.speed_kmh
                    trial_service = sortie_service_h + inst.inspect_time_h[j]
                    trial_energy = q * trial_distance
                    # If this closes the sortie now, can the UAV still remain within the day?
                    trial_work = uav_work[u] + trial_flight + trial_service
                    if trial_work <= inst.horizon_h + 1e-9:
                        chosen = v
                        chosen_leg = leg
                        break

                if chosen is None:
                    break

                j = chosen[0]
                seq.append(j)
                remaining.remove(chosen)
                sortie_distance += chosen_leg
                sortie_energy += q * chosen_leg
                sortie_flight_h += chosen_leg / dt.speed_kmh
                sortie_service_h += inst.inspect_time_h[j]
                current_j = j

            if not seq:
                break

            # Safe return.
            back = return_distance(inst, i, current_j)
            sortie_distance += back
            sortie_energy += q * back
            sortie_flight_h += back / dt.speed_kmh

            # Recharge after a sortie only if work remains for this UAV / other sorties may follow.
            charge_h = sortie_energy / CHARGING_POWER_W if remaining else 0.0
            # If recharge makes this UAV exceed the horizon, keep the completed sortie but stop using it.
            work_add = sortie_flight_h + sortie_service_h
            uav_work[u] += work_add
            if remaining and uav_work[u] + charge_h <= inst.horizon_h + 1e-9:
                uav_work[u] += charge_h
            else:
                charge_h = 0.0

            r = Route(i, u, k, seq, sortie_distance, sortie_flight_h,
                      sortie_service_h, charge_h, sortie_energy)
            routes.append(r)
            total_distance += sortie_distance
            total_energy += sortie_energy

    unserved = impossible + remaining
    return dict(
        routes=routes,
        unserved=unserved,
        distance_km=total_distance,
        total_time_h=sum(uav_work),
        makespan_h=max(uav_work) if uav_work else 0.0,
        energy_Wh=total_energy,
        feasible=(len(unserved) == 0)
    )


def recompute_required_types(inst, sol):
    """Choose the lowest allowed homogeneous type able to serve all tasks of each platform.

    If no allowed type can serve all assigned tasks, keep the highest allowed type; the
    strict route decoder will mark the solution infeasible. This prevents a homogeneous
    experiment from silently switching to a forbidden UAV type.
    """
    allowed = _allowed_types(inst)
    for i in open_set(sol):
        assigned = [j for j in range(inst.J) if sol.assign_j.get(j,None)==i]
        if not assigned:
            sol.platform_type[i] = allowed[0]
            continue
        chosen = None
        for k in allowed:
            ok = True
            for j in assigned:
                if k < inst.task_required_type[j]:
                    ok = False; break
                d = inst.d_ip[i][j]
                dt = inst.drone_types[k]
                q = dt.energy_mu_Wh_km + inst.z_beta * dt.energy_sigma_Wh_km
                if d > dt.radius_km + 1e-9 or q*(2.0*d) > dt.battery_Wh + 1e-9:
                    ok = False; break
            if ok:
                chosen = k; break
        sol.platform_type[i] = chosen if chosen is not None else allowed[-1]

def compute_solution_obj(inst, sol):
    opens = open_set(sol)
    feasible = True

    if len(opens) < inst.MIN_OPEN or len(opens) > inst.MAX_OPEN:
        feasible = False
    if total_uavs(sol) > MAX_TOTAL_UAVS:
        feasible = False
    for i in opens:
        n = int(sol.platform_uav_count.get(i,0))
        if n < MIN_UAVS_PER_OPEN_PLATFORM or n > MAX_UAVS_PER_PLATFORM:
            feasible = False

    # Assignment coverage at point level.
    covered = {}
    for j in range(inst.J):
        i = sol.assign_j.get(j,None)
        covered[j] = int(i is not None and i in opens)
    sol.covered = covered
    sol.coverage_ratio = sum(covered.values()) / float(inst.J)
    if sol.coverage_ratio + 1e-12 < inst.min_cover_ratio:
        feasible = False

    recompute_required_types(inst, sol)

    # Capital cost in planning-horizon equivalent units.
    platform_cost = len(opens) * inst.platform_equiv_wan
    drone_cost = 0.0
    for i in opens:
        k = int(sol.platform_type.get(i,0))
        n = int(sol.platform_uav_count.get(i,1))
        drone_cost += n * inst.drone_equiv_wan[k]

    # Real route decoding.
    sol.routes = {}
    total_distance = 0.0
    total_route_time = 0.0
    max_makespan = 0.0
    unserved_visits = 0
    route_feasible = True
    for i in opens:
        dec = decode_platform_routes(inst, sol, i)
        sol.routes[i] = dec["routes"]
        total_distance += dec["distance_km"]
        total_route_time += dec["total_time_h"]
        max_makespan = max(max_makespan, dec["makespan_h"])
        unserved_visits += len(dec["unserved"])
        if not dec["feasible"]:
            route_feasible = False

    if not route_feasible:
        feasible = False

    # A visit that cannot be scheduled is penalized in addition to point-level noncoverage.
    uncovered_points = sum(1 for j in range(inst.J) if covered[j] == 0)
    route_energy_cost = ENERGY_COST_YUAN_PER_KM * total_distance / 10000.0
    route_time_cost = ROUTE_TIME_COST_WAN_PER_H * total_route_time
    makespan_cost = ROUTE_MAKESPAN_COST_WAN_PER_H * max_makespan
    uncovered_penalty = UNCOVERED_PENALTY_WAN_PER_POINT * (uncovered_points + unserved_visits)

    obj = (platform_cost + drone_cost + route_energy_cost + route_time_cost +
           makespan_cost + uncovered_penalty)

    sol.platform_capex_wan = platform_cost
    sol.drone_capex_wan = drone_cost
    sol.route_energy_cost_wan = route_energy_cost
    sol.route_time_cost_wan = route_time_cost
    sol.route_makespan_cost_wan = makespan_cost
    sol.uncovered_penalty_wan = uncovered_penalty
    sol.total_route_distance_km = total_distance
    sol.total_route_time_h = total_route_time
    sol.route_makespan_h = max_makespan
    sol.total_uavs = total_uavs(sol)
    sol.unserved_visits = unserved_visits
    sol.obj = obj
    sol.feasible = feasible
    return obj, feasible


# ============================================================================
# 6. construction and feasibility repair
# ============================================================================

def platform_score(inst, i):
    # lower is better: average distance to monitoring points + capex-equivalent term
    dbar = sum(inst.d_ip[i]) / float(inst.J)
    return dbar + 3.0*inst.platform_equiv_wan


def assign_all_points_by_distance(inst, sol, only_unassigned=False):
    opens = open_set(sol)
    if not opens:
        return
    for j in range(inst.J):
        if only_unassigned and sol.assign_j.get(j,None) is not None:
            continue
        cand = []
        for i in opens:
            mk = min_type_for_point(inst, i, j)
            if mk is None:
                continue
            # Distance + marginal daily UAV type cost proxy.
            c = inst.d_ip[i][j] + 5.0*inst.drone_equiv_wan[mk]
            cand.append((c, i))
        sol.assign_j[j] = min(cand)[1] if cand else None


def estimate_platform_uav_need(inst, sol, i):
    """Cheap workload estimator used by repair before the exact route decoder."""
    js = [j for j in range(inst.J) if sol.assign_j.get(j,None) == i]
    if not js:
        return 1
    k = int(sol.platform_type.get(i,0))
    dt = inst.drone_types[k]
    workload = 0.0
    for j in js:
        freq = inst.req_freq[j]
        workload += freq * (2.0*inst.d_ip[i][j]/dt.speed_kmh + inst.inspect_time_h[j])
    need = int(math.ceil(workload / max(1e-9, 0.80*inst.horizon_h)))
    return max(1, min(MAX_UAVS_PER_PLATFORM, need))


def make_initial_solution(inst, seed=7):
    rng = random.Random(seed)
    sol = Solution()
    default_k = _default_type(inst)
    for i in range(inst.I):
        sol.open_platform[i] = 0
        sol.platform_uav_count[i] = 0
    ranked = sorted(inst.valid_platform_indices, key=lambda i: platform_score(inst,i))
    chosen = ranked[:inst.MIN_OPEN]
    for i in chosen:
        sol.open_platform[i] = 1
        sol.platform_type[i] = default_k
        sol.platform_uav_count[i] = 1

    assign_all_points_by_distance(inst, sol, only_unassigned=False)
    recompute_required_types(inst, sol)
    for i in chosen:
        sol.platform_uav_count[i] = estimate_platform_uav_need(inst, sol, i)

    while total_uavs(sol) > MAX_TOTAL_UAVS:
        reducible = [i for i in open_set(sol) if sol.platform_uav_count.get(i,1) > 1]
        if not reducible:
            break
        reducible.sort(key=lambda i: sum(1 for j in range(inst.J) if sol.assign_j.get(j,None)==i))
        sol.platform_uav_count[reducible[0]] -= 1

    compute_solution_obj(inst, sol)
    return targeted_feasibility_repair(inst, sol, rng, max_rounds=4)

def _open_best_new_platform(inst, sol, rng):
    if len(open_set(sol)) >= inst.MAX_OPEN:
        return False
    closed = [i for i in inst.valid_platform_indices if sol.open_platform.get(i,0)==0]
    if not closed:
        return False
    current_opens = open_set(sol)
    def gain_score(i):
        gain = 0.0
        for j in range(inst.J):
            old = min((inst.d_ip[o][j] for o in current_opens), default=1e9)
            new = inst.d_ip[i][j]
            gain += max(0.0, old-new)
        return gain
    best = max(closed, key=gain_score)
    sol.open_platform[best] = 1
    sol.platform_type[best] = _default_type(inst)
    sol.platform_uav_count[best] = 1
    return True

def targeted_feasibility_repair(inst, sol, rng, max_rounds=4):
    s = clone_solution(sol)

    # Ensure minimum platforms.
    while len(open_set(s)) < inst.MIN_OPEN:
        if not _open_best_new_platform(inst, s, rng):
            break

    # Closed-platform assignments become unassigned.
    opens = set(open_set(s))
    for j in range(inst.J):
        if s.assign_j.get(j,None) not in opens:
            s.assign_j[j] = None
    assign_all_points_by_distance(inst, s, only_unassigned=True)

    for _ in range(max_rounds):
        recompute_required_types(inst, s)
        for i in open_set(s):
            s.platform_uav_count[i] = max(1, min(MAX_UAVS_PER_PLATFORM,
                                                  int(s.platform_uav_count.get(i,1))))
        compute_solution_obj(inst, s)
        if s.feasible:
            return s

        # First add UAV capacity on platforms with the largest assigned visit load.
        overloaded = sorted(
            open_set(s),
            key=lambda i: sum(inst.req_freq[j] for j in range(inst.J) if s.assign_j.get(j,None)==i),
            reverse=True
        )
        changed = False
        for i in overloaded:
            if total_uavs(s) >= MAX_TOTAL_UAVS:
                break
            if s.platform_uav_count.get(i,1) < MAX_UAVS_PER_PLATFORM:
                s.platform_uav_count[i] += 1
                changed = True
                break
        if changed:
            continue

        # Then add a platform and reassign globally.
        if _open_best_new_platform(inst, s, rng):
            assign_all_points_by_distance(inst, s, only_unassigned=False)
            continue

        # Last resort: reassign globally.
        assign_all_points_by_distance(inst, s, only_unassigned=False)

    compute_solution_obj(inst, s)
    return s


# ============================================================================
# 7. destroy and repair operators
# ============================================================================

def destruction_ratio(rng):
    return rng.uniform(0.10,0.30)


def destroy_platform_removal(inst, sol, rng):
    """Exact-open location search via one-for-one platform swaps."""
    s = clone_solution(sol)
    opens = open_set(s)
    if not opens:
        return s
    if inst.MIN_OPEN == inst.MAX_OPEN and len(opens) == inst.MIN_OPEN:
        closed = [i for i in inst.valid_platform_indices if i not in opens]
        if not closed:
            return s
        scored = []
        for i in opens:
            load = sum(inst.req_freq[j] for j in range(inst.J) if s.assign_j.get(j,None)==i)
            scored.append(((1.0+inst.platform_equiv_wan)/(1.0+load),i))
        scored.sort(reverse=True)
        out_i = scored[0][1] if rng.random()<0.80 else rng.choice(opens)
        sample_js = list(range(inst.J)); rng.shuffle(sample_js); sample_js=sample_js[:80]
        ranked=[]
        for i in closed:
            ds=[inst.d_ip[i][j] for j in sample_js if min_type_for_point(inst,i,j) is not None]
            ranked.append((sum(ds)/float(len(ds)) if ds else 1e100,i))
        ranked.sort(); in_i=ranked[0][1] if rng.random()<0.80 else rng.choice(closed)
        moved=max(MIN_UAVS_PER_OPEN_PLATFORM,min(MAX_UAVS_PER_PLATFORM,s.platform_uav_count.get(out_i,1)))
        s.open_platform[out_i]=0; s.platform_uav_count[out_i]=0; s.platform_type.pop(out_i,None)
        s.open_platform[in_i]=1; s.platform_uav_count[in_i]=moved; s.platform_type[in_i]=_default_type(inst)
        for j in range(inst.J):
            if s.assign_j.get(j,None)==out_i: s.assign_j[j]=None
        return s
    return s

def destroy_uav_reduction(inst, sol, rng):
    s = clone_solution(sol)
    opens = open_set(s); rng.shuffle(opens)
    for i in opens:
        if s.platform_uav_count.get(i,1) > 1:
            s.platform_uav_count[i] -= 1
            return s
        k = s.platform_type.get(i,_default_type(inst))
        prev = _previous_allowed_type(inst,k)
        if prev < k:
            s.platform_type[i] = prev
            return s
    return s

def destroy_task_release(inst, sol, rng):
    s = clone_solution(sol)
    assigned = [j for j in range(inst.J) if s.assign_j.get(j,None) is not None]
    if not assigned:
        return s
    m = max(1, int(destruction_ratio(rng)*len(assigned)))
    # High-distance assignments first, with some randomness.
    assigned.sort(key=lambda j: inst.d_ip[s.assign_j[j]][j], reverse=True)
    pool = assigned[:max(m, int(0.50*len(assigned)))]
    rng.shuffle(pool)
    for j in pool[:m]:
        s.assign_j[j] = None
    return s


def destroy_route_pressure(inst, sol, rng):
    s = clone_solution(sol)
    # Use actual last decoded work by platform when available.
    opens = open_set(s)
    if not opens:
        return s
    score = []
    for i in opens:
        dist = sum(r.distance_km for r in s.routes.get(i,[]))
        score.append((dist, i))
    score.sort(reverse=True)
    i = score[0][1]
    js = [j for j in range(inst.J) if s.assign_j.get(j,None)==i]
    js.sort(key=lambda j: inst.d_ip[i][j], reverse=True)
    m = max(1, int(destruction_ratio(rng)*max(1,len(js))))
    for j in js[:m]:
        s.assign_j[j] = None
    return s


def set_repair_flags(inst, nfz=True, energy=True, compatibility=True):
    """Enable/disable only targeted repair guidance; objective and constraints remain unchanged."""
    inst.repair_flags = {
        "nfz": bool(nfz),
        "energy": bool(energy),
        "compatibility": bool(compatibility),
    }
    return inst


def _repair_enabled(inst, name):
    return bool(getattr(
        inst,
        "repair_flags",
        {"nfz": True, "energy": True, "compatibility": True},
    ).get(name, True))


def _direct_ip_km(inst, i, j):
    a = inst.platform_xy[i]
    b = inst.point_xy[j]
    return math.hypot(a[0]-b[0], a[1]-b[1]) / 1000.0


def _nfz_guided_move(inst, s, rng, max_points=4):
    """Try high-detour assignments first; accept only strict feasible improvement."""
    compute_solution_obj(inst, s)
    best = clone_solution(s)
    opens = open_set(best)
    scored = []
    for j in range(inst.J):
        old = best.assign_j.get(j, None)
        if old is None:
            continue
        ratio = inst.d_ip[old][j] / max(1e-9, _direct_ip_km(inst, old, j))
        scored.append((ratio, j))
    scored.sort(reverse=True)
    for _, j in scored[:max_points]:
        old = best.assign_j.get(j, None)
        candidates = [x for x in opens if x != old and min_type_for_point(inst, x, j) is not None]
        candidates = sorted(candidates, key=lambda x: inst.d_ip[x][j])[:2]
        for i in candidates:
            cand = clone_solution(best)
            cand.assign_j[j] = i
            compute_solution_obj(inst, cand)
            if cand.feasible and cand.obj < best.obj - IMPROVEMENT_TOL:
                best = cand
                break
    return best


def _energy_guided_move(inst, s, rng, max_points=4):
    """Move energy-tight visits to assignments with larger beta-quantile energy margin."""
    compute_solution_obj(inst, s)
    best = clone_solution(s)
    opens = open_set(best)
    pressure = []
    for j in range(inst.J):
        old = best.assign_j.get(j, None)
        if old is None:
            continue
        k = best.platform_type.get(old, _default_type(inst))
        dt = inst.drone_types[k]
        q = dt.energy_mu_Wh_km + inst.z_beta * dt.energy_sigma_Wh_km
        margin = dt.battery_Wh - q * (2.0 * inst.d_ip[old][j])
        pressure.append((margin, j))
    pressure.sort()
    for _, j in pressure[:max_points]:
        old = best.assign_j.get(j, None)
        candidates = []
        for i in opens:
            if i == old:
                continue
            mk = min_type_for_point(inst, i, j)
            if mk is None:
                continue
            dt = inst.drone_types[mk]
            q = dt.energy_mu_Wh_km + inst.z_beta * dt.energy_sigma_Wh_km
            margin = dt.battery_Wh - q * (2.0 * inst.d_ip[i][j])
            candidates.append((-margin, inst.d_ip[i][j], i))
        for _, _, i in sorted(candidates)[:2]:
            cand = clone_solution(best)
            cand.assign_j[j] = i
            compute_solution_obj(inst, cand)
            if cand.feasible and cand.obj < best.obj - IMPROVEMENT_TOL:
                best = cand
                break
    return best


def _compatibility_guided_move(inst, s, rng, max_points=4):
    """Consolidate high-capability tasks to reduce unnecessary platform-wide type upgrades."""
    compute_solution_obj(inst, s)
    best = clone_solution(s)
    opens = open_set(best)
    js = [j for j in range(inst.J) if best.assign_j.get(j, None) is not None]
    js.sort(key=lambda j: inst.task_required_type[j], reverse=True)
    for j in js[:max_points]:
        old = best.assign_j.get(j, None)
        cand_sites = []
        for i in opens:
            if i == old:
                continue
            mk = min_type_for_point(inst, i, j)
            if mk is None:
                continue
            cand_sites.append((max(0, mk-best.platform_type.get(i, _default_type(inst))), inst.d_ip[i][j], i))
        for _, _, i in sorted(cand_sites)[:2]:
            cand = clone_solution(best)
            cand.assign_j[j] = i
            compute_solution_obj(inst, cand)
            if cand.feasible and cand.obj < best.obj - IMPROVEMENT_TOL:
                best = cand
                break
    return best


def repair_platform(inst, sol, rng):
    s = clone_solution(sol)
    while len(open_set(s)) < inst.MIN_OPEN:
        if not _open_best_new_platform(inst, s, rng):
            break
    # If many tasks are still unassigned, add one platform if allowed.
    unassigned = sum(1 for j in range(inst.J) if s.assign_j.get(j,None) is None)
    if unassigned > 0.05*inst.J and len(open_set(s)) < inst.MAX_OPEN:
        _open_best_new_platform(inst, s, rng)
    assign_all_points_by_distance(inst, s, only_unassigned=True)
    return targeted_feasibility_repair(inst, s, rng, max_rounds=2)


def repair_uav(inst, sol, rng):
    s = clone_solution(sol)
    assign_all_points_by_distance(inst, s, only_unassigned=True)
    recompute_required_types(inst, s)
    if _repair_enabled(inst, "energy"):
        opens = open_set(s)
        loads = sorted(
            opens,
            key=lambda i: sum(inst.req_freq[j] for j in range(inst.J) if s.assign_j.get(j, None) == i),
            reverse=True,
        )
        for i in loads:
            need = estimate_platform_uav_need(inst, s, i)
            if need > s.platform_uav_count.get(i, 1) and total_uavs(s) < MAX_TOTAL_UAVS:
                s.platform_uav_count[i] += 1
                break
    s = targeted_feasibility_repair(inst, s, rng, max_rounds=2)
    if _repair_enabled(inst, "energy"):
        s = _energy_guided_move(inst, s, rng, max_points=4)
    return s


def repair_assignment(inst, sol, rng):
    s = clone_solution(sol)
    assign_all_points_by_distance(inst, s, only_unassigned=True)
    s = targeted_feasibility_repair(inst, s, rng, max_rounds=2)
    if _repair_enabled(inst, "compatibility"):
        s = _compatibility_guided_move(inst, s, rng, max_points=4)
    return s


def _surrogate_move_gain(inst, sol, j, new_i):
    old_i = sol.assign_j.get(j,None)
    if old_i is None or new_i == old_i:
        return 0.0
    # Positive = estimated improvement.
    old_d = inst.d_ip[old_i][j]
    new_d = inst.d_ip[new_i][j]
    return old_d - new_d


def repair_route_rebalance(inst, sol, rng):
    s = clone_solution(sol)
    assign_all_points_by_distance(inst, s, only_unassigned=True)
    opens = open_set(s)
    js = list(range(inst.J))
    rng.shuffle(js)
    for j in js[:min(40, inst.J)]:
        old = s.assign_j.get(j, None)
        if old is None:
            continue
        best = old
        best_gain = 0.0
        for i in opens:
            if i == old or min_type_for_point(inst, i, j) is None:
                continue
            g = _surrogate_move_gain(inst, s, j, i)
            if g > best_gain:
                best_gain = g
                best = i
        if best != old:
            s.assign_j[j] = best
    s = targeted_feasibility_repair(inst, s, rng, max_rounds=2)
    if _repair_enabled(inst, "nfz"):
        s = _nfz_guided_move(inst, s, rng, max_points=4)
    return s


DESTROY_OPS = {
    "DP-platform-removal": destroy_platform_removal,
    "DU-uav-reduction": destroy_uav_reduction,
    "DA-task-release": destroy_task_release,
    "DR-route-pressure": destroy_route_pressure,
}
REPAIR_OPS = {
    "RP-platform-repair": repair_platform,
    "RU-uav-repair": repair_uav,
    "RA-assignment-repair": repair_assignment,
    "RR-route-rebalance": repair_route_rebalance,
}
ACTIONS = [d+"+"+r for d in DESTROY_OPS for r in REPAIR_OPS]


# ============================================================================
# 8. true Q-learning
# ============================================================================

class QLearner(object):
    def __init__(self):
        self.Q = {}
        self.epsilon = Q_EPSILON_START
        self.states = set()
        self.update_count = 0
        self.use_count = {}

    def q(self, state, action):
        return self.Q.get((state,action), 0.0)

    def choose(self, state, rng):
        self.states.add(state)
        if rng.random() < self.epsilon:
            action = rng.choice(ACTIONS)
        else:
            vals = [(self.q(state,a), a) for a in ACTIONS]
            mx = max(v[0] for v in vals)
            ties = [a for q,a in vals if abs(q-mx) <= 1e-12]
            action = rng.choice(ties)
        self.use_count[action] = self.use_count.get(action,0)+1
        return action

    def update(self, state, action, reward, next_state):
        self.states.add(next_state)
        old = self.q(state,action)
        target = reward + Q_GAMMA * max(self.q(next_state,a) for a in ACTIONS)
        new = old + Q_ALPHA*(target-old)
        self.Q[(state,action)] = new
        self.update_count += 1
        self.epsilon = max(Q_EPSILON_MIN, self.epsilon*Q_EPSILON_DECAY)
        return new


def search_state(best_obj, current_obj, recent_improved, stagnation, feasible):
    if recent_improved:
        trend = "improving"
    elif stagnation >= STATE_STAG_SEVERE:
        trend = "severe_stagnation"
    elif stagnation >= STATE_STAG_MILD:
        trend = "mild_stagnation"
    else:
        trend = "stable"
    return (trend, "feasible" if feasible else "infeasible")


def split_action(action):
    d,r = action.split("+",1)
    return d,r


def apply_action(inst, sol, action, rng):
    d,r = split_action(action)
    cand = DESTROY_OPS[d](inst, sol, rng)
    cand = REPAIR_OPS[r](inst, cand, rng)
    compute_solution_obj(inst, cand)
    return cand


def sa_accept(delta, T, rng):
    if delta <= 0:
        return True
    if T <= 1e-12:
        return False
    return rng.random() < math.exp(-delta/T)


# ============================================================================
# 9. TQ-specific elite intensification
# ============================================================================

def try_resource_downsize(inst, sol):
    best = clone_solution(sol)
    for _ in range(TQ_RESOURCE_DOWNSIZE_PASSES):
        improved = False
        for i in list(open_set(best)):
            if best.platform_uav_count.get(i,1) > 1:
                cand = clone_solution(best); cand.platform_uav_count[i]-=1
                compute_solution_obj(inst,cand)
                if cand.feasible and cand.obj < best.obj-IMPROVEMENT_TOL:
                    best=cand; improved=True; continue
            k=best.platform_type.get(i,_default_type(inst)); prev=_previous_allowed_type(inst,k)
            if prev < k:
                can=True
                for j in range(inst.J):
                    if best.assign_j.get(j,None)!=i: continue
                    mk=min_type_for_point(inst,i,j)
                    if mk is None or mk>prev: can=False; break
                if can:
                    cand=clone_solution(best); cand.platform_type[i]=prev
                    compute_solution_obj(inst,cand)
                    if cand.feasible and cand.obj < best.obj-IMPROVEMENT_TOL:
                        best=cand; improved=True
        if not improved: break
    return best

def elite_task_moves(inst, sol, rng, samples=TQ_TASK_MOVE_SAMPLES):
    best = clone_solution(sol)
    opens = open_set(best)
    if len(opens) <= 1:
        return best
    # Bias toward expensive/long current assignments.
    js = list(range(inst.J))
    js.sort(key=lambda j: inst.d_ip[best.assign_j[j]][j] if best.assign_j.get(j,None) is not None else -1,
            reverse=True)
    pool = js[:min(len(js), max(samples, 80))]
    rng.shuffle(pool)
    for j in pool[:samples]:
        old = best.assign_j.get(j,None)
        if old is None:
            continue
        candidates = sorted([i for i in opens if i != old and min_type_for_point(inst,i,j) is not None],
                            key=lambda i: inst.d_ip[i][j])[:4]
        for i in candidates:
            cand = clone_solution(best)
            cand.assign_j[j] = i
            cand = targeted_feasibility_repair(inst,cand,rng,max_rounds=1)
            compute_solution_obj(inst,cand)
            if cand.feasible and cand.obj < best.obj - IMPROVEMENT_TOL:
                best = cand
                break
    return best


def elite_platform_swaps(inst, sol, rng, samples=TQ_PLATFORM_SWAP_SAMPLES):
    best = clone_solution(sol)
    opens = open_set(best)
    closed = [i for i in inst.valid_platform_indices if i not in opens]
    pairs = list(itertools.product(opens, closed))
    rng.shuffle(pairs)
    # Also test deterministic promising swaps first.
    pairs = pairs[:samples]
    for out_i, in_i in pairs:
        cand = clone_solution(best)
        cand.open_platform[out_i] = 0
        cand.platform_uav_count[out_i] = 0
        cand.platform_type.pop(out_i,None)
        cand.open_platform[in_i] = 1
        cand.platform_uav_count[in_i] = 1
        cand.platform_type[in_i] = _default_type(inst)
        for j in range(inst.J):
            if cand.assign_j.get(j,None)==out_i:
                cand.assign_j[j] = None
        assign_all_points_by_distance(inst,cand,only_unassigned=True)
        cand = targeted_feasibility_repair(inst,cand,rng,max_rounds=2)
        compute_solution_obj(inst,cand)
        if cand.feasible and cand.obj < best.obj - IMPROVEMENT_TOL:
            best = cand
    return best


def tq_elite_intensification(inst, sol, rng):
    best = clone_solution(sol)
    for _ in range(TQ_INTENSIFY_ROUNDS):
        before = best.obj
        best = elite_task_moves(inst,best,rng)
        best = try_resource_downsize(inst,best)
        best = elite_platform_swaps(inst,best,rng)
        best = try_resource_downsize(inst,best)
        if best.obj >= before - IMPROVEMENT_TOL:
            break
    compute_solution_obj(inst,best)
    return best


# ============================================================================
# 10. four algorithms
# ============================================================================

def greedy_only(inst, seed):
    st = time.time()
    s = make_initial_solution(inst,seed)
    compute_solution_obj(inst,s)
    s.runtime_s = time.time()-st
    s.time_to_best_s = s.runtime_s
    return s


def alns_only(inst, seed, iters=HEURISTIC_ITERS):
    rng = random.Random(seed)
    st = time.time()
    cur = make_initial_solution(inst,seed)
    best = clone_solution(cur)
    for it in range(1,iters+1):
        action = rng.choice(ACTIONS)
        cand = apply_action(inst,cur,action,rng)
        if cand.feasible and cand.obj <= cur.obj + 1e-12:
            cur = cand
        if cand.feasible and cand.obj < best.obj - IMPROVEMENT_TOL:
            best = clone_solution(cand)
    best.runtime_s = time.time()-st
    best.time_to_best_s = best.runtime_s
    return best


def alns_sa_only(inst, seed, iters=HEURISTIC_ITERS):
    rng = random.Random(seed)
    st = time.time()
    cur = make_initial_solution(inst,seed)
    best = clone_solution(cur)
    T0 = max(SA_MIN_T0, SA_INITIAL_RATIO*max(cur.obj,1e-6))
    T = T0
    stagnation = 0
    for it in range(1,iters+1):
        action = rng.choice(ACTIONS)
        cand = apply_action(inst,cur,action,rng)
        delta = cand.obj-cur.obj
        if cand.feasible and sa_accept(delta,T,rng):
            cur = cand
        if cand.feasible and cand.obj < best.obj - IMPROVEMENT_TOL:
            best = clone_solution(cand); stagnation = 0
        else:
            stagnation += 1
        T *= SA_COOLING
        if stagnation >= SA_REHEAT_AFTER:
            T = max(T,SA_REHEAT_RATIO*T0); stagnation = 0
    best.runtime_s = time.time()-st
    best.time_to_best_s = best.runtime_s
    return best


def tq_alns(inst, seed, iters=HEURISTIC_ITERS, trace_file=None):
    rng = random.Random(seed)
    st = time.time()
    cur = make_initial_solution(inst,seed)
    best = clone_solution(cur)
    time_to_best = 0.0
    ql = QLearner()
    T0 = max(SA_MIN_T0, SA_INITIAL_RATIO*max(cur.obj,1e-6))
    T = T0
    stagnation = 0
    recent_improve_flags = []
    trace = []

    for it in range(1,iters+1):
        recent_improved = any(recent_improve_flags[-STATE_RECENT_WINDOW:]) if recent_improve_flags else True
        state = search_state(best.obj,cur.obj,recent_improved,stagnation,cur.feasible)
        action = ql.choose(state,rng)
        prev_obj = cur.obj
        cand = apply_action(inst,cur,action,rng)
        accepted = False
        if cand.feasible and sa_accept(cand.obj-cur.obj,T,rng):
            cur = cand
            accepted = True

        new_best = cand.feasible and cand.obj < best.obj - IMPROVEMENT_TOL
        if new_best:
            best = clone_solution(cand)
            time_to_best = time.time()-st
            stagnation = 0
        else:
            stagnation += 1

        improved_current = cand.feasible and cand.obj < prev_obj - IMPROVEMENT_TOL
        recent_improve_flags.append(bool(improved_current or new_best))

        if not cand.feasible:
            reward = REWARD_INFEASIBLE
        else:
            rel = max(-0.05,min(0.05,(prev_obj-cand.obj)/max(abs(prev_obj),1e-9)))
            reward = REWARD_REL_IMPROVEMENT_SCALE*rel
            if new_best:
                reward += REWARD_NEW_BEST_BONUS
            elif accepted and cand.obj > prev_obj + IMPROVEMENT_TOL:
                reward += REWARD_ACCEPTED_WORSE

        next_recent = any(recent_improve_flags[-STATE_RECENT_WINDOW:])
        next_state = search_state(best.obj,cur.obj,next_recent,stagnation,cur.feasible)
        qv = ql.update(state,action,reward,next_state)

        T *= SA_COOLING
        if stagnation >= SA_REHEAT_AFTER:
            T = max(T,SA_REHEAT_RATIO*T0)

        if SAVE_Q_TRACE:
            trace.append([it,seed,prev_obj,cand.obj,cur.obj,best.obj,int(cand.feasible),
                          int(accepted),int(new_best),reward,qv,ql.epsilon,T,action,str(next_state)])
        if PRINT_EVERY and it % PRINT_EVERY == 0:
            print("[TQ %4d] current=%.6f best=%.6f open=%d uavs=%d cov=%.3f eps=%.3f T=%.4f" %
                  (it,cur.obj,best.obj,len(open_set(best)),total_uavs(best),best.coverage_ratio,ql.epsilon,T))

    if TQ_USE_ELITE_INTENSIFICATION:
        before = best.obj
        best2 = tq_elite_intensification(inst,best,rng)
        if best2.feasible and best2.obj < best.obj - IMPROVEMENT_TOL:
            best = best2
            time_to_best = time.time()-st
            print("[TQ-INTENSIFY] %.8f -> %.8f" % (before,best.obj))

    best.runtime_s = time.time()-st
    best.time_to_best_s = time_to_best
    best.q_states_visited = len(ql.states)
    best.q_updates = ql.update_count

    if trace_file and SAVE_Q_TRACE:
        with open(trace_file,"w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f)
            w.writerow(["iter","seed","prev_obj","candidate_obj","current_obj","best_obj",
                        "candidate_feasible","accepted","new_best","reward","updated_q",
                        "epsilon","temperature","action","next_state"])
            w.writerows(trace)
    return best

