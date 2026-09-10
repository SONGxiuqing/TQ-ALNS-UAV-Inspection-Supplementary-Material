# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import math
import time
import csv
import os

# =============================================================================
# 第六章敏感性分析唯一参数配置区
# 修改实验时只改本区域，后续旧版函数中的默认值不会覆盖这里的正式设置。
# =============================================================================
CH6_DEBUG_MODE = False

# 固定基础实例：所有参数水平共享同一空间布局，仅改变一个敏感性参数。
CH6_INSTANCE_SEED = 20260805
CH6_SEARCH_SEEDS = tuple(range(1001, 1011))

# TQ-ALNS停止准则：时间上限与迭代上限，先到者停止。
CH6_TIME_LIMIT_SEC = 600.0
CH6_MAX_ITERATIONS = 8000
CH6_HEARTBEAT_SEC = 60.0

# 调试模式设置。首次检查代码时可将CH6_DEBUG_MODE改为True。
CH6_DEBUG_TIME_LIMIT_SEC = 10.0
CH6_DEBUG_MAX_ITERATIONS = 80
CH6_DEBUG_SEARCH_SEEDS = (1001,)

# 平台与机队数量内生决定。
CH6_MIN_OPEN_PLATFORMS = 1
CH6_MAX_OPEN_PLATFORMS = 5
CH6_MAX_UAVS_PER_PLATFORM = 6
CH6_INITIAL_OPEN_PLATFORMS = 3

# 三组正文敏感性实验。
CH6_BETA_LEVELS = (0.80, 0.85, 0.90, 0.95, 0.99)
CH6_DEMAND_LEVELS = (0.80, 1.00, 1.20, 1.50)
CH6_BATTERY_LEVELS = (0.80, 1.00, 1.20)

# 基准值。
CH6_BASE_BETA = 0.95
CH6_BASE_DEMAND_SCALE = 1.00
CH6_BASE_BATTERY_SCALE = 1.00

# 输出设置。
CH6_FORCE_RERUN = False
CH6_OUTPUT_FOLDER = os.path.join("..", "results", "chapter6_sensitivity_TQALNS")
CH6_FIG_DPI = 600

# ============================================================
# 非固定算例随机种子读取：
# 优先读取 RUN_SEED，其次读取 SEARCH_SEED；都没有时使用当前时间生成随机种子。
# 这样每次独立运行会生成不同算例数据，不再固定为 seed=7。
# ============================================================
def _get_int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _dynamic_run_seed(default=7):
    for _name in ("RUN_SEED", "SEARCH_SEED"):
        _v = os.environ.get(_name, None)
        if _v is not None:
            try:
                return int(_v)
            except Exception:
                pass
    try:
        return int(time.time_ns() % 2147483647)
    except Exception:
        return int(time.time() * 1000) % 2147483647
# ============================================================


try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception:
    plt = None
    Patch = None


# =========================================================
# 几何工具：点在多边形/线段相交
# =========================================================
def _orient(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax, ay, bx, by, cx, cy):
    return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)


def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    o1 = _orient(ax, ay, bx, by, cx, cy)
    o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay)
    o4 = _orient(cx, cy, dx, dy, bx, by)

    if o1 == 0 and _on_segment(ax, ay, bx, by, cx, cy):
        return True
    if o2 == 0 and _on_segment(ax, ay, bx, by, dx, dy):
        return True
    if o3 == 0 and _on_segment(cx, cy, dx, dy, ax, ay):
        return True
    if o4 == 0 and _on_segment(cx, cy, dx, dy, bx, by):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def point_in_poly(px, py, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if _orient(x1, y1, x2, y2, px, py) == 0 and _on_segment(x1, y1, x2, y2, px, py):
            return True
        if ((y1 > py) != (y2 > py)):
            x_int = x1 + (py - y1) * (x2 - x1) / float(y2 - y1)
            if x_int > px:
                inside = not inside
    return inside


def segment_intersects_poly(ax, ay, bx, by, poly):
    if point_in_poly(ax, ay, poly) or point_in_poly(bx, by, poly):
        return True
    n = len(poly)
    for i in range(n):
        cx, cy = poly[i]
        dx, dy = poly[(i + 1) % n]
        if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
            return True
    return False


def euclid_m(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def poly_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def polyline_length(pts):
    s = 0.0
    for i in range(len(pts) - 1):
        s += euclid_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
    return s


def polyline_intersects_poly(pts, poly):
    for i in range(len(pts) - 1):
        if segment_intersects_poly(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], poly):
            return True
    return False


# =========================================================
# 绕行：对矩形禁飞区生成折线路径（不穿越禁飞区）
# =========================================================
def detour_single_rect(A, B, rect_poly, margin=200.0):
    ax, ay = A
    bx, by = B
    if not segment_intersects_poly(ax, ay, bx, by, rect_poly):
        return [A, B]

    xmin, xmax, ymin, ymax = poly_bbox(rect_poly)
    x_candidates = [xmin - margin, xmax + margin]
    y_candidates = [ymin - margin, ymax + margin]

    best_path = None
    best_len = 1e100

    for x_out in x_candidates:
        for y_out in y_candidates:
            pts = [A, (ax, y_out), (x_out, y_out), (x_out, by), B]

            compact = [pts[0]]
            for p in pts[1:]:
                if p[0] != compact[-1][0] or p[1] != compact[-1][1]:
                    compact.append(p)

            if polyline_intersects_poly(compact, rect_poly):
                continue

            L = polyline_length(compact)
            if L < best_len:
                best_len = L
                best_path = compact

    return best_path if best_path is not None else [A, B]


def detour_path(A, B, nofly_polys, margin=200.0, max_iter=12):
    path = [A, B]
    for _ in range(max_iter):
        changed = False
        new_path = [path[0]]
        for i in range(len(path) - 1):
            p = path[i]
            q = path[i + 1]
            seg_handled = False
            for poly in nofly_polys:
                if segment_intersects_poly(p[0], p[1], q[0], q[1], poly):
                    det = detour_single_rect(p, q, poly, margin=margin)
                    for t in det[1:]:
                        new_path.append(t)
                    seg_handled = True
                    changed = True
                    break
            if not seg_handled:
                new_path.append(q)
        path = new_path
        if not changed:
            break

    compact = [path[0]]
    for p in path[1:]:
        if p[0] != compact[-1][0] or p[1] != compact[-1][1]:
            compact.append(p)
    return compact


# =========================================================
# 数据结构
# =========================================================
class DroneType(object):
    def __init__(self, name, purchase_cost, maintain_cost,
                 speed_kmh, battery_Wh, energy_mu_Wh_per_km, energy_sigma_Wh_per_km,
                 max_radius_km):
        self.name = name
        self.purchase_cost = float(purchase_cost)  # 万元/台
        self.maintain_cost = float(maintain_cost)  # 万元/年/台
        self.speed = float(speed_kmh)  # km/h
        self.battery = float(battery_Wh)  # Wh
        self.energy_mu = float(energy_mu_Wh_per_km)  # Wh/km
        self.energy_sigma = float(energy_sigma_Wh_per_km)  # Wh/km
        self.max_radius = float(max_radius_km)  # km（单程半径）


class Route(object):
    def __init__(self, seq=None):
        self.seq = seq[:] if seq else []


class PlatformPlan(object):
    def __init__(self, routes=None):
        self.routes = routes[:] if routes else []  # list[Route]


class Solution(object):
    def __init__(self):
        self.open_platform = {}  # i->0/1
        self.platform_type = {}  # i->k（该平台唯一无人机机型）
        self.assign_j = {}  # j->i or None（监测点分配到平台）
        self.assign = {}  # (i,j,k)->1（compute里重建）
        self.covered = {}  # j->0/1
        self.routes = {}  # i->PlatformPlan
        self.fleet = {}  # (i,k)->1（每开平台恰好1架无人机）

        self.obj = 1e100
        self.feasible = False
        self.coverage_ratio = 0.0
        self.runtime_s = 0.0


class Instance(object):
    pass


# =========================================================
# 中规模实例生成（用于表6统一对比）
# 平台候选点: 15个，从中选择4个开放平台
# 监测点: 50个
# 危险源: 3个
# 禁飞区: 3个
# 开放平台/无人机数量: 4
# 最低覆盖率: 90%
# 成本单位：万元
# =========================================================
def build_instance(seed=7):
    rnd = random.Random(seed)
    inst = Instance()

    inst.W = 10000.0
    inst.H = 10000.0

    # 中规模参数：15个候选平台、50个监测点、3个危险源、3个禁飞区
    inst.J = 50  # 监测点数量
    inst.I = 15  # 平台候选点总数
    inst.Hz = 3  # 危险源数量
    inst.Z = 3  # 禁飞区数量

    # 约束参数
    inst.MIN_OPEN = 4  # 最小开放平台数
    inst.MAX_OPEN = 4  # 最大开放平台数；表6中规模固定开放4个平台，便于横向比较
    inst.min_cover_ratio = 0.90  # 最低覆盖率90%
    inst.safe_dist_hazard_m = 500.0
    inst.T_day = 16.0
    inst.alpha = 0.95
    inst.z_alpha = 1.645

    # 成本参数
    inst.energy_cost_yuan_per_km = 0.8  # 元/km
    inst.penalty_uncovered_wan = 1.5  # 万元/个

    # 用于计算Gap的参考目标值
    # 说明：ALNS属于启发式算法，没有精确最优值/下界时，严格Gap无法直接计算。
    # 如有Gurobi/CPLEX最优值、下界或对比算法最好目标值，可改为具体数值，例如：
    # inst.ref_obj_wan = 210.35
    inst.ref_obj_wan = None

    inst.req_freq = [rnd.randint(1, 3) for _ in range(inst.J)]
    inst.inspect_time_h = [rnd.uniform(0.08, 0.30) for _ in range(inst.J)]

    # 无人机购置成本为十万级别，单位：万元/台
    inst.drone_types = [
        DroneType("K0-轻型巡检机", 12, 1.0, 60, 800, 20, 4, 5.0),
        DroneType("K1-中型长航机", 26, 1.4, 80, 1200, 25, 5, 7.0),
        DroneType("K2-重型重载机", 48, 2.0, 100, 2000, 35, 7, 10.0),
    ]
    inst.K = len(inst.drone_types)

    inst.hazards = [(rnd.uniform(0, inst.W), rnd.uniform(0, inst.H)) for _ in range(inst.Hz)]

    # 禁飞区：随机轴对齐矩形（3-4个）
    inst.nofly = []
    for _ in range(inst.Z):
        cx = rnd.uniform(2000, inst.W - 2000)
        cy = rnd.uniform(2000, inst.H - 2000)
        w = rnd.uniform(1800, 2600)
        h = rnd.uniform(1800, 2600)
        poly = [(cx - w / 2, cy - h / 2),
                (cx + w / 2, cy - h / 2),
                (cx + w / 2, cy + h / 2),
                (cx - w / 2, cy + h / 2)]
        inst.nofly.append(poly)

    # 生成平台候选点（避开禁飞区和危险源）
    inst.platform_xy = []
    inst.platform_fixed_wan = []
    for _ in range(inst.I):
        attempts = 0
        while attempts < 1000:
            x = rnd.uniform(0, inst.W)
            y = rnd.uniform(0, inst.H)

            in_nofly = False
            for poly in inst.nofly:
                if point_in_poly(x, y, poly):
                    in_nofly = True
                    break

            near_hazard = False
            if not in_nofly:
                for hx, hy in inst.hazards:
                    if euclid_m(x, y, hx, hy) < inst.safe_dist_hazard_m:
                        near_hazard = True
                        break

            if not in_nofly and not near_hazard:
                inst.platform_xy.append((x, y))
                inst.platform_fixed_wan.append(25.0)  # 万元/个
                break

            attempts += 1

        if attempts >= 1000:
            x = rnd.choice([rnd.uniform(0, 1000), rnd.uniform(inst.W - 1000, inst.W)])
            y = rnd.choice([rnd.uniform(0, 1000), rnd.uniform(inst.H - 1000, inst.H)])
            inst.platform_xy.append((x, y))
            inst.platform_fixed_wan.append(25.0)  # 万元/个

    # 生成监测点（避开禁飞区）
    inst.point_xy = []
    for _ in range(inst.J):
        attempts = 0
        while attempts < 1000:
            x = rnd.uniform(0, inst.W)
            y = rnd.uniform(0, inst.H)

            in_nofly = False
            for poly in inst.nofly:
                if point_in_poly(x, y, poly):
                    in_nofly = True
                    break

            if not in_nofly:
                inst.point_xy.append((x, y))
                break

            attempts += 1

        if attempts >= 1000:
            x = rnd.choice([rnd.uniform(0, 500), rnd.uniform(inst.W - 500, inst.W)])
            y = rnd.choice([rnd.uniform(0, 500), rnd.uniform(inst.H - 500, inst.H)])
            inst.point_xy.append((x, y))

    # 平台合法性：不在禁飞区内 + 远离危险源
    inst.platform_valid = [True] * inst.I
    for i in range(inst.I):
        x, y = inst.platform_xy[i]
        ok = True
        for poly in inst.nofly:
            if point_in_poly(x, y, poly):
                ok = False
                break
        if ok:
            for hx, hy in inst.hazards:
                if euclid_m(x, y, hx, hy) < inst.safe_dist_hazard_m:
                    ok = False
                    break
        inst.platform_valid[i] = ok

    inst.valid_platform_indices = [i for i in range(inst.I) if inst.platform_valid[i]]
    # 这里保留15个候选平台中的所有合法平台，再由算法选择开放4个平台。
    # 不建议把有效候选平台强行固定为4个，否则 MIN_OPEN=4 时没有选址优化空间，四种算法会退化为同一结果。
    if len(inst.valid_platform_indices) < getattr(inst, "MIN_OPEN", 4):
        inst.valid_platform_indices = list(range(inst.I))

    # 预计算平台->点 距离与可达性（穿越禁飞区用绕行系数惩罚）
    detour_factor = 1.40
    inst.d_ij_km = [[None for _ in range(inst.J)] for _ in range(inst.I)]
    inst.reachable = [[0 for _ in range(inst.J)] for _ in range(inst.I)]
    for i in range(inst.I):
        ax, ay = inst.platform_xy[i]
        for j in range(inst.J):
            bx, by = inst.point_xy[j]
            base_km = euclid_m(ax, ay, bx, by) / 1000.0
            cross_cnt = 0
            for poly in inst.nofly:
                if segment_intersects_poly(ax, ay, bx, by, poly):
                    cross_cnt += 1
            if cross_cnt >= 3:
                inst.d_ij_km[i][j] = None
                inst.reachable[i][j] = 0
            else:
                dist = base_km * (detour_factor if cross_cnt > 0 else 1.0)
                inst.d_ij_km[i][j] = dist
                inst.reachable[i][j] = 1

    return inst


# =========================================================
# 基础工具
# =========================================================
def clone_solution(sol):
    s = Solution()
    s.open_platform = dict(sol.open_platform)
    s.platform_type = dict(sol.platform_type)
    s.assign_j = dict(sol.assign_j)
    s.assign = dict(sol.assign)
    s.covered = dict(sol.covered)
    s.fleet = dict(sol.fleet)
    s.obj = sol.obj
    s.feasible = sol.feasible
    s.coverage_ratio = sol.coverage_ratio
    s.runtime_s = getattr(sol, "runtime_s", 0.0)

    new_routes = {}
    for pid, plan in sol.routes.items():
        if plan is None:
            new_routes[pid] = None
            continue
        copied = []
        for rr in plan.routes:
            copied.append(Route(rr.seq[:]))
        new_routes[pid] = PlatformPlan(routes=copied)
    s.routes = new_routes
    return s


def _open_set(sol):
    return [i for i, xi in sol.open_platform.items() if xi == 1]


def calc_gap_percent(best_obj, ref_obj):
    """
    Gap(%) = (best_obj - ref_obj) / |ref_obj| * 100
    ref_obj 可以是精确最优值、下界或对比算法最好目标值。
    如果没有参考值，则返回 None，并在结果表中显示 N/A。
    """
    if ref_obj is None:
        return None
    if abs(ref_obj) < 1e-12:
        return None
    return 100.0 * (best_obj - ref_obj) / abs(ref_obj)


# =========================================================
# 平台机型：判断单点所需最小机型k
# =========================================================
def _min_type_for_dist(inst, d_km):
    if d_km is None:
        return None
    for k in range(inst.K):
        dt = inst.drone_types[k]
        if d_km > dt.max_radius + 1e-9:
            continue
        e_q = dt.energy_mu + inst.z_alpha * dt.energy_sigma
        need_Wh = e_q * (2.0 * d_km)
        if need_Wh <= dt.battery + 1e-6:
            return k
    return None


def _ensure_platform_type_feasible(current_k, required_k):
    if current_k is None:
        return required_k
    if required_k is None:
        return current_k
    return max(current_k, required_k)


# =========================================================
# 评估：目标函数 + 可行性
# 关键：每开平台恰好1架无人机 => 无人机数量 == 开放平台数量
# =========================================================
def compute_solution_obj(inst, sol):
    open_set = set(_open_set(sol))

    covered_cnt = 0
    sol.covered = {}
    for j in range(inst.J):
        i = sol.assign_j.get(j, None)
        if i is None:
            sol.covered[j] = 0
        else:
            sol.covered[j] = 1
            covered_cnt += 1
    sol.coverage_ratio = covered_cnt / float(inst.J)

    feasible = True
    if sol.coverage_ratio + 1e-12 < inst.min_cover_ratio:
        feasible = False
    if len(open_set) < getattr(inst, "MIN_OPEN", 1):
        feasible = False
    max_open = getattr(inst, "MAX_OPEN", None)
    if max_open is not None and len(open_set) > max_open:
        feasible = False
    for i in open_set:
        if not inst.platform_valid[i]:
            feasible = False
            break

    platform_cost = sum(inst.platform_fixed_wan[i] for i in open_set)
    uncovered_penalty = 0.0
    energy_cost_wan = 0.0

    sol.assign = {}
    sol.platform_type = dict(sol.platform_type)

    # 先推平台需要机型（可升级）
    for j in range(inst.J):
        i = sol.assign_j.get(j, None)
        if i is None:
            continue
        if i not in open_set:
            feasible = False
            continue
        d = inst.d_ij_km[i][j]
        if d is None or inst.reachable[i][j] == 0:
            feasible = False
            continue
        req_k = _min_type_for_dist(inst, d)
        if req_k is None:
            feasible = False
            continue
        sol.platform_type[i] = _ensure_platform_type_feasible(sol.platform_type.get(i, None), req_k)

    # 再逐点检查
    for j in range(inst.J):
        i = sol.assign_j.get(j, None)
        if i is None:
            uncovered_penalty += inst.penalty_uncovered_wan
            continue
        if i not in open_set:
            uncovered_penalty += inst.penalty_uncovered_wan
            feasible = False
            sol.assign_j[j] = None
            sol.covered[j] = 0
            continue

        k = sol.platform_type.get(i, 0)
        d = inst.d_ij_km[i][j]
        if d is None or inst.reachable[i][j] == 0:
            uncovered_penalty += inst.penalty_uncovered_wan
            feasible = False
            sol.assign_j[j] = None
            sol.covered[j] = 0
            continue

        dt = inst.drone_types[k]
        if d > dt.max_radius + 1e-9:
            uncovered_penalty += inst.penalty_uncovered_wan
            feasible = False
            sol.assign_j[j] = None
            sol.covered[j] = 0
            continue

        e_q = dt.energy_mu + inst.z_alpha * dt.energy_sigma
        need_Wh = e_q * (2.0 * d)
        if need_Wh > dt.battery + 1e-6:
            uncovered_penalty += inst.penalty_uncovered_wan
            feasible = False
            sol.assign_j[j] = None
            sol.covered[j] = 0
            continue

        freq = inst.req_freq[j]
        energy_yuan = inst.energy_cost_yuan_per_km * (2.0 * d) * freq
        energy_cost_wan += energy_yuan / 10000.0
        sol.assign[(i, j, k)] = 1

    # 每个开平台恰好1架无人机
    sol.fleet = {}
    drone_cost = 0.0
    for i in open_set:
        k = sol.platform_type.get(i, 0)
        sol.fleet[(i, k)] = 1
        dt = inst.drone_types[k]
        drone_cost += dt.purchase_cost
        drone_cost += (dt.maintain_cost / 365.0)

    sol.feasible = feasible
    sol.obj = platform_cost + drone_cost + energy_cost_wan + uncovered_penalty
    return sol.obj, sol.feasible


# =========================================================
# 初解：开若干低成本平台 + 贪心分配（平台机型可升级）
# =========================================================
def make_initial_solution(inst, seed=7, init_open=4):
    sol = Solution()
    for i in range(inst.I):
        sol.open_platform[i] = 0
    sol.platform_type = {}

    valid = inst.valid_platform_indices[:]
    valid.sort(key=lambda i: inst.platform_fixed_wan[i])
    must_open = max(getattr(inst, "MIN_OPEN", 1), init_open, 3)
    chosen = valid[:max(2, must_open)]

    for i in chosen:
        sol.open_platform[i] = 1
        sol.platform_type[i] = 0

    for j in range(inst.J):
        best_i = None
        best_cost = 1e100
        best_new_k = None

        freq = inst.req_freq[j]
        ins_t = inst.inspect_time_h[j]

        for i in chosen:
            d = inst.d_ij_km[i][j]
            if d is None or inst.reachable[i][j] == 0:
                continue
            req_k = _min_type_for_dist(inst, d)
            if req_k is None:
                continue
            cur_k = sol.platform_type.get(i, 0)
            new_k = max(cur_k, req_k)

            dt = inst.drone_types[new_k]
            fly_h = (2.0 * d) / dt.speed
            energy_wan = (inst.energy_cost_yuan_per_km * (2.0 * d) * freq) / 10000.0
            time_wan = 0.02 * freq * (fly_h + ins_t)
            upgrade_wan = 0.12 * (inst.drone_types[new_k].purchase_cost - inst.drone_types[cur_k].purchase_cost)
            c = energy_wan + time_wan + max(0.0, upgrade_wan)

            if c < best_cost:
                best_cost = c
                best_i = i
                best_new_k = new_k

        sol.assign_j[j] = best_i
        if best_i is not None:
            sol.platform_type[best_i] = best_new_k

    compute_solution_obj(inst, sol)
    return sol


# =========================================================
# SA 接受准则
# =========================================================
def sa_accept(delta, T, rnd):
    if delta <= 0:
        return True
    if T <= 1e-12:
        return False
    return rnd.random() < math.exp(-delta / T)


# =========================================================
# Q-learning（算子组合动作）
# =========================================================
class QLearner(object):
    def __init__(self, actions, alpha=0.15, epsilon=0.18):
        self.actions = actions[:]
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = dict((a, 0.0) for a in actions)

    def choose(self, rnd):
        if rnd.random() < self.epsilon:
            return rnd.choice(self.actions)
        best_a, best_q = None, -1e100
        for a in self.actions:
            q = self.Q.get(a, 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a if best_a is not None else rnd.choice(self.actions)

    def update(self, action, reward):
        old = self.Q.get(action, 0.0)
        self.Q[action] = (1.0 - self.alpha) * old + self.alpha * reward


# =========================================================
# Destroy / Repair 算子：4×4 组合动作池
# =========================================================
# 本版本与论文中的四层决策结构保持一致：
#   Destroy:  D_P 平台移除、D_U 无人机降配/减少、D_A 任务分配破坏、D_R 路径结构破坏
#   Repair:   R_P 平台补充/重选、R_U 无人机补充/升级、R_A 任务重分配、R_R 路径重规划
# Q-learning 将每一个 destroy-repair 组合 a=(D_m,R_n) 作为一个动作，因此共有 4×4=16 个动作。
# =========================================================

def _destruction_ratio(inst, rnd):
    """论文中 φ ~ U(0.10, 0.30)，这里换算为被破坏监测点数量。"""
    return rnd.uniform(0.10, 0.30)


def _energy_quantile(inst, k):
    dt = inst.drone_types[k]
    return dt.energy_mu + inst.z_alpha * dt.energy_sigma


def _can_type_serve_distance(inst, k, d_km):
    """检查给定机型 k 是否能够完成平台-监测点往返并满足能耗机会约束的保守等价。"""
    if d_km is None:
        return False
    dt = inst.drone_types[k]
    if d_km > dt.max_radius + 1e-9:
        return False
    need_Wh = _energy_quantile(inst, k) * (2.0 * d_km)
    return need_Wh <= dt.battery + 1e-6


def _assignment_incremental_cost(inst, s, i, j, new_k):
    """给监测点 j 分配到平台 i 且平台机型调整为 new_k 后的近似增量成本。"""
    d = inst.d_ij_km[i][j]
    if d is None:
        return 1e100
    freq = inst.req_freq[j]
    ins_t = inst.inspect_time_h[j]
    dt = inst.drone_types[new_k]
    fly_h = (2.0 * d) / dt.speed
    energy_wan = (inst.energy_cost_yuan_per_km * (2.0 * d) * freq) / 10000.0
    time_wan = 0.02 * freq * (fly_h + ins_t)
    cur_k = s.platform_type.get(i, 0)
    upgrade_wan = 0.12 * max(0.0, inst.drone_types[new_k].purchase_cost - inst.drone_types[cur_k].purchase_cost)
    return energy_wan + time_wan + upgrade_wan


def _best_assignment_for_point(inst, s, j, candidate_platforms=None, allow_upgrade=True):
    """为未分配监测点寻找最优平台-机型组合。"""
    if candidate_platforms is None:
        candidate_platforms = _open_set(s)

    best_i, best_cost, best_new_k = None, 1e100, None
    for i in candidate_platforms:
        if s.open_platform.get(i, 0) != 1:
            continue
        if not inst.platform_valid[i]:
            continue
        d = inst.d_ij_km[i][j]
        if d is None or inst.reachable[i][j] == 0:
            continue

        req_k = _min_type_for_dist(inst, d)
        if req_k is None:
            continue
        cur_k = s.platform_type.get(i, 0)
        new_k = max(cur_k, req_k) if allow_upgrade else cur_k
        if not _can_type_serve_distance(inst, new_k, d):
            continue

        c = _assignment_incremental_cost(inst, s, i, j, new_k)
        if c < best_cost:
            best_i, best_cost, best_new_k = i, c, new_k
    return best_i, best_cost, best_new_k


def _add_best_platform(inst, s, rnd, focus_unassigned=True):
    """选择一个新增平台。优先选择能够覆盖未分配点、平均距离短且固定成本低的平台。"""
    max_open = getattr(inst, "MAX_OPEN", None)
    opens = _open_set(s)
    if max_open is not None and len(opens) >= max_open:
        return False

    cand = [i for i in inst.valid_platform_indices if s.open_platform.get(i, 0) == 0]
    if not cand:
        return False
    rnd.shuffle(cand)

    best_i, best_score = None, -1e100
    for i in cand:
        gain = 0
        dist_sum = 0.0
        for j in range(inst.J):
            if focus_unassigned and s.assign_j.get(j, None) is not None:
                continue
            d = inst.d_ij_km[i][j]
            if d is None or inst.reachable[i][j] == 0:
                continue
            if _min_type_for_dist(inst, d) is not None:
                gain += 1
                dist_sum += d
        avg_dist = dist_sum / float(gain) if gain > 0 else 1e6
        score = 1000.0 * gain - avg_dist - inst.platform_fixed_wan[i]
        if score > best_score:
            best_i, best_score = i, score

    if best_i is None:
        return False
    s.open_platform[best_i] = 1
    s.platform_type[best_i] = 0
    return True


def _ensure_minimum_platforms(inst, s, rnd):
    min_open = getattr(inst, "MIN_OPEN", 1)
    while len(_open_set(s)) < min_open:
        if not _add_best_platform(inst, s, rnd, focus_unassigned=True):
            break


def _remove_invalid_assignments(inst, s):
    """删除当前平台关闭、路径不可达或当前机型无法服务的任务分配。"""
    opens = set(_open_set(s))
    for j, i in list(s.assign_j.items()):
        if i is None:
            continue
        if i not in opens or (not inst.platform_valid[i]):
            s.assign_j[j] = None
            continue
        d = inst.d_ij_km[i][j]
        k = s.platform_type.get(i, 0)
        if d is None or inst.reachable[i][j] == 0 or (not _can_type_serve_distance(inst, k, d)):
            s.assign_j[j] = None


def _assign_unassigned_points(inst, s, rnd, allow_upgrade=True):
    """把未分配监测点重新插入到可行平台。"""
    opens = _open_set(s)
    if not opens:
        _add_best_platform(inst, s, rnd, focus_unassigned=True)
        opens = _open_set(s)

    # 优先处理距离所有平台较远或可选平台较少的点，避免后期修复困难。
    unassigned = [j for j in range(inst.J) if s.assign_j.get(j, None) is None]
    def scarcity_score(j):
        feasible_cnt = 0
        min_d = 1e100
        for i in opens:
            d = inst.d_ij_km[i][j]
            if d is None or inst.reachable[i][j] == 0:
                continue
            if _min_type_for_dist(inst, d) is not None:
                feasible_cnt += 1
                min_d = min(min_d, d)
        return (feasible_cnt, -min_d)
    unassigned.sort(key=scarcity_score)

    for j in unassigned:
        best_i, _, best_k = _best_assignment_for_point(inst, s, j, _open_set(s), allow_upgrade=allow_upgrade)
        if best_i is not None:
            s.assign_j[j] = best_i
            if best_k is not None:
                s.platform_type[best_i] = max(s.platform_type.get(best_i, 0), best_k)


def _upgrade_uavs_for_assigned_tasks(inst, s):
    """在每个平台只有一架无人机的简化代码结构下，无人机补充用“机型升级”表示。"""
    opens = set(_open_set(s))
    for i in opens:
        s.platform_type[i] = s.platform_type.get(i, 0)

    changed = False
    for j, i in list(s.assign_j.items()):
        if i is None or i not in opens:
            continue
        d = inst.d_ij_km[i][j]
        if d is None or inst.reachable[i][j] == 0:
            s.assign_j[j] = None
            continue
        req_k = _min_type_for_dist(inst, d)
        if req_k is None:
            s.assign_j[j] = None
            continue
        cur_k = s.platform_type.get(i, 0)
        if req_k > cur_k:
            s.platform_type[i] = req_k
            changed = True
    return changed


def _local_reassign_improvement(inst, s, rnd, ratio=0.20):
    """局部重插入：尝试把部分高成本点转移到更合适的平台。"""
    opens = _open_set(s)
    if not opens:
        return
    assigned = [j for j in range(inst.J) if s.assign_j.get(j, None) is not None]
    if not assigned:
        return

    # 优先选择长距离、高能耗边际成本的点进行重插入。
    def current_cost(j):
        i = s.assign_j.get(j, None)
        if i is None:
            return 0.0
        k = s.platform_type.get(i, 0)
        d = inst.d_ij_km[i][j]
        if d is None:
            return 1e100
        return _assignment_incremental_cost(inst, s, i, j, k)

    assigned.sort(key=current_cost, reverse=True)
    m = max(1, int(ratio * len(assigned)))
    selected = assigned[:m]
    rnd.shuffle(selected)

    for j in selected:
        old_i = s.assign_j.get(j, None)
        old_k = s.platform_type.get(old_i, 0) if old_i is not None else 0
        old_c = current_cost(j)
        best_i, best_c, best_k = _best_assignment_for_point(inst, s, j, opens, allow_upgrade=True)
        if best_i is not None and best_c + 1e-9 < old_c:
            s.assign_j[j] = best_i
            s.platform_type[best_i] = max(s.platform_type.get(best_i, 0), best_k)
            # 原平台是否降级不在这里立即处理，避免破坏其他任务可行性。


def targeted_feasibility_repair(inst, sol, rnd, max_rounds=3):
    """统一的可行性修复入口，对应流程图中的 Feasibility repair。"""
    s = clone_solution(sol)
    for _ in range(max_rounds):
        _ensure_minimum_platforms(inst, s, rnd)
        _upgrade_uavs_for_assigned_tasks(inst, s)
        _remove_invalid_assignments(inst, s)
        _assign_unassigned_points(inst, s, rnd, allow_upgrade=True)
        _upgrade_uavs_for_assigned_tasks(inst, s)
        compute_solution_obj(inst, s)
        if s.feasible:
            break
        # 覆盖率不足时继续补平台；若已达 MAX_OPEN，则只能通过任务重分配和机型升级修复。
        _add_best_platform(inst, s, rnd, focus_unassigned=True)
    return s


# -------------------------
# 4类 Destroy operators
# -------------------------
def destroy_platform_removal(inst, sol, rnd):
    """D_P：平台移除。优先移除任务负荷低、单位贡献弱的平台。"""
    s = clone_solution(sol)
    opens = _open_set(s)
    if len(opens) <= 1:
        return s

    phi = _destruction_ratio(inst, rnd)
    remove_cnt = max(1, int(phi * len(opens)))
    remove_cnt = min(remove_cnt, max(1, len(opens) - 1))

    scored = []
    for i in opens:
        load = sum(1 for j, ii in s.assign_j.items() if ii == i)
        score = inst.platform_fixed_wan[i] / float(load + 1)
        scored.append((score, i))
    scored.sort(reverse=True)

    # 80%选择贡献弱平台，20%随机移除，保留搜索多样性。
    if rnd.random() < 0.80:
        removed = [i for _, i in scored[:remove_cnt]]
    else:
        rnd.shuffle(opens)
        removed = opens[:remove_cnt]

    for i in removed:
        s.open_platform[i] = 0
        s.platform_type.pop(i, None)
    for j, ii in list(s.assign_j.items()):
        if ii in removed:
            s.assign_j[j] = None
    return s


def destroy_uav_reduction(inst, sol, rnd):
    """D_U：无人机减少/降配。在当前代码中每个平台一架无人机，因此用机型降级表示。"""
    s = clone_solution(sol)
    opens = [i for i in _open_set(s) if s.platform_type.get(i, 0) > 0]
    if not opens:
        return s

    phi = _destruction_ratio(inst, rnd)
    reduce_cnt = max(1, int(phi * len(opens)))
    rnd.shuffle(opens)
    selected = opens[:reduce_cnt]

    for i in selected:
        s.platform_type[i] = max(0, s.platform_type.get(i, 0) - 1)

    # 降配后，把当前机型已不能服务的任务释放出来，交给 repair 阶段重新匹配或升级。
    _remove_invalid_assignments(inst, s)
    return s


def destroy_task_assignment(inst, sol, rnd):
    """D_A：任务分配破坏。释放高边际成本、长距离或随机任务。"""
    s = clone_solution(sol)
    assigned = [j for j in range(inst.J) if s.assign_j.get(j, None) is not None]
    if not assigned:
        return s

    phi = _destruction_ratio(inst, rnd)
    m = max(1, int(phi * len(assigned)))

    def marginal_cost(j):
        i = s.assign_j.get(j, None)
        if i is None:
            return 0.0
        k = s.platform_type.get(i, 0)
        return _assignment_incremental_cost(inst, s, i, j, k)

    assigned.sort(key=marginal_cost, reverse=True)
    high_cost_part = assigned[:max(m, int(0.50 * len(assigned)))]
    rnd.shuffle(high_cost_part)
    selected = high_cost_part[:m]

    for j in selected:
        s.assign_j[j] = None
    return s


def destroy_route_structure(inst, sol, rnd):
    """D_R：路径结构破坏。释放同一平台内长距离/能耗裕度较紧的局部路线片段。"""
    s = clone_solution(sol)
    opens = _open_set(s)
    if not opens:
        return s

    # 选择一个任务量较大的平台作为被破坏的路径区域。
    platform_loads = []
    for i in opens:
        js = [j for j in range(inst.J) if s.assign_j.get(j, None) == i]
        platform_loads.append((len(js), i, js))
    platform_loads.sort(reverse=True)
    _, chosen_i, js = platform_loads[0]
    if not js:
        return s

    phi = _destruction_ratio(inst, rnd)
    m = max(1, int(phi * len(js)))

    k = s.platform_type.get(chosen_i, 0)
    qk = _energy_quantile(inst, k)

    def route_pressure(j):
        d = inst.d_ij_km[chosen_i][j]
        if d is None:
            return 1e100
        # 距离越远、能耗裕度越小，越优先破坏。
        margin = inst.drone_types[k].battery - qk * (2.0 * d)
        return d - 0.001 * margin

    js.sort(key=route_pressure, reverse=True)
    selected = js[:m]
    for j in selected:
        s.assign_j[j] = None

    # 路径段被破坏后，清空该平台的可视化 route，后续 R_R 会重新构建。
    if chosen_i in s.routes:
        s.routes[chosen_i] = PlatformPlan(routes=[])
    return s


# -------------------------
# 4类 Repair operators
# -------------------------
def repair_platform_addition(inst, sol, rnd):
    """R_P：平台补充/重选。优先补入能降低未覆盖惩罚和工作负荷压力的平台。"""
    s = clone_solution(sol)
    _ensure_minimum_platforms(inst, s, rnd)

    compute_solution_obj(inst, s)
    # 如果覆盖不足且未达到平台上限，继续补平台。
    tries = 0
    while s.coverage_ratio + 1e-12 < inst.min_cover_ratio and tries < 10:
        tries += 1
        if not _add_best_platform(inst, s, rnd, focus_unassigned=True):
            break
        _assign_unassigned_points(inst, s, rnd, allow_upgrade=True)
        compute_solution_obj(inst, s)

    # 平台补充后仍需要做基本任务插入，保证候选解能进入后续 SA 评价。
    _assign_unassigned_points(inst, s, rnd, allow_upgrade=True)
    return targeted_feasibility_repair(inst, s, rnd, max_rounds=2)


def repair_uav_supplement(inst, sol, rnd):
    """R_U：无人机补充/升级。当前代码采用平台机型升级来表示补充更高能力 UAV。"""
    s = clone_solution(sol)
    _ensure_minimum_platforms(inst, s, rnd)
    _upgrade_uavs_for_assigned_tasks(inst, s)
    _assign_unassigned_points(inst, s, rnd, allow_upgrade=True)
    _upgrade_uavs_for_assigned_tasks(inst, s)
    return targeted_feasibility_repair(inst, s, rnd, max_rounds=2)


def repair_task_reassignment(inst, sol, rnd):
    """R_A：任务重分配。将释放任务插入到可行的平台-无人机类型组合。"""
    s = clone_solution(sol)
    _ensure_minimum_platforms(inst, s, rnd)
    _remove_invalid_assignments(inst, s)
    _assign_unassigned_points(inst, s, rnd, allow_upgrade=True)
    return targeted_feasibility_repair(inst, s, rnd, max_rounds=2)


def repair_route_replanning(inst, sol, rnd):
    """R_R：路径重规划。通过局部重插入和路线重建改善路径结构。"""
    s = clone_solution(sol)
    _ensure_minimum_platforms(inst, s, rnd)
    _remove_invalid_assignments(inst, s)
    _assign_unassigned_points(inst, s, rnd, allow_upgrade=True)
    _local_reassign_improvement(inst, s, rnd, ratio=0.25)
    build_routes(inst, s, seed=rnd.randint(1, 10**9))
    return targeted_feasibility_repair(inst, s, rnd, max_rounds=2)


# 为保持向后兼容，保留原函数名的包装。
def destroy_rand_remove_platform(inst, sol, rnd):
    return destroy_platform_removal(inst, sol, rnd)


def destroy_worst_platform(inst, sol, rnd):
    return destroy_platform_removal(inst, sol, rnd)


def destroy_rand_unassign_points(inst, sol, rnd):
    return destroy_task_assignment(inst, sol, rnd)


def repair_greedy(inst, sol, rnd):
    return repair_task_reassignment(inst, sol, rnd)


def repair_local_improve(inst, sol, rnd):
    return repair_route_replanning(inst, sol, rnd)


# =========================================================
# 路线示意（可视化用）
# =========================================================
def build_routes(inst, sol, seed=7):
    rnd = random.Random(seed)
    open_set = set(_open_set(sol))
    sol.routes = {}
    for i in open_set:
        sol.routes[i] = PlatformPlan(routes=[])

    by_platform = {}
    for j in range(inst.J):
        i = sol.assign_j.get(j, None)
        if i is None:
            continue
        if i not in open_set:
            continue
        by_platform.setdefault(i, []).append(j)

    for i, js in by_platform.items():
        rnd.shuffle(js)
        cur = 0
        while cur < len(js):
            step = rnd.randint(3, 8)
            sol.routes[i].routes.append(Route(js[cur:cur + step]))
            cur += step


# =========================================================
# ALNS 主循环 + Q-learning
# =========================================================
def alns_qlearning(inst, seed=7, iters=600):
    rnd = random.Random(seed)
    cur = make_initial_solution(inst, seed=seed, init_open=getattr(inst, "MIN_OPEN", 1))
    best = clone_solution(cur)

    destroy_ops = {
        "DP-平台移除": destroy_platform_removal,
        "DU-无人机降配": destroy_uav_reduction,
        "DA-任务释放": destroy_task_assignment,
        "DR-路径破坏": destroy_route_structure,
    }
    repair_ops = {
        "RP-平台补充": repair_platform_addition,
        "RU-无人机补充": repair_uav_supplement,
        "RA-任务重分配": repair_task_reassignment,
        "RR-路径重规划": repair_route_replanning,
    }

    actions = [d + "+" + r for d in destroy_ops for r in repair_ops]
    ql = QLearner(actions, alpha=0.15, epsilon=0.18)

    T0 = max(10.0, 0.15 * cur.obj)
    T = T0
    no_improve = 0

    st = time.time()
    last = st

    for it in range(1, iters + 1):
        act = ql.choose(rnd)
        dname, rname = act.split("+", 1)

        cand = destroy_ops[dname](inst, cur, rnd)
        cand = repair_ops[rname](inst, cand, rnd)
        compute_solution_obj(inst, cand)
        if not cand.feasible:
            cand = targeted_feasibility_repair(inst, cand, rnd, max_rounds=2)
            compute_solution_obj(inst, cand)

        delta = cand.obj - cur.obj
        accepted = sa_accept(delta, T, rnd)
        if accepted and cand.feasible:
            cur = cand

        reward = 0.0
        if cand.feasible and cand.obj < best.obj - 1e-9:
            best = clone_solution(cand)
            reward = 6.0
            no_improve = 0
        elif accepted and cand.feasible:
            reward = 1.0
            no_improve += 1
        else:
            reward = 0.0
            no_improve += 1

        if not cand.feasible:
            reward -= 1.0

        ql.update(act, reward)

        T *= 0.995
        if no_improve > 100:
            T = 0.6 * T0
            no_improve = 0

        if it % 100 == 0:
            now = time.time()
            block = now - last
            avg_all = (now - st) / float(it)
            avg_100 = block / 100.0
            print("[迭代 %4d] 当前=%.2f 可行=%s 覆盖=%.3f | 最优=%.2f 覆盖=%.3f 开平台=%d 无人机=%d 温度=%.2f | "
                  "本100次=%.2fs 全局均值=%.4fs/次 最近均值=%.4fs/次"
                  % (it, cur.obj, str(cur.feasible), cur.coverage_ratio,
                     best.obj, best.coverage_ratio, len(_open_set(best)), len(_open_set(best)), T,
                     block, avg_all, avg_100))
            last = now

    total = time.time() - st
    best.runtime_s = total

    print("\nALNS 完成：%.2fs（平均 %.4fs/次）" % (total, total / float(iters)))
    print("最优目标值=%.2f（万元），可行=%s，覆盖率=%.3f，开平台=%d，无人机=%d（MIN_OPEN=%d）"
          % (best.obj, str(best.feasible), best.coverage_ratio,
             len(_open_set(best)), len(_open_set(best)), inst.MIN_OPEN))

    top = sorted(ql.Q.items(), key=lambda kv: kv[1], reverse=True)[:6]
    print("\n动作Q值Top：")
    for a, q in top:
        print("  %-18s Q=%.3f" % (a, q))

    return best


# =========================================================
# 表6统一对比算法：Greedy / ALNS-only / ALNS+SA / ALNS+Q-learning+SA
# 说明：四个算法共用同一个 build_instance、compute_solution_obj、初始解与约束口径。
# =========================================================
def _finalize_solution_for_compare(inst, sol, seed=7):
    compute_solution_obj(inst, sol)
    try:
        build_routes(inst, sol, seed=seed)
    except Exception:
        pass
    return sol


def greedy_only(inst, seed=7):
    """
    Greedy：只生成初始解，不进行 destroy-repair，不使用 SA，不使用 Q-learning。
    """
    st = time.time()
    cur = make_initial_solution(
        inst,
        seed=seed,
        init_open=getattr(inst, "MIN_OPEN", 1)
    )
    cur = _finalize_solution_for_compare(inst, cur, seed=seed)
    cur.runtime_s = time.time() - st
    return cur


def alns_only(inst, seed=7, iters=600):
    """
    ALNS-only：destroy-repair + 贪婪接受。
    不使用 SA，不使用 Q-learning。
    """
    rnd = random.Random(seed)
    st = time.time()

    cur = make_initial_solution(
        inst,
        seed=seed,
        init_open=getattr(inst, "MIN_OPEN", 1)
    )
    compute_solution_obj(inst, cur)
    best = clone_solution(cur)

    destroy_ops = {
        "DP-平台移除": destroy_platform_removal,
        "DU-无人机降配": destroy_uav_reduction,
        "DA-任务释放": destroy_task_assignment,
        "DR-路径破坏": destroy_route_structure,
    }
    repair_ops = {
        "RP-平台补充": repair_platform_addition,
        "RU-无人机补充": repair_uav_supplement,
        "RA-任务重分配": repair_task_reassignment,
        "RR-路径重规划": repair_route_replanning,
    }

    for _ in range(1, iters + 1):
        dname = rnd.choice(list(destroy_ops.keys()))
        rname = rnd.choice(list(repair_ops.keys()))

        cand = destroy_ops[dname](inst, cur, rnd)
        cand = repair_ops[rname](inst, cand, rnd)
        compute_solution_obj(inst, cand)
        if not cand.feasible:
            cand = targeted_feasibility_repair(inst, cand, rnd, max_rounds=2)
            compute_solution_obj(inst, cand)

        # ALNS-only：只接受可行且不劣于当前解的候选解
        if cand.feasible and cand.obj <= cur.obj + 1e-9:
            cur = cand

        # 始终保存历史最优解，避免迭代算法结果劣于初始 Greedy 解
        if cand.feasible and cand.obj < best.obj - 1e-9:
            best = clone_solution(cand)

    best = _finalize_solution_for_compare(inst, best, seed=seed)
    best.runtime_s = time.time() - st
    return best


def alns_sa_only(inst, seed=7, iters=600):
    """
    ALNS+SA：destroy-repair + 模拟退火接受准则。
    不使用 Q-learning。
    """
    rnd = random.Random(seed)
    st = time.time()

    cur = make_initial_solution(
        inst,
        seed=seed,
        init_open=getattr(inst, "MIN_OPEN", 1)
    )
    compute_solution_obj(inst, cur)
    best = clone_solution(cur)

    destroy_ops = {
        "DP-平台移除": destroy_platform_removal,
        "DU-无人机降配": destroy_uav_reduction,
        "DA-任务释放": destroy_task_assignment,
        "DR-路径破坏": destroy_route_structure,
    }
    repair_ops = {
        "RP-平台补充": repair_platform_addition,
        "RU-无人机补充": repair_uav_supplement,
        "RA-任务重分配": repair_task_reassignment,
        "RR-路径重规划": repair_route_replanning,
    }

    T0 = max(10.0, 0.15 * cur.obj)
    T = T0
    cooling = 0.995

    for _ in range(1, iters + 1):
        dname = rnd.choice(list(destroy_ops.keys()))
        rname = rnd.choice(list(repair_ops.keys()))

        cand = destroy_ops[dname](inst, cur, rnd)
        cand = repair_ops[rname](inst, cand, rnd)
        compute_solution_obj(inst, cand)
        if not cand.feasible:
            cand = targeted_feasibility_repair(inst, cand, rnd, max_rounds=2)
            compute_solution_obj(inst, cand)

        delta = cand.obj - cur.obj
        accepted = sa_accept(delta, T, rnd)
        if accepted and cand.feasible:
            cur = cand

        if cand.feasible and cand.obj < best.obj - 1e-9:
            best = clone_solution(cand)

        T *= cooling

    best = _finalize_solution_for_compare(inst, best, seed=seed)
    best.runtime_s = time.time() - st
    return best


def run_table6_medium_comparison(seed=7, iters=600, save_csv=True):
    """
    统一生成中规模表6对比结果。
    返回 results, inst0, best_proposed。
    """
    import copy

    inst0 = build_instance(seed=seed)
    algorithms = [
        ("Greedy", lambda inst: greedy_only(inst, seed=seed)),
        ("ALNS-only", lambda inst: alns_only(inst, seed=seed, iters=iters)),
        ("ALNS+SA", lambda inst: alns_sa_only(inst, seed=seed, iters=iters)),
        ("ALNS+Q-learning+SA", lambda inst: alns_qlearning(inst, seed=seed, iters=iters)),
    ]

    results = []
    best_proposed = None

    for alg_name, alg_func in algorithms:
        # 每个算法使用同一个实例的深拷贝，避免前一个算法修改实例或解对象后影响后一个算法。
        inst = copy.deepcopy(inst0)
        sol = alg_func(inst)
        compute_solution_obj(inst, sol)

        runtime_s = getattr(sol, "runtime_s", 0.0)
        open_cnt = len(_open_set(sol))
        uncovered_cnt = sum(1 for j in range(inst.J) if sol.assign_j.get(j, None) is None)

        results.append({
            "Instance size": "Medium-scale",
            "Algorithm": alg_name,
            "Objective value": sol.obj,
            "Coverage rate (%)": sol.coverage_ratio * 100.0,
            "Number of open platforms": open_cnt,
            "Computational time (s)": runtime_s,
            "Feasible": sol.feasible,
            "Uncovered points": uncovered_cnt,
            "Solution": sol,
        })

        if alg_name == "ALNS+Q-learning+SA":
            best_proposed = sol

    feasible_objs = [r["Objective value"] for r in results if r["Feasible"]]
    ref_obj = min(feasible_objs) if feasible_objs else None

    for r in results:
        if ref_obj is None or abs(ref_obj) < 1e-12 or (not r["Feasible"]):
            r["Deviation from BKS (%)"] = None
        else:
            r["Deviation from BKS (%)"] = 100.0 * (r["Objective value"] - ref_obj) / abs(ref_obj)

    if save_csv:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            base_dir = os.getcwd()
        csv_path = os.path.join(base_dir, "table6_medium_comparison.csv")
        fieldnames = [
            "Instance size", "Algorithm", "Objective value", "Deviation from BKS (%)",
            "Coverage rate (%)", "Number of open platforms", "Computational time (s)",
            "Feasible", "Uncovered points"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = dict((k, r.get(k, "")) for k in fieldnames)
                if row["Deviation from BKS (%)"] is None:
                    row["Deviation from BKS (%)"] = "N/A"
                writer.writerow(row)
        print("\n已保存中规模对比结果CSV：%s" % csv_path)

    return results, inst0, best_proposed


def print_table6_results(results):
    print("\nMedium-scale benchmark comparison")
    print("Instance size\tAlgorithm\tObjective value\tDeviation from BKS (%)\tCoverage rate (%)\tNumber of open platforms\tComputational time (s)\tFeasibility / Remarks")
    for r in results:
        dev = r["Deviation from BKS (%)"]
        dev_str = "N/A" if dev is None else "%.2f" % dev
        remark = "Feasible" if r["Feasible"] else "Infeasible"
        print("%s\t%s\t%.2f\t%s\t%.2f\t%d\t%.2f\t%s" % (
            r["Instance size"],
            r["Algorithm"],
            r["Objective value"],
            dev_str,
            r["Coverage rate (%)"],
            r["Number of open platforms"],
            r["Computational time (s)"],
            remark
        ))

    greedy = None
    for r in results:
        if r["Algorithm"] == "Greedy":
            greedy = r
            break
    if greedy is not None:
        for r in results:
            if r["Algorithm"] != "Greedy" and r["Feasible"] and greedy["Feasible"]:
                if r["Objective value"] > greedy["Objective value"] + 1e-6:
                    print("[WARNING] %s 的目标值高于 Greedy。若该结果用于论文，请检查算子、接受准则和可行性修复逻辑。" % r["Algorithm"])


# =========================================================
# 可视化：明确体现绕行（红虚线=直线对比, 绿实线=绕行路径）
# 禁飞区：统一红色
# =========================================================
def plot_polyline(ax, pts, alpha=0.25, linewidth=1.0, color=None, linestyle='-'):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, alpha=alpha, linewidth=linewidth, color=color, linestyle=linestyle)


def _pick_zoom_window(inst, sol, zoom_w=3500.0, zoom_h=3500.0):
    open_ids = _open_set(sol)
    if open_ids:
        cx = sum(inst.platform_xy[i][0] for i in open_ids) / float(len(open_ids))
        cy = sum(inst.platform_xy[i][1] for i in open_ids) / float(len(open_ids))
    else:
        cx = sum(inst.point_xy[j][0] for j in range(inst.J)) / float(inst.J)
        cy = sum(inst.point_xy[j][1] for j in range(inst.J)) / float(inst.J)

    xmin = max(0.0, cx - zoom_w / 2.0)
    xmax = min(inst.W, cx + zoom_w / 2.0)
    ymin = max(0.0, cy - zoom_h / 2.0)
    ymax = min(inst.H, cy + zoom_h / 2.0)

    if xmax - xmin < zoom_w:
        if xmin == 0.0:
            xmax = min(inst.W, zoom_w)
        elif xmax == inst.W:
            xmin = max(0.0, inst.W - zoom_w)
    if ymax - ymin < zoom_h:
        if ymin == 0.0:
            ymax = min(inst.H, zoom_h)
        elif ymax == inst.H:
            ymin = max(0.0, inst.H - zoom_h)

    return xmin, xmax, ymin, ymax


def _in_window(x, y, win):
    xmin, xmax, ymin, ymax = win
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def draw_scene(inst, sol, ax, title,
               window=None,
               show_nofly_label=False,
               show_candidate_label=False,
               label_all_candidates_in_window=False,
               detour_margin=220.0,
               max_assign_lines=150,
               max_route_paths=25):
    open_set = set(_open_set(sol))

    # 1) 禁飞区（统一红色）
    for zid, poly in enumerate(inst.nofly):
        if window is not None:
            xmin, xmax, ymin, ymax = window
            bx1, bx2, by1, by2 = poly_bbox(poly)
            if bx2 < xmin or bx1 > xmax or by2 < ymin or by1 > ymax:
                continue

        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(xs, ys, linewidth=1.6, color="red")
        ax.fill(xs, ys, alpha=0.18, color="red")

        if show_nofly_label:
            cx = sum([p[0] for p in poly]) / float(len(poly))
            cy = sum([p[1] for p in poly]) / float(len(poly))
            ax.text(cx, cy, "禁飞区%d" % zid, fontsize=9, weight="bold", color="red")

    # 2) 危险源
    hx, hy = [], []
    for (x, y) in inst.hazards:
        if window is None or _in_window(x, y, window):
            hx.append(x)
            hy.append(y)
    ax.scatter(hx, hy, marker="*", s=120, label="危险源（★）")

    # 3) 平台候选点
    px_ok, py_ok, px_bad, py_bad = [], [], [], []
    for i in range(inst.I):
        x, y = inst.platform_xy[i]
        if window is not None and (not _in_window(x, y, window)):
            continue
        if inst.platform_valid[i]:
            px_ok.append(x)
            py_ok.append(y)
        else:
            px_bad.append(x)
            py_bad.append(y)

    ax.scatter(px_ok, py_ok, s=10, alpha=0.30, label="平台候选点（有效）")
    ax.scatter(px_bad, py_bad, s=10, alpha=0.10, label="平台候选点（无效）")

    if show_candidate_label and label_all_candidates_in_window and window is not None:
        for i in range(inst.I):
            x, y = inst.platform_xy[i]
            if not _in_window(x, y, window):
                continue
            ax.text(x + 18, y + 18, "P%d" % i, fontsize=6)

    # 4) 已开放平台（橙色方块）
    opx, opy = [], []
    for i in open_set:
        x, y = inst.platform_xy[i]
        if window is None or _in_window(x, y, window):
            opx.append(x)
            opy.append(y)

    ax.scatter(opx, opy, marker="s", s=90, color="orange",
               label="橙色方块：已开放平台/基站（每个平台1架无人机）")
    for i in open_set:
        x, y = inst.platform_xy[i]
        if window is None or _in_window(x, y, window):
            k = sol.platform_type.get(i, 0)
            ax.text(x + 35, y + 35, "OP%d(K%d)" % (i, k), fontsize=8, weight="bold")

    # 5) 监测点（覆盖/未覆盖）
    cov_x, cov_y, unc_x, unc_y = [], [], [], []
    for j in range(inst.J):
        x, y = inst.point_xy[j]
        if window is not None and (not _in_window(x, y, window)):
            continue
        if sol.assign_j.get(j, None) is None:
            unc_x.append(x)
            unc_y.append(y)
        else:
            cov_x.append(x)
            cov_y.append(y)

    ax.scatter(cov_x, cov_y, s=14, label="已覆盖监测点")
    if unc_x:
        ax.scatter(unc_x, unc_y, marker="x", s=45, label="未覆盖监测点（×）")

    # 6) 绕行可视化：红虚线=直线对比（可能穿越禁飞区），绿实线=绕行路径
    js = list(range(inst.J))
    random.shuffle(js)
    drawn = 0

    for j in js:
        i = sol.assign_j.get(j, None)
        if i is None:
            continue

        A = inst.platform_xy[i]
        B = inst.point_xy[j]

        if window is not None:
            if (not _in_window(A[0], A[1], window)) and (not _in_window(B[0], B[1], window)):
                continue

        cross_any = False
        for poly in inst.nofly:
            if segment_intersects_poly(A[0], A[1], B[0], B[1], poly):
                cross_any = True
                break

        pts = detour_path(A, B, inst.nofly, margin=detour_margin)

        if cross_any:
            plot_polyline(ax, [A, B], alpha=0.22, linewidth=0.9, linestyle="--", color="red")

        plot_polyline(ax, pts, alpha=0.38, linewidth=1.1, linestyle="-", color="green")

        drawn += 1
        if drawn >= max_assign_lines:
            break

    # 7) 巡检路线示意（粗线，也用绕行折线）
    route_drawn = 0
    if sol.routes:
        for i in list(open_set):
            plan = sol.routes.get(i, None)
            if plan is None or not plan.routes:
                continue
            px, py = inst.platform_xy[i]
            for rr in plan.routes[:3]:
                seq = rr.seq
                if not seq:
                    continue
                nodes = [(px, py)] + [inst.point_xy[j] for j in seq] + [(px, py)]

                if window is not None:
                    ok = False
                    for (x, y) in nodes:
                        if _in_window(x, y, window):
                            ok = True
                            break
                    if not ok:
                        continue

                for t in range(len(nodes) - 1):
                    pts2 = detour_path(nodes[t], nodes[t + 1], inst.nofly, margin=detour_margin)
                    plot_polyline(ax, pts2, alpha=0.65, linewidth=2.0, linestyle="-", color="green")

                route_drawn += 1
                if route_drawn >= max_route_paths:
                    break
            if route_drawn >= max_route_paths:
                break

    # 视窗设置
    if window is None:
        ax.set_xlim(0, inst.W)
        ax.set_ylim(0, inst.H)
    else:
        xmin, xmax, ymin, ymax = window
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()

    line_straight = ax.plot([0, 1], [0, 1], linestyle="--", color="red", alpha=0.35)[0]
    line_detour = ax.plot([0, 1], [0, 1], linestyle="-", color="green", alpha=0.55)[0]
    handles += [line_straight, line_detour]
    labels += ["红虚线：直线对比（可能穿越禁飞区）", "绿实线：绕行路径（避开禁飞区）"]
    line_straight.remove()
    line_detour.remove()

    if Patch is not None:
        handles.append(Patch(facecolor="red", edgecolor="red", alpha=0.18))
        labels.append("红色半透明矩形：禁飞区")

    ax.legend(handles, labels, loc="upper right", fontsize=8)


def visualize_dual(inst, sol,
                   overview_file="solution_overview_medium.png",
                   zoom_file="solution_zoom_medium.png",
                   detour_margin=220.0,
                   zoom_w=3500.0,
                   zoom_h=3500.0):
    if plt is None:
        print("\n[警告] matplotlib 导入失败，无法生成可视化。请先执行：pip install matplotlib")
        return

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_dir = os.getcwd()

    overview_path = os.path.join(base_dir, overview_file)
    zoom_path = os.path.join(base_dir, zoom_file)

    # 总览图
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    draw_scene(
        inst, sol, ax1,
        title="中规模实例总览：目标=%.2f万元 覆盖=%.2f%% 开平台=%d 无人机=%d 禁飞区=%d"
              % (sol.obj, sol.coverage_ratio * 100.0, len(_open_set(sol)), len(_open_set(sol)), inst.Z),
        window=None,
        show_nofly_label=False,
        show_candidate_label=False,
        label_all_candidates_in_window=False,
        detour_margin=detour_margin,
        max_assign_lines=80,
        max_route_paths=15
    )
    plt.tight_layout()
    plt.savefig(overview_path, dpi=220)
    plt.close(fig1)
    print("已保存总览图：", overview_path)

    # 局部放大图
    win = _pick_zoom_window(inst, sol, zoom_w=zoom_w, zoom_h=zoom_h)
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    draw_scene(
        inst, sol, ax2,
        title="局部放大：红虚线=直线对比；绿实线=绕行路径；红色矩形=禁飞区",
        window=win,
        show_nofly_label=True,
        show_candidate_label=True,
        label_all_candidates_in_window=True,
        detour_margin=detour_margin,
        max_assign_lines=100,
        max_route_paths=20
    )
    plt.tight_layout()
    plt.savefig(zoom_path, dpi=260)
    plt.close(fig2)
    print("已保存局部图：", zoom_path)


# =========================================================
# 结果输出
# =========================================================
def make_feasibility_note(best, inst):
    if best.feasible:
        return "可行，满足最低覆盖率、最小开放平台数、平台合法性、禁飞区绕避与无人机续航约束"
    return "不可行，可能存在覆盖率不足、平台无效、监测点不可达或无人机续航约束不满足"


def make_gap_string_and_note(inst, best):
    gap = calc_gap_percent(best.obj, getattr(inst, "ref_obj_wan", None))
    if gap is None:
        return "N/A", "未设置精确最优值/下界/参考目标值，启发式算法无法直接计算严格Gap"
    return "%.2f" % gap, "已根据参考目标值计算Gap"





# =============================================================================
# 第六章敏感性分析专用扩展
# 说明：
# 1. 保留原TQ-ALNS破坏—修复、Q-learning和SA框架；
# 2. 固定同一中规模空间实例，避免将随机实例差异误认为参数效应；
# 3. 平台数量在[1,5]内生决定；
# 4. 每个平台允许配置多架同型UAV，UAV数量根据工作负荷内生确定；
# 5. 动态更新机会约束置信水平、巡检频次和电池容量；
# 6. 支持时间限制、断点续跑、统计汇总和Figure 6–8自动绘制。
# =============================================================================

import copy
import hashlib
import json
import statistics
import traceback
from datetime import datetime
from statistics import NormalDist


# 保存原始中规模实例生成函数，后续构造固定基础实例时调用。
_build_instance_medium_original = build_instance
_clone_solution_original = clone_solution


def _normal_quantile(beta):
    beta = float(beta)
    if not 0.0 < beta < 1.0:
        raise ValueError("beta必须位于(0,1)内，当前值=%s" % beta)
    return NormalDist().inv_cdf(beta)


def _total_uav_count(sol):
    total = 0
    for value in getattr(sol, "fleet", {}).values():
        try:
            total += int(value)
        except Exception:
            pass
    return total


def clone_solution(sol):
    """扩展原复制函数，使敏感性分析新增指标能够随解一起复制。"""
    s = _clone_solution_original(sol)
    extra_attrs = (
        "total_uavs",
        "total_workload_h",
        "estimated_makespan_h",
        "total_route_distance_km",
        "min_energy_margin_Wh",
        "iterations_done",
        "stop_reason",
        "overloaded_platforms",
    )
    for name in extra_attrs:
        if hasattr(sol, name):
            setattr(s, name, copy.deepcopy(getattr(sol, name)))
    return s


def build_sensitivity_base_instance(seed=CH6_INSTANCE_SEED):
    """
    构造第六章唯一固定基础实例。
    规模沿用上传代码的中规模设置：15个候选平台、50个监测点、
    3个危险源和3个禁飞区。
    """
    inst = _build_instance_medium_original(seed=int(seed))

    inst.MIN_OPEN = int(CH6_MIN_OPEN_PLATFORMS)
    inst.MAX_OPEN = int(CH6_MAX_OPEN_PLATFORMS)
    inst.max_uavs_per_platform = int(CH6_MAX_UAVS_PER_PLATFORM)

    inst.beta = float(CH6_BASE_BETA)
    inst.alpha = inst.beta
    inst.z_alpha = _normal_quantile(inst.beta)

    inst.base_req_freq = list(inst.req_freq)
    inst.base_battery_Wh = [float(dt.battery) for dt in inst.drone_types]
    inst.base_energy_mu = [float(dt.energy_mu) for dt in inst.drone_types]
    inst.base_energy_sigma = [float(dt.energy_sigma) for dt in inst.drone_types]

    inst.demand_scale = float(CH6_BASE_DEMAND_SCALE)
    inst.battery_scale = float(CH6_BASE_BATTERY_SCALE)
    inst.instance_seed = int(seed)

    # 巡检服务能耗（Wh/次），用于路线级电量审计。
    # 该参数与论文中的固定巡检作业能耗对应。
    inst.service_energy_Wh_per_visit = [25.0, 40.0, 70.0]

    # 预计算监测点之间的安全距离。若直线穿越禁飞区，沿用原代码的1.40绕行系数。
    inst.d_jj_km = [
        [0.0 for _ in range(inst.J)]
        for _ in range(inst.J)
    ]
    detour_factor = 1.40
    for u in range(inst.J):
        ax, ay = inst.point_xy[u]
        for v in range(u + 1, inst.J):
            bx, by = inst.point_xy[v]
            base_km = euclid_m(ax, ay, bx, by) / 1000.0
            cross_cnt = 0
            for poly in inst.nofly:
                if segment_intersects_poly(ax, ay, bx, by, poly):
                    cross_cnt += 1
            dist = base_km * (detour_factor if cross_cnt > 0 else 1.0)
            inst.d_jj_km[u][v] = dist
            inst.d_jj_km[v][u] = dist

    return inst


def apply_sensitivity_level(base_inst, group, level):
    """
    单因素敏感性设计：每次从同一个基础实例深复制，然后只改变一个参数。
    demand_scale作用于重复巡检频次，不改变监测点坐标与数量。
    """
    inst = copy.deepcopy(base_inst)

    # 先恢复全部基准值。
    inst.beta = float(CH6_BASE_BETA)
    inst.alpha = inst.beta
    inst.z_alpha = _normal_quantile(inst.beta)
    inst.req_freq = list(inst.base_req_freq)
    inst.demand_scale = float(CH6_BASE_DEMAND_SCALE)
    inst.battery_scale = float(CH6_BASE_BATTERY_SCALE)

    for k, dt in enumerate(inst.drone_types):
        dt.battery = float(inst.base_battery_Wh[k])
        dt.energy_mu = float(inst.base_energy_mu[k])
        dt.energy_sigma = float(inst.base_energy_sigma[k])

    level = float(level)

    if group == "beta":
        inst.beta = level
        inst.alpha = level
        inst.z_alpha = _normal_quantile(level)

    elif group == "demand_scale":
        inst.demand_scale = level
        # 四舍五入到整数巡检频次；每个监测点至少巡检1次。
        inst.req_freq = [
            max(1, int(math.floor(float(r) * level + 0.5)))
            for r in inst.base_req_freq
        ]

    elif group == "battery_scale":
        inst.battery_scale = level
        for k, dt in enumerate(inst.drone_types):
            dt.battery = float(inst.base_battery_Wh[k]) * level

    else:
        raise ValueError("未知敏感性参数组：%s" % group)

    return inst


def _construct_energy_feasible_sorties(inst, platform_i, assigned_points, k):
    """
    针对给定平台与机型构造确定性的最近邻巡检架次。

    每个监测点按req_freq展开为重复任务。单个架次从平台出发，
    连续访问若干任务后返回同一平台。加入下一任务前同时检查：
    1) 所有任务点均位于机型最大服务半径内；
    2) 路线飞行能耗与巡检服务能耗之和不超过电池容量；
    3) 能耗系数采用beta分位数，因此beta与电池容量会真实改变路线分段。
    """
    dt = inst.drone_types[k]
    q_energy = dt.energy_mu + inst.z_alpha * dt.energy_sigma
    service_energy = float(inst.service_energy_Wh_per_visit[k])

    tasks = []
    for j, d in assigned_points:
        if d is None or d > dt.max_radius + 1e-9:
            return None
        one_task_energy = q_energy * (2.0 * d) + service_energy
        if one_task_energy > dt.battery + 1e-9:
            return None
        tasks.extend([int(j)] * int(inst.req_freq[j]))

    if not tasks:
        return {
            "workload_h": 0.0,
            "distance_km": 0.0,
            "min_margin_Wh": dt.battery,
            "sortie_count": 0,
        }

    remaining = tasks[:]
    total_distance = 0.0
    total_inspection_h = 0.0
    min_margin = float("inf")
    sortie_count = 0

    while remaining:
        # 先服务距离平台最远的任务，避免把难插入任务留到最后。
        first_pos = max(
            range(len(remaining)),
            key=lambda pos: inst.d_ij_km[platform_i][remaining[pos]]
        )
        first = remaining.pop(first_pos)

        route = [first]
        route_distance = inst.d_ij_km[platform_i][first]
        route_service_energy = service_energy
        current = first

        while remaining:
            best = None
            for pos, j in enumerate(remaining):
                segment = inst.d_jj_km[current][j]
                return_distance = inst.d_ij_km[platform_i][j]
                candidate_distance = route_distance + segment
                candidate_service_energy = route_service_energy + service_energy
                candidate_energy = (
                    q_energy * (candidate_distance + return_distance)
                    + candidate_service_energy
                )
                if candidate_energy <= dt.battery + 1e-9:
                    key = (segment, return_distance, j)
                    if best is None or key < best[0]:
                        best = (
                            key, pos, j,
                            candidate_distance,
                            candidate_service_energy
                        )

            if best is None:
                break

            _, pos, j, route_distance, route_service_energy = best
            remaining.pop(pos)
            route.append(j)
            current = j

        route_distance += inst.d_ij_km[platform_i][current]
        route_energy = q_energy * route_distance + route_service_energy
        margin = dt.battery - route_energy

        if margin < -1e-6:
            return None

        total_distance += route_distance
        total_inspection_h += sum(
            inst.inspect_time_h[j] for j in route
        )
        min_margin = min(min_margin, margin)
        sortie_count += 1

    workload_h = total_distance / dt.speed + total_inspection_h
    return {
        "workload_h": workload_h,
        "distance_km": total_distance,
        "min_margin_Wh": min_margin,
        "sortie_count": sortie_count,
    }


def _all_types_serving_platform(inst, platform_i, assigned_points):
    """
    返回能够服务平台全部已分配任务的机型方案。
    路线距离、工作负荷和能量裕度均由路线级架次构造得到。
    """
    options = []
    if not assigned_points:
        dt = inst.drone_types[0]
        options.append({
            "k": 0,
            "workload_h": 0.0,
            "required_uavs": 1,
            "distance_km": 0.0,
            "min_margin_Wh": dt.battery,
            "fleet_cost_wan": dt.purchase_cost + dt.maintain_cost / 365.0,
            "sortie_count": 0,
        })
        return options

    for k, dt in enumerate(inst.drone_types):
        route_metrics = _construct_energy_feasible_sorties(
            inst, platform_i, assigned_points, k
        )
        if route_metrics is None:
            continue

        workload_h = float(route_metrics["workload_h"])
        required_uavs = max(
            1,
            int(math.ceil(workload_h / float(inst.T_day) - 1e-12))
        )
        fleet_cost = required_uavs * (
            dt.purchase_cost + dt.maintain_cost / 365.0
        )

        options.append({
            "k": k,
            "workload_h": workload_h,
            "required_uavs": required_uavs,
            "distance_km": float(route_metrics["distance_km"]),
            "min_margin_Wh": float(route_metrics["min_margin_Wh"]),
            "fleet_cost_wan": fleet_cost,
            "sortie_count": int(route_metrics["sortie_count"]),
        })

    return options

def compute_solution_obj(inst, sol):
    """
    第六章统一目标与可行性审计。

    目标单位：万元。
    UAV数量根据平台工作负荷内生确定：
        n_i = ceil(workload_i / T_day)
    因而需求增加时可以先在既有平台增配UAV，而不必机械地新建平台。
    """
    open_set = set(_open_set(sol))
    feasible = True

    if len(open_set) < inst.MIN_OPEN or len(open_set) > inst.MAX_OPEN:
        feasible = False

    for i in open_set:
        if i < 0 or i >= inst.I or not inst.platform_valid[i]:
            feasible = False

    # 清理无效分配；此阶段只要求存在至少一种可行机型。
    cleaned_assign = {}
    for j in range(inst.J):
        i = sol.assign_j.get(j, None)
        if i is None or i not in open_set:
            cleaned_assign[j] = None
            continue

        d = inst.d_ij_km[i][j]
        if d is None or inst.reachable[i][j] == 0:
            cleaned_assign[j] = None
            continue

        any_type = False
        for k in range(inst.K):
            if _can_type_serve_distance(inst, k, d):
                any_type = True
                break
        cleaned_assign[j] = i if any_type else None

    sol.assign_j = cleaned_assign
    sol.assign = {}
    sol.covered = {}
    sol.platform_type = {}
    sol.fleet = {}

    by_platform = dict((i, []) for i in open_set)
    for j in range(inst.J):
        i = sol.assign_j.get(j, None)
        if i is None:
            sol.covered[j] = 0
        else:
            sol.covered[j] = 1
            by_platform[i].append((j, inst.d_ij_km[i][j]))

    covered_cnt = sum(sol.covered.values())
    sol.coverage_ratio = covered_cnt / float(inst.J)
    if sol.coverage_ratio + 1e-12 < inst.min_cover_ratio:
        feasible = False

    platform_cost = sum(inst.platform_fixed_wan[i] for i in open_set)
    uncovered_penalty = (
        inst.J - covered_cnt
    ) * inst.penalty_uncovered_wan

    drone_cost = 0.0
    energy_cost_wan = 0.0
    total_workload_h = 0.0
    total_distance_km = 0.0
    min_energy_margin = float("inf")
    platform_completion_times = []
    overloaded = []

    for i in sorted(open_set):
        options = _all_types_serving_platform(inst, i, by_platform.get(i, []))
        if not options:
            feasible = False
            overloaded.append(i)
            # 为保证目标仍可比较，给予一个高成本占位配置。
            chosen = {
                "k": inst.K - 1,
                "workload_h": 0.0,
                "required_uavs": inst.max_uavs_per_platform + 1,
                "distance_km": 0.0,
                "min_margin_Wh": -1.0,
                "fleet_cost_wan": (
                    inst.max_uavs_per_platform + 1
                ) * (
                    inst.drone_types[-1].purchase_cost
                    + inst.drone_types[-1].maintain_cost / 365.0
                ),
            }
        else:
            capacity_feasible = [
                op for op in options
                if op["required_uavs"] <= inst.max_uavs_per_platform
            ]
            if capacity_feasible:
                # 以机队配置成本最小为主，完成时间和机型编号为次级准则。
                chosen = min(
                    capacity_feasible,
                    key=lambda op: (
                        op["fleet_cost_wan"],
                        op["workload_h"] / op["required_uavs"],
                        op["k"],
                    )
                )
            else:
                feasible = False
                overloaded.append(i)
                chosen = min(
                    options,
                    key=lambda op: (
                        op["required_uavs"],
                        op["fleet_cost_wan"],
                        op["k"],
                    )
                )

        k = int(chosen["k"])
        n_uav = int(chosen["required_uavs"])
        sol.platform_type[i] = k
        sol.fleet[(i, k)] = n_uav

        drone_cost += float(chosen["fleet_cost_wan"])
        total_workload_h += float(chosen["workload_h"])
        total_distance_km += float(chosen["distance_km"])
        min_energy_margin = min(
            min_energy_margin,
            float(chosen["min_margin_Wh"])
        )
        platform_completion_times.append(
            float(chosen["workload_h"]) / max(1, n_uav)
        )

        # 能源成本按实际构造的多任务架次总距离计算。
        energy_cost_wan += (
            inst.energy_cost_yuan_per_km
            * float(chosen["distance_km"])
            / 10000.0
        )

        for j, d in by_platform.get(i, []):
            sol.assign[(i, j, k)] = 1

    sol.total_uavs = _total_uav_count(sol)
    sol.total_workload_h = total_workload_h
    sol.estimated_makespan_h = (
        max(platform_completion_times) if platform_completion_times else 0.0
    )
    sol.total_route_distance_km = total_distance_km
    sol.min_energy_margin_Wh = (
        min_energy_margin if min_energy_margin != float("inf") else 0.0
    )
    sol.overloaded_platforms = overloaded

    sol.obj = (
        platform_cost
        + drone_cost
        + energy_cost_wan
        + uncovered_penalty
    )
    sol.feasible = bool(feasible)
    return sol.obj, sol.feasible


def make_initial_solution(inst, seed=7, init_open=CH6_INITIAL_OPEN_PLATFORMS):
    """构造与固定实例对应的可重复初始解。"""
    rnd = random.Random(int(seed))
    sol = Solution()
    sol.open_platform = dict((i, 0) for i in range(inst.I))
    sol.assign_j = dict((j, None) for j in range(inst.J))
    sol.platform_type = {}

    target_open = max(
        inst.MIN_OPEN,
        min(int(init_open), inst.MAX_OPEN)
    )

    # 使用覆盖增益和平均距离选择初始平台，而不是按候选点编号截取。
    while len(_open_set(sol)) < target_open:
        added = _add_best_platform(
            inst, sol, rnd, focus_unassigned=True
        )
        if not added:
            break

    _assign_unassigned_points(inst, sol, rnd, allow_upgrade=True)
    compute_solution_obj(inst, sol)

    # 如覆盖不足，继续补平台并修复，直至达到上限。
    while (
        not sol.feasible
        and len(_open_set(sol)) < inst.MAX_OPEN
    ):
        if not _add_best_platform(
            inst, sol, rnd, focus_unassigned=True
        ):
            break
        _assign_unassigned_points(inst, sol, rnd, allow_upgrade=True)
        compute_solution_obj(inst, sol)

    return sol


def _platform_workload_from_solution(inst, sol, platform_i):
    k = sol.platform_type.get(platform_i, 0)
    dt = inst.drone_types[k]
    total = 0.0
    points = []
    for j in range(inst.J):
        if sol.assign_j.get(j, None) != platform_i:
            continue
        d = inst.d_ij_km[platform_i][j]
        if d is None:
            continue
        w = int(inst.req_freq[j]) * (
            (2.0 * d) / dt.speed + inst.inspect_time_h[j]
        )
        total += w
        points.append((w, j))
    return total, points


def _rebalance_overloaded_platforms(inst, sol, rnd):
    """
    当某个平台所需UAV数超过上限时，将高工作负荷任务转移至其他平台。
    """
    s = clone_solution(sol)
    compute_solution_obj(inst, s)

    for overloaded_i in list(getattr(s, "overloaded_platforms", [])):
        total_w, points = _platform_workload_from_solution(
            inst, s, overloaded_i
        )
        cap_h = inst.max_uavs_per_platform * inst.T_day
        points.sort(reverse=True)

        for _, j in points:
            if total_w <= cap_h + 1e-9:
                break

            alternatives = [
                i for i in _open_set(s) if i != overloaded_i
            ]
            best_i, _, best_k = _best_assignment_for_point(
                inst, s, j,
                candidate_platforms=alternatives,
                allow_upgrade=True
            )
            if best_i is None:
                continue

            s.assign_j[j] = best_i
            if best_k is not None:
                s.platform_type[best_i] = max(
                    s.platform_type.get(best_i, 0),
                    best_k
                )
            compute_solution_obj(inst, s)
            total_w, _ = _platform_workload_from_solution(
                inst, s, overloaded_i
            )

    return s


def targeted_feasibility_repair(inst, sol, rnd, max_rounds=5):
    """覆盖、可达性、机队容量和平台数量的统一修复。"""
    s = clone_solution(sol)

    for _ in range(max_rounds):
        _ensure_minimum_platforms(inst, s, rnd)
        _remove_invalid_assignments(inst, s)
        _assign_unassigned_points(inst, s, rnd, allow_upgrade=True)
        compute_solution_obj(inst, s)

        if getattr(s, "overloaded_platforms", []):
            if len(_open_set(s)) < inst.MAX_OPEN:
                _add_best_platform(inst, s, rnd, focus_unassigned=False)
            s = _rebalance_overloaded_platforms(inst, s, rnd)
            compute_solution_obj(inst, s)

        if s.feasible:
            break

        if len(_open_set(s)) < inst.MAX_OPEN:
            added = _add_best_platform(
                inst, s, rnd, focus_unassigned=True
            )
            if not added:
                break
        else:
            _local_reassign_improvement(inst, s, rnd, ratio=0.30)

    compute_solution_obj(inst, s)
    return s


def alns_qlearning(
    inst,
    seed=7,
    iters=CH6_MAX_ITERATIONS,
    time_limit_s=CH6_TIME_LIMIT_SEC,
    verbose=False,
    heartbeat_sec=CH6_HEARTBEAT_SEC,
):
    """
    TQ-ALNS主循环。迭代上限与时间上限采用先到先停。
    """
    rnd = random.Random(int(seed))
    cur = make_initial_solution(
        inst,
        seed=seed,
        init_open=CH6_INITIAL_OPEN_PLATFORMS
    )
    cur = targeted_feasibility_repair(inst, cur, rnd, max_rounds=5)
    compute_solution_obj(inst, cur)
    best = clone_solution(cur)

    destroy_ops = {
        "DP-平台移除": destroy_platform_removal,
        "DU-无人机降配": destroy_uav_reduction,
        "DA-任务释放": destroy_task_assignment,
        "DR-路径破坏": destroy_route_structure,
    }
    repair_ops = {
        "RP-平台补充": repair_platform_addition,
        "RU-无人机补充": repair_uav_supplement,
        "RA-任务重分配": repair_task_reassignment,
        "RR-路径重规划": repair_route_replanning,
    }

    actions = [d + "+" + r for d in destroy_ops for r in repair_ops]
    ql = QLearner(actions, alpha=0.15, epsilon=0.18)

    T0 = max(10.0, 0.15 * cur.obj)
    T = T0
    no_improve = 0

    start = time.time()
    last_heartbeat = start
    iterations_done = 0
    stop_reason = "iteration_limit"

    for it in range(1, int(iters) + 1):
        elapsed = time.time() - start
        if elapsed >= float(time_limit_s):
            stop_reason = "time_limit"
            break

        act = ql.choose(rnd)
        dname, rname = act.split("+", 1)

        cand = destroy_ops[dname](inst, cur, rnd)
        cand = repair_ops[rname](inst, cand, rnd)
        compute_solution_obj(inst, cand)

        if not cand.feasible:
            cand = targeted_feasibility_repair(
                inst, cand, rnd, max_rounds=4
            )
            compute_solution_obj(inst, cand)

        delta = cand.obj - cur.obj
        accepted = cand.feasible and sa_accept(delta, T, rnd)
        if accepted:
            cur = cand

        if cand.feasible and cand.obj < best.obj - 1e-9:
            best = clone_solution(cand)
            reward = 6.0
            no_improve = 0
        elif accepted:
            reward = 1.0
            no_improve += 1
        else:
            reward = -1.0 if not cand.feasible else 0.0
            no_improve += 1

        ql.update(act, reward)

        T *= 0.995
        if no_improve > 120:
            T = max(T, 0.60 * T0)
            no_improve = 0

        iterations_done = it

        now = time.time()
        if verbose and now - last_heartbeat >= float(heartbeat_sec):
            print(
                "[HEARTBEAT] seed=%d iter=%d elapsed=%.1fs "
                "best=%.4f万元 platforms=%d UAVs=%d coverage=%.2f%%"
                % (
                    seed,
                    it,
                    now - start,
                    best.obj,
                    len(_open_set(best)),
                    _total_uav_count(best),
                    100.0 * best.coverage_ratio,
                ),
                flush=True,
            )
            last_heartbeat = now

    best.runtime_s = time.time() - start
    best.iterations_done = iterations_done
    best.stop_reason = stop_reason
    compute_solution_obj(inst, best)

    try:
        build_routes(inst, best, seed=int(seed))
    except Exception:
        pass

    if verbose:
        print(
            "[TQ-ALNS] 完成：seed=%d, obj=%.4f万元, feasible=%s, "
            "platforms=%d, UAVs=%d, coverage=%.2f%%, runtime=%.2fs, stop=%s"
            % (
                seed,
                best.obj,
                str(best.feasible),
                len(_open_set(best)),
                _total_uav_count(best),
                100.0 * best.coverage_ratio,
                best.runtime_s,
                best.stop_reason,
            ),
            flush=True,
        )

    return best


# =============================================================================
# 批量实验、断点续跑、统计汇总与绘图
# =============================================================================

def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def _active_settings():
    if CH6_DEBUG_MODE:
        return {
            "seeds": tuple(CH6_DEBUG_SEARCH_SEEDS),
            "time_limit": float(CH6_DEBUG_TIME_LIMIT_SEC),
            "iters": int(CH6_DEBUG_MAX_ITERATIONS),
        }
    return {
        "seeds": tuple(CH6_SEARCH_SEEDS),
        "time_limit": float(CH6_TIME_LIMIT_SEC),
        "iters": int(CH6_MAX_ITERATIONS),
    }


def _sensitivity_scenarios():
    if CH6_DEBUG_MODE:
        return [
            ("beta", CH6_BASE_BETA),
            ("demand_scale", CH6_BASE_DEMAND_SCALE),
            ("battery_scale", CH6_BASE_BATTERY_SCALE),
        ]
    scenarios = []
    scenarios.extend(("beta", x) for x in CH6_BETA_LEVELS)
    scenarios.extend(("demand_scale", x) for x in CH6_DEMAND_LEVELS)
    scenarios.extend(("battery_scale", x) for x in CH6_BATTERY_LEVELS)
    return scenarios


def _config_id():
    payload = {
        "instance_seed": CH6_INSTANCE_SEED,
        "seeds": list(_active_settings()["seeds"]),
        "time_limit": _active_settings()["time_limit"],
        "iters": _active_settings()["iters"],
        "min_open": CH6_MIN_OPEN_PLATFORMS,
        "max_open": CH6_MAX_OPEN_PLATFORMS,
        "max_uavs_per_platform": CH6_MAX_UAVS_PER_PLATFORM,
        "levels": {
            "beta": list(CH6_BETA_LEVELS),
            "demand_scale": list(CH6_DEMAND_LEVELS),
            "battery_scale": list(CH6_BATTERY_LEVELS),
        },
        "version": "TQALNS_CH6_20260805_v3",
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _prepare_output_dir():
    root = os.path.join(
        _script_dir(),
        CH6_OUTPUT_FOLDER + "_" + _config_id()
    )
    for sub in ("", "figures", "tables", "logs", "base_instance"):
        path = os.path.join(root, sub) if sub else root
        if not os.path.exists(path):
            os.makedirs(path)
    return root


RAW_FIELDS = [
    "config_id",
    "run_id",
    "group",
    "level",
    "seed",
    "status",
    "feasible",
    "objective_wan",
    "open_platforms",
    "deployed_uavs",
    "coverage_rate_pct",
    "runtime_s",
    "estimated_makespan_h",
    "total_workload_h",
    "total_route_distance_km",
    "min_energy_margin_Wh",
    "total_inspection_tasks",
    "iterations_done",
    "stop_reason",
    "message",
    "timestamp",
]


def _read_raw_rows(raw_path):
    if not os.path.exists(raw_path):
        return []
    with open(raw_path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_raw_rows(raw_path, rows):
    temp_path = raw_path + ".tmp"
    with open(temp_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((k, row.get(k, "")) for k in RAW_FIELDS))
    os.replace(temp_path, raw_path)


def _upsert_raw_row(raw_path, new_row):
    rows = _read_raw_rows(raw_path)
    kept = [
        r for r in rows
        if not (
            r.get("config_id") == new_row.get("config_id")
            and r.get("run_id") == new_row.get("run_id")
        )
    ]
    kept.append(new_row)
    _write_raw_rows(raw_path, kept)


def _to_float_or_none(value):
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _to_bool_value(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in (
        "1", "true", "yes", "y", "feasible"
    )


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _sd(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return statistics.stdev(vals)


def _ci95(values):
    vals = [float(v) for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0 if len(vals) == 1 else None
    # n=10附近采用t临界值2.262；调试模式n=1时为0。
    tcrit = 2.262 if len(vals) <= 10 else 1.96
    return tcrit * statistics.stdev(vals) / math.sqrt(len(vals))


def summarize_sensitivity(raw_rows, config_id_value):
    current = [
        r for r in raw_rows
        if r.get("config_id") == config_id_value
    ]
    groups = {}
    for r in current:
        key = (r.get("group"), float(r.get("level")))
        groups.setdefault(key, []).append(r)

    summary = []
    metric_fields = [
        "objective_wan",
        "open_platforms",
        "deployed_uavs",
        "coverage_rate_pct",
        "runtime_s",
        "estimated_makespan_h",
        "total_workload_h",
        "total_route_distance_km",
        "min_energy_margin_Wh",
        "total_inspection_tasks",
    ]

    group_order = {
        "beta": 0,
        "demand_scale": 1,
        "battery_scale": 2,
    }

    for (group, level), rows in groups.items():
        completed = [r for r in rows if r.get("status") == "completed"]
        feasible_rows = [
            r for r in completed
            if _to_bool_value(r.get("feasible"))
        ]

        out = {
            "group": group,
            "level": level,
            "runs": len(rows),
            "completed_runs": len(completed),
            "feasible_runs": len(feasible_rows),
            "feasibility_rate_pct": (
                100.0 * len(feasible_rows) / len(rows)
                if rows else None
            ),
        }

        for field in metric_fields:
            source = completed if field == "runtime_s" else feasible_rows
            vals = [
                _to_float_or_none(r.get(field))
                for r in source
            ]
            vals = [v for v in vals if v is not None]
            out[field + "_mean"] = _mean(vals)
            out[field + "_sd"] = _sd(vals)
            out[field + "_ci95"] = _ci95(vals)

        summary.append(out)

    summary.sort(
        key=lambda r: (
            group_order.get(r["group"], 99),
            r["level"]
        )
    )
    return summary


def _write_dict_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_excel_if_available(path, raw_rows, summary_rows):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment
    except Exception as exc:
        print("[提示] 未安装openpyxl，跳过Excel输出：%s" % exc)
        return

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Summary"

    if summary_rows:
        headers = list(summary_rows[0].keys())
        ws1.append(headers)
        for row in summary_rows:
            ws1.append([row.get(h, "") for h in headers])
        for cell in ws1[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        ws1.freeze_panes = "A2"
        for col in ws1.columns:
            width = min(
                30,
                max(10, max(len(str(c.value or "")) for c in col) + 2)
            )
            ws1.column_dimensions[col[0].column_letter].width = width

    ws2 = wb.create_sheet("Raw runs")
    if raw_rows:
        headers = RAW_FIELDS
        ws2.append(headers)
        for row in raw_rows:
            ws2.append([row.get(h, "") for h in headers])
        for cell in ws2[1]:
            cell.font = Font(bold=True)
        ws2.freeze_panes = "A2"

    wb.save(path)


def _save_base_instance_data(inst, out_dir):
    base_dir = os.path.join(out_dir, "base_instance")

    with open(
        os.path.join(base_dir, "candidate_platforms.csv"),
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["platform_id", "x_m", "y_m", "valid", "fixed_cost_wan"])
        for i in range(inst.I):
            writer.writerow([
                i,
                inst.platform_xy[i][0],
                inst.platform_xy[i][1],
                inst.platform_valid[i],
                inst.platform_fixed_wan[i],
            ])

    with open(
        os.path.join(base_dir, "monitoring_points.csv"),
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "point_id", "x_m", "y_m",
            "base_frequency", "inspection_time_h"
        ])
        for j in range(inst.J):
            writer.writerow([
                j,
                inst.point_xy[j][0],
                inst.point_xy[j][1],
                inst.base_req_freq[j],
                inst.inspect_time_h[j],
            ])

    with open(
        os.path.join(base_dir, "uav_types.csv"),
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "type_id", "name", "purchase_cost_wan",
            "maintenance_cost_wan", "speed_kmh",
            "base_battery_Wh", "energy_mu_Wh_per_km",
            "energy_sigma_Wh_per_km", "max_radius_km"
        ])
        for k, dt in enumerate(inst.drone_types):
            writer.writerow([
                k, dt.name, dt.purchase_cost, dt.maintain_cost,
                dt.speed, inst.base_battery_Wh[k],
                inst.base_energy_mu[k], inst.base_energy_sigma[k],
                dt.max_radius,
            ])


def _plot_one_group(summary_rows, group, output_path):
    if plt is None:
        print("[提示] matplotlib不可用，跳过绘图。")
        return

    rows = [r for r in summary_rows if r["group"] == group]
    rows.sort(key=lambda r: r["level"])
    if not rows:
        return

    x = [r["level"] for r in rows]

    plots = [
        ("objective_wan", "Total cost (10,000 RMB)"),
        ("open_platforms", "Opened platforms"),
        ("deployed_uavs", "Deployed UAVs"),
        ("coverage_rate_pct", "Coverage rate (%)"),
        ("runtime_s", "Runtime (s)"),
        ("feasibility_rate_pct", "Feasibility rate (%)"),
    ]

    xlabels = {
        "beta": r"Confidence level $\beta$",
        "demand_scale": r"Inspection-demand scale $\kappa$",
        "battery_scale": "Battery-capacity scale",
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    axes = list(axes.ravel())

    for ax, (metric, ylabel) in zip(axes, plots):
        if metric == "feasibility_rate_pct":
            y = [r.get(metric) for r in rows]
            ax.plot(x, y, marker="o", linewidth=1.4)
        else:
            y = [r.get(metric + "_mean") for r in rows]
            err = [r.get(metric + "_ci95") for r in rows]
            clean_err = [
                0.0 if v is None else float(v) for v in err
            ]
            ax.errorbar(
                x, y, yerr=clean_err,
                marker="o", capsize=3, linewidth=1.4
            )

        ax.set_xlabel(xlabels[group])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=int(CH6_FIG_DPI),
        bbox_inches="tight"
    )
    plt.close(fig)


def _generate_figures(summary_rows, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    names = {
        "beta": "Figure_3_beta_sensitivity.png",
        "demand_scale": "Figure_4_demand_sensitivity.png",
        "battery_scale": "Figure_5_battery_sensitivity.png",
    }
    for group, filename in names.items():
        _plot_one_group(
            summary_rows,
            group,
            os.path.join(fig_dir, filename)
        )


def _write_diagnostics(summary_rows, out_dir):
    lines = []
    for group in ("beta", "demand_scale", "battery_scale"):
        rows = [r for r in summary_rows if r["group"] == group]
        rows.sort(key=lambda r: r["level"])
        if not rows:
            continue

        first = rows[0]
        last = rows[-1]
        c0 = first.get("objective_wan_mean")
        c1 = last.get("objective_wan_mean")
        if c0 not in (None, 0) and c1 is not None:
            change = 100.0 * (c1 - c0) / abs(c0)
        else:
            change = None

        lines.append("[%s]" % group)
        lines.append(
            "Levels: %s -> %s" % (
                first["level"], last["level"]
            )
        )
        if change is not None:
            lines.append(
                "Endpoint total-cost change: %+.2f%%" % change
            )
        lines.append(
            "Opened platforms: %.3f -> %.3f"
            % (
                first.get("open_platforms_mean") or 0.0,
                last.get("open_platforms_mean") or 0.0,
            )
        )
        lines.append(
            "Deployed UAVs: %.3f -> %.3f"
            % (
                first.get("deployed_uavs_mean") or 0.0,
                last.get("deployed_uavs_mean") or 0.0,
            )
        )
        lines.append(
            "Minimum feasibility rate: %.2f%%"
            % min(
                r.get("feasibility_rate_pct") or 0.0
                for r in rows
            )
        )
        lines.append(
            "Use the observed results when rewriting Section 6; "
            "do not retain the old 14.84%, 41.36%, or 14.48% claims "
            "unless reproduced by these runs."
        )
        lines.append("")

    with open(
        os.path.join(out_dir, "sensitivity_diagnostics.txt"),
        "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(lines))


def run_chapter6_sensitivity():
    settings = _active_settings()
    scenarios = _sensitivity_scenarios()
    config_id_value = _config_id()
    out_dir = _prepare_output_dir()
    raw_path = os.path.join(out_dir, "sensitivity_raw_runs.csv")
    summary_path = os.path.join(
        out_dir, "tables", "sensitivity_summary.csv"
    )
    excel_path = os.path.join(
        out_dir, "tables", "Table_9_Chapter6_sensitivity.xlsx"
    )

    base_inst = build_sensitivity_base_instance(
        seed=CH6_INSTANCE_SEED
    )
    _save_base_instance_data(base_inst, out_dir)

    config_payload = {
        "config_id": config_id_value,
        "debug_mode": CH6_DEBUG_MODE,
        "instance_seed": CH6_INSTANCE_SEED,
        "search_seeds": list(settings["seeds"]),
        "time_limit_sec": settings["time_limit"],
        "max_iterations": settings["iters"],
        "min_open_platforms": CH6_MIN_OPEN_PLATFORMS,
        "max_open_platforms": CH6_MAX_OPEN_PLATFORMS,
        "max_uavs_per_platform": CH6_MAX_UAVS_PER_PLATFORM,
        "initial_open_platforms": CH6_INITIAL_OPEN_PLATFORMS,
        "beta_levels": list(CH6_BETA_LEVELS),
        "demand_levels": list(CH6_DEMAND_LEVELS),
        "battery_levels": list(CH6_BATTERY_LEVELS),
        "cost_unit": "10,000 RMB",
        "algorithm": "TQ-ALNS",
    }
    with open(
        os.path.join(out_dir, "experiment_config.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            config_payload, f,
            ensure_ascii=False, indent=2
        )

    existing = _read_raw_rows(raw_path)
    completed_ids = set()
    if not CH6_FORCE_RERUN:
        for r in existing:
            if (
                r.get("config_id") == config_id_value
                and r.get("status") == "completed"
            ):
                completed_ids.add(r.get("run_id"))

    total_runs = len(scenarios) * len(settings["seeds"])
    counter = 0

    print("=" * 108)
    print("Chapter 6 sensitivity analysis using TQ-ALNS")
    print("Output directory: %s" % out_dir)
    print("Mode: %s" % ("debug" if CH6_DEBUG_MODE else "paper"))
    print(
        "Fixed instance: I=%d, J=%d, hazards=%d, no-fly zones=%d, seed=%d"
        % (
            base_inst.I, base_inst.J,
            base_inst.Hz, base_inst.Z,
            CH6_INSTANCE_SEED,
        )
    )
    print(
        "Endogenous platforms=[%d,%d], max UAVs/platform=%d"
        % (
            CH6_MIN_OPEN_PLATFORMS,
            CH6_MAX_OPEN_PLATFORMS,
            CH6_MAX_UAVS_PER_PLATFORM,
        )
    )
    print(
        "Runs=%d, time limit/run=%.1fs, iteration limit=%d"
        % (
            total_runs,
            settings["time_limit"],
            settings["iters"],
        )
    )
    print("=" * 108, flush=True)

    for scenario_idx, (group, level) in enumerate(
        scenarios, start=1
    ):
        print("\n" + "=" * 108)
        print(
            "Scenario %d/%d: %s = %g"
            % (scenario_idx, len(scenarios), group, level)
        )
        print("=" * 108, flush=True)

        for seed in settings["seeds"]:
            counter += 1
            run_id = "%s|%.8g|%d" % (
                group, float(level), int(seed)
            )

            if run_id in completed_ids:
                print(
                    "[RESUME] Skip %s=%g, seed=%d (%d/%d)"
                    % (
                        group, level, seed,
                        counter, total_runs
                    ),
                    flush=True,
                )
                continue

            print(
                "[TQ-ALNS] Start %s=%g, seed=%d (%d/%d)"
                % (
                    group, level, seed,
                    counter, total_runs
                ),
                flush=True,
            )

            row = {
                "config_id": config_id_value,
                "run_id": run_id,
                "group": group,
                "level": float(level),
                "seed": int(seed),
                "status": "error",
                "feasible": False,
                "objective_wan": "",
                "open_platforms": "",
                "deployed_uavs": "",
                "coverage_rate_pct": "",
                "runtime_s": "",
                "estimated_makespan_h": "",
                "total_workload_h": "",
                "total_route_distance_km": "",
                "min_energy_margin_Wh": "",
                "total_inspection_tasks": "",
                "iterations_done": "",
                "stop_reason": "",
                "message": "",
                "timestamp": datetime.now().isoformat(),
            }

            try:
                inst = apply_sensitivity_level(
                    base_inst, group, level
                )
                sol = alns_qlearning(
                    inst,
                    seed=int(seed),
                    iters=settings["iters"],
                    time_limit_s=settings["time_limit"],
                    verbose=True,
                    heartbeat_sec=CH6_HEARTBEAT_SEC,
                )
                compute_solution_obj(inst, sol)

                row.update({
                    "status": "completed",
                    "feasible": bool(sol.feasible),
                    "objective_wan": float(sol.obj),
                    "open_platforms": len(_open_set(sol)),
                    "deployed_uavs": _total_uav_count(sol),
                    "coverage_rate_pct": 100.0 * sol.coverage_ratio,
                    "runtime_s": float(sol.runtime_s),
                    "estimated_makespan_h": float(
                        getattr(sol, "estimated_makespan_h", 0.0)
                    ),
                    "total_workload_h": float(
                        getattr(sol, "total_workload_h", 0.0)
                    ),
                    "total_route_distance_km": float(
                        getattr(sol, "total_route_distance_km", 0.0)
                    ),
                    "min_energy_margin_Wh": float(
                        getattr(sol, "min_energy_margin_Wh", 0.0)
                    ),
                    "total_inspection_tasks": int(sum(inst.req_freq)),
                    "iterations_done": int(
                        getattr(sol, "iterations_done", 0)
                    ),
                    "stop_reason": str(
                        getattr(sol, "stop_reason", "")
                    ),
                    "message": "",
                })

            except Exception as exc:
                row["message"] = (
                    "%s: %s\n%s"
                    % (
                        type(exc).__name__,
                        str(exc),
                        traceback.format_exc(),
                    )
                )
                print(
                    "[ERROR] %s=%g seed=%d: %s"
                    % (group, level, seed, exc),
                    flush=True,
                )

            _upsert_raw_row(raw_path, row)

            print(
                "[RESULT] status=%s feasible=%s obj=%s万元 "
                "platforms=%s UAVs=%s coverage=%s%% runtime=%ss"
                % (
                    row["status"],
                    row["feasible"],
                    row["objective_wan"],
                    row["open_platforms"],
                    row["deployed_uavs"],
                    row["coverage_rate_pct"],
                    row["runtime_s"],
                ),
                flush=True,
            )

    raw_rows = _read_raw_rows(raw_path)
    summary_rows = summarize_sensitivity(
        raw_rows, config_id_value
    )
    _write_dict_csv(summary_path, summary_rows)

    current_raw = [
        r for r in raw_rows
        if r.get("config_id") == config_id_value
    ]
    _write_excel_if_available(
        excel_path, current_raw, summary_rows
    )
    _generate_figures(summary_rows, out_dir)
    _write_diagnostics(summary_rows, out_dir)

    print("\n" + "=" * 108)
    print("第六章敏感性分析完成")
    print("Raw results : %s" % raw_path)
    print("Summary CSV: %s" % summary_path)
    print("Excel table: %s" % excel_path)
    print("Figures     : %s" % os.path.join(out_dir, "figures"))
    print("=" * 108)

    return raw_rows, summary_rows


def main():
    run_chapter6_sensitivity()


if __name__ == "__main__":
    main()
