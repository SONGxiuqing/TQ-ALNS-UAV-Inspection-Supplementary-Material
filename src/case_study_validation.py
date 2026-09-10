# -*- coding: utf-8 -*-
"""
Chapter 7 case-study data validation and route reconstruction
Aligned with the current manuscript:
- 20 candidate sites reported; 5 selected platform coordinates supplied.
- 95 monitoring points, 5 hazardous sources, 3 fixed no-fly zones.
- Planning horizon = 24 h.
- beta = 0.95; hazardous-source buffer = 500 m.
- Table 12 assignment:
  P01 Type A M001-M017
  P02 Type B M018-M039
  P03 Type C M040-M060
  P04 Type B M061-M078
  P05 Type C M079-M095

Important:
This script validates/reconstructs the published five-platform route solution.
It does NOT pretend to re-optimize 20 candidate sites when the 15 unselected
candidate coordinates are unavailable. If a full 20-site file is supplied,
set FULL_CANDIDATE_PLATFORM_FILE accordingly for a complete location audit.
"""
from pathlib import Path
import csv, math, re, json
import pandas as pd

# =============================================================================
# 0. PARAMETER CONFIGURATION
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_ROOT / "7Case Study"

SELECTED_PLATFORM_FILE = BASE_DIR / "平台12345位置坐标数据.xlsx"
MONITORING_POINT_FILE = BASE_DIR / "所有监测点95.xlsx"
HAZARD_FILE = BASE_DIR / "危险源12345位置数据.xlsx"
NO_FLY_FILE = BASE_DIR / "禁飞区顶点.csv"
BOUNDARY_GEOJSON = BASE_DIR / "边界线.geojson"

# Optional: full 20-candidate-site file. Keep empty if unavailable.
FULL_CANDIDATE_PLATFORM_FILE = ""

OUTPUT_DIR = REPO_ROOT / "results" / "case_study"

TOTAL_CANDIDATE_PLATFORM_COUNT = 20
MAX_OPENED_PLATFORMS = 5
PLANNING_HORIZON_H = 24.0
BETA = 0.95
Z_BETA = 1.645
HAZARD_SAFETY_DISTANCE_M = 500.0
PLATFORM_COST_RMB = 250000.0
ENERGY_COST_RMB_PER_KM = 0.8
UNCOVERED_PENALTY_RMB = 15000.0
NO_FLY_DETOUR_MARGIN_M = 160.0

UAV_CONFIG = {
    "P01": dict(uav_id="UAV-A01", uav_type="Type A", prototype="DJI Matrice 4T",
                speed_kmh=60.0, battery_Wh=800.0, energy_mu=20.0, energy_sigma=4.0,
                point_start=1, point_end=17,
                role="Auxiliary facilities, dense edge points, and utility-related routine inspection"),
    "P02": dict(uav_id="UAV-B01", uav_type="Type B", prototype="RY-V30",
                speed_kmh=75.0, battery_Wh=1200.0, energy_mu=25.0, energy_sigma=5.0,
                point_start=18, point_end=39,
                role="Peripheral storage area, port-supporting belt, and corridor inspection"),
    "P03": dict(uav_id="UAV-C01", uav_type="Type C", prototype="TR100",
                speed_kmh=90.0, battery_Wh=2000.0, energy_mu=35.0, energy_sigma=7.0,
                point_start=40, point_end=60,
                role="Core process units and hazardous-source surroundings"),
    "P04": dict(uav_id="UAV-B02", uav_type="Type B", prototype="RY-V30",
                speed_kmh=75.0, battery_Wh=1200.0, energy_mu=25.0, energy_sigma=5.0,
                point_start=61, point_end=78,
                role="Storage tank area and adjacent main pipe-rack corridor"),
    "P05": dict(uav_id="UAV-C02", uav_type="Type C", prototype="TR100",
                speed_kmh=90.0, battery_Wh=2000.0, energy_mu=35.0, energy_sigma=7.0,
                point_start=79, point_end=95,
                role="Public engineering area and emergency-support-related facilities"),
}

# =============================================================================
# 1. DATA READING
# =============================================================================
def read_selected_platforms(path):
    df = pd.read_excel(path, header=None)
    rows = []
    for idx, r in df.iterrows():
        try:
            lon, lat = float(r.iloc[0]), float(r.iloc[1])
        except Exception:
            continue
        raw_name = str(r.iloc[3]) if len(r) > 3 and pd.notna(r.iloc[3]) else f"平台{idx+1}"
        m = re.search(r"平台\s*(\d+)", raw_name)
        num = int(m.group(1)) if m else idx + 1
        rows.append(dict(source_platform_number=num, raw_name=raw_name, lon=lon, lat=lat))
    rows.sort(key=lambda x: x["source_platform_number"])
    rows = rows[:5]
    for i, p in enumerate(rows, 1):
        p["platform_id"] = f"P{i:02d}"
    if len(rows) != 5:
        raise ValueError(f"Expected 5 selected platforms, found {len(rows)}.")
    return rows

def read_monitoring_points(path):
    df = pd.read_excel(path)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError("Monitoring-point file must contain X and Y columns.")
    name_col = "Name" if "Name" in df.columns else None
    pts = []
    for idx, r in df.iterrows():
        if pd.isna(r["X"]) or pd.isna(r["Y"]):
            continue
        source_name = str(r[name_col]) if name_col else str(idx + 1)
        try:
            source_num = int(float(source_name))
        except Exception:
            source_num = idx + 1
        pts.append(dict(source_numeric_id=source_num, source_name=source_name,
                        lon=float(r["X"]), lat=float(r["Y"])))
    pts.sort(key=lambda x: x["source_numeric_id"])
    if len(pts) != 95:
        raise ValueError(f"Expected 95 monitoring points, found {len(pts)}.")
    # Normalize IDs to the manuscript's M001-M095 convention.
    for i, p in enumerate(pts, 1):
        p["point_id"] = f"M{i:03d}"
    return pts

def read_hazards(path):
    df = pd.read_excel(path, header=None)
    hazards = []
    for idx, r in df.iterrows():
        try:
            lon, lat = float(r.iloc[0]), float(r.iloc[1])
        except Exception:
            continue
        raw_name = str(r.iloc[3]) if len(r) > 3 and pd.notna(r.iloc[3]) else f"危险源{idx+1}"
        m = re.search(r"H\s*(\d+)", raw_name, re.I)
        hid = f"H{int(m.group(1)):02d}" if m else f"H{idx+1:02d}"
        hazards.append(dict(hazard_id=hid, raw_name=raw_name, lon=lon, lat=lat))
    if len(hazards) != 5:
        raise ValueError(f"Expected 5 hazardous sources, found {len(hazards)}.")
    return hazards

def read_nofly(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    groups = {}
    for _, r in df.iterrows():
        if pd.isna(r.get("X")) or pd.isna(r.get("Y")):
            continue
        name = str(r.get("Name", "No-fly zone")).strip()
        groups.setdefault(name, []).append((float(r["X"]), float(r["Y"])))
    for name, poly in groups.items():
        if len(poly) > 1 and abs(poly[0][0]-poly[-1][0]) < 1e-12 and abs(poly[0][1]-poly[-1][1]) < 1e-12:
            poly.pop()
    if len(groups) != 3:
        raise ValueError(f"Expected 3 fixed no-fly zones, found {len(groups)}.")
    return groups

# =============================================================================
# 2. GEOMETRY
# =============================================================================
def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def orient(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

def on_segment(a,b,c,eps=1e-9):
    return min(a[0],b[0])-eps <= c[0] <= max(a[0],b[0])+eps and min(a[1],b[1])-eps <= c[1] <= max(a[1],b[1])+eps

def segments_intersect(a,b,c,d):
    o1,o2,o3,o4 = orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)
    if abs(o1)<1e-9 and on_segment(a,b,c): return True
    if abs(o2)<1e-9 and on_segment(a,b,d): return True
    if abs(o3)<1e-9 and on_segment(c,d,a): return True
    if abs(o4)<1e-9 and on_segment(c,d,b): return True
    return (o1>0)!=(o2>0) and (o3>0)!=(o4>0)

def point_in_poly(p,poly):
    x,y=p
    inside=False
    for i in range(len(poly)):
        a,b=poly[i],poly[(i+1)%len(poly)]
        if abs(orient(a,b,p))<1e-9 and on_segment(a,b,p):
            return True
        if (a[1]>y)!=(b[1]>y):
            xi=a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if xi>x: inside=not inside
    return inside

def segment_intersects_poly(a,b,poly):
    if point_in_poly(a,poly) or point_in_poly(b,poly):
        return True
    return any(segments_intersect(a,b,poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly)))

def compact(path):
    out=[]
    for p in path:
        if not out or distance(out[-1],p)>1e-8:
            out.append(p)
    return out

def path_length(path):
    return sum(distance(path[i],path[i+1]) for i in range(len(path)-1))

def poly_bbox(poly):
    xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
    return min(xs),max(xs),min(ys),max(ys)

# =============================================================================
# 3. CASE PREPARATION AND ROUTES
# =============================================================================
def assign_local_coordinates(platforms, points, hazards, nfz_ll):
    lons=[p["lon"] for p in platforms]+[p["lon"] for p in points]+[h["lon"] for h in hazards]
    lats=[p["lat"] for p in platforms]+[p["lat"] for p in points]+[h["lat"] for h in hazards]
    for poly in nfz_ll.values():
        lons += [x for x,y in poly]; lats += [y for x,y in poly]
    lon0=min(lons); lat0=min(lats); latmid=(min(lats)+max(lats))/2.0
    mx=111320.0*math.cos(math.radians(latmid)); my=110540.0
    def conv(lon,lat): return ((lon-lon0)*mx,(lat-lat0)*my)
    for arr in (platforms,points,hazards):
        for item in arr:
            item["x_m"],item["y_m"]=conv(item["lon"],item["lat"])
    nfz_xy={name:[conv(lon,lat) for lon,lat in poly] for name,poly in nfz_ll.items()}
    return nfz_xy

def nearest_neighbor(start, pts):
    remaining=pts[:]; order=[]; cur=start
    while remaining:
        nxt=min(remaining,key=lambda p:distance(cur,(p["x_m"],p["y_m"])))
        order.append(nxt); remaining.remove(nxt); cur=(nxt["x_m"],nxt["y_m"])
    return order

def straight_route_length(start,order):
    coords=[start]+[(p["x_m"],p["y_m"]) for p in order]+[start]
    return path_length(coords)

def two_opt(start,order,max_iter=60):
    if len(order)<4: return order
    best=order[:]; best_len=straight_route_length(start,best)
    for _ in range(max_iter):
        improved=False
        n=len(best)
        for i in range(0,n-2):
            for j in range(i+2,n+1):
                cand=best[:i]+list(reversed(best[i:j]))+best[j:]
                L=straight_route_length(start,cand)
                if L+1e-6<best_len:
                    best,best_len=cand,L; improved=True
        if not improved: break
    return best

def build_detour_functions(nfz_xy):
    def clear(a,b):
        return not any(segment_intersects_poly(a,b,poly) for poly in nfz_xy.values())
    def candidates(a,b,poly):
        minx,maxx,miny,maxy=poly_bbox(poly)
        left,right=minx-NO_FLY_DETOUR_MARGIN_M,maxx+NO_FLY_DETOUR_MARGIN_M
        bottom,top=miny-NO_FLY_DETOUR_MARGIN_M,maxy+NO_FLY_DETOUR_MARGIN_M
        ax,ay=a; bx,by=b
        return [
            compact([a,(ax,top),(bx,top),b]),
            compact([a,(ax,bottom),(bx,bottom),b]),
            compact([a,(left,ay),(left,by),b]),
            compact([a,(right,ay),(right,by),b]),
            compact([a,(ax,top),(right,top),(right,by),b]),
            compact([a,(ax,top),(left,top),(left,by),b]),
            compact([a,(ax,bottom),(right,bottom),(right,by),b]),
            compact([a,(ax,bottom),(left,bottom),(left,by),b]),
        ]
    def detour_segment(a,b):
        if clear(a,b): return [a,b]
        best=None; best_len=float("inf")
        for poly in nfz_xy.values():
            if not segment_intersects_poly(a,b,poly): continue
            for cand in candidates(a,b,poly):
                if all(clear(cand[i],cand[i+1]) for i in range(len(cand)-1)):
                    L=path_length(cand)
                    if L<best_len: best,best_len=cand,L
        return best if best is not None else [a,b]
    return clear,detour_segment

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    platforms=read_selected_platforms(SELECTED_PLATFORM_FILE)
    points=read_monitoring_points(MONITORING_POINT_FILE)
    hazards=read_hazards(HAZARD_FILE)
    nfz_ll=read_nofly(NO_FLY_FILE)
    nfz_xy=assign_local_coordinates(platforms,points,hazards,nfz_ll)
    clear,detour_segment=build_detour_functions(nfz_xy)

    pmap={p["platform_id"]:p for p in platforms}
    ptmap={p["point_id"]:p for p in points}

    # Safety audit of the five selected platforms.
    for pid,p in pmap.items():
        pxy=(p["x_m"],p["y_m"])
        min_h=min(distance(pxy,(h["x_m"],h["y_m"])) for h in hazards)
        if min_h < HAZARD_SAFETY_DISTANCE_M:
            raise RuntimeError(f"{pid} violates 500 m hazardous-source safety distance.")
        if any(point_in_poly(pxy,poly) for poly in nfz_xy.values()):
            raise RuntimeError(f"{pid} lies inside a fixed no-fly zone.")

    summary=[]
    assignments=[]
    for pid,cfg in UAV_CONFIG.items():
        assigned=[ptmap[f"M{i:03d}"] for i in range(cfg["point_start"],cfg["point_end"]+1)]
        for p in assigned:
            assignments.append(dict(point_id=p["point_id"],assigned_platform=pid,
                                    uav_id=cfg["uav_id"],uav_type=cfg["uav_type"],prototype=cfg["prototype"]))
        start=(pmap[pid]["x_m"],pmap[pid]["y_m"])
        order=two_opt(start,nearest_neighbor(start,assigned))
        nodes=[start]+[(p["x_m"],p["y_m"]) for p in order]+[start]
        detoured=[]
        for i in range(len(nodes)-1):
            part=detour_segment(nodes[i],nodes[i+1])
            detoured.extend(part if i==0 else part[1:])
        length_km=path_length(detoured)/1000.0
        energy_q95=length_km*(cfg["energy_mu"]+Z_BETA*cfg["energy_sigma"])
        nfz_safe=all(clear(detoured[i],detoured[i+1]) for i in range(len(detoured)-1))
        energy_safe=energy_q95 <= cfg["battery_Wh"]+1e-9
        summary.append(dict(
            platform=pid,uav_id=cfg["uav_id"],uav_type=cfg["uav_type"],prototype=cfg["prototype"],
            monitoring_points=f"M{cfg['point_start']:03d}-M{cfg['point_end']:03d}",
            number=cfg["point_end"]-cfg["point_start"]+1,
            route_length_km=length_km,flight_time_h=length_km/cfg["speed_kmh"],
            energy_q95_Wh=energy_q95,battery_Wh=cfg["battery_Wh"],
            NFZ_safe=nfz_safe,beta_0_95_energy_feasible=energy_safe
        ))

    pd.DataFrame(assignments).to_csv(OUTPUT_DIR/"monitoring_point_assignments.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(OUTPUT_DIR/"route_summary.csv",index=False,encoding="utf-8-sig")

    # Honest data-availability warning.
    if not FULL_CANDIDATE_PLATFORM_FILE:
        (OUTPUT_DIR/"data_availability_note.txt").write_text(
            "The manuscript reports 20 candidate sites, but the supplied platform file contains "
            "only the five selected platform coordinates. The other 15 candidate coordinates "
            "were not fabricated. Supply a 20-site file to perform a complete location-selection audit.",
            encoding="utf-8"
        )

    print("Chapter 7 case reconstruction completed.")
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    main()
