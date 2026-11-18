"""Phase 8c: orbit-realistic inter-satellite links (descriptive, no training).

Orbits
  Public two-line elements (TLEs) from CelesTrak for Sentinel-2A/2B/2C, EDRS-C and Eutelsat 9B (host of EDRS-A),
  fetched once and saved with the download date in data/orbits/ (or passed with --tle-file, for offline runs).
  Propagated with skyfield (SGP4) over --hours from the newest TLE epoch, every --step seconds.
  Synthetic satellites use circular two-body orbits (no J2) at Sentinel-2A's mean altitude and inclination.

Scenarios (links)
  S1  LEO -> GEO relay   Sentinel-2A/2B/2C -> EDRS-C and -> Eutelsat 9B (EDRS-A): the real Sentinel-2 relay path
  S2  same-plane LEO     Sentinel-2 pairs (real); synthetic leader-follower pairs at 1, 5, 20 degrees phase
  S3  constellation      synthetic Walker-delta 24/3/1: in-plane neighbour, and the same-index satellite in the
                         adjacent plane
  Line of sight is clear when the segment between the satellites stays >= --margin-km above the Earth.

Link classes (nominal rates, bit/s; override with --links name=rate ...). S1 uses EDRS's 1.8 Gbit/s only.
  s_band 2e6, ka_band 5e8, optical_1g8 1.8e9, optical_10g 1e10, optical_100g 1e11

Workload: a Sentinel-2-like imager (--swath-km 290) at the TLE altitude produces
  tiles/s = (swath / tile) x (ground speed / tile), tile = 64 px x 10 m = 640 m.

Payloads (bytes per tile): raw and zlib-compressed raw (8a); every split point's full map and pooled vector,
fp32 and int8 (8a); every Phase 3/4b variant (PCA, int8, quantile codes) of the audited encoders.
Utility (BigEarthNet macro AP) and leakage (AUC) attached where measured: Phase 4b runs.csv (best attacker on
validation) and Phase 8b runs8b.csv (deep MLP; a full map takes its map4 leakage and gap utility, both lower
bounds, marked in `source`).

Outputs (results/phase8/)
  isl_links.csv        per link: line-of-sight share, contact time per day, distances
  isl_windows.csv      every contact window
  isl_payloads.csv     payloads with bytes, utility and leakage
  isl_feasibility.csv  link x class x payload: required rate, live share, tiles per day, per-tile latency
  isl_frontier.csv     per link class: best utility among payloads it can send live, and that payload's leakage
  isl_README.json      workload, assumptions, sanity checks

Sanity checks (printed; the first two stop the script): Sentinel-2 period 99-102 min and altitude 760-810 km;
EDRS-C altitude 35,600-36,000 km; line-of-sight share of every real Sentinel-2 pair (reported).

Usage
  python scripts/isl8c.py                     # fetch TLEs (once per day) and run
  python scripts/isl8c.py --tle-file data/orbits/tle_2026-11-18.txt
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402

MU, RE, CLIGHT = 398600.4418, 6371.0, 299792.458
QUERIES = {"SENTINEL-2": "SENTINEL-2", "EDRS-C": "EDRS-C", "EUTELSAT 9B": "EUTELSAT%209B"}
CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?NAME={}&FORMAT=TLE"
LINKS = {"s_band": 2e6, "ka_band": 5e8, "optical_1g8": 1.8e9, "optical_10g": 1e10, "optical_100g": 1e11}
ORBITS = "data/orbits"


# ---------------------------------------------------------------- TLEs

def fetch_tles(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tle_{time.strftime('%Y-%m-%d', time.gmtime())}.txt")
    if os.path.exists(path):
        return path
    text = []
    for label, q in QUERIES.items():
        req = urllib.request.Request(CELESTRAK.format(q), headers={"User-Agent": "isl8c-research-script"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode().strip()
        if not body or "No GP data" in body:
            raise SystemExit(f"CelesTrak returned no TLE for {label}; pass --tle-file instead")
        text.append(body)
        time.sleep(1)
    open(path, "w").write("\n".join(text) + "\n")
    print(f"  saved TLEs to {path}")
    return path


def read_tles(path):
    lines = [l.rstrip() for l in open(path) if l.strip()]
    sats = {}
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
        assert l1.startswith("1 ") and l2.startswith("2 "), f"bad TLE block at line {i + 1} of {path}"
        sats[name] = (l1, l2)
    return sats


# ---------------------------------------------------------------- geometry

def los_and_dist(r1, r2, margin):
    """r1, r2: (3, T) km. Line of sight clear if the segment stays >= RE + margin from Earth's centre."""
    d = r2 - r1
    dd = (d * d).sum(0)
    t = np.clip(-(r1 * d).sum(0) / np.maximum(dd, 1e-9), 0.0, 1.0)
    closest = r1 + t * d
    return np.linalg.norm(closest, axis=0) >= RE + margin, np.sqrt(dd)


def circular(a, inc, raan, u0, t_s):
    """ECI positions (3, T) of a circular orbit; angles in radians, t_s seconds from start."""
    n = math.sqrt(MU / a ** 3)
    u = u0 + n * t_s
    cO, sO, ci, si = math.cos(raan), math.sin(raan), math.cos(inc), math.sin(inc)
    return a * np.stack([cO * np.cos(u) - sO * np.sin(u) * ci, sO * np.cos(u) + cO * np.sin(u) * ci,
                         np.sin(u) * si])


def windows(vis, dist, t0, step):
    out, i, n = [], 0, len(vis)
    while i < n:
        if vis[i]:
            j = i
            while j < n and vis[j]:
                j += 1
            seg = dist[i:j]
            out.append({"start_s": i * step, "end_s": j * step, "duration_s": (j - i) * step,
                        "start_utc": t0[i].utc_iso(), "min_km": round(float(seg.min()), 1),
                        "mean_km": round(float(seg.mean()), 1), "max_km": round(float(seg.max()), 1)})
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------- payloads

def best_rows(path, attackers=None):
    if not os.path.exists(path):
        return {}
    best = {}
    for r in csv.DictReader(open(path)):
        if r["split_seed"] != "0" or (attackers and r["attacker"] not in attackers):
            continue
        k = (r["encoder"], r["variant"], r["task"])
        if k not in best or float(r["val_score"]) > float(best[k]["val_score"]):
            best[k] = r
    return best


def payloads(compute_csv):
    out = []
    comp = list(csv.DictReader(open(compute_csv))) if os.path.exists(compute_csv) else []
    if not comp:
        print(f"  {compute_csv} missing: run compute8a.py first for split payloads and head latency")
    b4 = best_rows("results/phase4b/runs.csv")
    b8 = best_rows("results/phase8/runs8b.csv", {"deep_mlp"})

    def score(best, enc, var, task):
        r = best.get((enc, var, task))
        return float(r["test_score"]) if r else ""

    for r in comp:
        if r["arch"] == "raw":
            out.append({"payload": "raw", "kind": "raw", "bytes": int(r["raw_bytes"]), "head_ms": 0.0})
            out.append({"payload": "raw_zlib", "kind": "raw", "bytes": int(r["raw_zlib_bytes"]), "head_ms": 0.0})
            continue
        lat = float(r["lat_mps_b1_ms"]) if r.get("lat_mps_b1_ms") else ""
        for enc in r["encoders"].split():
            if enc.startswith("own_"):
                continue
            ps = f"{enc}@{r['split']}"
            for form, nbytes in (("map_fp32", r["map_fp32_bytes"]), ("map_int8", r["map_int8_bytes"]),
                                 ("gap_fp32", r["gap_fp32_bytes"]), ("gap_int8", r["gap_int8_bytes"])):
                leak_form = "map4" if form.startswith("map") else "gap"
                out.append({"payload": f"{ps}/{form}", "kind": "split", "encoder": enc, "split": r["split"],
                            "bytes": int(nbytes), "head_ms": lat,
                            "utility_ben": score(b8, ps, "gap", "utility_ben"),
                            "utility_eurosat": score(b8, ps, "gap", "utility_eurosat"),
                            "leakage": score(b8, ps, leak_form, "leakage"),
                            "spatial": score(b8, ps, "map4", "spatial"),
                            "source": "phase8b deep_mlp" + (" (map: map4 leakage, gap utility; lower bounds)"
                                                            if leak_form == "map4" else "")})
    final_lat = {}
    for r in comp:
        if r.get("is_final") == "True" and r.get("lat_mps_b1_ms"):
            for enc in r["encoders"].split():
                final_lat[enc] = float(r["lat_mps_b1_ms"])
    seen = set()
    for (enc, var, task), r in b4.items():
        if enc not in C.AUDITED or task != "leakage" or (enc, var) in seen:
            continue
        seen.add((enc, var))
        out.append({"payload": f"{enc}/{var}", "kind": "compressed", "encoder": enc, "split": "final",
                    "bytes": C.variant_bytes(enc, var, int(r["dims"])), "head_ms": final_lat.get(enc, ""),
                    "utility_ben": score(b4, enc, var, "utility_ben"),
                    "utility_eurosat": score(b4, enc, var, "utility_eurosat"), "leakage": float(r["test_score"]),
                    "spatial": score(b4, enc, var, "spatial"), "source": "phase4b best on val"})
    qf = "results/phase4b/analysis/quantisation.csv"  # quantile codes: Phase 4 MLP probe, EuroSAT utility only
    if os.path.exists(qf):
        q = {}
        for r in csv.DictReader(open(qf)):
            if r["encoder"] in C.AUDITED and r["probe"] == "mlp" and int(r["bits"]) < 32:
                q.setdefault((r["encoder"], r["level"], int(r["bits"])), {})[r["task"]] = (float(r["test"]),
                                                                                            float(r["bytes"]))
        for (enc, level, bits), d in sorted(q.items()):
            if "leakage" not in d or "utility" not in d:
                continue
            out.append({"payload": f"{enc}/{level}_q{bits}", "kind": "quantised", "encoder": enc, "split": "final",
                        "bytes": d["leakage"][1], "head_ms": final_lat.get(enc, ""), "utility_ben": "",
                        "utility_eurosat": d["utility"][0], "leakage": d["leakage"][0], "spatial": "",
                        "source": "phase4b quantisation (Phase 4 MLP probe)"})
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tle-file", default=None)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--step", type=float, default=10, help="seconds")
    ap.add_argument("--margin-km", type=float, default=100)
    ap.add_argument("--swath-km", type=float, default=290)
    ap.add_argument("--tile-km", type=float, default=0.64)
    ap.add_argument("--links", nargs="*", default=[], help="override or add link classes: name=rate_bps")
    ap.add_argument("--compute", default="results/phase8/compute.csv")
    ap.add_argument("--out", default="results/phase8")
    args = ap.parse_args()
    from skyfield.api import EarthSatellite, load

    links = dict(LINKS)
    for kv in args.links:
        k, v = kv.split("=")
        links[k] = float(v)
    os.makedirs(args.out, exist_ok=True)
    path = args.tle_file or fetch_tles(ORBITS)
    tles = read_tles(path)
    ts = load.timescale(builtin=True)
    sats = {n: EarthSatellite(l1, l2, n, ts) for n, (l1, l2) in tles.items()}
    s2 = sorted(n for n in sats if n.startswith("SENTINEL-2"))
    geo = [n for n in sats if n.startswith("EDRS") or n.startswith("EUTELSAT 9B")]
    assert s2 and geo, f"need Sentinel-2 and EDRS TLEs in {path}, found {list(sats)}"

    start = max(s.epoch.tt for s in sats.values())
    t_s = np.arange(0, args.hours * 3600 + 1e-9, args.step)
    t = ts.tt_jd(start + t_s / 86400.0)
    pos = {n: sats[n].at(t).position.km for n in s2 + geo}

    # sanity checks and Sentinel-2A-like orbit for the synthetic satellites
    ref = sats[s2[0]]
    n_rad = ref.model.no_kozai / 60.0  # rad/s
    a = (MU / n_rad ** 2) ** (1 / 3)
    period = 2 * math.pi / n_rad / 60
    alt = float(np.linalg.norm(pos[s2[0]], axis=0).mean() - RE)
    checks = {"tle_file": path, "start_utc": t[0].utc_iso(), "sentinel2_period_min": round(period, 2),
              "sentinel2_mean_altitude_km": round(alt, 1)}
    assert 99 <= period <= 102 and 760 <= alt <= 810, f"Sentinel-2 orbit check failed: {checks}"
    for g in geo:
        galt = float(np.linalg.norm(pos[g], axis=0).mean() - RE)
        checks[f"{g}_altitude_km"] = round(galt, 1)
        if g.startswith("EDRS"):
            assert 35600 <= galt <= 36000, f"EDRS-C altitude check failed: {galt:.0f} km"
    print("  checks:", json.dumps(checks))
    inc, raan, u0 = ref.model.inclo, ref.model.nodeo, ref.model.argpo + ref.model.mo

    pairs = []  # (scenario, link name, r1, r2, link classes)
    for n in s2:
        for g in geo:
            pairs.append(("S1 LEO->GEO relay", f"{n} -> {g}", pos[n], pos[g], {"optical_1g8": links["optical_1g8"]}))
    for i in range(len(s2)):
        for j in range(i + 1, len(s2)):
            pairs.append(("S2 same-plane LEO (real)", f"{s2[i]} <-> {s2[j]}", pos[s2[i]], pos[s2[j]], links))
    base = circular(a, inc, raan, u0, t_s)
    for dphi in (1, 5, 20):
        pairs.append(("S2 leader-follower (synthetic)", f"pair {dphi} deg", base,
                      circular(a, inc, raan, u0 + math.radians(dphi), t_s), links))
    T, P, F = 24, 3, 1
    s_per = T // P
    w0 = circular(a, inc, raan, u0, t_s)
    pairs.append(("S3 Walker 24/3/1 (synthetic)", "in-plane neighbour", w0,
                  circular(a, inc, raan, u0 + 2 * math.pi / s_per, t_s), links))
    pairs.append(("S3 Walker 24/3/1 (synthetic)", "adjacent plane, same index", w0,
                  circular(a, inc, raan + 2 * math.pi / P, u0 + 2 * math.pi * F / T, t_s), links))

    link_rows, win_rows = [], []
    for scen, name, r1, r2, cls in pairs:
        vis, dist = los_and_dist(r1, r2, args.margin_km)
        w = windows(vis, dist, t, args.step)
        for x in w:
            win_rows.append({"scenario": scen, "link": name, **x})
        share = float(vis.mean())
        link_rows.append({"scenario": scen, "link": name, "los_share": round(share, 4),
                          "contact_s_per_day": round(share * 86400, 0), "windows": len(w),
                          "mean_window_s": round(np.mean([x["duration_s"] for x in w]), 0) if w else 0,
                          "min_km": round(float(dist.min()), 1), "mean_km_in_contact":
                              round(float(dist[vis].mean()), 1) if vis.any() else "",
                          "max_km": round(float(dist.max()), 1), "classes": " ".join(cls)})
        print(f"  {scen:>30} | {name:<34} line of sight {share:6.1%}, {len(w):3d} windows, "
              f"distance {dist.min():8.0f}-{dist.max():8.0f} km", flush=True)
        if scen.startswith("S2 same-plane LEO (real)"):
            checks[f"los_share {name}"] = round(share, 4)

    v = math.sqrt(MU / a)
    vg = v * RE / a
    tiles_s = (args.swath_km / args.tile_km) * (vg / args.tile_km)
    pl = payloads(args.compute)
    raw_b = next((p["bytes"] for p in pl if p["payload"] == "raw"), 13 * 64 * 64 * 2)
    print(f"  workload: ground speed {vg:.2f} km/s, {tiles_s:.0f} tiles/s, raw {tiles_s * raw_b * 8 / 1e9:.2f} Gbit/s")

    feas = []
    for lr in link_rows:
        cls = {k: links[k] for k in lr["classes"].split()}
        contact = lr["contact_s_per_day"]
        dist_km = lr["mean_km_in_contact"] or 0.0
        for cname, rate in cls.items():
            for p in pl:
                req = tiles_s * p["bytes"] * 8
                lat = ((p["head_ms"] or 0.0) + 8 * p["bytes"] / rate * 1000
                       + ((dist_km or 0.0) / CLIGHT * 1000)) if contact else ""
                feas.append({"scenario": lr["scenario"], "link": lr["link"], "class": cname, "rate_bps": rate,
                             "payload": p["payload"], "bytes": p["bytes"], "required_bps": round(req),
                             "live_share": round(min(1.0, rate / req), 4) if contact else 0.0,
                             "tiles_per_day": round(rate * contact / (8 * p["bytes"])),
                             "latency_ms": round(lat, 3) if lat != "" else "",
                             "utility_ben": p.get("utility_ben", ""),
                             "utility_eurosat": p.get("utility_eurosat", ""), "leakage": p.get("leakage", "")})

    frontier = []
    for util in ("utility_ben", "utility_eurosat"):
        for cname, rate in links.items():
            live = [p for p in pl if p.get(util) not in ("", None) and rate >= tiles_s * p["bytes"] * 8]
            base = {"utility_measure": util, "class": cname, "rate_bps": rate,
                    "max_live_bytes": round(rate / (tiles_s * 8), 1)}
            if not live:
                frontier.append({**base, "best_payload": "none fits live"})
                continue
            b = max(live, key=lambda p: p[util])
            low = min((p for p in live if p.get("leakage") not in ("", None)), key=lambda p: p["leakage"],
                      default=None)
            frontier.append({**base, "live_payloads": len(live), "best_payload": b["payload"], "bytes": b["bytes"],
                             "utility": b[util], "leakage_of_best": b.get("leakage", ""),
                             "lowest_leakage_payload": low["payload"] if low else "",
                             "lowest_leakage": low["leakage"] if low else "",
                             "utility_of_lowest_leakage": low[util] if low else ""})

    def write(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(args.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {os.path.join(args.out, name)} ({len(rows)} rows)")

    write("isl_links.csv", link_rows)
    write("isl_windows.csv", win_rows)
    write("isl_payloads.csv", pl)
    write("isl_feasibility.csv", feas)
    write("isl_frontier.csv", frontier)
    json.dump({"checks": checks, "links_bps": links, "workload": {
        "swath_km": args.swath_km, "tile_km": args.tile_km, "ground_speed_km_s": round(vg, 3),
        "tiles_per_s": round(tiles_s, 1), "raw_bits_per_s": round(tiles_s * raw_b * 8)},
        "assumptions": ["nominal link rates, no link budget, pointing or acquisition time",
                        "synthetic orbits are circular two-body (no J2)", "no queueing or on-board storage",
                        "13 bands on the 10 m grid overstates Sentinel-2's raw volume"],
        "hours": args.hours, "step_s": args.step, "margin_km": args.margin_km},
        open(os.path.join(args.out, "isl_README.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
