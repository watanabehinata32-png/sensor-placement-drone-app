"""Phase 2 ステップ 1: 前処理（道路網の最短走行時間と、飛行の列挙）.

使い方:
    net = RoadNetwork(segments, nodes, sensors, cfg)   # 道路網の組み立て（最初に 1 回）
    res = net.run(closed_segments, comm_down, bases)    # 通行止め・通信断を変えるたびに呼ぶ

ルール（仮定は config.py と README を参照）:
- 車は通行止めでない区間だけを走る。高速道路も走れるが、乗り降りは IC ランプ経由のみ（道路網の形どおり）。
- 離陸（停車）は一般道上のみ。候補点 = 一般道区間の両端 + 区間上の等間隔点 + 各センサーから区間への最寄り点。
- 1 回の飛行: 離陸点 → センサーを順に回って各 3 分滞空 → 同じ離陸点（同じ車）に戻る。合計 15 分以内。
- どの離陸候補点からも 1 基飛行（片道 4 km）が成り立たない通信断センサーは「対応不能」。

道路網はグラフ（頂点 = 交差点や離陸候補点、辺 = その間の道、重み = 走行時間）として持ち、
最短走行時間はダイクストラ法（scipy の dijkstra）で求める。
"""
import itertools
import math
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .config import Config

EPS = 1e-6   # 小数の誤差の余裕


# ============================================================
# 折れ線（道路の形）を扱う小さな関数
# ============================================================
def polyline_with_length(coords):
    """折れ線の点の配列と、始点から各点までの道のり（m）の配列を返す."""
    points = np.asarray(coords, float)
    piece_lengths = np.hypot(*np.diff(points, axis=0).T)
    return points, np.concatenate([[0.0], np.cumsum(piece_lengths)])


def point_on_polyline(points, dist_along, s):
    """折れ線上で、始点から道のり s（m）の位置の座標."""
    s = min(max(s, 0.0), dist_along[-1])
    i = min(np.searchsorted(dist_along, s, side="right") - 1, len(points) - 2)   # s を含む小区間の番号
    t = (s - dist_along[i]) / max(dist_along[i + 1] - dist_along[i], EPS)        # その小区間の中の割合
    return points[i] + t * (points[i + 1] - points[i])


def closest_on_polyline(points, dist_along, p):
    """折れ線上で点 p にいちばん近い位置. 戻り値: (始点からの道のり, p までの距離)."""
    starts, ends = points[:-1], points[1:]
    piece = ends - starts
    piece_len2 = np.maximum((piece ** 2).sum(1), EPS)
    t = np.clip(((p - starts) * piece).sum(1) / piece_len2, 0, 1)   # 各小区間で p にいちばん近い割合
    nearest = starts + t[:, None] * piece
    dist = np.hypot(*(nearest - p).T)
    i = int(np.argmin(dist))
    return dist_along[i] + t[i] * math.sqrt(piece_len2[i]), float(dist[i])


def minutes_or_none(v):
    """有限なら小数 3 桁に丸め、行けない（inf）なら None（JSON にするため）."""
    return None if not np.isfinite(v) else round(float(v), 3)


# ============================================================
# 道路網
# ============================================================
class RoadNetwork:
    def __init__(self, segments, nodes, sensors, cfg=None):
        """道路網のグラフと、離陸候補点を作る（通行止めに関係なく 1 回だけ）."""
        t_start = time.time()
        self.cfg = cfg or Config()
        self.sensors = sensors
        reach = self.cfg.drone.one_way_reach_m              # ドローンが片道で行ける距離
        spacing = self.cfg.road.launch_spacing_m            # 離陸候補点の間隔
        no_launch = set(self.cfg.road.no_launch_highways)   # 停車できない道路の種類（高速道路など）
        speed_factor = self.cfg.road.speed_factor

        self.node_ids = list(nodes)
        self.vidx = {n: i for i, n in enumerate(self.node_ids)}   # ノード ID → 頂点の番号
        vertex_xy = [nodes[n] for n in self.node_ids]             # 頂点の座標（離陸候補点を足していく）
        self.segments = {s["segment_id"]: s for s in segments}
        sensor_ids = list(sensors)
        sensor_xy = np.array([sensors[s][:2] for s in sensor_ids])
        sensor_tree = cKDTree(sensor_xy)                          # 近いセンサーをすばやく探すための木

        launch_points = []          # 離陸候補点: dict(x, y, segment_id, vertex, kind)
        node_launch_segs = {}       # ノード → つながる一般道区間（どれか 1 本が開いていれば停車できる）
        # グラフの辺: 頂点 edge_from[i] と edge_to[i] の間を edge_seconds[i] 秒で走れる（区間 edge_segment[i] の一部）
        edge_from, edge_to, edge_seconds, edge_segment = [], [], [], []

        def add_edge(u, v, seconds, segment_id):
            edge_from.append(u)
            edge_to.append(v)
            edge_seconds.append(seconds)
            edge_segment.append(segment_id)

        for s in segments:
            sid = s["segment_id"]
            u, v = self.vidx[s["from_node"]], self.vidx[s["to_node"]]
            seconds = s["travel_time_s"] / speed_factor
            can_stop = s.get("launch_ok", True) and s["highway"] not in no_launch
            if not can_stop:
                add_edge(u, v, seconds, sid)        # 停車できない道は、そのまま 1 本の辺にする
                continue
            for n in (s["from_node"], s["to_node"]):
                node_launch_segs.setdefault(n, []).append(sid)
            # 道路データの端点座標とノード座標がずれていても良いよう、折れ線の両端をノードに合わせる
            points, _ = polyline_with_length(s["coords"])
            points[0], points[-1] = vertex_xy[u], vertex_xy[v]
            points, dist_along = polyline_with_length(points)
            length = dist_along[-1]
            # 区間の途中に離陸候補点を置く位置（始点からの道のり）: 等間隔の点 ＋ 近くのセンサーにいちばん近い点
            n_pieces = math.ceil(length / spacing)
            cut_positions = [k * length / n_pieces for k in range(1, n_pieces)]
            kinds = ["grid"] * len(cut_positions)
            if self.cfg.road.add_sensor_projections:
                middle = points[len(points) // 2]
                for j in sensor_tree.query_ball_point(middle, reach + length):
                    pos, dist = closest_on_polyline(points, dist_along, sensor_xy[j])
                    if dist <= reach + EPS and EPS < pos < length - EPS:
                        cut_positions.append(pos)
                        kinds.append("proj:" + sensor_ids[j])
            # 区間を離陸候補点で切って、短い辺の列にする（走行時間は長さに比例して分ける）
            prev_vertex, prev_pos = u, 0.0
            for k in np.argsort(cut_positions):
                pos = cut_positions[k]
                if pos - prev_pos < 1.0:            # 1 m 以内の重複はまとめる
                    continue
                x, y = point_on_polyline(points, dist_along, pos)
                new_vertex = len(vertex_xy)
                vertex_xy.append((x, y))
                launch_points.append(dict(x=float(x), y=float(y), segment_id=sid, vertex=new_vertex, kind=kinds[k]))
                add_edge(prev_vertex, new_vertex, seconds * (pos - prev_pos) / length, sid)
                prev_vertex, prev_pos = new_vertex, pos
            add_edge(prev_vertex, v, seconds * (length - prev_pos) / length, sid)
        # 一般道の端（ノード）も離陸候補点にする
        for n, segs in node_launch_segs.items():
            x, y = nodes[n]
            launch_points.append(dict(x=x, y=y, segment_id=None, node_id=n, vertex=self.vidx[n], kind="node",
                                      node_segments=segs))
        for i, p in enumerate(launch_points):
            p["launch_id"] = f"L{i:05d}"

        self.launch_points = launch_points
        self.n_vertices = len(vertex_xy)
        self.vxy = np.array(vertex_xy)
        self.eu, self.ev = np.array(edge_from), np.array(edge_to)
        self.ew = np.array(edge_seconds, float)
        self.eseg = np.array(edge_segment)
        self.lp_xy = np.array([(p["x"], p["y"]) for p in launch_points])
        self.lp_tree = cKDTree(self.lp_xy)
        # 各センサーから片道で届く離陸候補点と、その距離: {センサー: {離陸候補点の番号: 距離 m}}
        self.sensor_lp = {}
        for j, sid in enumerate(sensor_ids):
            near = self.lp_tree.query_ball_point(sensor_xy[j], reach + EPS)
            self.sensor_lp[sid] = {i: float(np.hypot(*(self.lp_xy[i] - sensor_xy[j]))) for i in near}
        self.build_seconds = time.time() - t_start

    def _open_graph(self, closed):
        """通行止めの区間を除いた道路網のグラフ（疎行列）."""
        is_open = ~np.isin(self.eseg, list(closed))
        n = self.n_vertices
        return csr_matrix((self.ew[is_open], (self.eu[is_open], self.ev[is_open])), shape=(n, n))

    # ------------------------------------------------------------------
    def run(self, closed_segments, comm_down, bases, keep_per_set=None):
        """通行止め区間・通信断センサー・拠点から前処理結果を作る.

        bases: [{"base_id", "node_id", "vehicles", "name"}]
        keep_per_set: 同じセンサーの組を回る飛行のうち、残す離陸点の数の上限（None = 支配されないもの全部）
        """
        t_start = time.time()
        drone = self.cfg.drone
        closed = set(closed_segments)
        unknown_closed = sorted(closed - set(self.segments))
        unknown_sensors = sorted(set(comm_down) - set(self.sensors))
        comm_down = [s for s in comm_down if s in self.sensors]

        # 1) 通行止めを除いた道路網で、拠点から各頂点までの最短走行時間（秒）
        graph = self._open_graph(closed)
        base_vertices = [self.vidx[b["node_id"]] for b in bases]
        sec_from_base = dijkstra(graph, directed=False, indices=base_vertices)   # [拠点, 頂点]
        reachable_vertex = np.isfinite(sec_from_base).any(0)                     # どれかの拠点から行ける頂点

        # 2) 使える離陸候補点: 拠点から行けて、乗っている一般道区間が開いている
        def usable(p):
            if not reachable_vertex[p["vertex"]]:
                return False
            if p["segment_id"] is not None:
                return p["segment_id"] not in closed
            return any(s not in closed for s in p["node_segments"])
        usable_lp = np.array([usable(p) for p in self.launch_points])

        # 3) 通信断センサーごとの、使える離陸候補点と、対応不能の判定
        near_lp = {}                 # センサー → {使える離陸候補点: 距離}
        unreachable = []
        nearest_km = {}
        for s in comm_down:
            near = {i: d for i, d in self.sensor_lp[s].items() if usable_lp[i]}
            near_lp[s] = near
            if near:
                nearest_km[s] = round(min(near.values()) / 1000, 3)
            else:
                unreachable.append(s)
                # 参考: 届かない場合の、いちばん近い使える離陸点までの距離
                if usable_lp.any():
                    d, _ = cKDTree(self.lp_xy[usable_lp]).query(self.sensors[s][:2])
                else:
                    d = math.inf
                nearest_km[s] = round(d / 1000, 3)
        targets = [s for s in comm_down if s not in unreachable]

        # 4) 飛行の列挙: 各離陸候補点から、届くセンサーの組（1〜4 基）を、いちばん短い回り方で回れるか
        meters_per_min = drone.speed_kmh * 1000 / 60
        sensors_near_lp = {}         # 離陸候補点 → 届くセンサーの一覧
        for s in targets:
            for i in near_lp[s]:
                sensors_near_lp.setdefault(i, []).append(s)
        target_xy = {s: np.array(self.sensors[s][:2]) for s in targets}
        flights_by_set = {}          # センサーの組 → [(離陸候補点, 回る順, 飛行分, 飛行距離 m), ...]
        for i, near_sensors in sensors_near_lp.items():
            origin = self.lp_xy[i]
            near_sensors = sorted(near_sensors)
            for k in range(1, min(drone.max_sensors_per_flight, len(near_sensors)) + 1):
                # 飛べる距離 = (15 分 − 滞空 3 分 × k) × 速さ
                distance_budget = (drone.max_flight_min - drone.hover_min * k) * meters_per_min + EPS
                if distance_budget <= 0:
                    break
                for combo in itertools.combinations(near_sensors, k):
                    best = None              # (回る順, 距離)
                    for order in itertools.permutations(combo):
                        if k > 1 and order[0] > order[-1]:
                            continue         # 逆回りは同じ距離なので片方だけ調べる
                        stops = [origin] + [target_xy[s] for s in order] + [origin]
                        dist = sum(float(np.hypot(*(stops[j + 1] - stops[j]))) for j in range(len(stops) - 1))
                        if dist <= distance_budget and (best is None or dist < best[1]):
                            best = (order, dist)
                    if best:
                        minutes = best[1] / meters_per_min + drone.hover_min * k
                        flights_by_set.setdefault(frozenset(combo), []).append((i, best[0], minutes, best[1]))

        # 5) 同じセンサーの組の中で「支配される」離陸点を落とす。
        #    比べるのは（各拠点からの走行時間, 飛行時間）。どれも別の離陸点以上なら、その離陸点を使う理由がない。
        #    （仮定: 離陸点どうしの走行（次の飛行への移動）は比べていない。車 3 台の条件で、間引く前の全部で解いても最適値が同じことは確認した）
        def drive_from_bases(i):
            return sec_from_base[:, self.launch_points[i]["vertex"]]
        flights = []
        n_before_pruning = 0
        for combo, options in flights_by_set.items():
            n_before_pruning += len(options)
            scored = [(np.append(drive_from_bases(i), minutes), (i, order, minutes, dist))
                      for i, order, minutes, dist in options]
            scored.sort(key=lambda t: (np.nanmin(t[0][:-1]), t[0][-1]))   # 拠点から近い順 → 飛行が短い順
            kept = []
            for score, flight in scored:
                if any(np.all(kept_score <= score + EPS) for kept_score, _ in kept):
                    continue                 # すでに残したものに支配される
                kept.append((score, flight))
                if keep_per_set and len(kept) >= keep_per_set:
                    break
            for _, (i, order, minutes, dist) in kept:
                flights.append(dict(sensors=list(order), launch=i, flight_min=round(minutes, 3),
                                    flight_km=round(dist / 1000, 3)))
        flights.sort(key=lambda f: (-len(f["sensors"]), f["sensors"], f["flight_min"]))
        used_lp = sorted({f["launch"] for f in flights})
        launch_id_of = {i: self.launch_points[i]["launch_id"] for i in used_lp}
        for k, f in enumerate(flights):
            f["flight_id"] = f"F{k:05d}"
            f["launch_id"] = launch_id_of[f.pop("launch")]

        # 6) 拠点と、使う離陸点のすべての組の最短走行時間
        point_vertices = base_vertices + [self.launch_points[i]["vertex"] for i in used_lp]
        labels = [b["base_id"] for b in bases] + [launch_id_of[i] for i in used_lp]
        sec_between = dijkstra(graph, directed=False, indices=point_vertices)[:, point_vertices]

        launch_out = []
        for i in used_lp:
            p = self.launch_points[i]
            launch_out.append(dict(
                launch_id=p["launch_id"], x=round(p["x"], 1), y=round(p["y"], 1),
                segment_id=p["segment_id"] or p.get("node_id"), on_node=p["segment_id"] is None,
                road_name=(self.segments[p["segment_id"]]["name"] if p["segment_id"] else None),
                sensors_in_reach=sorted(sensors_near_lp[i]),
                drive_min_from_base={b["base_id"]: minutes_or_none(sec_from_base[k, p["vertex"]] / 60)
                                     for k, b in enumerate(bases)},
            ))
        n_flights_of = {s: sum(1 for f in flights if s in f["sensors"]) for s in targets}
        n_bases = len(bases)
        return dict(
            summary=dict(
                comm_down=len(comm_down), reachable=len(targets), unreachable=len(unreachable),
                closed_segments=len(closed), launch_candidates_usable=int(usable_lp.sum()),
                launch_points_used=len(used_lp), flights=len(flights), flights_before_pruning=n_before_pruning,
                flights_by_size={k: sum(1 for f in flights if len(f["sensors"]) == k)
                                 for k in range(1, drone.max_sensors_per_flight + 1)},
                bases_reachable_from_each_other=bool(np.isfinite(sec_between[:n_bases, :n_bases]).all()),
                seconds=round(time.time() - t_start, 3),
            ),
            warnings=dict(unknown_closed_segments=unknown_closed, unknown_sensors=unknown_sensors),
            unreachable=sorted(unreachable),
            sensors={s: dict(status=("unreachable" if s in unreachable else "reachable"),
                             nearest_launch_km=nearest_km[s], flights=n_flights_of.get(s, 0),
                             x=self.sensors[s][0], y=self.sensors[s][1], municipality_id=self.sensors[s][2])
                     for s in comm_down},
            bases=bases,
            launch_points=launch_out,
            flights=flights,
            travel_time_min=dict(labels=labels, matrix=[[minutes_or_none(v / 60) for v in row] for row in sec_between]),
        )

    # ------------------------------------------------------------------
    def drive_paths(self, closed_segments, legs, bases):
        """走行ルートの折れ線（地図に描く用）.

        legs: [(出発のラベル, 到着のラベル), ...]（ラベルは拠点 ID か離陸点 ID）
        戻り値: {(出発, 到着): [(x, y, 出発からの走行秒数), ...]}（行けない組は入らない）
        """
        graph = self._open_graph(set(closed_segments))
        vertex_of_lp = {p["launch_id"]: p["vertex"] for p in self.launch_points}
        vertex_of_base = {b["base_id"]: self.vidx[b["node_id"]] for b in bases}

        def vertex(label):
            return vertex_of_base[label] if label in vertex_of_base else vertex_of_lp[label]
        sources = sorted({vertex(a) for a, _ in legs})
        if not sources:
            return {}
        # predecessors[r, w] = 出発点 sources[r] からの最短経路で、頂点 w の 1 つ手前の頂点
        dist, predecessors = dijkstra(graph, directed=False, indices=sources, return_predecessors=True)
        row_of = {v: r for r, v in enumerate(sources)}
        out = {}
        for a, b in legs:
            va, vb = vertex(a), vertex(b)
            r = row_of[va]
            if not np.isfinite(dist[r, vb]):
                continue
            path = [vb]                      # 到着から 1 つずつ手前へたどり、最後に逆順にする
            while path[-1] != va:
                path.append(predecessors[r, path[-1]])
            # 各点は (x, y, 出発からの走行秒数). 秒数はアニメーションで車を道路の速さどおりに動かすのに使う
            out[(a, b)] = [(float(self.vxy[k][0]), float(self.vxy[k][1]), float(dist[r, k])) for k in reversed(path)]
        return out
