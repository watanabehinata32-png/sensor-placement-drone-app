"""Phase 2 道路データの読み込み層.

合成データ (このフォルダの road_*.geojson) でも、後から届く企業の本データでも、
同じ「共通スキーマ」(README の「共通スキーマ」節) に揃えてから後段 (飛行列挙・車巡回など) に渡す。
形式の違いは設定 (config) で吸収し、後段のコードは変えない方針。

吸収できる違い:
  - 列名の違い (例: highway ではなく road_type, segment_id ではなく link_id)
  - 道路種別の値の違い (日本語名・OSM タグ・数値コード) → class_map で共通種別へ
  - ノード ID が無い (端点の座標をスナップしてノードを作る)
  - 長さ・速度・所要時間が無い (形状と種別の既定速度から計算)
  - MultiLineString、1 km を超える区間 (split_max_m を指定すれば分割)
  - 座標系の違い (pyproj があれば to_crs で変換)
  - GeoJSON 以外 (geopandas があれば Shapefile / GeoPackage なども)

使い方:
    from road_loader import load_roads, load_scenario, load_facilities
    net = load_roads("road_segments.geojson", nodes_path="road_nodes.geojson")
    sc = load_scenario("scenario_eq_tonan_west.json")
    g = net.graph(closed=sc.closed_segments)      # networkx.Graph (重み travel_time_s)

    # 企業データの例: 列名と種別コードが違い、ノード ID が無い
    net = load_roads("company_roads.geojson", config="company_config.json")

CLI (検証レポート):
    python road_loader.py road_segments.geojson --nodes road_nodes.geojson [--config cfg.json]
"""
import argparse
import json
import math
import os
from dataclasses import dataclass, field

import networkx as nx

# ---- 共通スキーマの定義 ----
CLASSES = ["motorway", "motorway_link", "trunk", "primary", "secondary", "tertiary", "other"]
DEFAULT_SPEED_KMH = {"motorway": 100, "motorway_link": 40, "trunk": 60, "primary": 50,
                     "secondary": 40, "tertiary": 30, "other": 30}
NO_LAUNCH = {"motorway", "motorway_link"}  # ドローンを離陸させない種別

# 種別の値の既定対応 (config の class_map で追加・上書き)
DEFAULT_CLASS_MAP = {
    "motorway": "motorway", "motorway_link": "motorway_link",
    "trunk": "trunk", "trunk_link": "trunk", "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary", "tertiary": "tertiary", "tertiary_link": "tertiary",
    "unclassified": "other", "residential": "other", "service": "other", "track": "other",
    "高速道路": "motorway", "高速自動車国道": "motorway", "自動車専用道路": "motorway", "ランプ": "motorway_link",
    "国道": "trunk", "一般国道": "trunk", "主要地方道": "primary",
    "一般県道": "secondary", "県道": "secondary", "一般都道府県道": "secondary", "市町村道": "tertiary",
}

# 列名の既定候補 (先に見つかったものを使う。config の fields で上書き)
DEFAULT_FIELDS = {
    "segment_id": ["segment_id", "link_id", "id", "osm_id", "fid"],
    "from_node": ["from_node", "u", "source", "start_node", "node_from"],
    "to_node": ["to_node", "v", "target", "end_node", "node_to"],
    "road_class": ["highway", "road_class", "road_type", "class", "種別"],
    "name": ["name", "road_name", "路線名"],
    "ref": ["ref", "route_no", "路線番号"],
    "length_m": ["length_m", "length", "len_m"],
    "maxspeed_kmh": ["maxspeed", "maxspeed_kmh", "speed_kmh", "speed"],
    "travel_time_s": ["travel_time_s", "travel_time", "time_s"],
    "oneway": ["oneway"],
    "launch_ok": ["drone_launch_ok", "launch_ok"],
    "node_id": ["node_id", "id", "osm_id"],
}


@dataclass
class Segment:
    segment_id: str
    u: str
    v: str
    coords: list  # [(x, y), ...] メートル座標
    road_class: str
    length_m: float
    maxspeed_kmh: float
    travel_time_s: float
    launch_ok: bool
    oneway: bool = False
    name: str = None
    ref: str = None
    raw: dict = field(default_factory=dict)  # 元データの属性 (そのまま保持)


@dataclass
class RoadNetwork:
    nodes: dict  # node_id -> {"x", "y", **元属性}
    segments: list
    warnings: list = field(default_factory=list)

    def by_id(self):
        return {s.segment_id: s for s in self.segments}

    def graph(self, closed=()):
        """通行止め (closed の segment_id) を除いた無向グラフ. 一方通行は無視 (災害時の緊急車両を想定)."""
        closed = set(closed)
        g = nx.MultiGraph()
        for s in self.segments:
            if s.segment_id not in closed:
                g.add_edge(s.u, s.v, key=s.segment_id, segment_id=s.segment_id, travel_time_s=s.travel_time_s,
                           length_m=s.length_m)
        return g

    def reachable_nodes(self, start_nodes, closed=()):
        g = self.graph(closed)
        out = set()
        for n in start_nodes:
            if n in g:
                out |= nx.node_connected_component(g, n)
        return out

    def launch_segments(self, start_nodes, closed=()):
        """車で到達でき、ドローンを離陸させられる区間."""
        reach = self.reachable_nodes(start_nodes, closed)
        closed = set(closed)
        return [s for s in self.segments if s.launch_ok and s.segment_id not in closed and s.u in reach]

    def validate(self, max_seg_m=1000.0):
        g = self.graph()
        exits = {n for n, p in self.nodes.items() if p.get("kind") == "boundary_exit"}
        dead = [n for n in g.nodes if g.degree(n) == 1 and n not in exits]
        classes = {}
        for s in self.segments:
            classes[s.road_class] = classes.get(s.road_class, 0) + 1
        return dict(
            segments=len(self.segments), nodes=len(self.nodes),
            components=nx.number_connected_components(g) if len(g) else 0,
            dead_ends=len(dead), dead_end_nodes=dead[:20],
            over_max_length=sum(1 for s in self.segments if s.length_m > max_seg_m + 1e-6),
            classes=classes, warnings=self.warnings[:20], n_warnings=len(self.warnings),
        )


@dataclass
class Scenario:
    scenario_id: str
    comm_down: list  # candidate_id のリスト
    closed_segments: set
    epicenter: tuple = None
    expected_unreachable: list = None  # ファイルに答えがあれば (検証用)
    raw: dict = field(default_factory=dict)


# ---------------- 読み込み ----------------

def _load_config(config):
    if config is None:
        return {}
    if isinstance(config, str):
        with open(config, encoding="utf-8") as f:
            return json.load(f)
    return dict(config)


def _read_features(path, cfg):
    """GeoJSON は標準ライブラリで、それ以外は geopandas で読む. 戻り値は (props, geometry dict) のリスト."""
    if path.lower().endswith((".geojson", ".json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        feats = d["features"] if d.get("type") == "FeatureCollection" else d
        out = [(f.get("properties") or {}, f["geometry"]) for f in feats]
        src_crs = cfg.get("crs") or ((d.get("crs") or {}).get("properties") or {}).get("name")
    else:
        import geopandas as gpd  # 任意依存
        gdf = gpd.read_file(path)
        src_crs = cfg.get("crs") or (gdf.crs.to_string() if gdf.crs else None)
        out = [(dict(r.drop("geometry")), json.loads(gpd.GeoSeries([r.geometry]).to_json())["features"][0]["geometry"])
               for _, r in gdf.iterrows()]
    to_crs = cfg.get("to_crs")
    if to_crs and src_crs and src_crs != to_crs:
        from pyproj import Transformer  # 任意依存
        tr = Transformer.from_crs(src_crs, to_crs, always_xy=True)

        def conv(c):
            return [list(tr.transform(*p[:2])) for p in c]
        for _, gm in out:
            if gm["type"] == "LineString":
                gm["coordinates"] = conv(gm["coordinates"])
            elif gm["type"] == "MultiLineString":
                gm["coordinates"] = [conv(c) for c in gm["coordinates"]]
            elif gm["type"] == "Point":
                gm["coordinates"] = conv([gm["coordinates"]])[0]
    return out


def _pick(props, key, cfg):
    names = cfg.get("fields", {}).get(key)
    names = [names] if isinstance(names, str) else (names or DEFAULT_FIELDS.get(key, []))
    for n in names:
        if n in props and props[n] not in (None, ""):
            return props[n]
    return None


def _length(coords):
    return sum(math.dist(a, b) for a, b in zip(coords, coords[1:]))


def _as_bool(v, default):
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("yes", "true", "1", "-1")
    return bool(v)


def _split(coords, max_len):
    L = _length(coords)
    n = max(1, math.ceil(L / max_len - 1e-9))
    if n == 1:
        return [coords]
    cum = [0.0]
    for a, b in zip(coords, coords[1:]):
        cum.append(cum[-1] + math.dist(a, b))

    def at(d):
        for i in range(len(coords) - 1):
            if cum[i + 1] >= d:
                t = 0 if cum[i + 1] == cum[i] else (d - cum[i]) / (cum[i + 1] - cum[i])
                return (coords[i][0] + t * (coords[i + 1][0] - coords[i][0]),
                        coords[i][1] + t * (coords[i + 1][1] - coords[i][1]))
        return tuple(coords[-1])
    cuts = [L * k / n for k in range(n + 1)]
    out = []
    for s, t in zip(cuts, cuts[1:]):
        out.append([at(s)] + [tuple(c) for c, d in zip(coords, cum) if s < d < t] + [at(t)])
    return out


def load_roads(segments_path, nodes_path=None, config=None):
    """道路区間 (と任意でノード) を読み、共通スキーマの RoadNetwork を返す.

    config (dict か JSON ファイルのパス) のキー:
      fields:      {共通名: 元の列名 or 候補リスト}  例 {"road_class": "road_type", "segment_id": "link_id"}
      class_map:   {元の値: 共通種別}               例 {"1": "motorway", "3": "trunk"}
      default_speed_kmh: {共通種別: km/h}
      snap_tol_m:  ノード ID が無いとき端点を同一視する距離 (既定 1.0)
      split_max_m: 指定すると、それより長い区間を等分割する (既定 なし)
      crs / to_crs: 座標変換 (pyproj が必要)
      launch_classes_excluded: ドローン離陸不可の種別 (既定 motorway, motorway_link)
    """
    cfg = _load_config(config)
    class_map = {**DEFAULT_CLASS_MAP, **{str(k): v for k, v in cfg.get("class_map", {}).items()}}
    speeds = {**DEFAULT_SPEED_KMH, **cfg.get("default_speed_kmh", {})}
    no_launch = set(cfg.get("launch_classes_excluded", NO_LAUNCH))
    tol = float(cfg.get("snap_tol_m", 1.0))
    warnings = []

    nodes = {}
    if nodes_path:
        for props, gm in _read_features(nodes_path, cfg):
            nid = _pick(props, "node_id", cfg)
            x, y = gm["coordinates"][:2]
            nodes[str(nid)] = dict(props, x=float(x), y=float(y))

    # 座標 → ノード ID (スナップ用の格子ハッシュ)
    grid = {}

    def cell(x, y):
        return (int(math.floor(x / max(tol, 1e-9))), int(math.floor(y / max(tol, 1e-9))))

    for nid, p in nodes.items():
        grid.setdefault(cell(p["x"], p["y"]), []).append(nid)

    def node_at(x, y):
        cx, cy = cell(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in grid.get((cx + dx, cy + dy), []):
                    if math.dist((nodes[nid]["x"], nodes[nid]["y"]), (x, y)) <= tol:
                        return nid
        nid = f"auto{len(nodes) + 1:06d}"
        nodes[nid] = dict(x=float(x), y=float(y), kind="auto")
        grid.setdefault((cx, cy), []).append(nid)
        return nid

    segments = []
    for i, (props, gm) in enumerate(_read_features(segments_path, cfg)):
        if gm["type"] == "LineString":
            parts = [gm["coordinates"]]
        elif gm["type"] == "MultiLineString":
            parts = gm["coordinates"]
        else:
            warnings.append(f"#{i}: geometry {gm['type']} を無視")
            continue
        sid = _pick(props, "segment_id", cfg)
        sid = str(sid) if sid is not None else f"seg{i + 1:06d}"
        raw_cls = _pick(props, "road_class", cfg)
        cls = class_map.get(str(raw_cls), None) if raw_cls is not None else None
        if cls is None:
            cls = "other"
            warnings.append(f"{sid}: 種別 {raw_cls!r} が class_map に無いので other 扱い")
        spd = _pick(props, "maxspeed_kmh", cfg)
        try:
            spd = float(str(spd).split()[0]) if spd is not None else float(speeds[cls])
        except ValueError:
            spd = float(speeds[cls])
        launch = _as_bool(_pick(props, "launch_ok", cfg), cls not in no_launch)
        oneway = _as_bool(_pick(props, "oneway", cfg), False)
        u0, v0 = _pick(props, "from_node", cfg), _pick(props, "to_node", cfg)

        pieces = []
        for part in parts:
            coords = [tuple(map(float, c[:2])) for c in part]
            pieces += _split(coords, cfg["split_max_m"]) if cfg.get("split_max_m") else [coords]
        for k, coords in enumerate(pieces):
            single = len(pieces) == 1
            u = str(u0) if (single and u0 is not None and str(u0) in nodes) else node_at(*coords[0])
            v = str(v0) if (single and v0 is not None and str(v0) in nodes) else node_at(*coords[-1])
            L = _length(coords)
            given_L = _pick(props, "length_m", cfg) if single else None
            if given_L is not None and abs(float(given_L) - L) > max(5.0, 0.05 * L):
                warnings.append(f"{sid}: 属性の長さ {given_L} と形状の長さ {L:.0f} が違う (形状を採用)")
            tt = _pick(props, "travel_time_s", cfg) if single else None
            segments.append(Segment(
                segment_id=sid if single else f"{sid}#{k + 1}", u=u, v=v, coords=coords, road_class=cls,
                length_m=round(L, 1), maxspeed_kmh=spd,
                travel_time_s=float(tt) if tt is not None else round(L / (spd / 3.6), 1),
                launch_ok=launch, oneway=oneway, name=_pick(props, "name", cfg), ref=_pick(props, "ref", cfg),
                raw=props))
    ids = [s.segment_id for s in segments]
    if len(ids) != len(set(ids)):
        warnings.append("segment_id に重複あり")
    return RoadNetwork(nodes=nodes, segments=segments, warnings=warnings)


def load_facilities(path, config=None):
    """ドローン拠点 (と IC/JCT). 戻り値: [{facility_id, kind, name, x, y, node_id?, dock_vehicles?}]."""
    cfg = _load_config(config)
    out = []
    for props, gm in _read_features(path, cfg):
        x, y = gm["coordinates"][:2]
        out.append(dict(
            facility_id=str(props.get("facility_id") or props.get("id") or props.get("name")),
            kind=props.get("kind", "drone_base"), name=props.get("name"),
            node_id=props.get("node_id"), x=float(x), y=float(y),
            dock_vehicles=props.get("dock_vehicles") or props.get("vehicles")))
    return out


def snap_to_network(net, x, y):
    """座標に最も近いノード ID (拠点に node_id が無いデータ用)."""
    return min(net.nodes, key=lambda n: math.dist((net.nodes[n]["x"], net.nodes[n]["y"]), (x, y)))


def load_scenario(path, config=None):
    """シナリオ JSON. キー名の揺れを吸収する (comm_down_sensors / down_sensors / failed_sensors など)."""
    cfg = _load_config(config)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    def first(*keys):
        for k in keys:
            if k in d:
                return d[k]
        return None
    down = first(*cfg.get("down_keys", ["comm_down_sensors", "down_sensors", "failed_sensors", "comm_down"])) or []
    down = [x["candidate_id"] if isinstance(x, dict) else x for x in down]
    closed = first(*cfg.get("closed_keys", ["closed_segments", "blocked_segments", "closed_links", "closures"])) or []
    closed = {str(x["segment_id"]) if isinstance(x, dict) else str(x) for x in closed}
    epi = d.get("epicenter")
    if isinstance(epi, dict):
        epi = (epi.get("x"), epi.get("y"))
    unreach = (d.get("summary") or {}).get("unreachable_ids") or d.get("unreachable")
    return Scenario(scenario_id=str(d.get("scenario_id", os.path.basename(path))), comm_down=[str(x) for x in down],
                    closed_segments=closed, epicenter=tuple(epi) if epi else None,
                    expected_unreachable=unreach, raw=d)


def unreachable_sensors(net, base_nodes, sensors_xy, closed=(), reach_m=4000.0):
    """通行止め後に、離陸可能な道路から reach_m 以内に入らないセンサー. 戻り値 {id: 最短距離 m}."""
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union
    segs = net.launch_segments(base_nodes, closed)
    U = unary_union([LineString(s.coords) for s in segs])
    dist = {k: Point(*xy).distance(U) for k, xy in sensors_xy.items()}
    return {k: d for k, d in dist.items() if d > reach_m}, dist


def main():
    ap = argparse.ArgumentParser(description="道路データを共通スキーマで読み込み、検証レポートを出す")
    ap.add_argument("segments")
    ap.add_argument("--nodes")
    ap.add_argument("--config")
    ap.add_argument("--max-seg-m", type=float, default=1000.0)
    a = ap.parse_args()
    net = load_roads(a.segments, a.nodes, a.config)
    print(json.dumps(net.validate(a.max_seg_m), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
