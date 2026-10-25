"""入力データの読み込み.

企業の本データに差し替えるときは、道路は road_loader の設定 JSON（列名・種別コード・座標系）を
load_roads(..., config="company_config.json") で渡す。センサーとシナリオは下の形式。
"""
import json

# 道路の読み込みは road_loader.py（phase2_roads/road_loader.py のコピー。列名・種別コード・座標系の違いを設定で吸収する）
from . import road_loader


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_roads(segments_path, nodes_path=None, config=None):
    """道路区間とノードを読む. config は road_loader の設定（企業データの列名など）.
    戻り値: (segments: list[dict], nodes: dict[node_id -> (x, y)])."""
    net = road_loader.load_roads(segments_path, nodes_path=nodes_path, config=config)
    nodes = {str(k): (float(v["x"]), float(v["y"])) for k, v in net.nodes.items()}
    segments = [dict(segment_id=str(s.segment_id), from_node=str(s.u), to_node=str(s.v), highway=s.road_class,
                     launch_ok=bool(s.launch_ok), name=s.name, length_m=s.length_m,
                     travel_time_s=float(s.travel_time_s), coords=[tuple(c[:2]) for c in s.coords])
                for s in net.segments]
    return segments, nodes


def load_sensors(path):
    """センサー位置. Phase 1 の配布データ（municipalities[].candidate_locations）か、
    {"sensors": [{"id", "x", "y"}]} 形式の JSON を読む. 戻り値: dict[id -> (x, y, municipality_id)]."""
    d = _read(path)
    out = {}
    if "municipalities" in d:
        for m in d["municipalities"]:
            for c in m["candidate_locations"]:
                out[c["id"]] = (float(c["x"]), float(c["y"]), m["id"])
    else:
        for s in d["sensors"]:
            out[s["id"]] = (float(s["x"]), float(s["y"]), s.get("municipality_id"))
    return out


def load_scenario(path):
    """シナリオ. 必須キー: comm_down_sensors（id の配列か {candidate_id} の配列）, closed_segments, drone_bases."""
    d = _read(path)
    down = [s if isinstance(s, str) else s["candidate_id"] for s in d["comm_down_sensors"]]
    bases = [dict(base_id=b["facility_id"], node_id=str(b["node_id"]), name=b.get("name"),
                  vehicles=int(b.get("vehicles", 1))) for b in d["drone_bases"]]
    return dict(
        scenario_id=d.get("scenario_id", "scenario"),
        title=d.get("title", ""),
        comm_down=down,
        closed_segments=set(map(str, d["closed_segments"])),
        bases=bases,
        raw=d,
    )
