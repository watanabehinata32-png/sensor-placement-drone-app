"""QUBO を何度か解いて、終了時刻の上限 T を二分探索で小さくしていく（QUBO 版の本体）.

使い方:
    inst = build_instance(pre, cfg, end_mode="recovery")
    out = solve_routing(inst, TabuSearch())      # Amplify なら AmplifySolver(client)

流れ:
1. 下限 LB を求める（どれか 1 基は、どの車がどこから飛んでもこれより早くは終われない）。
2. 上限 T を決めて「T までに終われる変数・組」だけの QUBO を作り、ソルバーで解く（T が小さいほど変数も減る）。
3. 答えが全センサーを回り制約も守っていれば、その終了時刻を新しい「これ以下で済む」値（hi）にする。
   T 以内の答えが出なければ、T を「これ以上かかりそう」な値（lo）にする。
4. lo と hi の間で T を選び直して 2. に戻る（二分探索）。答えが LB に届いたら最適と分かるので止める。
"""
import time

import numpy as np

from .routing import evaluate, route_times
from .routing_qubo import Weights, build_qubo, decode


def lower_bound(inst):
    """終了時刻の下限.

    各センサーについて「いちばん早く終われる車 1 台が、そのセンサーだけを回る」ときの終了時刻を求め、その最大値。
    ほかのセンサーも回るなら、これより早くはならない。
    """
    lb = 0.0
    for s in inst.targets:
        best_for_s = np.inf
        for f in inst.flights:
            if s not in f.sensors:
                continue
            for v in range(len(inst.vehicles)):
                drive = inst.drive_from_base(v, f.idx)
                if inst.end_mode == "return":
                    t = drive + inst.time_after_last(v, f.idx)                        # 拠点に帰るまで
                else:
                    t = drive + f.recover_off[f.sensors.index(s)]          # そのセンサーの回復まで
                best_for_s = min(best_for_s, t)
        if np.isfinite(best_for_s):
            lb = max(lb, best_for_s)
    return lb


def _end_time(inst, v, route):
    """車 v がルート route を回るときの終了時刻（選んだ定義で）."""
    _, end_recovery, end_return = route_times(inst, v, route)
    return end_recovery if inst.end_mode == "recovery" else end_return


def repair(inst, routes, spares):
    """回れていないセンサーを、全体の終了時刻がいちばん増えない「車・順番・飛行」に差し込む（貪欲法）.

    - QUBO の答えでセンサーが残ったときの後処理（使ったら画面に出す）
    - routes = {} から呼べば、ゼロからの手早い答え（厳密解版で数え上げの上限に使う）
    バッテリーの回数と予備の総数は守る。どこにも入らないセンサーは回れないまま残す。
    戻り値: (routes, spares, 追加したセンサーの一覧)
    """
    fleet = inst.cfg.fleet
    routes = {v: list(r) for v, r in routes.items()}
    # 予備バッテリーは「予備なしの回数を超えて飛ぶ車」だけが使う、と数え直す
    spares = {v: len(r) > fleet.flights_builtin for v, r in routes.items()}
    done = {s for r in routes.values() for f in r for s in inst.flights[f].sensors}
    added = []
    for s in inst.targets:
        if s in done:
            continue
        spares_left = fleet.spare_total - sum(spares.values())
        best = None                               # ((全体の終了, その車の終了), 車, 新しいルート)
        for v in range(len(inst.vehicles)):
            route = routes.get(v, [])
            # この車が飛べる回数: 予備を積んでいるか、まだ予備が残っていれば予備込みの回数
            can_use_spare = spares.get(v, False) or spares_left > 0
            limit = inst.max_flights(can_use_spare)
            if len(route) + 1 > limit:
                continue
            # この車以外でいちばん遅い終了時刻
            others_end = max([_end_time(inst, u, r) for u, r in routes.items() if u != v and r] + [0.0])
            for f in inst.flights:
                if s not in f.sensors or set(f.sensors) & done:
                    continue                      # このセンサーを回らない／もう回ったセンサーを含む飛行
                for pos in range(len(route) + 1):
                    candidate = route[:pos] + [f.idx] + route[pos:]
                    end = _end_time(inst, v, candidate)
                    if not np.isfinite(end):
                        continue
                    key = (max(end, others_end), end)
                    if best is None or key < best[0]:
                        best = (key, v, candidate)
        if best:
            _, v, candidate = best
            routes[v] = candidate
            if len(candidate) > fleet.flights_builtin:
                spares[v] = True                  # 予備を 1 個使う
            done |= {x for f in candidate for x in inst.flights[f].sensors}
            added.append(s)
    return routes, spares, added


def solve_routing(inst, solver, weights=None, K=None, rounds=8, tol=0.5, T_start=None, grow=1.12, progress=None):
    """QUBO を最大 rounds 回解いて、いちばん良い答えを返す.

    solver:   solve(prob) を持つもの（solvers.py）
    tol:      lo と hi の差がこれ（分）以下になったら止める
    grow:     まだ答えがないとき、T を何倍ずつ広げるか
    progress: 1 回解くごとに呼ぶ関数（画面に途中経過を出す用）
    戻り値 dict: ok, routes, spares, eval（時刻表）, lower_bound, history（各回の記録）, seconds など
    """
    weights = weights or Weights()
    t_start = time.time()
    lb = lower_bound(inst)
    history = []
    best = None                # (終了時刻, routes, spares, eval, 何回目)
    lo, hi = lb, None          # 最適値は lo〜hi のどこか（hi = これまでの最良。まだなければ None）
    T = lb + 1e-6 if T_start is None else T_start
    last_answer = None         # 最後に出た答え（全部は回れていなくても後処理で使う）

    for r in range(rounds):
        prob = build_qubo(inst, T, K=K, weights=weights)
        if prob.n == 0:
            # T が小さすぎて変数が 1 つも作れない → T を広げる
            history.append(dict(round=r + 1, T=round(T, 2), n_vars=0, valid=False, within_T=False, note="変数なし"))
            lo = T
            T = T * grow if hi is None else (T + hi) / 2
            continue

        x, info = solver.solve(prob)
        routes, spares, violations = decode(inst, prob, x)
        ev = evaluate(inst, routes, spares)
        valid = ev["feasible"] and not any(violations.values())     # 全センサーを回り、制約も守っている
        within_T = valid and ev["makespan"] <= T + 1e-6              # しかも上限 T までに終わっている
        last_answer = (routes, spares)
        history.append(dict(round=r + 1, T=round(float(T), 2), n_vars=prob.n, n_pairs=prob.meta["n_pairs"],
                            energy=round(float(info.get("energy", np.nan)), 4), solver_seconds=info.get("seconds"),
                            recovered=ev["recovered"], makespan=float(ev["makespan"]), valid=bool(valid),
                            within_T=bool(within_T), violations={k: v for k, v in violations.items() if v}))
        if progress:
            progress(history[-1])

        if valid:
            if best is None or ev["makespan"] < best[0] - 1e-9:
                best = (ev["makespan"], routes, spares, ev, r + 1)
            hi = ev["makespan"] if hi is None else min(hi, ev["makespan"])
        if best is not None and best[0] <= lb + 1e-6:
            break                                   # 下限に届いた = 最適
        if not within_T:
            lo = max(lo, T)                         # この T 以内の答えは見つからなかった
        if hi is not None:
            lo = min(lo, hi)
            if hi - lo <= tol:
                break                               # 十分に絞れた

        # 次の T
        if hi is None:
            T = T * grow                            # まだ答えがない → 広げる
        elif hi - lo > 2 * tol:
            T = max(lb + 1e-6, (lo + hi) / 2)       # lo と hi のまん中
        else:
            T = hi - tol                            # あと少し: 最良より tol だけ早い答えを探す

    repaired = []
    if best is None and last_answer is not None:
        # 一度も全部を回れなかった → 最後の答えに、回れなかったセンサーを貪欲法で足す
        routes, spares, repaired = repair(inst, last_answer[0], last_answer[1])
        ev = evaluate(inst, routes, spares)
        best = (ev["makespan"], routes, spares, ev, None)
    if best is None:
        return dict(ok=False, lower_bound=lb, history=history, seconds=round(time.time() - t_start, 2))
    makespan, routes, spares, ev, best_round = best
    return dict(ok=True, routes=routes, spares=spares, eval=ev, lower_bound=round(lb, 2), history=history,
                # 下限は「全センサーを回る」ときの値なので、最適と言えるのは全部回れて、後処理も使っていないときだけ
                best_round=best_round, repaired=repaired,
                proven_optimal=bool(ev["feasible"] and not repaired and makespan <= lb + 1e-6),
                seconds=round(time.time() - t_start, 2))
