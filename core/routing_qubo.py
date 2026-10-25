"""Phase 2 ステップ 2・3 の QUBO.

「どの離陸点から飛ぶか（飛行の選択）」「どの車が担当するか（割当）」「車が回る順番」を、1 つの QUBO にまとめて解く。

■ 変数（どれも 0 か 1）
  x[v, k, f] = 1 … 車 v が、後ろから数えて k 番目（k = 1 が最後の飛行）に、飛行 f を行う
                   （飛行 f = 「どの離陸点から・どのセンサーを回るか」の組。候補は routing.build_instance で作る）
  b[v]       = 1 … 車 v に予備バッテリーを積む（予備なしの回数を超えて飛ぶ車に必要）
  slack      … 「予備バッテリーは総数まで」という不等式を等式にするための補助の変数

■ なぜ「後ろから」数えるのか
  センサーの回復時刻の合計を小さくしたい。前から数えると「k 番目の移動のあと、何回飛ぶか」が
  ほかの変数で決まるので 3 次以上の式になってしまう。後ろから数えると
  「後ろから k 番目の移動は、それ以降の k 基の回復を遅らせる」と決まるので 2 次式（QUBO）で書ける。

■ エネルギー（時刻は分。上限時刻 T で割って 0〜1 くらいの大きさにそろえる）
  目的   w_end × （各車の終了時刻の合計）
       + w_lat × Σ（後ろから k 番目に入る移動の時間 × k^p） + w_lat × Σ（各飛行の、離陸から最後の回復まで）
         ・移動 = 「拠点 → 最初の離陸点」か「前の飛行の離陸 → 次の飛行の離陸」
         ・前の飛行の離陸 → 次の離陸 = 飛行時間 + max(充電 30 分, 次の離陸点までの走行)
  罰金   A × Σ_センサー（そのセンサーを回る変数の和 − 1）²      … 各センサーをちょうど 1 回
         3B × （同じ枠に 2 つ以上の飛行）                       … 1 つの枠に飛行は 1 つまで
         B × y[v,k+1] × (1 − y[v,k])                             … 枠は後ろから詰めて使う
         B × （上限 T までに終わり得ない、隣り合う 2 飛行の組）
         B × （予備なしの回数を超えたのに予備を積まない） + B × （予備の数が総数を超える）
  ここで y[v,k] = Σ_f x[v,k,f]（車 v の後ろから k 番目の枠が使われているか）。

■ 書き方
  Amplify の巡回セールスマンの例と同じく、x = gen.array("Binary", shape=(車, 枠, 飛行)) で変数を作り、
  「ちょうど 1 回」は one_hot、「1 つまで」「総数まで」は less_equal で書く（罰金の式は Amplify が作る）。
  決まった形にならないルール（枠を詰める・並びの禁止・予備を積む）だけ、罰金の式を自分で書く。
  できたモデルは、手元のソルバーでも使えるように係数の表（QuboProblem）に直す。

■ 「いちばん遅い車」（最大値）について
  最大値は 2 次式で書けないので、上限時刻 T を決めて「T までに終わり得ない変数・組」を外した QUBO を作り、
  T を二分探索で小さくしながら何回か解く（optimize.solve_routing）。
"""
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from amplify import AcceptableDegrees, ConstraintList, Model, Poly, VariableGenerator, less_equal, one_hot

from .routing import INF


@dataclass
class Weights:
    """エネルギーの重み（アプリのサイドバーで変えられる）."""
    w_end: float = 1.0        # 各車の終了時刻の合計の重み
    w_lat: float = 1.0        # 各センサーの回復時刻の合計の重み（車ごとの仕事を均す効果がある）
    penalty: float = 1.0      # 罰金の強さ（「目的が 1 飛行あたり最大どれだけ増えるか」の何倍にするか）
    spare_cost: float = 0.01  # 予備バッテリーを必要もないのに積まないための、小さな費用
    pos_power: float = 2.0    # 後ろから k 番目の移動にかける重み k^p の p（大きいほど 1 台に仕事が集まるのを嫌う）
    w_bal: float = 0.0        # 車ごとの仕事量（飛行 + 充電）の 2 乗和の重み（初期値は使わない）


@dataclass
class QuboProblem:
    """できあがった QUBO. エネルギー = const + Σ linear[i]·x_i + Σ quadratic[(i, j)]·x_i·x_j."""
    n: int                    # 変数の数
    linear: np.ndarray        # 1 次の係数（長さ n）
    quadratic: dict           # 2 次の係数 {(i, j): 係数}（i < j）
    const: float              # 定数
    x_keys: list              # x の番号 → (車 v, 後ろから k 番目, 飛行 f)
    spare_var: dict           # 車 v → b[v] の番号
    slack_vars: list          # 補助の変数の番号
    sensor_groups: list       # センサーごとに、そのセンサーを回る x の番号の一覧（焼きなましの「付け替え」に使う）
    T: float                  # 上限時刻
    K: int                    # 1 台あたりの枠の数
    meta: dict = field(default_factory=dict)
    var_index: dict = field(default_factory=dict)   # Amplify の変数の id → この表での番号
    model: object = None      # もとの Amplify のモデル（目的 + 制約）

    def matrix(self):
        """対称行列 W（W[i, j] = W[j, i] = 2 次の係数）. 手元のソルバーで使う."""
        W = np.zeros((self.n, self.n))
        for (i, j), c in self.quadratic.items():
            W[i, j] += c
            W[j, i] += c
        return W

    def upper(self):
        """上三角行列（W[i, j], i < j だけに係数）. Amplify に渡す形."""
        U = np.zeros((self.n, self.n))
        for (i, j), c in self.quadratic.items():
            U[i, j] += c
        return U

    def energy(self, x):
        """0/1 の列 x のエネルギー."""
        x = np.asarray(x, float)
        e = self.const + self.linear @ x
        for (i, j), c in self.quadratic.items():
            e += c * x[i] * x[j]
        return float(e)


def build_qubo(inst, T, K=None, weights=None):
    """上限時刻 T（分）の QUBO を作る.

    inst: routing.Instance（候補の飛行・走行時間・車）
    K:    1 台あたりの飛行の枠の数（None なら「予備込みで飛べる回数」と「対象センサー数」の小さい方）

    書き方は Amplify の巡回セールスマンの例と同じ形にしている:
      変数   x = gen.array("Binary", shape=(車, 枠, 飛行))       … 例の q = gen.array("Binary", shape=(N + 1, N))
      目的   Σ 移動時間 × x[v, k+1, g] × x[v, k, f]              … 例の Σ d[i, j] × q[t, i] × q[t + 1, j]
      制約   one_hot（各センサーちょうど 1 回）・less_equal（各枠 1 つまで）
    """
    wt = weights or Weights()
    t_start = time.time()
    fleet = inst.cfg.fleet
    flights = inst.flights
    n_flights = len(flights)
    n_vehicles = len(inst.vehicles)
    n_builtin = fleet.flights_builtin                      # 予備なしで飛べる回数
    max_per_vehicle = n_builtin + (fleet.flights_per_spare if fleet.spare_total > 0 else 0)
    K = min(K or max_per_vehicle, max_per_vehicle, max(len(inst.targets), 1))
    scale = 1.0 / T                                        # 時刻を T で割って大きさをそろえる

    # ---- 1. 時刻の表を作っておく（何度も使うので） ----
    # drive_from_base[v, f]: 車 v の拠点 → 飛行 f の離陸点 の走行時間
    drive_from_base = np.array([[inst.drive_from_base(v, f) for f in range(n_flights)] for v in range(n_vehicles)])
    # gap[g, f]: 飛行 g の離陸 → 次の飛行 f の離陸 の最短時間（飛行 + max(充電, 走行)）
    gap = np.array([[inst.gap(g, f) for f in range(n_flights)] for g in range(n_flights)])
    # time_after_last[v, f]: 最後の飛行 f の離陸 → 終了時刻（定義により「最後の回復まで」か「拠点に帰るまで」）
    time_after_last = np.array([[inst.time_after_last(v, f) for f in range(n_flights)] for v in range(n_vehicles)])
    # to_last_recovery[f]: 飛行 f の離陸 → そのうち最後のセンサーの回復まで
    to_last_recovery = np.array([f.last_recover_off for f in flights])

    # ---- 2. 「後ろから k 番目に f をしたら、終了までに少なくとも何分かかるか」 ----
    # min_to_end[v][k-1, f] は、センサーが重なってもよいことにした（ゆるめた）ときの最短時間なので、本当の時間以下になる。
    #   k = 1: 最後の飛行なので、そのあとの時間だけ
    #   k ≥ 2: 次にどの飛行 h をしても、gap[f, h] + （h を後ろから k−1 番目にしたときの最短）以上かかる
    min_to_end = []
    for v in range(n_vehicles):
        table = np.full((K, n_flights), INF)
        table[0] = time_after_last[v]
        for k in range(1, K):
            via = gap + table[k - 1][None, :]               # via[f, h] = f の次に h をする場合
            np.fill_diagonal(via, INF)                      # 同じ飛行を続けてはしない
            table[k] = via.min(axis=1)
        min_to_end.append(table)

    # ---- 3. 変数 x[v, k, f] を作る ----
    # 枠の番号 k は 1〜K をそのまま添字に使うため、配列は 0〜K+1 の K+2 個にする。
    # 0 番と K+1 番は「枠がない」ので 0 に固定する（TSP の例で q[N, :] = q[0, :] と固定しているのと同じ考え方）。
    # こうしておくと「1 つ前の枠（k+1 番目）」を、k = K のときも特別扱いせずに書ける。
    gen = VariableGenerator()
    x = gen.array("Binary", shape=(n_vehicles, K + 2, n_flights))
    possible = np.zeros((n_vehicles, K + 2, n_flights), dtype=bool)   # possible[v, k, f]: 本当に使う変数か
    for v in range(n_vehicles):
        for k in range(1, K + 1):
            for f in range(n_flights):
                reachable = np.isfinite(drive_from_base[v, f])                         # 拠点から離陸点へ行ける
                in_time = drive_from_base[v, f] + min_to_end[v][k - 1, f] <= T + 1e-9   # T に間に合いうる
                possible[v, k, f] = reachable and in_time
    for v in range(n_vehicles):
        for k in range(K + 2):
            for f in range(n_flights):
                if not possible[v, k, f]:
                    x[v, k, f] = 0                          # 使えない変数は 0 に固定（QUBO から消える）

    # used[v][k] = 車 v の後ろから k 番目の枠が使われているか（= その枠の x の和。0 か 1 になってほしい）
    used = [[x[v, k, :].sum() for k in range(K + 2)] for v in range(n_vehicles)]

    # ---- 4. 罰金の大きさ ----
    # 目的は 1 飛行あたり最大でおよそ (w_end + w_lat × k^p) × (移動時間 / T) 増える。移動時間 / T ≦ 1 なので
    # 罰金をこの大きさ以上にしておけば、制約を破って目的を下げても得にならない。
    # k には「実際に変数を作れたいちばん後ろからの番号」を使う。T が小さいと大きな k の変数は作られないので、
    # K（枠の数の上限）を使うより罰金が小さくて済む。罰金が大きすぎると、エネルギーの山が高くなって解きにくくなる。
    k_used = max((k for v, k, f in zip(*np.nonzero(possible))), default=1)
    one_flight_max = wt.w_end + wt.w_lat * k_used ** wt.pos_power
    A = wt.penalty * one_flight_max        # 各センサーちょうど 1 回の罰金
    B = wt.penalty * one_flight_max        # そのほかのルールの罰金

    # ---- 5. 目的（時間）と、並びの禁止 ----
    objective = Poly()
    forbidden = Poly()                     # 並べてはいけない 2 飛行の組（あとで B をかける）
    n_forbidden = 0
    for v in range(n_vehicles):
        for k in range(1, K + 1):
            # 後ろから k 番目に入る移動は「k 基ぶんの回復」を遅らせる（p で強める）。終了時刻にも 1 回入る。
            leg_weight = scale * (wt.w_end + wt.w_lat * k ** wt.pos_power)
            for f in range(n_flights):
                if not possible[v, k, f]:
                    continue
                # (a) この飛行の、離陸から最後の回復まで（回復時刻の合計の一部）
                objective += wt.w_lat * scale * to_last_recovery[f] * x[v, k, f]
                # (b) この飛行が車の最初の飛行（= 1 つ前の枠が空）なら「拠点 → 離陸点」を走る
                objective += drive_from_base[v, f] * leg_weight * x[v, k, f] * (1 - used[v][k + 1])
                # (c) 最後の飛行（k = 1）なら、そのあと終了までの時間
                if k == 1:
                    objective += wt.w_end * scale * time_after_last[v, f] * x[v, 1, f]
                # (d) 1 つ前の飛行 g（k+1 番目）→ この飛行 f（k 番目）の移動。TSP の d[i, j] × q[t, i] × q[t+1, j] と同じ形
                for g in range(n_flights):
                    if not possible[v, k + 1, g]:
                        continue
                    move = gap[g, f]
                    cannot = (
                        f == g                                          # 同じ飛行を 2 回
                        or not np.isfinite(move)                        # 走って行けない
                        or drive_from_base[v, g] + move + min_to_end[v][k - 1, f] > T + 1e-9  # T に間に合わない
                        or bool(set(flights[f].sensors) & set(flights[g].sensors))        # 同じセンサーを 2 回
                    )
                    if cannot:
                        forbidden += x[v, k + 1, g] * x[v, k, f]
                        n_forbidden += 1
                    else:
                        objective += move * leg_weight * x[v, k + 1, g] * x[v, k, f]

    # (e) （初期値では使わない）車ごとの仕事量の 2 乗和: w_bal × Σ_v (Σ (飛行時間 + 充電) / T × x)²
    if wt.w_bal:
        for v in range(n_vehicles):
            work = Poly()
            for k in range(1, K + 1):
                for f in range(n_flights):
                    if possible[v, k, f]:
                        work += (flights[f].flight_min + fleet.charge_min) * scale * x[v, k, f]
            objective += wt.w_bal * work ** 2

    # ---- 6. 制約（ルール） ----
    constraints = ConstraintList()
    extra_penalty = Poly()                 # Amplify の決まった形にならないルール（自分で式を書いて B をかける）
    unreachable_penalty = 0.0              # T までに回れないセンサーの数 × A

    # (f) 各センサーをちょうど 1 回。TSP の「各都市ちょうど 1 回」（列の合計 = 1）と同じ
    sensor_groups_vars = []
    for s in inst.targets:
        members = [x[v, k, f] for v, k, f in zip(*np.nonzero(possible)) if s in flights[f].sensors]
        sensor_groups_vars.append(members)
        if members:
            constraints += one_hot(sum(members, Poly())) * A
        else:
            unreachable_penalty += A       # T までにどうしても回れないセンサー（必ず罰金 A がかかる）

    # (g) 1 つの枠に飛行は 1 つまで。TSP の「各順番に都市は 1 つ」（行の合計 = 1）に近いが、飛ばない枠もあるので「1 以下」。
    #     (b) には負の係数があり、1 つの枠に 2 つ入れるとエネルギーが下がってしまうことがあるので 3 倍にする。
    for v in range(n_vehicles):
        for k in range(1, K + 1):
            if possible[v, k].sum() >= 2:
                constraints += less_equal(used[v][k], 1) * (3 * B)

    # (h) 枠は後ろから詰める: 1 つ前の枠（k+1 番目）が使われているのに、この枠（k 番目）が空なら罰金
    #     「後ろから k 番目の移動は k 基を遅らせる」という数え方が成り立つために必要
    for v in range(n_vehicles):
        for k in range(1, K + 1):
            extra_penalty += used[v][k + 1] * (1 - used[v][k])

    # (i) 予備バッテリー
    spare = {}                             # 車 v → b[v]（予備を積むか）
    if K > n_builtin:
        # 予備なしの回数を超える枠（後ろから n_builtin+1 番目）を使える車にだけ b[v] を作る
        b = gen.array("Binary", n_vehicles)
        for v in range(n_vehicles):
            if possible[v, n_builtin + 1].any():
                spare[v] = b[v]
        for v, b_v in spare.items():
            objective += wt.spare_cost * b_v                                  # 必要ないのに積まないための小さな費用
            extra_penalty += used[v][n_builtin + 1] * (1 - b_v)               # その枠を使うのに予備がないと罰金
        if len(spare) > fleet.spare_total:
            constraints += less_equal(sum(spare.values(), Poly()), fleet.spare_total) * B   # 予備は総数まで

    model = Model(objective + B * (forbidden + extra_penalty) + unreachable_penalty, constraints)

    # ---- 7. 手元のソルバーでも使えるよう、係数の表（QuboProblem）にする ----
    x_keys = [(int(v), int(k), int(f)) for v, k, f in zip(*np.nonzero(possible))]
    x_vars = [x[key].as_variable() for key in x_keys]
    prob = model_to_qubo_problem(model, x_vars, list(spare.values()))
    index_of = {key: i for i, key in enumerate(x_keys)}
    prob.x_keys = x_keys
    prob.spare_var = {v: prob.var_index[b_v.as_variable().id] for v, b_v in spare.items()}
    prob.sensor_groups = [[index_of[(int(v), int(k), int(f))] for v, k, f in zip(*np.nonzero(possible))
                           if s in flights[f].sensors] for s in inst.targets]
    prob.T, prob.K = T, K
    prob.meta = dict(A=A, B=B, k_used=int(k_used), n_x=len(x_keys), n_pairs=len(prob.quadratic),
                     n_forbidden_pairs=n_forbidden, build_seconds=round(time.time() - t_start, 3))
    prob.model = model
    return prob


def model_to_qubo_problem(model, x_vars, spare_vars):
    """Amplify のモデル（目的 + 制約）を、罰金込みの 1 つの 2 次式にして QuboProblem にする.

    Amplify は制約を罰金の式に直してくれる（less_equal の補助の変数もここで作られる）。
    変数の番号は「x の変数 → 予備バッテリーの変数 → 補助の変数」の順にそろえる（decode が x を先頭から読むため）。
    """
    intermediate, _ = model.to_intermediate_model(AcceptableDegrees(objective={"Binary": "Quadratic"}))
    energy = intermediate.objective
    for c in intermediate.constraints:
        energy += c.penalty                # c.penalty には重みがかけてある

    first_ids = [var.id for var in x_vars] + [var.id for var in spare_vars]
    other_ids = [var.id for var in energy.variables if var.id not in set(first_ids)]
    var_index = {var_id: i for i, var_id in enumerate(first_ids + other_ids)}
    n = len(var_index)

    linear = np.zeros(n)
    quadratic = {}
    const = 0.0
    for ids, coef in energy.as_dict().items():
        if len(ids) == 0:
            const += coef
        elif len(ids) == 1:
            linear[var_index[ids[0]]] += coef
        else:
            i, j = sorted((var_index[ids[0]], var_index[ids[1]]))
            if i == j:                     # x × x = x（0/1 なので）
                linear[i] += coef
            else:
                quadratic[(i, j)] = quadratic.get((i, j), 0.0) + coef
    quadratic = {key: c for key, c in quadratic.items() if c != 0.0}
    slack_vars = [var_index[var_id] for var_id in other_ids]
    return QuboProblem(n=n, linear=linear, quadratic=quadratic, const=const, x_keys=[], spare_var={},
                       slack_vars=slack_vars, sensor_groups=[], T=0.0, K=0, var_index=var_index)


def decode(inst, prob, x):
    """ソルバーの 0/1 の答えから、各車のルート（前から順の飛行の列）を作る.

    制約を破っていたら、その数を violations に数える（回れなかったセンサーを足すのは optimize.repair）。
    戻り値: (routes {車: [飛行, ...]}, spares {車: 予備を積むか}, violations {種類: 数})
    """
    x = np.asarray(x).round().astype(int)
    violations = dict(slot_multi=0, gap=0, duplicate=0, missing=0, spare=0)
    # 車ごと・枠ごとに、1 になっている飛行を集める
    chosen = {}                                   # (v, k) → [f, ...]
    for (v, k, f), value in zip(prob.x_keys, x[:len(prob.x_keys)]):
        if value:
            chosen.setdefault((v, k), []).append(f)

    routes, spares = {}, {}
    for v in range(len(inst.vehicles)):
        used_slots = []                           # 前から順（k の大きい方から）の (k, f)
        for k in range(prob.K, 0, -1):
            fs_in_slot = chosen.get((v, k), [])
            if len(fs_in_slot) > 1:
                violations["slot_multi"] += 1     # 1 つの枠に 2 つ以上（最初の 1 つだけ使う）
            if fs_in_slot:
                used_slots.append((k, fs_in_slot[0]))
        ks = [k for k, _ in used_slots]
        if ks and ks != list(range(ks[0], 0, -1)):
            violations["gap"] += 1                # 枠が後ろから詰まっていない（空きはつめて読む）
        routes[v] = [f for _, f in used_slots]
        if v in prob.spare_var:
            spares[v] = bool(x[prob.spare_var[v]])
        if len(routes[v]) > inst.cfg.fleet.flights_builtin and not spares.get(v, False):
            violations["spare"] += 1

    # 同じセンサーを 2 回以上回っていたら、2 回目以降の飛行を外す
    seen = set()
    for v in range(len(inst.vehicles)):
        kept = []
        for f in routes[v]:
            its_sensors = set(inst.flights[f].sensors)
            if its_sensors & seen:
                violations["duplicate"] += 1
                continue
            seen |= its_sensors
            kept.append(f)
        routes[v] = kept
    violations["missing"] = sum(1 for s in inst.targets if s not in seen)
    return routes, spares, violations


def encode(inst, prob, routes, spares=None):
    """ルートを QUBO の 0/1 の列にする（テストで「エネルギー = 目的の式」を確かめる用）.

    QUBO に無い変数（T に間に合わないとして外した変数）を使うルートなら None を返す。
    """
    x = np.zeros(prob.n, int)
    index_of = {key: i for i, key in enumerate(prob.x_keys)}
    for v, route in routes.items():
        for pos, f in enumerate(route):
            key = (v, len(route) - pos, f)        # 前から pos 番目 = 後ろから len − pos 番目
            if key not in index_of:
                return None
            x[index_of[key]] = 1
    for v, b_index in prob.spare_var.items():
        x[b_index] = int(bool((spares or {}).get(v, False)))
    if prob.slack_vars:
        # 補助の変数（Amplify が less_equal のために作ったもの）は数個しかないので、
        # 0/1 の組合せを全部試して、エネルギーがいちばん低くなるものにする
        best_x, best_e = None, None
        for pattern in itertools.product([0, 1], repeat=len(prob.slack_vars)):
            x[prob.slack_vars] = pattern
            e = prob.energy(x)
            if best_e is None or e < best_e:
                best_x, best_e = x.copy(), e
        x = best_x
    return x
