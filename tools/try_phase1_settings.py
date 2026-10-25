"""一段階目の初期値を決めるための小さな試し（Amplify AE で、いくつかの重みと制限時間を 1 回ずつ解く）.

sensor_app フォルダで:  python3 tools/try_phase1_settings.py
トークンは .env から読む（表示しない）。結果は 1 行ずつ「重み・時間 → 総コスト・満たさない市町村」を出す。
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from core import phase1  # noqa: E402
from core.solvers import make_amplify_client  # noqa: E402
from server.settings import APP_DIR, amplify_token  # noqa: E402

data = phase1.load_phase1_data(os.path.join(APP_DIR, 'data', 'data_transformed.json'))
covers = phase1.make_covers(data)
token = amplify_token()
assert token, '.env に AMPLIFY_TOKEN がありません'

# (λ_cov, λ_ab, 制限時間ミリ秒) の組。いまの初期値は (2, 1, 1000)
for lam_cov, lam_ab, time_ms in [(2, 1, 1000), (2, 1, 3000), (4, 1, 3000), (6, 1, 3000), (4, 2, 3000), (8, 2, 3000)]:
    model, x = phase1.build_model(data, covers, lam_one=20, lam_ab=lam_ab, lam_cov=lam_cov)
    place = phase1.solve_placement(model, x, make_amplify_client(token, time_ms))
    rows = phase1.check_placement(data, covers, place)
    bad = [f"{r['name']}(α{r['alpha']:.0f}/{r['need_alpha']:.0f} β{r['beta']:.0f}/{r['need_beta']:.0f} "
           f"カバー{r['coverage'] * 100:.1f}/{r['need_coverage'] * 100:.0f})" for r in rows if not r['ok']]
    print(f"λ_cov={lam_cov} λ_ab={lam_ab} 時間={time_ms}ms → 総コスト {(data['cost'] * place).sum():.0f}・"
          f"満たさない: {'、'.join(bad) or 'なし'}")
