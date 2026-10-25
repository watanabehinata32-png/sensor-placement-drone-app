"""設定の読み込み（Amplify のトークン）.

トークンは sensor_app/.env に次の 1 行で書く（.env は人に渡したり GitHub に上げたりしない）:
    AMPLIFY_TOKEN=ここにトークン
環境変数 AMPLIFY_TOKEN があれば、そちらを優先する（公開するサーバーではこちらを使う）。
トークンの値は画面にもログにも出さない。
"""
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sensor_app フォルダ
ENV_FILE = os.path.join(APP_DIR, '.env')


def read_env_file(path):
    """.env を「名前=値」の辞書にする（# で始まる行と空行は読み飛ばす）."""
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            name, value = line.split('=', 1)
            values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def amplify_token():
    """Amplify AE のトークン（なければ空文字）. 毎回読むので、.env を書き換えたら起動し直さなくても効く."""
    return os.environ.get('AMPLIFY_TOKEN') or read_env_file(ENV_FILE).get('AMPLIFY_TOKEN', '')
