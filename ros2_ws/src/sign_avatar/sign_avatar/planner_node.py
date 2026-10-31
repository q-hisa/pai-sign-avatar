"""sign_planner: 日本語の文を、登録済みの手話の文に対応づけて再生を依頼する。

入力: パラメータ texts（'|'区切り。起動時に順番に再生）／トピック /sign/text_in（実行中に1文）
出力: /sign/command  {"ids": ["sample4", ...], "texts": [...]}
対応づけ: 既定は文字列の類似度（API不要）。use_llm:=true でLLM（Anthropic API）を使い、失敗したら類似度に戻る。
"""
import difflib
import json
import os
import re
import unicodedata

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .common import RELIABLE, T_COMMAND, T_TEXT_IN, load_library

THRESHOLD = 0.35


def normalize(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    return re.sub(r'[\s、。,.!?！？「」]', '', s)


def match_similarity(text, library):
    """(id, score) を返す。部分一致は加点。"""
    t = normalize(text)
    best, best_score = None, 0.0
    for sid, it in library.items():
        ref = normalize(it['text'])
        score = difflib.SequenceMatcher(None, t, ref).ratio()
        if t and (t in ref or ref in t):
            score = max(score, 0.6 + 0.4 * min(len(t), len(ref)) / max(len(t), len(ref)))
        if score > best_score:
            best, best_score = sid, score
    return best, best_score


def match_llm(text, library, model):
    """LLMで登録文のidを選ぶ。選べなければ None。"""
    import anthropic
    client = anthropic.Anthropic()
    options = '\n'.join(f'{sid}: {it["text"]}' for sid, it in library.items())
    prompt = ('次の入力文に最も意味が近い登録文のidを1つだけ答えてください。'
              '該当がなければ none と答えてください。説明は不要です。\n'
              f'登録文:\n{options}\n入力文: {text}\nid:')
    msg = client.messages.create(model=model, max_tokens=20,
                                 messages=[{'role': 'user', 'content': prompt}])
    ans = msg.content[0].text.strip().split()[0].strip('`"\'.')
    return ans if ans in library else None


class Planner(Node):
    def __init__(self):
        super().__init__('sign_planner')
        self.declare_parameter('library', '')
        self.declare_parameter('texts', '')
        self.declare_parameter('use_llm', False)
        self.declare_parameter('llm_model', 'claude-sonnet-5-5')
        self.declare_parameter('start_delay', 2.0)
        self.library = load_library(self.get_parameter('library').value or None)
        self.pub = self.create_publisher(String, T_COMMAND, RELIABLE)
        self.create_subscription(String, T_TEXT_IN, self.on_text, RELIABLE)
        self.get_logger().info('登録文: ' + ' / '.join(f'{k}={v["text"]}' for k, v in self.library.items()))
        self._timer = self.create_timer(float(self.get_parameter('start_delay').value), self.on_start)

    def resolve(self, text):
        sid, how = None, 'similarity'
        if self.get_parameter('use_llm').value and os.environ.get('ANTHROPIC_API_KEY'):
            try:
                sid, how = match_llm(text, self.library, self.get_parameter('llm_model').value), 'llm'
            except Exception as e:  # LLMが使えなくても止めない
                self.get_logger().warn(f'LLMでの対応づけに失敗。類似度に切り替えます: {e}')
        if sid is None:
            how = 'similarity'
            sid, score = match_similarity(text, self.library)
            if score < THRESHOLD:
                sid = None
        self.get_logger().info(f'「{text}」→ {sid} ({how})')
        return sid

    def publish_ids(self, texts):
        ids, used = [], []
        for t in texts:
            sid = self.resolve(t)
            if sid is None:
                self.get_logger().warn(f'登録された手話の文に対応づけられませんでした: 「{t}」')
                continue
            ids.append(sid)
            used.append(t)
        if ids:
            self.pub.publish(String(data=json.dumps({'ids': ids, 'texts': used}, ensure_ascii=False)))
        return ids

    def on_start(self):
        self._timer.cancel()
        # 再生ノードが接続するまで待つ（最大30秒）
        if self.pub.get_subscription_count() == 0:
            self._waited = getattr(self, '_waited', 0) + 1
            if self._waited < 15:
                self._timer = self.create_timer(2.0, self.on_start)
                return
        raw = self.get_parameter('texts').value
        texts = [t.strip() for t in raw.split('|') if t.strip()]
        if texts:
            self.publish_ids(texts)

    def on_text(self, msg):
        self.publish_ids([msg.data])


def main():
    rclpy.init()
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
