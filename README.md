# Attention Visualization Examples

本リポジトリでは、以下の二種類の「アテンション可視化」を行うこーどです。

1. **Self-Attention の重み行列を用いた可視化**  
2. **勾配（Gradient）を利用した単語重要度の可視化 (Attention by Gradient)**

---

## 1. Self-Attention の重み行列を用いた可視化

従来のアテンション可視化例では、Transformer 内部のレイヤーごと・ヘッドごとの **Self-Attention 重み** をそのまま取り出して可視化しています。  
本リポジトリでは、以下のステップで実装しています:

1. `transformers` ライブラリを使って **トークナイザとモデル** を読み込み
2. 推論時に `output_attentions=True` を設定し、各レイヤー・各ヘッドの **アテンション行列** を得る
3. BOS/EOS などの特殊トークンを除外した上で、注目したい Query トークンに対するアテンションベクトルを取り出し
4. **Min-Max 正規化** した値をスコアとして、HTML または画像で可視化

具体的には `visualize_attention_as_image_excluding_bos_eos` 関数内で行っています。

---

## 2. 勾配（Gradient）を利用したアテンション可視化 (Attention by Gradient)

論文等で提案されているアプローチの一つで、[PromptRobust: Towards Evaluating the Robustness of LargeLanguage Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528)論文内で提唱された、**ロスに対する勾配の大きさ** を可視化し、トークン（または単語）が損失にどの程度影響を与えているかを見える化する手法です。  
本リポジトリでは、以下のステップで実装しています:

1. **自己回帰型言語モデル** を例として、`labels=input_ids` を指定し損失 (loss) を計算  
2. `loss.backward()` を呼び出して **埋め込み層 (Embedding) の勾配** を取得  
3. **BOS/EOS** などの特殊トークンを除外  
4. 入力に使われたトークン ID に対応する勾配ベクトルの **L2 ノルム** を算出 (これを “勾配スコア” とみなす)  
5. **サブワード単位で分割されたトークン**（例: "Hello", "##wor", "##ld"）をまとめ、1 単語の勾配スコアに合計  
6. 最終的に **Min-Max 正規化** し、HTML 上で背景色として可視化

具体的には `compute_attention_by_gradient` 関数内で行っています。

---

## ファイル構成

- `attention_visualization.py` (など適宜ファイル名):
  - Self-Attention の重み行列を用いた可視化関連の関数
  - 勾配を使ったアテンション可視化 (Attention by Gradient) 関連の関数
  - 補助的な関数 (HTML 生成、カラー変換、正規化など)

- `README.md` (このファイル):
  - リポジトリ概要・使用方法・実行例など

---

## 実行方法

### 1. Self-Attention 可視化の例

```bash
python attention_visualization.py
```

- デフォルトでは `visualize_attention_as_image_excluding_bos_eos()` 関数が呼ばれ、  
  `attention_demo.html` などのファイルに可視化結果が生成されます。

### 2. Gradient による可視化の例

```bash
python attention_visualization.py
```

- `compute_attention_by_gradient()` 関数が呼ばれ、  
  `gradient_attention_demo.html` に勾配ベースの可視化結果が生成されます。

> **Note**: ファイル名やスクリプト名は本リポジトリの構成に合わせて読み替えてください。

---

## 依存関係

- Python 3.9 以上 (推奨)
- [PyTorch](https://pytorch.org/)  
- [Transformers](https://github.com/huggingface/transformers)  
- そのほか、HTML 出力のために標準ライブラリのみ利用

以下のように `requirements.txt` があれば参照し、インストールできます。

```bash
pip install -r requirements.txt
```
