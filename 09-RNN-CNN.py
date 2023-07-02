## 第9章：RNN,CNN
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import string
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from gensim.models import KeyedVectors
from torch.nn import functional as F
import optuna
from transformers import BertTokenizer, BertModel

# 学習データの準備(問題51)
# データの読込
df = pd.read_csv(
    "text/newsCorpora.csv",
    header=None,
    sep="\t",
    names=[
        "ID",
        "TITLE",
        "URL",
        "PUBLISHER",
        "CATEGORY",
        "STORY",
        "HOSTNAME",
        "TIMESTAMP",
    ],
)

# データの抽出
df = df.loc[
    df["PUBLISHER"].isin(
        ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
    ),
    ["TITLE", "CATEGORY"],
]

# データの分割
train, valid_test = train_test_split(
    df, test_size=0.2, shuffle=True, random_state=123, stratify=df["CATEGORY"]
)
valid, test = train_test_split(
    valid_test,
    test_size=0.5,
    shuffle=True,
    random_state=123,
    stratify=valid_test["CATEGORY"],
)

## 80.ID番号への変換
# 単語の頻度集計
d = defaultdict(int)
table = str.maketrans(
    string.punctuation, " " * len(string.punctuation)
)  # 記号をスペースに置換するテーブル
for text in train["TITLE"]:
    for word in text.translate(table).split():
        d[word] += 1
d = sorted(d.items(), key=lambda x: x[1], reverse=True)

# 単語ID辞書の作成
word2id = {word: i + 1 for i, (word, cnt) in enumerate(d) if cnt > 1}  # 出現頻度が2回以上の単語を登録

# print(f"ID数: {len(set(word2id.values()))}\n")
# print("頻度上位20語")
# for key in list(word2id)[:20]:
#     print(f"{key}: {word2id[key]}")


def tokenizer(text, word2id=word2id, unk=0):
    """入力テキストをスペースで分割しID列に変換(辞書になければunkで指定した数字を設定)"""
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return [word2id.get(word, unk) for word in text.translate(table).split()]


def func80():
    # 確認
    text = train.iloc[1, train.columns.get_loc("TITLE")]
    print(f"テキスト: {text}")
    print(f"ID列: {tokenizer(text)}")


## 81.RNNによる予測
# ニューラルネットを用いた自然言語処理、特にテキスト分類における処理の流れ
# 1.文をトークン（単語など）の列に分割　→ スペースで分割
# 2.それぞれのトークンをベクトルに変換　→ 埋め込み層で変換
# 3.トークンベクトルを文ベクトルとして１つに集約　→ RNNやCNNで集約
# 4.文ベクトルを入力としてラベルを分類　→ 全結合層で分類
class RNN(nn.Module):
    # RNNモデルの構造定義
    def __init__(self, vocab_size, emb_size, padding_idx, output_size, hidden_size):
        # 親クラスのコンストラクタ
        super().__init__()
        # RNNの隠れ層のサイズを指定
        # RNNでは時系列データの処理において、隠れ状態（内部の状態）を保持する
        self.hidden_size = hidden_size
        # 入力データの単語埋め込みを行うためのEmbedding層を作成
        # vocab_size：語彙のサイズ
        # emb_size：埋め込みベクトルのサイズ
        # padding_idx：パディング用のインデックス
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        # RNN層を作成
        # emb_size：入力の特徴ベクトルのサイズ
        # hidden_size：隠れ状態のサイズ
        # nonlinearity="tanh"：活性化関数としてtanh関数を使用
        # batch_size="True"：入力tensorの形状を（バッチサイズ、シーケンス長、特徴量の次元数）とする
        self.rnn = nn.RNN(emb_size, hidden_size, nonlinearity="tanh", batch_first=True)
        # RNN層の出力を指定されたクラスの数にマッピングするための全結合層（Linear層）を作成
        # hidden_size：入力の特徴ベクトルの次元数
        # output_size：出力のクラス数
        self.fc = nn.Linear(hidden_size, output_size)

    # RNNモデルの順伝播処理を定義
    # 入力データを順方向に処理し、RNN層を通じて予測結果を得るまでの処理
    def forward(self, x):
        # 入力データ'x'のバッチサイズ取得
        self.batch_size = x.size()[0]
        # 初期の隠れ状態'hidden'を作成
        # x.device：'x'のデバイス（CPUまたはGPU）を指定
        hidden = self.init_hidden(x.device)  # h0のゼロベクトルを作成
        # 入力データ'x'を単語埋め込み層'self.emb'に通して単語埋め込みベクトル'emb'を取得
        # emb.size() = (batch_size, seq_len, emb_size)
        emb = self.emb(x)
        # 単語埋め込みベクトル'emb'と初期の隠れ状態'hidden'をRNN層'self.rnn'に入力し、
        # 出力'out'と最終的な隠れ状態'hidden'を取得
        # out.size() = (batch_size, seq_len, hidden_size)
        out, hidden = self.rnn(emb, hidden)
        # 'out'の最後の時刻の隠れ状態を抽出し、全結合層'self.fc'に入力して予測結果'out'を取得
        # out.size() = (batch_size, output_size)
        out = self.fc(out[:, -1, :])
        return out

    # 隠れ状態の初期化を行う補助関数
    # RNNの初期の隠れ状態をゼロベクトルで初期化する役割
    def init_hidden(self, device):
        # 指定されたデバイス上に指定したサイズのゼロベクトル'hidden' を作成
        # self.batch_size：RNNモデルの純伝播処理中に取得したバッチサイズ
        hidden = torch.zeros(1, self.batch_size, self.hidden_size, device=device)
        return hidden


# PyTorchのDatasetクラスを継承して作られたカスタムデータセットのクラス
# Datasetクラス：データセットを表現するためのクラス
class CreateDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        # X：入力データ
        self.X = X
        # y:ラベルデータ
        self.y = y
        # tokenizer：トークン化を行うための関数
        self.tokenizer = tokenizer

    # len(Dataset)で返す値を指定
    def __len__(self):
        return len(self.y)

    # Dataset[index]で返す値を指定
    def __getitem__(self, index):
        # 指定したインデックスに対応する入力データをトークン化し、'input'として保持
        # 対応するラベルデータも取得して'labels'として保持
        text = self.X.iloc[index]
        inputs = self.tokenizer(text)

        return {
            "inputs": torch.tensor(inputs, dtype=torch.int64),
            "labels": torch.tensor(self.y[index], dtype=torch.int64),
        }


# ラベルベクトルの作成
category_dict = {"b": 0, "t": 1, "e": 2, "m": 3}
# 訓練データのカテゴリラベルを数値に変換
# train["CATEGORY"]：訓練データのカテゴリ列
# map(lambda x: category_dict[x])：カテゴリラベルを数値に変換
# values：Numpy配列として'y_train'に格納
y_train = train["CATEGORY"].map(lambda x: category_dict[x]).values
y_valid = valid["CATEGORY"].map(lambda x: category_dict[x]).values
y_test = test["CATEGORY"].map(lambda x: category_dict[x]).values

# Datasetの作成
# train["TITLE"]：訓練データのタイトル列
# y_train：訓練データのカテゴリラベルを数値化したもの
# tokenizer：テキストをトークン化する関数
dataset_train = CreateDataset(train["TITLE"], y_train, tokenizer)
dataset_valid = CreateDataset(valid["TITLE"], y_valid, tokenizer)
dataset_test = CreateDataset(test["TITLE"], y_test, tokenizer)


def func81():
    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1  # 辞書のID数 + パディングID
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50

    # モデルの定義
    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    # 先頭10件の予測値取得
    for i in range(10):
        X = dataset_train[i]["inputs"]
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))
    print(f"len(Dataset)の出力: {len(dataset_train)}")
    print("Dataset[index]の出力:")
    for var in dataset_train[1]:
        print(f"  {var}: {dataset_train[1][var]}")
    # print(train["TITLE"])
    # print(y_train)
    # print(dataset_train[1])


## 82.確率的勾配降下法による学習
# 与えられたモデルとデータセットに対して損失と正解率を計算
# model：評価対象のモデル
# dataset：評価対象のデータセット
# device：計算に使用するデバイス。デフォルトではNone
# criterion：損失関数を指定するための引数。デフォルトではNone
def calculate_loss_and_accuracy(model, dataset, device=None, criterion=None):
    # １つのデータを１バッチとして読み込むための設定を行う
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0.0
    total = 0
    correct = 0
    # 計算グラフの構築と勾配の追跡を無効化
    with torch.no_grad():
        # dataloaderからバッチごとにデータを取得
        for data in dataloader:
            # デバイスを指定し、データ転送
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)

            # 順伝播
            # データをモデルに入力して出力を取得
            outputs = model(inputs)

            # 損失計算
            # criterionが指定されている場合はlossに加算
            if criterion != None:
                loss += criterion(outputs, labels).item()

            # 正解率計算
            # 出力の最大値のインデックスを取得し、予測結果として使用
            pred = torch.argmax(outputs, dim=-1)
            # バッチ内のデータ数をtotalに加算
            total += len(inputs)
            # 予測結果とラベルが一致する場合、correctをインクリメント
            correct += (pred == labels).sum().item()

    # データセットの長さでlossと正解率を割って平均化
    return loss / len(dataset), correct / total


# 与えられた訓練データセットと検証データセットを使用してモデルの学習を実行し、
# 損失と正解率のログを返すための関数
def train_model(
    dataset_train,
    dataset_valid,
    batch_size,
    model,
    criterion,
    optimizer,
    num_epochs,
    collate_fn=None,
    device=None,
):
    """モデルの学習を実行し、損失・正解率のログを返す"""
    # デバイスの指定
    model.to(device)

    # dataloaderの作成
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)

    # スケジューラの設定
    # 最初のエポックから最後のエポックまでの間で学習率を徐々に減少させる
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, eta_min=1e-5, last_epoch=-1
    )

    # 学習
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):  # 指定されたエポック数のループ
        # 開始時刻の記録
        s_time = time.time()

        # 訓練モードに設定
        model.train()
        for data in dataloader_train:  # バッチごとにデータ取得
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            # データを指定されたデバイスに転送
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            # モデルにデータを入力して出力を指定
            outputs = model(inputs)
            # 損失を計算し、誤差逆伝播と重みの更新
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 評価モードに設定
        model.eval()

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy(
            model, dataset_train, device, criterion=criterion
        )
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, dataset_valid, device, criterion=criterion
        )
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # チェックポイントの保存
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"pt/checkpoint{epoch + 1}.pt",
        )

        # 終了時刻の記録
        e_time = time.time()

        # ログを出力
        print(
            f"epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec"
        )

        # 検証データの損失が3エポック連続で低下しなかった場合は学習終了
        if (
            epoch > 2
            and log_valid[epoch - 3][0]
            <= log_valid[epoch - 2][0]
            <= log_valid[epoch - 1][0]
            <= log_valid[epoch][0]
        ):
            break

        # スケジューラを1ステップ進める
        scheduler.step()

    return {"train": log_train, "valid": log_valid}


def visualize_logs(log):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log["train"]).T[0], label="train")
    ax[0].plot(np.array(log["valid"]).T[0], label="valid")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[1].plot(np.array(log["train"]).T[1], label="train")
    ax[1].plot(np.array(log["valid"]).T[1], label="valid")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    plt.show()


def func82():
    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1
    NUM_EPOCHS = 10

    # モデルの定義
    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # モデルの学習
    log = train_model(
        dataset_train,
        dataset_valid,
        BATCH_SIZE,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
    )
    # ログの可視化
    visualize_logs(log)

    # 正解率の算出
    # 損失は使用しないため'_'
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test)
    print(f"正解率（学習データ）：{acc_train:.3f}")
    print(f"正解率（評価データ）：{acc_test:.3f}")


## 83.ミニバッチ化・GPU上での学習
class Padsequence:
    """Dataloaderからミニバッチを取り出すごとに最大系列長でパディング"""

    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x["inputs"].shape[0], reverse=True)
        sequences = [x["inputs"] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.padding_idx
        )
        labels = torch.LongTensor([x["labels"] for x in sorted_batch])

        return {"inputs": sequences_padded, "labels": labels}


def func83():
    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # モデルの定義
    model = RNN(VOCAB_SIZE, EMB_SIZE, PADDING_IDX, OUTPUT_SIZE, HIDDEN_SIZE)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # デバイスの指定
    device = torch.device("mps")

    # モデルの学習
    log = train_model(
        dataset_train,
        dataset_valid,
        BATCH_SIZE,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
        collate_fn=Padsequence(PADDING_IDX),
        device=device,
    )

    # ログの可視化
    visualize_logs(log)

    # 正解率の算出
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train, device)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test, device)
    print(f"正解率（学習データ）：{acc_train:.3f}")
    print(f"正解率（評価データ）：{acc_test:.3f}")


# 84.単語ベクトルの導入
# 学習済みモデルのロード
model = KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin.gz", binary=True
)

# 学習済み単語ベクトルの取得
# 語彙のサイズ
VOCAB_SIZE = len(set(word2id.values())) + 1
# 単語ベクトルの次元数
EMB_SIZE = 300
# 重み行列の初期化
weights = np.zeros((VOCAB_SIZE, EMB_SIZE))
words_in_pretrained = 0
# word2idに含まれる各単語について、学習済み単語ベクトルモデルから単語ベクトルを取得
for i, word in enumerate(word2id.keys()):
    # 単語が学習済みモデルに存在する場合、その単語のベクトルを重み行列の対応する行に設定
    try:
        weights[i] = model[word]
        words_in_pretrained += 1
    # 単語が学習済みモデルに存在しない場合、ガウス分布に基づくランダムな値でベクトルを初期化
    except KeyError:
        weights[i] = np.random.normal(scale=0.4, size=(EMB_SIZE,))
# torch.from_numpyを使用してNumPy配列からPyTorchのtensorに変換
weights = torch.from_numpy(weights.astype((np.float32)))

print(f"学習済みベクトル利用単語数: {words_in_pretrained} / {VOCAB_SIZE}")
print(weights.size())


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        padding_idx,
        output_size,
        hidden_size,
        num_layers,
        emb_weights=None,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1  # 単方向：1、双方向：2
        if emb_weights != None:  # 指定があれば埋め込み層の重みをemb_weightsで初期化
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx
            )
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(
            emb_size,
            hidden_size,
            num_layers,
            nonlinearity="tanh",
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        self.batch_size = x.size()[0]
        hidden = self.init_hidden(x.device)  # h0のゼロベクトルを作成
        emb = self.emb(x)
        # emb.size() = (batch_size, seq_len, emb_size)
        out, hidden = self.rnn(emb, hidden)
        # out.size() = (batch_size, seq_len, hidden_size * num_directions)
        out = self.fc(out[:, -1, :])
        # out.size() = (batch_size, output_size)
        return out

    def init_hidden(self, device):
        hidden = torch.zeros(
            self.num_layers * self.num_directions,
            self.batch_size,
            self.hidden_size,
            device=device,
        )
        return hidden


def func84():
    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    NUM_LAYERS = 1
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # モデルの定義
    model = RNN(
        VOCAB_SIZE,
        EMB_SIZE,
        PADDING_IDX,
        OUTPUT_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        emb_weights=weights,
    )

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # デバイスの指定
    device = torch.device("mps")

    # モデルの学習
    log = train_model(
        dataset_train,
        dataset_valid,
        BATCH_SIZE,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
        collate_fn=Padsequence(PADDING_IDX),
        device=device,
    )

    # ログの可視化
    visualize_logs(log)

    # 正解率の算出
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train, device)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test, device)
    print(f"正解率（学習データ）：{acc_train:.3f}")
    print(f"正解率（評価データ）：{acc_test:.3f}")


## 85.双方向RNN・多層化
def func85():
    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    HIDDEN_SIZE = 50
    NUM_LAYERS = 2  # NUM_LAYERSを2に設定
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # モデルの定義
    model = RNN(
        VOCAB_SIZE,
        EMB_SIZE,
        PADDING_IDX,
        OUTPUT_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        emb_weights=weights,
        bidirectional=True,  # 双方向を指定する引数をTrueに
    )

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # デバイスの指定
    device = torch.device("cuda")

    # モデルの学習
    log = train_model(
        dataset_train,
        dataset_valid,
        BATCH_SIZE,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
        collate_fn=Padsequence(PADDING_IDX),
        device=device,
    )

    # ログの可視化
    visualize_logs(log)

    # 正解率の算出
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train, device)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test, device)
    print(f"正解率（学習データ）：{acc_train:.3f}")
    print(f"正解率（評価データ）：{acc_test:.3f}")


## 86.畳み込みニューラルネットワーク（CNN）
class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        padding_idx,
        output_size,
        out_channels,
        kernel_heights,
        stride,
        padding,
        emb_weights=None,
    ):
        super().__init__()
        if emb_weights != None:  # 指定があれば埋め込み層の重みをemb_weightsで初期化
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx
            )
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.conv = nn.Conv2d(
            1, out_channels, (kernel_heights, emb_size), stride, (padding, 0)
        )
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(out_channels, output_size)

    def forward(self, x):
        # x.size() = (batch_size, seq_len)
        emb = self.emb(x).unsqueeze(1)
        # emb.size() = (batch_size, 1, seq_len, emb_size)
        conv = self.conv(emb)
        # conv.size() = (batch_size, out_channels, seq_len, 1)
        act = F.relu(conv.squeeze(3))
        # act.size() = (batch_size, out_channels, seq_len)
        max_pool = F.max_pool1d(act, act.size()[2])
        # max_pool.size() = (batch_size, out_channels, 1) -> seq_len方向に最大値を取得
        out = self.fc(self.drop(max_pool.squeeze(2)))
        # out.size() = (batch_size, output_size)
        return out


def func86():
    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1

    # モデルの定義
    model = CNN(
        VOCAB_SIZE,
        EMB_SIZE,
        PADDING_IDX,
        OUTPUT_SIZE,
        OUT_CHANNELS,
        KERNEL_HEIGHTS,
        STRIDE,
        PADDING,
        emb_weights=weights,
    )

    # 先頭10件の予測値取得
    for i in range(10):
        X = dataset_train[i]["inputs"]
        print(torch.softmax(model(X.unsqueeze(0)), dim=-1))


## 87.確率的勾配降下法によるCNNの学習
def func87():
    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = 300
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = 100
    KERNEL_HEIGHTS = 3
    STRIDE = 1
    PADDING = 1
    LEARNING_RATE = 5e-2
    BATCH_SIZE = 64
    NUM_EPOCHS = 10

    # モデルの定義
    model = CNN(
        VOCAB_SIZE,
        EMB_SIZE,
        PADDING_IDX,
        OUTPUT_SIZE,
        OUT_CHANNELS,
        KERNEL_HEIGHTS,
        STRIDE,
        PADDING,
        emb_weights=weights,
    )

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # デバイスの指定
    device = torch.device("mps")

    # モデルの学習
    log = train_model(
        dataset_train,
        dataset_valid,
        BATCH_SIZE,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
        collate_fn=Padsequence(PADDING_IDX),
        device=device,
    )

    # ログの可視化
    visualize_logs(log)

    # 正解率の算出
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train, device)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test, device)
    print(f"正解率（学習データ）：{acc_train:.3f}")
    print(f"正解率（評価データ）：{acc_test:.3f}")


## 88.パラメータチューニング
# 幅3のフィルターのみ　→　２、３、４の３種類の幅フィルターを利用
class textCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        padding_idx,
        output_size,
        out_channels,
        conv_params,
        drop_rate,
        emb_weights=None,
    ):
        super().__init__()
        if emb_weights != None:  # 指定があれば埋め込み層の重みをemb_weightsで初期化
            self.emb = nn.Embedding.from_pretrained(
                emb_weights, padding_idx=padding_idx
            )
        else:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1, out_channels, (kernel_height, emb_size), padding=(padding, 0)
                )
                for kernel_height, padding in conv_params
            ]
        )
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(len(conv_params) * out_channels, output_size)

    def forward(self, x):
        # x.size() = (batch_size, seq_len)
        emb = self.emb(x).unsqueeze(1)
        # emb.size() = (batch_size, 1, seq_len, emb_size)
        conv = [F.relu(conv(emb)).squeeze(3) for i, conv in enumerate(self.convs)]
        # conv[i].size() = (batch_size, out_channels, seq_len + padding * 2 - kernel_height + 1)
        max_pool = [F.max_pool1d(i, i.size(2)) for i in conv]
        # max_pool[i].size() = (batch_size, out_channels, 1) -> seq_len方向に最大値を取得
        max_pool_cat = torch.cat(max_pool, 1)
        # max_pool_cat.size() = (batch_size, len(conv_params) * out_channels, 1)  -> フィルター別の結果を結合
        out = self.fc(self.drop(max_pool_cat.squeeze(2)))
        # out.size() = (batch_size, output_size)
        return out


def objective(trial):
    # チューニング対象パラメータのセット
    emb_size = int(trial.suggest_discrete_uniform("emb_size", 100, 400, 100))
    out_channels = int(trial.suggest_discrete_uniform("out_channels", 50, 200, 50))
    drop_rate = trial.suggest_discrete_uniform("drop_rate", 0.0, 0.5, 0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 5e-4, 5e-2)
    momentum = trial.suggest_discrete_uniform("momentum", 0.5, 0.9, 0.1)
    batch_size = int(trial.suggest_discrete_uniform("batch_size", 16, 128, 16))

    # 固定パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    CONV_PARAMS = [[2, 0], [3, 1], [4, 2]]
    NUM_EPOCHS = 30

    # モデルの定義
    model = textCNN(
        VOCAB_SIZE,
        emb_size,
        PADDING_IDX,
        OUTPUT_SIZE,
        out_channels,
        CONV_PARAMS,
        drop_rate,
        emb_weights=weights,
    )

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # デバイスの指定
    device = torch.cuda.set_device(0)

    # モデルの学習
    log = train_model(
        dataset_train,
        dataset_valid,
        batch_size,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
        collate_fn=Padsequence(PADDING_IDX),
        device=device,
    )

    # 損失の算出
    loss_valid, _ = calculate_loss_and_accuracy(
        model, dataset_valid, device, criterion=criterion
    )

    return loss_valid


def func88():
    # 最適化
    study = optuna.create_study()
    study.optimize(objective, timeout=7200)

    # 結果の表示
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {:.3f}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # パラメータの設定
    VOCAB_SIZE = len(set(word2id.values())) + 1
    EMB_SIZE = int(trial.params["emb_size"])
    PADDING_IDX = len(set(word2id.values()))
    OUTPUT_SIZE = 4
    OUT_CHANNELS = int(trial.params["out_channels"])
    CONV_PARAMS = [[2, 0], [3, 1], [4, 2]]
    DROP_RATE = trial.params["drop_rate"]
    LEARNING_RATE = trial.params["learning_rate"]
    BATCH_SIZE = int(trial.params["batch_size"])
    NUM_EPOCHS = 30

    # モデルの定義
    model = textCNN(
        VOCAB_SIZE,
        EMB_SIZE,
        PADDING_IDX,
        OUTPUT_SIZE,
        OUT_CHANNELS,
        CONV_PARAMS,
        DROP_RATE,
        emb_weights=weights,
    )
    print(model)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # デバイスの指定
    device = torch.cuda.set_device(0)

    # モデルの学習
    log = train_model(
        dataset_train,
        dataset_valid,
        BATCH_SIZE,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
        collate_fn=Padsequence(PADDING_IDX),
        device=device,
    )

    # ログの可視化
    visualize_logs(log)

    # 正解率の算出
    _, acc_train = calculate_loss_and_accuracy(model, dataset_train, device)
    _, acc_test = calculate_loss_and_accuracy(model, dataset_test, device)
    print(f"正解率（学習データ）：{acc_train:.3f}")
    print(f"正解率（評価データ）：{acc_test:.3f}")


## 89.事前学習済み言語モデルからの転移学習


# Datasetの定義
class NewsDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        text = self.X.iloc[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            "labels": torch.Tensor(self.y[index]),
        }


# 正解ラベルのone-hot化
y_train = pd.get_dummies(train, columns=["CATEGORY"])[
    ["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]
].values
y_valid = pd.get_dummies(valid, columns=["CATEGORY"])[
    ["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]
].values
y_test = pd.get_dummies(test, columns=["CATEGORY"])[
    ["CATEGORY_b", "CATEGORY_e", "CATEGORY_t", "CATEGORY_m"]
].values

# Datasetの作成
max_len = 20
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset_train = NewsDataset(train["TITLE"], y_train, tokenizer, max_len)
dataset_valid = NewsDataset(valid["TITLE"], y_valid, tokenizer, max_len)
dataset_test = NewsDataset(test["TITLE"], y_test, tokenizer, max_len)

for var in dataset_train[0]:
    print(f"{var}: {dataset_train[0][var]}")


# BERT分類モデルの定義
class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate, otuput_size):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, otuput_size)  # BERTの出力に合わせて768次元を指定

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask, return_dict=False)
        out = self.fc(self.drop(out))
        return out


def calculate_loss_and_accuracy(model, criterion, loader, device):
    """損失・正解率を計算"""
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            # デバイスの指定
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            labels = data["labels"].to(device)

            # 順伝播
            outputs = model(ids, mask)

            # 損失計算
            loss += criterion(outputs, labels).item()

            # 正解率計算
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()  # バッチサイズの長さの予測ラベル配列
            labels = torch.argmax(labels, dim=-1).cpu().numpy()  # バッチサイズの長さの正解ラベル配列
            total += len(labels)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total


def train_model(
    dataset_train,
    dataset_valid,
    batch_size,
    model,
    criterion,
    optimizer,
    num_epochs,
    device=None,
):
    """モデルの学習を実行し、損失・正解率のログを返す"""
    # デバイスの指定
    model.to(device)

    # dataloaderの作成
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=len(dataset_valid), shuffle=False
    )

    # 学習
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # 開始時刻の記録
        s_time = time.time()

        # 訓練モードに設定
        model.train()
        for data in dataloader_train:
            # デバイスの指定
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            labels = data["labels"].to(device)

            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy(
            model, criterion, dataloader_train, device
        )
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, criterion, dataloader_valid, device
        )
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # チェックポイントの保存
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"pt/checkpoint{epoch + 1}.pt",
        )

        # 終了時刻の記録
        e_time = time.time()

        # ログを出力
        print(
            f"epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec"
        )

    return {"train": log_train, "valid": log_valid}


# 正解率の算出
def calculate_accuracy(model, dataset, device):
    # Dataloaderの作成
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            # デバイスの指定
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            labels = data["labels"].to(device)

            # 順伝播 + 予測値の取得 + 正解数のカウント
            outputs = model.forward(ids, mask)
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            correct += (pred == labels).sum().item()

    return correct / total


def func89():
    # パラメータの設定
    DROP_RATE = 0.4
    OUTPUT_SIZE = 4
    BATCH_SIZE = 32
    NUM_EPOCHS = 4
    LEARNING_RATE = 2e-5

    # モデルの定義
    model = BERTClass(DROP_RATE, OUTPUT_SIZE)

    # 損失関数の定義
    criterion = torch.nn.BCEWithLogitsLoss()

    # オプティマイザの定義
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    # デバイスの指定
    device = "mps"

    # モデルの学習
    log = train_model(
        dataset_train,
        dataset_valid,
        BATCH_SIZE,
        model,
        criterion,
        optimizer,
        NUM_EPOCHS,
        device=device,
    )

    # ログの可視化
    x_axis = [x for x in range(1, len(log["train"]) + 1)]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(x_axis, np.array(log["train"]).T[0], label="train")
    ax[0].plot(x_axis, np.array(log["valid"]).T[0], label="valid")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[1].plot(x_axis, np.array(log["train"]).T[1], label="train")
    ax[1].plot(x_axis, np.array(log["valid"]).T[1], label="valid")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    plt.show()

    print(f"正解率（学習データ）：{calculate_accuracy(model, dataset_train, device):.3f}")
    print(f"正解率（検証データ）：{calculate_accuracy(model, dataset_valid, device):.3f}")
    print(f"正解率（評価データ）：{calculate_accuracy(model, dataset_test, device):.3f}")
    # 正解率（学習データ）：0.397
    # 正解率（検証データ）：0.951
    # 正解率（評価データ）：0.952


## 実行
func89()
