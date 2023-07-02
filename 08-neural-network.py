## 第8章：ニューラルネット
## wget https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
## unzip NewsAggregatorDataset.zip

import pandas as pd
from sklearn.model_selection import train_test_split
import gdown
from gensim.models import KeyedVectors
import string
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import time
from torch.nn import functional as F
from torch import optim

# データの読込
df = pd.read_csv(
    "./newsCorpora.csv",
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

# 事例数の確認
# print("【学習データ】")
# print(train["CATEGORY"].value_counts())
# print("【検証データ】")
# print(valid["CATEGORY"].value_counts())
# print("【評価データ】")
# print(test["CATEGORY"].value_counts())


# 学習済み単語ベクトルのダウンロード
# url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
# output = "GoogleNews-vectors-negative300.bin.gz"
# gdown.download(url, output, quiet=True)

# ダウンロードファイルのロード
model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary=True
)


## 70.単語ベクトルの和による特徴量
def transform_w2v(text):
    # 記号をスペースに置換
    table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    # スペースで分割してリスト化
    words = text.translate(table).split()
    # 1語ずつベクトル化
    vec = [model[word] for word in words if word in model]

    # 平均ベクトルをTensor型に変換して出力
    # Pytorchによるニューラルネットのインプットとして利用するためTensor型に
    return torch.tensor(sum(vec) / len(vec))


# 特徴ベクトルの作成
X_train = torch.stack([transform_w2v(text) for text in train["TITLE"]])
X_valid = torch.stack([transform_w2v(text) for text in valid["TITLE"]])
X_test = torch.stack([transform_w2v(text) for text in test["TITLE"]])

# ラベルベクトルの作成
category_dict = {"b": 0, "t": 1, "e": 2, "m": 3}
y_train = torch.tensor(train["CATEGORY"].map(lambda x: category_dict[x]).values)
y_valid = torch.tensor(valid["CATEGORY"].map(lambda x: category_dict[x]).values)
y_test = torch.tensor(test["CATEGORY"].map(lambda x: category_dict[x]).values)


def func70():
    print("特徴ベクトルのサイズ：", X_train.size())
    print("特徴ベクトル：", X_train)
    # torch.Size([10672, 300])
    # 二次元であり、１次元目の要素数10672、２次元目の要素数300を表す

    print("ラベルベクトルのサイズ：", y_train.size())
    print("ラベルベクトル：", y_train)
    # torch.Size([10672])
    # 一次元であり、要素数10672を表す

    # 保存
    torch.save(X_train, "pt/X_train.pt")
    torch.save(X_valid, "pt/X_valid.pt")
    torch.save(X_test, "pt/X_test.pt")
    torch.save(y_train, "pt/y_train.pt")
    torch.save(y_valid, "pt/y_valid.pt")
    torch.save(y_test, "pt/y_test.pt")


## 71.単層ニューラルネットワークによる予測
class SLPNet(nn.Module):  # 単層ニューラルネットワーク定義
    # ネットワークを構成するレイヤー定義
    def __init__(self, input_size, output_size):
        super().__init__()
        # nn.Linear:全結合層
        self.fc = nn.Linear(input_size, output_size, bias=False)
        # 正規乱数で重みを初期化
        nn.init.normal_(self.fc.weight, 0.0, 1.0)

    # インプットデータが順伝播時に通るレイヤー定義
    def forward(self, x):
        # 重みを正規乱数で初期化
        x = self.fc(x)
        return x


# 単層ニューラルネットワークの初期化(入力サイズ300、出力サイズ4)
model = SLPNet(300, 4)
# X_trainの最初のサンプルに対する予測結果の確率分布
# softmax():予測結果を確率分布に変換
# dim=-1:確率分布を計算する次元として最後の次元を指定
y_hat_1 = torch.softmax(model(X_train[:1]), dim=-1)
# X_trainの最初の４つのサンプルに対する予測結果の確率分布
Y_hat = torch.softmax(model.forward(X_train[:4]), dim=-1)


def func71():
    print("y_hat_1:", y_hat_1)
    print("Y_hat:", Y_hat)


## 72.損失と勾配の計算
# モデルの出力ベクトルとラベルベクトルを入力することで平均損失を計算
criterion = nn.CrossEntropyLoss()
# softmax前の値を入力ベクトルに設定
l_1 = criterion(model(X_train[:1]), y_train[:1])
# 勾配をゼロで初期化
model.zero_grad()
# 勾配を計算
l_1.backward()

l = criterion(model(X_train[:4]), y_train[:4])
model.zero_grad()
l.backward()


def func72():
    print(f"損失: {l_1:.4f}")
    print(f"勾配:\n{model.fc.weight.grad}")
    print(f"損失: {l:.4f}")
    print(f"勾配:\n{model.fc.weight.grad}")


## 73.確率的勾配降下法による学習
class NewsDataset(Dataset):
    def __init__(self, X, y):  # datasetの構成要素を指定
        self.X = X
        self.y = y

    def __len__(self):  # len(dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, idx):  # dataset[idx]で返す値を指定
        return [self.X[idx], self.y[idx]]


# Datasetの作成
dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

# Dataloaderの作成
# Datasetを入力とし、batch_sizeにまとめたデータを順に取り出すことが可能
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(
    dataset_valid, batch_size=len(dataset_valid), shuffle=False
)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)


# モデルの定義
model = SLPNet(300, 4)

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# オプティマイザの定義
# モデルのパラメータの最適化を行う
# model.parameters():モデルのパラメータを取得する関数
# モデルのパラメータを渡し、更新する対象とする
# lr=1e-1：学習率。
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# 学習
num_epochs = 20
for epoch in range(num_epochs):
    # 訓練モードに設定
    model.train()
    loss_train = 0.0
    # dataloader_trainからミニバッチごとにデータとラベルを取得して処理を行う
    for i, (inputs, labels) in enumerate(dataloader_train):
        # 勾配をゼロで初期化
        optimizer.zero_grad()

        # 順伝播 + 誤差逆伝播 + 重み更新
        # モデルに入力データを与え、予測を取得
        outputs = model(inputs)
        # 予測と正解ラベルの損失を計算
        loss = criterion(outputs, labels)
        # 誤差逆伝播を行い、勾配を計算
        loss.backward()
        # オプティマイザによる重みの更新
        optimizer.step()

        # 損失を記録
        loss_train += loss.item()

    # バッチ単位の平均損失計算
    loss_train = loss_train / i

    # 検証データの損失計算
    # 検証モードに設定
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(dataloader_valid))
        outputs = model(inputs)
        loss_valid = criterion(outputs, labels)


def func73():
    print(
        f"epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}"
    )


## 74.正解率の計測
def calculate_accuracy(model, loader):
    # 評価モードに設定
    model.eval()
    total = 0
    correct = 0
    # データローダーからデータとラベルを取得し、モデルに入力して予測を取得
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            # 予測の最大値を持つクラスを選択
            # 予測の次元（dim=-1）に沿って最大値を取得するため、確率分布の予測をクラスラベルに変換
            pred = torch.argmax(outputs, dim=-1)
            # 合計のデータ数を更新し、予測が正解と一致するデータポイントの数をカウント
            total += len(inputs)
            # 正確性を計算し、正解数を総データ数で割ってパーセンテージで表す
            correct += (pred == labels).sum().item()
    # モデルの正確性（0から1の範囲の値）を返す
    return correct / total


acc_train = calculate_accuracy(model, dataloader_train)
acc_test = calculate_accuracy(model, dataloader_test)


def func74():
    print(f"正解率（学習データ）：{acc_train:.3f}")
    print(f"正解率（評価データ）：{acc_test:.3f}")


# 75.損失と正解のプロット
def calculate_loss_and_accuracy(model, criterion, loader):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total


def func75():
    # モデルの定義
    model = SLPNet(300, 4)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # 学習
    num_epochs = 30
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # 訓練モードに設定
        model.train()
        for inputs, labels in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy(
            model, criterion, dataloader_train
        )
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, criterion, dataloader_valid
        )
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])
        # ログを出力
        print(
            f"epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}"
        )

    # 視覚化
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log_train).T[0], label="train")
    ax[0].plot(np.array(log_valid).T[0], label="valid")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[1].plot(np.array(log_train).T[1], label="train")
    ax[1].plot(np.array(log_valid).T[1], label="valid")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    plt.show()


## 76.チェックポイント
def func76():
    # モデルの定義
    model = SLPNet(300, 4)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # 学習
    num_epochs = 10
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # 訓練モードに設定
        model.train()
        for inputs, labels in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy(
            model, criterion, dataloader_train
        )
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, criterion, dataloader_valid
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
        # ログを出力
        print(
            f"epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}"
        )


## 77.ミニバッチ化
def train_model(
    dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs
):
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
        for inputs, labels in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy(
            model, criterion, dataloader_train
        )
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, criterion, dataloader_valid
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
            f"checkpoint{epoch + 1}.pt",
        )

        # 終了時刻の記録
        e_time = time.time()

        # ログを出力
        print(
            f"epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(e_time - s_time):.4f}sec"
        )

    return {"train": log_train, "valid": log_valid}


def func77():
    # datasetの作成
    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)

    # モデルの定義
    model = SLPNet(300, 4)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # モデルの学習
    for batch_size in [2**i for i in range(11)]:
        print(f"バッチサイズ: {batch_size}")
        log = train_model(
            dataset_train, dataset_valid, batch_size, model, criterion, optimizer, 1
        )


# 78.GPU上での学習
def calculate_loss_and_accuracy(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
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
    # GPUに送る
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
        for inputs, labels in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
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


def func78():
    # datasetの作成
    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)

    # モデルの定義
    model = SLPNet(300, 4)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # デバイスの指定
    device = torch.device("mps")

    # モデルの学習
    for batch_size in [2**i for i in range(11)]:
        print(f"バッチサイズ: {batch_size}")
        log = train_model(
            dataset_train,
            dataset_valid,
            batch_size,
            model,
            criterion,
            optimizer,
            1,
            device=device,
        )


## 79.多層ニューラルネットワーク
class MLPNet(nn.Module):
    def __init__(self, input_size, mid_size, output_size, mid_layers):
        super().__init__()
        self.mid_layers = mid_layers
        self.fc = nn.Linear(input_size, mid_size)
        self.fc_mid = nn.Linear(mid_size, mid_size)
        self.fc_out = nn.Linear(mid_size, output_size)
        self.bn = nn.BatchNorm1d(mid_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        for _ in range(self.mid_layers):
            x = F.relu(self.bn(self.fc_mid(x)))
        x = F.relu(self.fc_out(x))

        return x


def calculate_loss_and_accuracy(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
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
    # GPUに送る
    model.to(device)

    # dataloaderの作成
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=len(dataset_valid), shuffle=False
    )

    # スケジューラの設定
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs, eta_min=1e-5, last_epoch=-1
    )

    # 学習
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        # 開始時刻の記録
        s_time = time.time()

        # 訓練モードに設定
        model.train()
        for inputs, labels in dataloader_train:
            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.forward(inputs)
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


def calculate_accuracy(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()

    return correct / total


def func79():
    # datasetの作成
    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)

    # モデルの定義
    model = MLPNet(300, 200, 4, 1)

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # オプティマイザの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # デバイスの指定
    device = torch.device("mps")

    # モデルの学習
    log = train_model(
        dataset_train, dataset_valid, 64, model, criterion, optimizer, 1000, device
    )

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

    # 正解率の確認
    acc_train = calculate_accuracy(model, dataloader_train, device)
    acc_test = calculate_accuracy(model, dataloader_test, device)
    print(f"正解率（学習データ）：{acc_train:.3f}")
    print(f"正解率（評価データ）：{acc_test:.3f}")


## 実行
func79()
