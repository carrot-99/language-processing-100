## 第6章：機械学習
import pandas as pd
import numpy as np
import string
import re
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


## 50.データの入手・整形
def func50():
    # データ読み込み
    df = pd.read_csv(
        "./text/newsCorpora.csv",
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

    # データ抽出
    df = df.loc[
        df["PUBLISHER"].isin(
            [
                "Reuters",
                "Huffington Post",
                "Businessweek",
                "Contactmusic.com",
                "Daily Mail",
            ]
        ),
        ["TITLE", "CATEGORY"],
    ]

    # データ分割
    # stratifyオプションによって指定したカラム（カテゴリ）の構成比が分割語の各データで等しくなるように分割される
    # train（学習データ）：80%、valid_test（検証・評価データ）：20%
    train, valid_test = train_test_split(
        df, test_size=0.2, shuffle=True, random_state=123, stratify=df["CATEGORY"]
    )
    # valid（検証データ）：10%、test（評価データ）：10%
    valid, test = train_test_split(
        valid_test,
        test_size=0.5,
        shuffle=True,
        random_state=123,
        stratify=valid_test["CATEGORY"],
    )

    # データの保存
    train.to_csv("./text/train.txt", sep="\t", index=False)
    valid.to_csv("./text/valid.txt", sep="\t", index=False)
    test.to_csv("./text/test.txt", sep="\t", index=False)

    # 事例数の確認
    # print("【学習データ】")
    # print(train["CATEGORY"].value_counts())
    # print("【検証データ】")
    # print(valid["CATEGORY"].value_counts())
    # print("【評価データ】")
    # print(test["CATEGORY"].value_counts())

    return train, valid, test


## 51.特徴量抽出
def preprocessing(text):  # 前処理
    table = str.maketrans(
        string.punctuation, " " * len(string.punctuation)
    )  # punctuationで定義されている全ての記号をスペースに置換
    text = text.translate(table)  # 記号をスペースに置換
    text = text.lower()  # 小文字化
    text = re.sub("[0-9]+", "0", text)  # 数字列を0に置換

    return text


def func51():
    train, valid, test = func50()
    # データ再結合
    df = pd.concat([train, valid, test], axis=0)  # axis=0で縦に結合
    df.reset_index(
        drop=True, inplace=True
    )  # indexを振りなおす　dropによって元のインデックスを新しい列として追加せずに削除 inplaceによって元のdf自体を変更

    # 前処理 TITLE列の各テキストデータに対して前処理
    df["TITLE"] = df["TITLE"].map(lambda x: preprocessing(x))

    # dfの先頭部分を出力
    # print(df.head())

    # データ分割
    # test情報を使わないため分割
    train_valid = df[: len(train) + len(valid)]
    test = df[len(train) + len(valid) :]

    # TfidfVectorizer
    vec_tfidf = TfidfVectorizer(
        min_df=10, ngram_range=(1, 2)
    )  # ngram_rangeでTF-IDFを計算する単語の長さを指定 今回はuni-gram,bi-gram

    # ベクトル化
    X_train_valid = vec_tfidf.fit_transform(train_valid["TITLE"])  # testの情報は使わない
    X_test = vec_tfidf.transform(test["TITLE"])

    # ベクトルをデータフレームに変換
    # pd.DataFrame()：密な行列をデータフレームに変換
    # X_train_valid.toarray()：X_train_validのベクトル表現を密な行列（NumPy配列）に変換
    # vec_tfidf.get_feature_names()：vec_tfidfによって抽出された特徴量の単語リスト取得
    X_train_valid = pd.DataFrame(
        X_train_valid.toarray(), columns=vec_tfidf.get_feature_names_out()
    )
    X_test = pd.DataFrame(X_test.toarray(), columns=vec_tfidf.get_feature_names_out())

    # データ分割
    X_train = X_train_valid[: len(train)]
    X_valid = X_train_valid[len(train) :]

    # データ保存
    X_train.to_csv("./text/X_train.txt", sep="\t", index=False)
    X_valid.to_csv("./text/X_valid.txt", sep="\t", index=False)
    X_test.to_csv("./text/X_test.txt", sep="\t", index=False)

    # print(X_train.head())
    return X_train, train, X_test, test, X_valid, valid


## 52.学習
def func52():
    X_train, train, X_test, test, X_valid, valid = func51()
    # モデルの学習
    lg = LogisticRegression(random_state=123, max_iter=10000)
    lg.fit(X_train, train["CATEGORY"])
    # print(lg)
    return X_train, train, X_test, test, lg, X_valid, valid


## 53.予測
def score_lg(lg, X):
    # lg.predict_proba(X)：Xの各サンプルに対して各クラスの確率を予測する
    # np.max(lg.predict_proba(X), axis=1)：各サンプルの最大確率を計算し、ロジスティック回帰モデルの予測の信頼度を表す
    # lg.predict(X)：特徴行列Xの各サンプルに対してクラスラベルを予測する
    # 各サンプルの最大確率（予測の信頼度）を表す値と各サンプルの予測されたクラスラベルの二つの要素を含むリストを返す
    return [np.max(lg.predict_proba(X), axis=1), lg.predict(X)]


def func53():
    X_train, train, X_test, test, lg = func52()
    train_pred = score_lg(lg, X_train)
    test_pred = score_lg(lg, X_test)
    # print(train_pred)
    return train_pred, test_pred


## 54.正解率の計測
def func54():
    X_train, train, X_test, test, lg = func52()
    train_pred, test_pred = func53()
    # accuracy_score():予測されたクラスラベル（train_pred[1]）と真のクラスラベル（train["CATEGORY"]）を比較して正解率を計算
    train_accuracy = accuracy_score(train["CATEGORY"], train_pred[1])
    test_accuracy = accuracy_score(test["CATEGORY"], test_pred[1])
    print(f"正解率（学習データ）：{train_accuracy:f}")
    print(f"正解率（評価データ）：{test_accuracy:f}")


## 55.混同行列の作成
## 混同行列:分類モデルの性能を評価するための行列
##         真陽性(True Positive)、真陰性(True Negative)、偽陽性(False Positive)、偽陰性(False Negative)の各カテゴリを示す
def func55():
    X_train, train, X_test, test, lg = func52()
    train_pred, test_pred = func53()
    # 学習データ
    # confusion_matrix():予測されたクラスラベルと真のクラスラベルから混同行列を作成
    train_cm = confusion_matrix(train["CATEGORY"], train_pred[1])
    print(train_cm)
    sns.heatmap(train_cm, annot=True, cmap="Blues")
    plt.show()


## 56.適合率、再現率、F1スコアの計測
def calculate_scores(y_true, y_pred):
    X_train, train, X_test, test, lg = func52()
    train_pred, test_pred = func53()
    # 適合率
    # 正と予測されたサンプルのうち、実際に正であるものの割合
    precision = precision_score(
        test["CATEGORY"], test_pred[1], average=None, labels=["b", "e", "t", "m"]
    )  # Noneを指定するとクラスごとの精度をndarrayで返す
    precision = np.append(
        precision, precision_score(y_true, y_pred, average="micro")
    )  # 末尾にマイクロ平均を追加
    precision = np.append(
        precision, precision_score(y_true, y_pred, average="macro")
    )  # 末尾にマクロ平均を追加

    # 再現率
    # 実際に正であるサンプルのうち、正と予測されたものの割合
    recall = recall_score(
        test["CATEGORY"], test_pred[1], average=None, labels=["b", "e", "t", "m"]
    )
    recall = np.append(recall, recall_score(y_true, y_pred, average="micro"))
    recall = np.append(recall, recall_score(y_true, y_pred, average="macro"))

    # F1スコア
    # 適合率と再現率の調和平均
    f1 = f1_score(
        test["CATEGORY"], test_pred[1], average=None, labels=["b", "e", "t", "m"]
    )
    f1 = np.append(f1, f1_score(y_true, y_pred, average="micro"))
    f1 = np.append(f1, f1_score(y_true, y_pred, average="macro"))

    # 結果を結合してデータフレーム化
    scores = pd.DataFrame(
        {"適合率": precision, "再現率": recall, "F1スコア": f1},
        index=["b", "e", "t", "m", "マイクロ平均", "マクロ平均"],
    )

    return scores


def func56():
    X_train, train, X_test, test, lg = func52()
    train_pred, test_pred = func53()
    print(calculate_scores(test["CATEGORY"], test_pred[1]))


## 57.特徴量の重みの確認
def func57():
    X_train, train, X_test, test, lg = func52()
    # X_trainの特徴量名を格納
    features = X_train.columns.values
    index = [i for i in range(1, 11)]
    # 各カテゴリ(lg.classes_)ごとに処理を行う
    # coef:重み係数
    for c, coef in zip(lg.classes_, lg.coef_):
        print(f"【カテゴリ】{c}")
        # coefを降順ソート
        best10 = pd.DataFrame(
            features[np.argsort(coef)[::-1][:10]], columns=["重複度上位"], index=index
        ).T
        # coefを昇順ソート
        worst10 = pd.DataFrame(
            features[np.argsort(coef)[:10]], columns=["重複度下位"], index=index
        ).T
        # best10とworst10を結合して表示
        display(pd.concat([best10, worst10], axis=0))
        print("\n")


## 58.正則化パラメータの更新
def func58():
    X_train, train, X_test, test, lg, X_valid, valid = func52()
    result = []
    for C in tqdm(np.logspace(-5, 4, 10, base=10)):
        # モデルの学習
        lg = LogisticRegression(random_state=123, max_iter=10000, C=C)
        lg.fit(X_train, train["CATEGORY"])

        # 予測値の取得
        train_pred = score_lg(lg, X_train)
        valid_pred = score_lg(lg, X_valid)
        test_pred = score_lg(lg, X_test)

        # 正解率の算出
        train_accuracy = accuracy_score(train["CATEGORY"], train_pred[1])
        valid_accuracy = accuracy_score(valid["CATEGORY"], valid_pred[1])
        test_accuracy = accuracy_score(test["CATEGORY"], test_pred[1])

        # 結果の格納
        result.append([C, train_accuracy, valid_accuracy, test_accuracy])

    # 視覚化
    result = np.array(result).T
    plt.plot(result[0], result[1], label="train")
    plt.plot(result[0], result[2], label="valid")
    plt.plot(result[0], result[3], label="test")
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.xlabel("C")
    plt.legend()
    plt.show()


## 59.ハイパーパラメータの探索
def objective_lg(trial):  # 最適化対象を関数で指定
    X_train, train, X_test, test, lg, X_valid, valid = func52()
    # チューニング対象パラメータのセット
    l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
    C = trial.suggest_float("C", 1e-4, 1e4, log=True)

    # モデルの学習
    lg = LogisticRegression(
        random_state=123,
        max_iter=10000,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
    )
    lg.fit(X_train, train["CATEGORY"])

    # 予測値の取得
    valid_pred = score_lg(lg, X_valid)

    # 正解率の算出
    valid_accuracy = accuracy_score(valid["CATEGORY"], valid_pred[1])

    return valid_accuracy


def func59():
    # 最適化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_lg, timeout=3600)

    # 結果の表示
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {:f}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("   {}: {}".format(key, value))


## 実行
func59()
