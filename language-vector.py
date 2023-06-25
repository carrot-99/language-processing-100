# import gdown
# url = "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM"
# output = "GoogleNews-vectors-negative300.bin.gz"
# gdown.download(url, output, quiet=False)

## 第7章：単語ベクトル
# import cython
# import bhtsne
from gensim.models import KeyedVectors
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE


## 60.単語ベクトルの読み込みと表示
model = KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin.gz", binary=True
)


def func60():
    print(model["United_States"])


## 61.単語の類似度
def func61():
    print(model.similarity("United_States", "U.S."))


## 62.類似度の高い単語10件
def func62():
    print(model.most_similar("United_States", topn=10))


## 63.加法構成性によるアナロジー
def func63():
    vec = model["Spain"] - model["Madrid"] + model["Athens"]
    print(
        model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10)
    )


## 64.アナロジーデータでの実験
## wget http://download.tensorflow.org/data/questions-words.txt
def func64():
    with open("./text/questions-words.txt", "r") as f1, open(
        "./text/questions-words-add.txt", "w"
    ) as f2:
        for line in f1:  # f1から1行ずつ読込み、求めた単語と類似度を追加してf2に書込む
            line = line.split()
            if line[0] == ":":
                category = line[1]
            else:
                word, cos = model.most_similar(
                    positive=[line[1], line[2]], negative=[line[0]], topn=1
                )[0]
                f2.write(" ".join([category] + line + [word, str(cos) + "\n"]))


## 65.アナロジータスクでの正解率
def func65():
    with open("./text/questions-words-add.txt", mode="r") as f:
        sem1 = 0
        sem2 = 0
        syn1 = 0
        syn2 = 0
        for line in f:
            line = line.split()
            if not line[0].startswith("gram"):  # 意味的アナロジー評価用カテゴリの場合
                sem1 += 1
                if line[4] == line[5]:
                    sem2 += 1
            else:  # 文法的アナロジー評価用カテゴリの場合
                syn1 += 1
                if line[4] == line[5]:
                    syn2 += 1
    print(f"意味的アナロジー正解率： {sem2/sem1:f}")
    print(f"文法的アナロジー正解率： {syn2/syn1:f}")


## 66.WordSimilarity-353での評価
## wget http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip
## unzip wordsim353.zip
def func66():
    ws353 = []  # 類似度の計算結果リスト
    with open("./text/combined.csv", mode="r") as f:
        next(f)  # header行をスキップ
        for line in f:  # 単語ベクトルと類似度を計算
            line = [s.strip() for s in line.split(",")]  # カンマで分割し、前後の空白削除
            line.append(model.similarity(line[0], line[1]))  # line[0]とline[1]の類似度を計算
            ws353.append(line)
    # check
    # for i in range(5):
    #     print(ws353[i])

    # スピアマン相関係数の計算
    human = np.array(ws353).T[2]  # 人間の類似度を取得
    w2v = np.array(ws353).T[3]  # 単語ベクトルの類似度を取得
    correlation, pvalue = spearmanr(human, w2v)  # 類似度のスピアマン相関係数を計算

    print(f"スピアマン相関係数: {correlation:f}")


## 67.k-meansクラスタリング
def func67():
    # 国名の取得
    countries = set()
    with open("./text/questions-words-add.txt") as f:
        for line in f:
            line = line.split()  # blank区切りの文字列をリストに変換
            if line[0] in ["capital-common-countries", "capital-world"]:
                countries.add(line[2])
            elif line[0] in ["currency", "gram6-nationality-adjective"]:
                countries.add(line[1])
    countries = list(countries)

    # 単語ベクトルの取得
    countries_vec = [model[country] for country in countries]

    # k-meansクラスタリング
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(countries_vec)  # K-meansクラスタリングの学習を実施

    # check
    # for i in range(5):
    #     cluster = np.where(kmeans.labels_ == i)[0]
    #     print("cluster", i)
    #     print(", ".join([countries[k] for k in cluster]))

    return countries, countries_vec


## 68.Ward法によるクラスタリング
## Ward法：クラスタ間の分散を最小化するようにクラスタリングを行う手法
def func68():
    countries, countries_vec = func67()
    plt.figure(figsize=(15, 5))  # 図のサイズ指定
    Z = linkage(countries_vec, method="ward")  # Ward法でクラスタリング
    dendrogram(Z, labels=countries)  # 階層的クラスタリングの結果をデンドログラムで表示
    plt.show()


## 69.t-SNEによる可視化
def func69():
    countries, countries_vec = func67()
    countries_vec = np.array(countries_vec)
    embedded = TSNE(n_components=2).fit_transform(countries_vec)
    plt.figure(figsize=(10, 10))
    plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
    for (x, y), name in zip(embedded, countries):
        plt.annotate(name, (x, y))
    plt.show()


## 実行
func69()
