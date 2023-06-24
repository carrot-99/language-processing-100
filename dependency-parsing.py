## 第5章：係り受け解析

## MeCabインストール済み
## CRF++、CaboChaのインストール
## brew install crf++
## brew install cabocha

import pydot
from IPython.display import Image, display_png
from graphviz import Digraph
from itertools import combinations
import re

filename = "ai.ja.txt.parsed"


## 40.係り受け解析結果の読み込み（形態素）
class Morph:
    def __init__(self, morph):
        surface, attr = morph.split("\t")
        attr = attr.split(",")
        self.surface = surface
        self.base = attr[6]
        self.pos = attr[0]
        self.pos1 = attr[1]


def func40():
    sentences = []
    morphs = []
    with open(filename, mode="r") as f:
        for line in f:
            if line[0] == "*":  # *から始まる行は係り受け関係を表すらしい。
                continue
            elif line != "EOS\n":  # 文末じゃなければMorphクラスを適用
                morphs.append(Morph(line))
            else:  # 文末の場合はmorphsリストをsentencesリストに追加
                sentences.append(morphs)
                morphs = []
    for m in sentences[2]:
        print(vars(m))


## 41.係り受け解析結果の読み込み（文節・係り受け）
class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []


class Sentence:
    def __init__(self, chunks):
        self.chunks = chunks
        for i, chunk in enumerate(self.chunks):  # srcs作成には1文の全ての文節情報を必要とするためここで作成
            if chunk.dst != -1 and chunk.dst < len(self.chunks):
                self.chunks[chunk.dst].srcs.append(i)


def func41():
    sentences = []
    chunks = []
    morphs = []
    with open(filename, mode="r") as f:
        for line in f:
            if line[0] == "*":  # 直前の文節の情報にChunkを適用しChunksリストに追加, 直後の文節の係先を取得
                if len(morphs) > 0:
                    chunks.append((Chunk(morphs, dst)))
                    morphs = []
                dst = int(line.split(" ")[2].rstrip("D"))
            elif line != "EOS\n":  # 文末以外にMorphsを適用し、morphsリストに追加
                morphs.append(Morph(line))
            else:  # 文末の場合は直前の文節の情報にChunkを適用しChunksリストに追加, ChunksリストにSentenceを適用しSentencesリストに追加
                sentence_chunks = chunks.copy()  # chunksリストをコピーする
                chunks.append(Chunk(morphs, dst))
                sentences.append(Sentence(sentence_chunks))
                morphs = []
                chunks = []
                dst = None
    # for chunk in sentences[2].chunks:
    #     print([morph.surface for morph in chunk.morphs], chunk.dst, chunk.srcs)
    return sentences


## 42.係り元と係り先の文節の表示
def func42():
    sentences = func41()
    sentence = sentences[2]
    for chunk in sentence.chunks:
        modifier = ""
        modifiee = ""
        if chunk.dst != -1 and chunk.dst < len(sentence.chunks):
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modifier += morph.surface
            for morph in sentence.chunks[int(chunk.dst)].morphs:
                if morph.pos != "記号":
                    modifiee += morph.surface
            print(modifier, modifiee, sep=",")


## 43.名詞を含む文節が動詞を含む文節に係るものを抽出
def func43():
    sentences = func41()
    sentence = sentences[2]
    for chunk in sentence.chunks:
        modifier = ""
        modifier_pos = []
        modifiee = ""
        modifiee_pos = []
        if chunk.dst != -1 and chunk.dst < len(sentence.chunks):
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modifier += morph.surface
                    modifier_pos.append(morph.pos)
            for morph in sentence.chunks[int(chunk.dst)].morphs:
                if morph.pos != "記号":
                    modifiee += morph.surface
                    modifiee_pos.append(morph.pos)
            if "名詞" in modifier_pos and "動詞" in modifiee_pos:
                print(modifier, modifiee, sep=",")


## 44.係り受け木の可視化
def func44():
    sentences = func41()
    sentence = sentences[7]
    edges = []
    for chunk in sentence.chunks:
        modifier = ""
        modifiee = ""
        if chunk.dst != -1 and chunk.dst < len(sentence.chunks):
            for morph in chunk.morphs:
                if morph.pos != "記号":
                    modifier += morph.surface
            for morph in sentence.chunks[int(chunk.dst)].morphs:
                if morph.pos != "記号":
                    modifiee += morph.surface
            edges.append([modifier, modifiee])
    n = pydot.Node("node")
    n.fontname = "IPAGothic"
    g = pydot.graph_from_edges(edges, directed=True)
    g.add_node(n)
    g.write_png("./img/ans44.png")
    display_png(Image("./img/ans44.png"))


## 45.動詞の格パターンの抽出
## cat ./text/ans45.txt | sort | uniq -c | sort -nr | head -n 10
## cat ./text/ans45.txt | grep '行う' | sort | uniq -c | sort -nr | head -n 5
## cat ./text/ans45.txt | grep 'なる' | sort | uniq -c | sort -nr | head -n 5
## cat ./text/ans45.txt | grep '与える' | sort | uniq -c | sort -nr | head -n 5
def func45():
    sentences = func41()
    with open("./text/ans45.txt", mode="w") as f:
        for sentence in sentences:
            for chunk in sentence.chunks:
                for morph in chunk.morphs:
                    if morph.pos == "動詞":  # chunkの左から順番に動詞を探す
                        temp = []
                        for src in chunk.srcs:  # 見つけた動詞の係元chunkから助詞を探す
                            temp = temp + [
                                morph.surface
                                for morph in sentence.chunks[src].morphs
                                if morph.pos == "助詞"
                            ]
                        if len(temp) > 0:  # 助詞があったら重複削除して辞書順にソート
                            temp = sorted(list(set(temp)))
                            line = "{}\t{}".format(morph.base, " ".join(temp))
                            print(line, file=f)
                        break


## 46.動詞の格フレーム情報の抽出
def func46():
    sentences = func41()
    with open("./text/ans46.txt", mode="w") as f:
        for sentence in sentences:
            for chunk in sentence.chunks:
                for morph in chunk.morphs:
                    if morph.pos == "動詞":  # chunkの左から順番に動詞を探す
                        temp = []
                        temp2 = []
                        for src in chunk.srcs:  # 見つけた動詞の係元chunkから助詞を探す
                            temp3 = [
                                morph.surface
                                for morph in sentence.chunks[src].morphs
                                if morph.pos == "助詞"
                            ]
                            if len(temp3) > 0:
                                temp = temp + temp3
                                temp2.append(
                                    "".join(
                                        morph.surface
                                        for morph in sentence.chunks[src].morphs
                                        if morph.pos != "記号"
                                    )
                                )
                        if len(temp) > 0:  # 助詞があったら重複削除して辞書順にソート
                            temp = sorted(list(set(temp)))
                            line = "{}\t{}\t{}".format(
                                morph.base, " ".join(temp), " ".join(temp2)
                            )
                            print(line, file=f)
                        break


## 47.機能動詞構文のマイニング
## cat ./text/ans47.txt | cut -f 1 | sort | uniq -c | sort -nr | head -n 10
## cat ./text/ans47.txt | cut -f 1,2 | sort | uniq -c | sort -nr | head -n 10
def func47():
    sentences = func41()
    with open("./text/ans47.txt", mode="w") as f:
        for sentence in sentences:
            for chunk in sentence.chunks:
                for morph in chunk.morphs:
                    if morph.pos == "動詞":  # chunkの左から順番に動詞を探す
                        for i, src in enumerate(
                            chunk.srcs
                        ):  # 見つけた動詞の係り元chunkが「サ変接続名詞+を（助詞）」で構成されるか確認
                            if (
                                len(sentence.chunks[src].morphs) == 2
                                and sentence.chunks[src].morphs[0].pos1 == "サ変接続"
                                and sentence.chunks[src].morphs[1].surface == "を"
                            ):
                                predicate = "".join(
                                    [
                                        sentence.chunks[src].morphs[0].surface,
                                        sentence.chunks[src].morphs[1].surface,
                                        morph.base,
                                    ]
                                )
                                temp = []
                                temp2 = []
                                for src in (
                                    chunk.srcs[:i] + chunk.srcs[i + 1 :]
                                ):  # 残りの係元chunkから助詞を探す
                                    temp3 = [
                                        morph.surface
                                        for morph in sentence.chunks[src].morphs
                                        if morph.pos == "助詞"
                                    ]
                                    if len(temp3) > 0:  # 助詞を含むchunkの場合は助詞と項を取得
                                        temp = temp + temp3
                                        temp2.append(
                                            "".join(
                                                morph.surface
                                                for morph in sentence.chunks[src].morphs
                                                if morph.pos != "記号"
                                            )
                                        )
                                if len(temp) > 0:  # 助詞があったら重複削除して辞書順にソート、項と合わせて出力
                                    temp = sorted(list(set(temp)))
                                    line = "{}\t{}\t{}".format(
                                        predicate, " ".join(temp), " ".join(temp2)
                                    )
                                    print(line, file=f)
                                break


## 48.名詞から根へのパスの抽出
def func48():
    sentences = func41()
    for sentence in sentences:
        for chunk in sentence.chunks:
            if "名詞" in [morph.pos for morph in chunk.morphs]:
                path = [
                    "".join(
                        morph.surface for morph in chunk.morphs if morph.pos != "記号"
                    )
                ]
                while chunk.dst != -1 and chunk.dst < len(sentence.chunks):
                    path.append(
                        "".join(
                            morph.surface
                            for morph in sentence.chunks[chunk.dst].morphs
                            if morph != "記号"
                        )
                    )
                    chunk = sentence.chunks[chunk.dst]
                print(" -> ".join(path))


## 49.名詞間の係り受けパスの抽出
def func49():
    sentences = func41()
    sentence = sentences[2]
    nouns = []
    for i, chunk in enumerate(sentence.chunks):
        if "名詞" in [morph.pos for morph in chunk.morphs]:
            nouns.append(i)
    for i, j in combinations(nouns, 2):
        path_i = []
        path_j = []
        while i != j:
            if i < j:
                path_i.append(i)
                i = sentence.chunks[i].dst
            else:
                path_j.append(j)
                j = sentence.chunks[j].dst
        if len(path_j) == 0:  # case 1
            chunk_X = "".join(
                [
                    morph.surface if morph.pos != "名詞" else "X"
                    for morph in sentence.chunks[path_i[0]].morphs
                ]
            )
            chunk_Y = "".join(
                [
                    morph.surface if morph.pos != "名詞" else "Y"
                    for morph in sentence.chunks[i].morphs
                ]
            )
            chunk_X = re.sub("X+", "X", chunk_X)  # Xが連続する場合、1文字のXに置換
            chunk_Y = re.sub("Y+", "Y", chunk_Y)  # Yが連続する場合、1文字のYに置換
            path_X2Y = (
                [chunk_X]
                + [
                    "".join(morph.surface for morph in sentence.chunks[n].morphs)
                    for n in path_i[1:]
                ]
                + [chunk_Y]
            )
            print(" -> ".join(path_X2Y))
        else:  # case 2
            chunk_X = "".join(
                [
                    morph.surface if morph.pos != "名詞" else "X"
                    for morph in sentence.chunks[path_i[0]].morphs
                ]
            )
            chunk_Y = "".join(
                [
                    morph.surface if morph.pos != "名詞" else "Y"
                    for morph in sentence.chunks[path_j[0]].morphs
                ]
            )
            chunk_k = "".join([morph.surface for morph in sentence.chunks[i].morphs])
            chunk_X = re.sub("X+", "X", chunk_X)
            chunk_Y = re.sub("Y+", "Y", chunk_Y)
            path_X = [chunk_X] + [
                "".join(morph.surface for morph in sentence.chunks[n].morphs)
                for n in path_i[1:]
            ]
            path_Y = [chunk_Y] + [
                "".join(morph.surface for morph in sentence.chunks[n].morphs)
                for n in path_j[1:]
            ]
            print(" | ".join([" -> ".join(path_X), " -> ".join(path_Y), chunk_k]))


## 実行
func49()
