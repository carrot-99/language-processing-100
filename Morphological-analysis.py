## 第4章：形態素解析
import MeCab
from collections import defaultdict
from matplotlib import pyplot as pyp
import japanize_matplotlib

# python 3.10.0
# pip 23.1.2
# mecab -o ./neko.txt.mecab ./neko.txt

filename = "./text/neko.txt.mecab"
sentences = []

# def start():
#     mecab = MeCab.Tagger()

#     with open("./text/neko.txt") as in_file, open(
#         "./text/neko.txt.mecab", mode="w"
#     ) as out_file:
#         out_file.write(mecab.parse(in_file.read()))


## 30.形態素解析結果の読み込み


def func30():
    morphs = []
    with open(filename, mode="r") as f:
        for line in f:  # 1行ずつ読み込み
            if line != "EOS\n":  # 文末以外：形態素解析情報を辞書型に格納して形態素リストに追加
                fields = line.split("\t")
                if len(fields) != 2 or fields[0] == "":  # 文頭以外の空白と改行文字をスキップ
                    continue
                else:
                    attr = fields[1].split(",")
                    morph = {
                        "surface": fields[0],  # 表層形
                        "base": attr[6],  # 基本形
                        "pos": attr[0],  # 品詞
                        "pos1": attr[1],  # 品詞細分類1
                    }
                    morphs.append(morph)
            else:
                sentences.append(morphs)
                morphs = []
    # for morph in sentences[2]:
    #     print(morph)
    return sentences


## 31.動詞
def func31():
    ans = set()
    for sentence in sentences:
        for morph in sentence:
            if morph["pos"] == "動詞":
                ans.add(morph["surface"])
    return ans


## 32.動詞の基本形
def func32():
    ans = set()
    for sentence in sentences:
        for morph in sentence:
            if morph["pos"] == "動詞":
                ans.add(morph["base"])
    return ans


## 33.「AのB」
def func33():
    ans = set()
    for sentence in sentences:
        for i in range(1, len(sentence) - 1):
            if (
                sentence[i - 1]["pos"] == "名詞"
                and sentence[i]["surface"] == "の"
                and sentence[i + 1]["pos"] == "名詞"
            ):
                ans.add(
                    sentence[i - 1]["surface"]
                    + sentence[i]["surface"]
                    + sentence[i + 1]["surface"]
                )
    return ans


## 34.名詞の連接
def func34():
    ans = set()
    for sentence in sentences:
        num = 0
        stc = ""
        for morph in sentence:
            if morph["pos"] == "名詞":
                num += 1
                stc = "".join([stc, morph["surface"]])
            elif num >= 2:
                ans.add(stc)
                num = 0
                stc = ""
            else:
                num = 0
                stc = ""
        if num >= 2:
            ans.add(stc)
    return ans


## 35.単語の出現頻度
def func35():
    ans = defaultdict(int)  # 初期値0（これをしないとキーがないってエラーになる）
    for sentence in sentences:
        for morph in sentence:
            if morph["pos"] != "記号":
                ans[morph["base"]] += 1  # 単語をキーにした辞書型変数の値を増やしていく
    ans = sorted(
        ans.items(), key=lambda x: x[1], reverse=True
    )  # .items()：辞書のキーと値を同時に取得
    return ans


## 36.頻度上位10語
def func36():
    ans = func35()
    words = [a[0] for a in ans[:10]]
    counts = [a[1] for a in ans[:10]]
    pyp.bar(words, counts)
    pyp.show()


## 37.「猫」と共起頻度の高い上位10語
def func37():
    ans = defaultdict(int)
    for sentence in sentences:
        if "猫" in [morph["base"] for morph in sentence]:  # 「猫」と同じ文に含まれる単語
            for morph in sentence:
                if morph["pos"] == "名詞" or morph["pos"] == "動詞":  # 名詞と動詞のみ抽出
                    ans[morph["base"]] += 1
    ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)
    ans = [item for item in ans if item[0] != "猫"]  # "猫"を除外
    words = [a[0] for a in ans[:10]]
    counts = [a[1] for a in ans[:10]]
    pyp.bar(words, counts)
    pyp.show()


## 38.ヒストグラム
def func38():
    ans = defaultdict(int)
    for sentence in sentences:
        for morph in sentence:
            if morph["pos"] != "記号":
                ans[morph["base"]] += 1
    pyp.hist(list(ans.values()), bins=100)

    pyp.title("単語の出現頻度")
    pyp.xlabel("出現頻度")
    pyp.ylabel("単語の種類数")
    pyp.show()


## 39.Zipfの法則
def func39():
    ans = func35()
    pyp.xscale("log")
    pyp.yscale("log")
    ranks = [i + 1 for i in range(len(ans))]
    values = [i[1] for i in ans]
    pyp.plot(ranks, values, marker="o")

    pyp.title("Zipfの法則")
    pyp.xlabel("単語の出現頻度順位")
    pyp.ylabel("出現頻度")

    pyp.show()


## 実行
sentences = func30()
func39()
# print(func35())
