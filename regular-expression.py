## 第3章：正規表現
import json
import re
import requests
import webbrowser


filename = "jawiki-country.json"
text = ""


## gunzip ./jawiki-country.json.gz
## 20.JSONデータの読み込み
def func20(text):
    with open(filename, mode="r") as f:
        for line in f:
            line = json.loads(line)
            if line["title"] == "イギリス":
                text = line["text"]
                break
    # print(text)
    return text


## 21.カテゴリ名を含む行を抽出
## r:Pythonの文字列リテラル（文字列表記）で使用されるプレフィックス。バックスラッシュが正規表現パターンの一部として解釈される.
## .*:0文字以上の任意の文字列
## ():()内の文字列が格納される
## ^:文字列（行）の先頭を意味する正規表現パターンのメタ文字。
## $:行末を意味する正規表現パターンのメタ文字。
## re.MULTILINE:各行の先頭、末尾にもマッチするようになる
def func21():
    pat = r"^(.*\[\[Category:.*\]\].*)$"
    ret = "\n".join(re.findall(pat, text, re.MULTILINE))
    print(ret)


## 22.カテゴリ名の抽出
## (?:...):カッコ内の正規表現にマッチはするが拾ってこない
def func22():
    pat = r"^.*\[\[Category:(.*?)(?:\|.*)?\]\].*$"
    ret = "\n".join(re.findall(pat, text, re.MULTILINE))
    print(ret)


## 23.セクション構造
## ()が３つ：３つの要素を持つタプル
def func23():
    pat = r"^(\={2,})\s*(.+?)\s*(\={2,}).*$"
    ret = ""
    for i in re.findall(pat, text, re.MULTILINE):
        print("".join(i[1] + ":" + str(len(i[0]) - 1)))


## 24.ファイル参照の抽出
def func24():
    pat = r"\[\[ファイル:(.+?)\|"
    ret = "\n".join(re.findall(pat, text))
    print(ret)


## 25.テンプレートの抽出
def func25():
    pat1 = r"^\{\{基礎情報.*?$(.*?)^\}\}"
    templete = re.findall(pat1, text, re.MULTILINE + re.DOTALL)
    # print(templete)

    pat2 = r"^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))"
    ret = dict(re.findall(pat2, templete[0], re.MULTILINE + re.DOTALL))
    for i, j in ret.items():
        print(i + ":" + j)


## 26.強調マークアップの除去
def remove_markup(text):
    # 2~5個の'を除去
    pat = r"\'{2,5}"
    text = re.sub(pat, "", text)
    return text


def func26():
    pat1 = r"^\{\{基礎情報.*?$(.*?)^\}\}"
    templete = re.findall(pat1, text, re.MULTILINE + re.DOTALL)

    pat2 = r"^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))"
    ret = dict(re.findall(pat2, templete[0], re.MULTILINE + re.DOTALL))
    ret2 = {}
    for i, j in ret.items():
        ret2[i] = remove_markup(j)
    for i, j in ret2.items():
        print(i + ":" + j)


## 27.内部リンクの除去
def remove_markup2(text):
    # 2~5個の'を除去
    pat = r"\'{2,5}"
    text = re.sub(pat, "", text)

    # [[]]を除去
    pat2 = r"\[\[(?:[^|]*?\|)??([^|]*?)\]\]"
    text = re.sub(pat2, r"\1", text)

    return text


def func27():
    pat1 = r"^\{\{基礎情報.*?$(.*?)^\}\}"
    templete = re.findall(pat1, text, re.MULTILINE + re.DOTALL)

    pat2 = r"^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))"
    ret = dict(re.findall(pat2, templete[0], re.MULTILINE + re.DOTALL))
    ret2 = {}
    for i, j in ret.items():
        ret2[i] = remove_markup2(j)
    for i, j in ret2.items():
        print(i + ":" + j)


## 28.MediaWikiマークアップの除去
def remove_markup3(text):
    # 強調マークアップの除去
    # 2~5個の'を除去
    pat = r"\'{2,5}"
    text = re.sub(pat, "", text)

    # 内部リンクマークアップの除去
    # [[リンク先のテキスト|表示するテキスト]]のうち「リンク先のテキスト」のみ抽出
    pat2 = r"\[\[(?:[^|]*?\|)??([^|]*?)\]\]"
    text = re.sub(pat2, r"\1", text)

    # 外部リンクマークアップの除去
    # URLを空文字列に置換
    pat3 = r"https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+"
    text = re.sub(pat3, "", text)

    # htmlタグの除去
    # <p>とかを空文字列に置換
    pat4 = r"<.+?>"
    text = re.sub(pat4, "", text)

    # テンプレートの除去
    pat5 = r"\{\{(?:lang|仮リンク)(?:[^|]*?\|)*?([^|]*?)\}\}"
    text = re.sub(pat5, r"\1", text)

    return text


def func28():
    pat1 = r"^\{\{基礎情報.*?$(.*?)^\}\}"
    templete = re.findall(pat1, text, re.MULTILINE + re.DOTALL)

    pat2 = r"^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))"
    ret = dict(re.findall(pat2, templete[0], re.MULTILINE + re.DOTALL))
    ret2 = {}
    for i, j in ret.items():
        ret2[i] = remove_markup3(j)
    for i, j in ret2.items():
        print(i + ":" + j)


# 29.国旗画像のURLを取得する
# def get_url(titles):
#     S = requests.Session()
#     URL = "https://en.wikipedia.org/w/api.php"
#     PARAMS = {
#         "action": "query",
#         "format": "json",
#         "prop": "imageinfo",
#         "titles": titles,
#     }
#     R = S.get(url=URL, params=PARAMS)
#     DATA = R.json()

#     PAGES = DATA["query"]["pages"]

#     for v in PAGES.values():
#         return v["imageinfo"][0]["user"]


def get_url(text):
    url_file = text["国旗画像"].replace(" ", "_")
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&titles=File:"
        + url_file
        + "&prop=imageinfo&iiprop=url&format=json"
    )
    data = requests.get(url)
    return re.search(r'"url":"(.+?)"', data.text).group(1)


def func29():
    # pat1 = r"^\{\{基礎情報.*?$(.*?)^\}\}"
    # templete = re.findall(pat1, text, re.MULTILINE + re.DOTALL)

    # pat2 = r"^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))"
    # ret = dict(re.findall(pat2, templete[0], re.MULTILINE + re.DOTALL))
    # ret2 = {}
    # for i, j in ret.items():
    #     ret2[i] = remove_markup3(j)
    # for i, j in ret2.items():
    #     print(i + ":" + j)

    # if __name__ == "__main__":
    #     dic = dict(ret2)
    #     for k, v in dic.items():
    #         if k == "国旗画像":
    #             url = get_url("File:" + v)
    #             print(url)
    #             webbrowser.open(url)
    pat1 = r"^\{\{基礎情報.*?$(.*?)^\}\}"
    templete = re.findall(pat1, text, re.MULTILINE + re.DOTALL)

    pat2 = r"^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))"
    ret = dict(re.findall(pat2, templete[0], re.MULTILINE + re.DOTALL))
    # ret2 = {}
    # for i, j in ret.items():
    #     ret2[i] = remove_markup3(j)
    # for i, j in ret2.items():
    #     print(i + ":" + j)
    print(get_url(ret))


## 実行
text = func20(text)
func29()
