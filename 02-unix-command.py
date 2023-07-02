## 第2章：UNIXコマンド
import sys
import numpy as np
import collections

file1 = "text/popular-names.txt"
file2 = "text/popular-names2.txt"
col1 = "text/col1.txt"
col2 = "text/col2.txt"
merge = "text/merge.txt"
split = "text/split.txt"
sort = "text/sort.txt"


## 10.行数のカウント
# wc -l ./popular-names.txt
def func10():
    line_count = 0
    with open(file1) as f:
        for line in f:
            line_count += 1
    f.close
    print(line_count)


## 11.タブをスペースに変換
# sed -e 's/\t/ /g' ./popular-names.txt| head -n 5
def func11():
    with open(file1, mode="r") as f:
        lines = f.read()
    f.close()

    lines = lines.replace("\t", " ")
    with open(file2, mode="w") as f:
        f.write(lines)
    f.close()


## 12.1列目をcol1.txtに、2列目をcol2.txtに保存
# cut -f 1 ./popular-names.txt| head -n 5
# cut -f 2 ./popular-names.txt| head -n 5
def pick_col1(str):
    return str.split(" ")[0] + "\n"


def pick_col2(str):
    return str.split(" ")[1] + "\n"


def func12():
    with open(file2, mode="r") as f:
        lines = f.read()
    f.close()

    # temp = lines.split(" ")
    line = lines.split("\n")
    ans1201 = map(pick_col1, line)
    ans1202 = map(pick_col2, line)

    with open(col1, mode="w") as f:
        f.write("".join(list(ans1201)).rstrip("\n"))
    f.close()

    with open(col2, mode="w") as f:
        f.write("".join(list(ans1202)).rstrip("\n"))
    f.close()


## 13.col1.txtとcol2.txtをマージ
## paste ./col1.txt ./col2.txt | head -n 5
def func13():
    with open(col1, mode="r") as f:
        lines1 = f.read()
    f.close()

    with open(col2, mode="r") as f:
        lines2 = f.read()
    f.close()

    lineslist1 = lines1.split("\n")
    lineslist2 = lines2.split("\n")

    ans13 = ""

    for i in range(len(lineslist1)):
        temp = lineslist1[i] + "\t" + lineslist2[i] + "\n"
        ans13 += temp

    with open(merge, mode="w") as f:
        f.write(ans13.rstrip("\n"))
    f.close()


## 14.先頭からN行を出力
## head -n 5 ./merge.txt
def func14():
    N = int(sys.argv[1])

    with open(merge, mode="r") as f:
        lines = f.read()
    f.close()

    lineslist = lines.split("\n")
    ans14 = ""
    for i in range(N):
        ans14 += lineslist[i] + "\n"
    print(ans14)


## 15.末尾のN行を出力
## tail -n 5 ./popular-names.txt
def func15():
    N = int(sys.argv[1])

    with open(file1, mode="r") as f:
        lines = f.read()
    f.close()

    lineslist = lines.split("\n")
    ans15 = ""
    for i in reversed(range(N)):
        ans15 += lineslist[len(lineslist) - i - 1] + "\n"
    print(ans15)


## 16.ファイルをN分割する
## split -l 278 -d ./popular-names.txt sp

# def split_list(lst, N):
#     ret = []
#     for i in range(0, len(lst), N):
#         ret.append(lst[i:i + N])
#     return ret


def func16():
    N = int(sys.argv[1])

    with open(file1, mode="r") as f:
        lines = f.read()
    f.close()

    lineslist = lines.split("\n")
    temp = list(np.array_split(lineslist, N))

    ans16 = []

    temp2 = ""

    for i in temp:
        for j in i:
            temp2 += "".join(j) + "\n"
        ans16.append(temp2)

    print(ans16[0])


## 17.1列目の文字列の異なり
## cut -f 1 ./popular-names.txt | sort | uniq | wc -l
def func17():
    with open(col1, mode="r") as f:
        lines = f.read()
    f.close()

    print(len(set(lines.split("\n"))))


## 18.各行を3コラム目の数値の降順にソート
## cat ./popular-names.txt | sort -rnk 3 | head -n 5
def func18():
    with open(file1, mode="r") as f:
        lines = f.read()
    f.close()

    lineslist = lines.split("\n")
    lineslist2 = []
    for i in lineslist:
        temp = i.split("\t")
        lineslist2.append(temp)

    sortedlist = sorted(lineslist2, key=lambda x: int(x[2]), reverse=True)

    ans18 = ""

    for i in sortedlist:
        temp = i[0] + "\t" + i[1] + "\t" + i[2] + "\t" + i[3] + "\n"
        ans18 += temp

    print(ans18[0:97])


## 19.各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
## cut -f 1 ./popular-names.txt | sort | uniq -c | sort -rn
def func19():
    with open(col1, mode="r") as f:
        lines = f.read()
    f.close()

    lineslist = lines.split("\n")

    c = collections.Counter(lineslist)
    temp = c.most_common()
    ans19 = ""
    for i in temp:
        temp2 = i[0] + "\t" + str(i[1]) + "\n"
        ans19 += temp2
    # ans19 = "".join(temp)
    print(ans19.rstrip("\n"))


## 実行
func10()
