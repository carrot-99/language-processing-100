## 第1章：準備運動
import random

## 00.文字列の逆順

def func00():
    str00 = 'stressed'
    ans00 = str00[::-1]
    print('文字列の逆順:', ans00)


## 01.「パタトクカシー」
def func01():
    str01 = 'パタトクカシー'
    ans01 = str01[::2]
    print('パトカー:', ans01)


## 02.「パトカー」＋「タクシー」＝「パタトクカシー」
def func02():
    str0201 = 'パトカー'
    str0202 = 'タクシー'
    ans02   =  ''

    for i in range(0,4):
        temp02 = (str0201 + str0202)[i::4]
        ans02 += temp02

    print('パタトクカシー:', ans02)


## 03.円周率

def list_len(str):
    return len(str)

def func03():

    str03 = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
    str03 = str03.replace(',', '')
    str03 = str03.replace('.', '')
    temp0301 = str03.split(' ')
    ans03 = map(func03, temp0301)

    print('円周率:', list(ans03))


## 04.元素記号

def func04():
    str04 = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
    ans04 = str04.split(' ')
    temp0402 = [1,5,6,7,8,9,15,16,19]

    for i in range(1,len(ans04)+1):
        if i in temp0402:
            ans04[i-1] = ans04[i-1][:1]
        else:
            ans04[i-1] = ans04[i-1][:2]

    print('元素記号:', ans04)


## 05.n-gram

def ngram(n, lst):
    ans = []
    for i in range(len(lst)-1):
        ans.append(lst[i:i+n]) 
    return ans
   
def func05():
    str05 = 'I am an NLPer'
    ans0501 = ngram(2, str05.split(' '))
    ans0502 = ngram(2, list(str05))

    print('単語bi-gram:', ans0501)
    print('文字bi-gram:', ans0502)


## 06.集合

def func06():
    str0601 = 'paraparaparadise'
    str0602 = 'paragraph'
    X06     = set(ngram(2, str0601))
    Y06     = set(ngram(2, str0602))
    ans0601 = X06 | Y06
    ans0602 = X06 & Y06
    ans0603 = X06 - Y06

    print('X:', X06)
    print('Y:', Y06)
    print('和集合:', ans0601)
    print('積集合:', ans0602)
    print('差集合:', ans0603)
    print('Xにseが含まれるか:', 'se' in X06)
    print('Yにseが含まれるか:', 'se' in Y06)


## 07.テンプレートによる文生成

def gene_sentence(x, y, z):
    print(str(x) + '時の' + str(y) + 'は' + str(z))

def func07():
    gene_sentence(12, '気温', 22.4)

## 08.暗号文

def cipher(str):
    ret = ''
    for x in str:
        if x.islower():
            ret += chr((219 - ord(x)))
        else:
            ret += x
    return ret

def func08():
    ans0801 = cipher('I am a carrot.')
    print('暗号化:', ans0801)
    ans0802 = cipher(ans0801)
    print('複合化:', ans0802)


## 09.Typoglycemia

def typoglycemia(str):
    if len(str) <= 4:
        return str
    else:
        lst = list(str)
        temp = ''.join(random.sample(lst[1:len(lst)-1], len(lst)-2))
        return lst[0] + temp + lst[-1]

def func09():
    str09 = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'
    lst09 = str09.split(' ')
    map09 = map(typoglycemia, lst09)
    ans09 = map(lambda s: s + ' ', list(map09))
    print(''.join(list(ans09)))

## 実行
func09()
