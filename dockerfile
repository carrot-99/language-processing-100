# 一旦枠作ったけど諦めた
FROM python:3.10.0

# アプリケーションディレクトリを作成する
# WORKDIR /usr/src/python

# pipのアップデート
# RUN pip install --upgrade pip

# pipでインストールしたいモジュールをrequirements.txtに記述しておいて、
# コンテナ内でpipにインストールさせる
# requirements.txtの書き方は[pip freeze]コマンドから参考に出来る
# COPY requirements.txt ./
# RUN pip install -r requirements.txt

# アプリケーションコードをコンテナにコピー
# COPY . .

# EXPOSE 8000
# CMD [ "python", "app.py" ]