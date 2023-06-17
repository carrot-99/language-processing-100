# ベースイメージの指定
FROM python:3.11.4

# 作業ディレクトリを設定
WORKDIR /app

# ホスト上のrequirements.txtをコンテナ内の/appディレクトリにコピー
COPY requirements.txt .

# 依存パッケージのインストール
RUN pip install --no-cache-dir -r requirements.txt

# ホスト上のコードをコンテナ内の/appディレクトリにコピー
COPY warm-up-excercise.py .

# コンテナ内で実行するコマンドを指定
CMD ["python", "warm-up-excercise.py"]
