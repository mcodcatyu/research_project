#官方輕量版 Python 基礎鏡像
FROM python:3.10-slim

#設定容器內的工作目錄
WORKDIR /app

#先複製requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案內的所有程式碼進容器
COPY . .

#Streamlit 的預設port
EXPOSE 8501

#容器啟動時自動執行Streamlit網頁
CMD ["streamlit", "run", "app.py", "--server.port=8501","--server.address=0.0.0.0"]