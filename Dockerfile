FROM python:3.9.20

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos necessários para o container
COPY requirements.txt .
COPY main.py .
COPY best_lstm_model_v4.keras .

# Faz o upgrade do pip wheel e setuptools
RUN pip install --upgrade pip wheel setuptools

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta usada pelo FastAPI
EXPOSE 8000

# Comando para rodar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]