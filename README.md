### **Deep Learning: API de Séries Temporais**

#### **Descrição do Projeto**
A API foi desenvolvida para realizar previsões de séries temporais utilizando um modelo de rede neural LSTM. O objetivo principal é prever valores futuros com base em dados históricos fornecidos, o modelo utilizado por esse repositório foi treinado com dados da empresa AAPL disponibilizados pela API do Yahoo Finance, é lógico concluir que essa API tem finalidade de prever o valor de ações **somente** da *Apple*. A aplicação suporta previsões de 7, 31 ou 365 dias úteis e fornece informações adicionais, como intervalos de confiança.

---

### **Funcionalidades Principais**
1. **Previsões:**
   - Rotas para previsão de 7 dias úteis, 31 dias úteis ou 365 dias úteis.
   - Baseia-se nos dados históricos fornecidos ou na data atual para iniciar as previsões.

2. **Monitoramento:**
   - Logs detalhados com informações sobre IP, localização (país), tempo de resposta e status da requisição.

3. **Escalabilidade:**
   - Preparado para ser executado em ambientes escaláveis, como Docker e Kubernetes.

4. **Segurança:**
   - Validação de dados com Pydantic para garantir a integridade da entrada e saída.

---

### **Endpoints da API**

#### **1. POST /predict**
- **Descrição:** Realiza previsões para os próximos 7 dias úteis, com base na data atual.
- **Entrada:**
  - Um JSON contendo a série histórica com pelo menos 7 dias:
    ```json
    {
      "data": [
        {"date": "2024-11-20", "value": 229.0},
        {"date": "2024-11-21", "value": 228.52},
        {"date": "2024-11-22", "value": 229.87}
      ]
    }
    ```
  - **Campos:**
    - `date`: Data no formato `YYYY-MM-DD`.
    - `value`: Valor correspondente à data.

- **Saída:**
  - Lista com previsões para os próximos 7 dias úteis:
    ```json
    [
      {
        "date": "2024-11-23",
        "predicted_value": 230.45,
        "lower_bound": 225.30,
        "upper_bound": 235.60
      }
    ]
    ```
    - `date`: Data da previsão.
    - `predicted_value`: Valor previsto.
    - `lower_bound`: Limite inferior do intervalo de confiança.
    - `upper_bound`: Limite superior do intervalo de confiança.

---

#### **2. POST /predict_next_31_days**
- **Descrição:** Realiza previsões para os próximos 31 dias úteis, com base na data atual.
- **Entrada:**
  - Mesmo formato do endpoint `/predict`.
- **Saída:**
  - Lista com previsões para os próximos 31 dias úteis, seguindo o mesmo formato da saída do `/predict`.

---

#### **3. POST /predict_from_last_date**
- **Descrição:** Realiza previsões para os próximos 7 dias úteis, com base na última data fornecida no histórico.
- **Entrada:**
  - JSON com a série histórica:
    ```json
    {
      "data": [
        {"date": "2024-11-20", "value": 229.0},
        {"date": "2024-11-21", "value": 228.52},
        {"date": "2024-11-22", "value": 229.87}
      ]
    }
    ```
  - **Campos:**
    - `date`: Data no formato `YYYY-MM-DD`.
    - `value`: Valor correspondente à data.

- **Saída:**
  - Lista com previsões para os próximos 7 dias úteis, começando imediatamente após a última data do histórico:
    ```json
    [
      {
        "date": "2024-11-23",
        "predicted_value": 230.45,
        "lower_bound": 225.30,
        "upper_bound": 235.60
      }
    ]
    ```

---

#### **4. POST /predict_next_year**
- **Descrição:** Realiza previsões para os próximos 365 dias úteis, com base na data atual.
- **Entrada:**
  - Mesmo formato do endpoint `/predict`.
- **Saída:**
  - Lista com previsões para os próximos 365 dias úteis, seguindo o mesmo formato da saída do `/predict`.

---

### **Monitoramento**
A API registra logs detalhados em um arquivo chamado `monitoring.log` na raiz do projeto. Os logs incluem:
- IP do cliente.
- Tempo de resposta da requisição.
- Status HTTP.
- Rota Acessada

**Exemplo de Log:**
```
2024-11-27 10:20:15 - INFO - Path: /predict, Method: POST, Status: 200, Time: 0.02s, IP: 127.0.0.1
```

---

### **Como Executar**
1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute a aplicação:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Acesse a documentação interativa:**
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Redoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### **Escalabilidade**
- **Docker:** Implemente contêineres para simplificar o deploy.
- **Kubernetes:** Gerencie múltiplas réplicas para lidar com grandes volumes de requisições.
- **Prometheus e Grafana:** Configure para monitorar métricas em tempo real.

---

### **Contato**
Para dúvidas ou sugestões:
- **E-mail:** rodrigo.siliunas12@gmail.com
- **GitHub:** https://github.com/RodrigoSiliunas

