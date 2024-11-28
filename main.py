import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


def get_next_weekdays(start_date: datetime, num_days: int) -> list[datetime]:
    """
    Calcula os próximos dias úteis a partir de uma data inicial.

    Esta função ignora fins de semana (sábado e domingo) e retorna apenas
    dias úteis (segunda a sexta-feira).

    Args:
        start_date (datetime): A data inicial para o cálculo dos dias úteis.
        num_days (int): O número de dias úteis que deseja calcular.

    Returns:
        List[datetime]: Uma lista contendo os próximos `num_days` dias úteis
        a partir de `start_date`.

    Exemplo:
        >>> from datetime import datetime
        >>> get_next_weekdays(datetime(2024, 11, 27), 5)
        [datetime.datetime(2024, 11, 27, 0, 0),
         datetime.datetime(2024, 11, 28, 0, 0),
         datetime.datetime(2024, 11, 29, 0, 0),
         datetime.datetime(2024, 12, 2, 0, 0),
         datetime.datetime(2024, 12, 3, 0, 0)]
    """
    current_date = start_date
    weekdays = []

    while len(weekdays) < num_days:
        if current_date.weekday() < 5:
            weekdays.append(current_date)
        current_date += timedelta(days=1)

    return weekdays


class HistoricalData(BaseModel):
    date: str
    value: float


class PredictRequest(BaseModel):
    data: list[HistoricalData]


class PredictionResponse(BaseModel):
    date: str
    predicted_value: float
    lower_bound: float
    upper_bound: float

# Inicialização do FastAPI
app = FastAPI()

model_path = 'best_lstm_model_v4.keras'
model = tf.keras.models.load_model(model_path) 
RMSE = 6.14

def process_historical_data(data: list[HistoricalData]) -> pd.DataFrame:
    """
    Processa os dados históricos recebidos, garantindo que estejam no formato correto.
    Usa todos os valores enviados, garantindo um mínimo de 7 registros.

    Args:
        data (List[HistoricalData]): Dados históricos enviados pelo usuário.

    Returns:
        pd.DataFrame: DataFrame contendo os dados históricos processados.

    Raises:
        HTTPException: Caso os dados estejam incompletos ou inválidos.
    """
    # Converte os dados recebidos para DataFrame
    historical_data = pd.DataFrame([d.model_dump() for d in data])

    # Verifica se o DataFrame tem as colunas esperadas
    if ('date' not in historical_data.columns) or ('value' not in historical_data.columns):
        raise HTTPException(status_code=400, detail="Os dados devem conter as colunas 'date' e 'value'.")

    # Garante que a coluna 'date' está no formato datetime
    historical_data['date'] = pd.to_datetime(historical_data['date'])

    # Verifica se há pelo menos 7 dias de dados históricos
    if len(historical_data) < 7:
        raise HTTPException(
            status_code=400,
            detail="Os dados históricos fornecidos devem conter pelo menos 7 dias de informações."
        )

    # Ordena os dados por data
    historical_data = historical_data.sort_values('date')

    return historical_data



def make_predictions(historical_data: pd.DataFrame, days_to_predict: int) -> list[dict]:
    """
    Realiza as previsões com base nos dados históricos processados.

    Usa todos os valores históricos como entrada, garantindo um mínimo de 7 dias.

    Args:
        historical_data (pd.DataFrame): DataFrame com os dados históricos.
        days_to_predict (int): Número de dias a serem previstos.

    Returns:
        List[dict]: Lista de previsões com data, valor previsto e intervalo de confiança.
    """
    # Usa todos os valores históricos como entrada para a normalização
    input_values = historical_data['value'].values[-max(len(historical_data), 7):].reshape(-1, 1)

    # Normaliza os valores
    mean_value = input_values.mean()
    std_value = input_values.std()
    input_values_scaled = (input_values - mean_value) / std_value
    input_values_scaled = input_values_scaled.reshape(1, input_values.shape[0], 1)

    # Gera as datas futuras
    next_dates = [(datetime.now() + timedelta(days=i)) for i in range(1, days_to_predict + 1)]

    predictions = []
    for i in range(days_to_predict):
        # Faz a previsão
        predicted_value_scaled = model.predict(input_values_scaled)[0][0]

        # Reverte a normalização para obter o valor original
        predicted_value = (predicted_value_scaled * std_value) + mean_value
        predictions.append({
            "date": next_dates[i].strftime('%Y-%m-%d'),
            "predicted_value": round(predicted_value, 2),
            "lower_bound": round(predicted_value - RMSE, 2),
            "upper_bound": round(predicted_value + RMSE, 2)
        })

        # Atualiza os valores de entrada com o novo valor previsto (reescalonado)
        predicted_value_scaled = (predicted_value - mean_value) / std_value
        input_values_scaled = np.concatenate(
            (input_values_scaled[:, 1:, :], np.array([[[predicted_value_scaled]]])), axis=1
        )

    return predictions


# Configuração do logging
logging.basicConfig(
    filename='monitoring.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    encoding='utf-8'
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Captura o IP do cliente
    client_host = request.client.host  # IP do cliente
    forwarded_for = request.headers.get('X-Forwarded-For')
    ip = forwarded_for.split(',')[0] if forwarded_for else client_host

    response = await call_next(request)
    process_time = time.time() - start_time

    logging.info(f"Path: {request.url.path}, Method: {request.method}, Status: {response.status_code}, Time: {process_time:.2f}s, IP: {ip}")
    return response

@app.post(
    "/predict",
    response_model=list[PredictionResponse],
    summary="Realiza previsões para os próximos 7 dias úteis",
    description="Recebe dados históricos, prevê os próximos 7 dias úteis e retorna os resultados."
)
def predict(data: PredictRequest):
    try:
        # Processa os dados históricos
        historical_data = process_historical_data(data.data)

        # Faz as previsões
        predictions = make_predictions(historical_data, days_to_predict=7)

        results = []
        for prediction in predictions:
            # Acessa os campos do dicionário
            predicted_value = prediction['predicted_value']
            lower_bound = prediction['lower_bound']
            upper_bound = prediction['upper_bound']

            # Adiciona ao resultado formatado
            results.append(PredictionResponse(
                date=prediction['date'],
                predicted_value=round(predicted_value, 2),
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2)
            ))

        # Retorna as previsões em JSON
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post(
    "/predict_from_last_date",
    response_model=list[PredictionResponse],
    summary="Realiza previsões para os próximos 7 dias úteis com base na última data enviada",
    description=(
        "Recebe dados históricos, identifica a última data enviada e prevê os próximos 7 dias úteis "
        "baseados nessa data."
    )
)
def predict_from_last_date(data: PredictRequest):
    """
    Rota para realizar previsões dos próximos 7 dias úteis com base na última
    data fornecida no histórico enviado.

    Args:
        data (PredictRequest): Dados históricos no formato JSON.

    Returns:
        list[PredictionResponse]: Lista com as previsões para os próximos 7 dias úteis.
    """
    try:
        # Processa os dados históricos
        # Processa os dados históricos
        historical_data = process_historical_data(data.data)

        # Determina a última data fornecida nos dados históricos
        last_date = historical_data['date'].iloc[-1]

        # Calcula os próximos 7 dias úteis a partir da última data enviada
        next_dates = get_next_weekdays(last_date + timedelta(days=1), 7)

        # Faz as previsões
        predictions = make_predictions(historical_data, days_to_predict=7)

        results = []
        for date, prediction in zip(next_dates, predictions):
            # Acessa os campos do dicionário
            predicted_value = prediction['predicted_value']
            lower_bound = prediction['lower_bound']
            upper_bound = prediction['upper_bound']

            # Adiciona ao resultado formatado
            results.append(PredictionResponse(
                date=date.strftime('%Y-%m-%d'),
                predicted_value=round(predicted_value, 2),
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2)
            ))

        # Retorna as previsões em JSON
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict_next_31_days",
    response_model=list[PredictionResponse],
    summary="Realiza previsões para os próximos 31 dias corridos",
    description=(
        "Recebe pelo menos 7 dias de dados históricos (data e valor) em formato JSON, normaliza os valores, "
        "faz previsões para os próximos 31 dias corridos usando um modelo de Machine Learning "
        "e retorna as previsões com intervalo de confiança (±RMSE)."
    ),
)
def predict_next_31_days(data: PredictRequest):
    """
    Realiza previsões para os próximos 31 dias corridos com base em dados históricos fornecidos.
    """
    try:
        # Processa os dados históricos
        historical_data = process_historical_data(data.data)

        # Faz as previsões
        predictions = make_predictions(historical_data, days_to_predict=31)

        results = []
        for prediction in predictions:
            # Acessa os campos do dicionário
            predicted_value = prediction['predicted_value']
            lower_bound = prediction['lower_bound']
            upper_bound = prediction['upper_bound']

            # Adiciona ao resultado formatado
            results.append(PredictionResponse(
                date=prediction['date'],
                predicted_value=round(predicted_value, 2),
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2)
            ))

        # Retorna as previsões em JSON
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict_next_year",
    response_model=list[PredictionResponse],
    summary="Realiza previsões para os próximos 365 dias corridos",
    description=(
        "Recebe pelo menos 7 dias de dados históricos (data e valor) em formato JSON, normaliza os valores, "
        "faz previsões para os próximos 365 dias corridos usando um modelo de Machine Learning "
        "e retorna as previsões com intervalo de confiança (±RMSE)."
    ),
)
def predict_next_year(data: PredictRequest):
    """
    Realiza previsões para os próximos 365 dias corridos com base em dados históricos fornecidos.
    """
    try:
        # Processa os dados históricos
        historical_data = process_historical_data(data.data)

        # Faz as previsões
        predictions = make_predictions(historical_data, days_to_predict=365)

        results = []
        for prediction in predictions:
            # Acessa os campos do dicionário
            predicted_value = prediction['predicted_value']
            lower_bound = prediction['lower_bound']
            upper_bound = prediction['upper_bound']

            # Adiciona ao resultado formatado
            results.append(PredictionResponse(
                date=prediction['date'],
                predicted_value=round(predicted_value, 2),
                lower_bound=round(lower_bound, 2),
                upper_bound=round(upper_bound, 2)
            ))

        # Retorna as previsões em JSON
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
