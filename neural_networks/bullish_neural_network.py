import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Add, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import backend as K
from ta.trend import MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from binance.client import Client
from datetime import datetime, timedelta
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from imblearn.combine import SMOTETomek
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sys
from tensorflow.keras.backend import clear_session
import glob



# Инициализация GPU
def initialize_strategy():
    # Проверяем наличие GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            strategy = tf.distribute.MirroredStrategy()
            print(f'Running on {len(gpus)} GPUs')
        except RuntimeError as e:
            print(e)
    else:
        strategy = tf.distribute.get_strategy()
        print('Running on CPU')
    return strategy
    
# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Имя файла для сохранения модели
nn_model_filename = os.path.join(os.getcwd(),'bullish_nn_model.h5')
log_file = 'training_log_bullish_nn.txt'

def save_logs_to_file(log_message):
    with open(log_file, 'a') as log_f:
        log_f.write(f"{datetime.now()}: {log_message}\n")
        
def calculate_cross_coin_features(data_dict):
    """
    Рассчитывает межмонетные признаки для всех пар.
    
    Args:
        data_dict (dict): Словарь DataFrame'ов по каждой монете
    Returns:
        dict: Словарь DataFrame'ов с добавленными признаками
    """
    btc_data = data_dict['BTCUSDT']
    
    for symbol, df in data_dict.items():
        # Корреляции с BTC
        df['btc_corr'] = df['close'].rolling(30).corr(btc_data['close'])
        
        # Относительная сила к BTC
        df['rel_strength_btc'] = (df['close'].pct_change() - 
                                btc_data['close'].pct_change())
        
        # Бета к BTC
        df['beta_btc'] = (df['close'].pct_change().rolling(30).cov(
            btc_data['close'].pct_change()) / 
            btc_data['close'].pct_change().rolling(30).var())
        
        # Опережение/следование за BTC
        df['lead_lag_btc'] = df['close'].pct_change().shift(-1).rolling(10).corr(
            btc_data['close'].pct_change())
            
        data_dict[symbol] = df
        
    return data_dict

def detect_anomalies(data):
    """
    Детектирует и фильтрует аномальные свечи.
    """
    # Рассчитываем z-score для разных метрик
    data['volume_zscore'] = ((data['volume'] - data['volume'].rolling(100).mean()) / 
                            data['volume'].rolling(100).std())
    data['price_zscore'] = ((data['close'] - data['close'].rolling(100).mean()) / 
                           data['close'].rolling(100).std())
    data['range_zscore'] = (((data['high'] - data['low']) - 
                            (data['high'] - data['low']).rolling(100).mean()) / 
                           (data['high'] - data['low']).rolling(100).std())
    
    # Фильтруем экстремальные выбросы
    data['is_anomaly'] = ((abs(data['volume_zscore']) > 4) | 
                         (abs(data['price_zscore']) > 4) | 
                         (abs(data['range_zscore']) > 4))
    
    return data

def validate_volume_confirmation_bullish(data):
    """
    Добавляет признаки подтверждения движений объемом для бычьего рынка.
    """
    # Объемное подтверждение тренда - более чувствительное к росту
    data['volume_trend_conf'] = np.where(
        (data['close'] > data['close'].shift(1)) & 
        (data['volume'] > data['volume'].rolling(10).mean()),  # Уменьшенное окно для быстрой реакции
        2,  # Усиленный сигнал для роста
        np.where(
            (data['close'] < data['close'].shift(1)) & 
            (data['volume'] > data['volume'].rolling(20).mean()),
            -1,
            0
        )
    )
    
    # Сила объемного подтверждения - с акцентом на рост
    data['volume_strength'] = (data['volume'] / 
                             data['volume'].rolling(10).mean() *  # Меньшее окно
                             np.where(data['volume_trend_conf'] > 0, 1.5, 1.0) *  # Усиление растущих движений
                             data['volume_trend_conf'])
    
    # Накопление объема - с фокусом на последовательный рост
    data['volume_accumulation'] = data['volume_trend_conf'].rolling(3).sum()  # Меньшее окно
    
    return data


def remove_noise(data):
    """
    Улучшенная фильтрация шума.
    """
    # Kalman filter для сглаживания цены
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data['close'].iloc[0]], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 10
    kf.R = 5
    kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])
    
    smoothed_prices = []
    for price in data['close']:
        kf.predict()
        kf.update(price)
        smoothed_prices.append(float(kf.x[0]))
    
    data['smoothed_close'] = smoothed_prices
    
    # Вычисляем "чистые" движения
    data['clean_returns'] = (data['smoothed_close'].pct_change() * 
                           (1 - data['is_anomaly']))
    
    return data

def preprocess_market_data(data_dict):
    """
    Комплексная предобработка данных с учетом межмонетных взаимосвязей.
    """
    # Добавляем межмонетные признаки
    data_dict = calculate_cross_coin_features(data_dict)
    
    for symbol, df in data_dict.items():
        # Детектируем аномалии
        df = detect_anomalies(df)
        
        # Добавляем подтверждение объемом
        df = validate_volume_confirmation(df)
        
        # Фильтруем шум
        df = remove_noise(df)
        
        data_dict[symbol] = df
    
    return data_dict

# Кастомная функция потерь для бычьего рынка, ориентированная на прибыль
def custom_profit_loss(y_true, y_pred):
    """
    Универсальная функция потерь для максимизации прибыли и минимизации убытков.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)  # Преобразование y_true к float32

    # Разница между предсказанием и реальным значением
    diff = y_pred - y_true

    # Логарифмический фактор для усиления больших ошибок
    log_factor = tf.math.log1p(tf.abs(diff) + 1e-7)
    
    # Штраф за недооценку роста
    underestimation_penalty = tf.where(y_true > y_pred, (y_true - y_pred) ** 2, 0.0)

    # Штраф за переоценку падения
    overestimation_penalty = tf.where(y_true < y_pred, (y_pred - y_true) ** 2, 0.0)

    # Прибыль и убыток
    gain = tf.math.maximum(diff, 0)
    loss = tf.math.abs(tf.math.minimum(diff, 0))

    # Итоговая функция потерь
    total_loss = tf.reduce_mean(
        loss * 2 +               # Усиленный акцент на убытках
        log_factor * 1.5 +       # Выделение больших ошибок
        underestimation_penalty * 3 -  # Усиленный штраф за упущенную прибыль
        gain * 1.5 +             # Акцент на получении прибыли
        overestimation_penalty * 2    # Штраф за переоценку падения
    )

    return total_loss



# Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.math.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.math.reduce_sum(output, axis=1)


def load_all_data(symbols, start_date, end_date, interval='1m'):
    """
    Загружает данные для всех символов и добавляет межмонетные признаки.
    
    Args:
        symbols (list): Список торговых пар
        start_date (datetime): Начальная дата
        end_date (datetime): Конечная дата
        interval (str): Интервал свечей
    
    Returns:
        pd.DataFrame: Объединенные данные со всеми признаками
    """
    # Словарь для хранения данных по каждой монете
    symbol_data_dict = {}
    
    logging.info(f"Начало загрузки данных для символов: {symbols}")
    logging.info(f"Период: с {start_date} по {end_date}, интервал: {interval}")
    
    # Загрузка данных параллельно
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(get_historical_data, symbol, interval, start_date, end_date): symbol 
                  for symbol in symbols}
        
        for future in futures:
            symbol = futures[future]
            try:
                logging.info(f"Ожидание данных для {symbol}")
                symbol_data = future.result()
                
                if symbol_data is not None:
                    # Добавляем базовые признаки для каждой монеты
                    symbol_data = detect_anomalies(symbol_data)
                    symbol_data = validate_volume_confirmation(symbol_data)
                    symbol_data = remove_noise(symbol_data)
                    
                    symbol_data_dict[symbol] = symbol_data
                    logging.info(f"Данные для {symbol} успешно загружены и обработаны, количество строк: {len(symbol_data)}")
            except Exception as e:
                logging.error(f"Ошибка при загрузке данных для {symbol}: {e}")
                save_logs_to_file(f"Ошибка при загрузке данных для {symbol}: {e}")
    
    if not symbol_data_dict:
        error_msg = "Не удалось получить данные ни для одного символа"
        logging.error(error_msg)
        save_logs_to_file(error_msg)
        raise ValueError(error_msg)
    
    # Добавляем межмонетные признаки
    try:
        logging.info("Добавление межмонетных признаков...")
        symbol_data_dict = calculate_cross_coin_features(symbol_data_dict)
        
        # Объединяем все данные
        all_data = []
        for symbol, df in symbol_data_dict.items():
            df['symbol'] = symbol
            all_data.append(df)
        
        data = pd.concat(all_data)
        
        # Проверяем наличие всех необходимых признаков
        expected_features = ['btc_corr', 'rel_strength_btc', 'beta_btc', 'lead_lag_btc',
                           'volume_strength', 'volume_accumulation', 'is_anomaly', 
                           'clean_returns']
        
        missing_features = [f for f in expected_features if f not in data.columns]
        if missing_features:
            logging.warning(f"Отсутствуют следующие признаки: {missing_features}")
        
        # Удаляем строки с пропущенными значениями
        initial_rows = len(data)
        data = data.dropna()
        dropped_rows = initial_rows - len(data)
        if dropped_rows > 0:
            logging.info(f"Удалено {dropped_rows} строк с пропущенными значениями")
        
        logging.info(f"Всего загружено и обработано {len(data)} строк данных")
        save_logs_to_file(f"Всего загружено и обработано {len(data)} строк данных")
        
        return data
        
    except Exception as e:
        error_msg = f"Ошибка при обработке межмонетных признаков: {e}"
        logging.error(error_msg)
        save_logs_to_file(error_msg)
        raise

# Загрузка данных с Binance
def get_historical_data(symbol, interval, start_date, end_date):
    logging.info(f"Начало загрузки данных для {symbol}, период: с {start_date} по {end_date}, интервал: {interval}")
    data = pd.DataFrame()
    client = Client()
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=30), end_date)
        try:
            logging.info(f"Загрузка данных для {symbol}: с {current_date} по {next_date}")
            klines = client.get_historical_klines(
                symbol, interval,
                current_date.strftime('%d %b %Y %H:%M:%S'),
                next_date.strftime('%d %b %Y %H:%M:%S'),
                limit=1000
            )
            if not klines:
                logging.warning(f"Нет данных для {symbol} за период с {current_date} по {next_date}")
                continue

            # Преобразуем данные в DataFrame
            temp_data = pd.DataFrame(
                klines,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ]
            )

            # Преобразуем `timestamp`
            temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'], unit='ms')
            temp_data.set_index('timestamp', inplace=True)

            # Оставляем только нужные колонки
            temp_data = temp_data[['open', 'high', 'low', 'close', 'volume']].astype(float)

            if 'timestamp' not in temp_data.columns and not isinstance(temp_data.index, pd.DatetimeIndex):
                raise ValueError(f"Колонка 'timestamp' отсутствует в данных для {symbol} за период {current_date} - {next_date}")

            data = pd.concat([data, temp_data])

            logging.info(f"Данные успешно добавлены для {symbol}, текущий размер: {len(data)} строк")

        except Exception as e:
            logging.error(f"Ошибка при загрузке данных для {symbol} за период {current_date} - {next_date}: {e}")
            save_logs_to_file(f"Ошибка при загрузке данных для {symbol} за период {current_date} - {next_date}: {e}")

        current_date = next_date

    if data.empty:
        logging.warning(f"Все данные для {symbol} пусты.")
    return data


def load_bullish_data(symbols, bullish_periods, interval="1m"):
    all_data = []
    logging.info(f"Начало загрузки данных за бычьи периоды для символов: {symbols}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for period in bullish_periods:
            start_date = datetime.strptime(period["start"], "%Y-%m-%d")
            end_date = datetime.strptime(period["end"], "%Y-%m-%d")
            for symbol in symbols:
                futures.append(executor.submit(get_historical_data, symbol, interval, start_date, end_date))
        
        for future in futures:
            try:
                symbol_data = future.result()
                if symbol_data is not None:
                    # Проверяем наличие колонки `timestamp` или корректного индекса
                    if 'timestamp' not in symbol_data.columns and not isinstance(symbol_data.index, pd.DatetimeIndex):
                        raise ValueError(f"Данные для {symbol} за период {start_date} - {end_date} не содержат колонку 'timestamp'.")

                    all_data.append(symbol_data)
                    logging.info(f"Данные добавлены для символа {symbol}. Текущий размер: {len(symbol_data)} строк.")

            except Exception as e:
                logging.error(f"Ошибка загрузки данных: {e}")

    if all_data:
        # Конкатенируем данные
        data = pd.concat(all_data, ignore_index=False)
        logging.info(f"Всего загружено {len(data)} строк данных после конкатенации")
        return data
    else:
        logging.error("Не удалось загрузить данные.")
        raise ValueError("Не удалось загрузить данные.")


'''def aggregate_to_2min(data):
    """
    Агрегация данных с интервала 1 минута до 2 минут.
    """
    logging.info("Агрегация данных с интервала 1 минута до 2 минут")
    
    # Проверка, установлен ли временной индекс
    if not isinstance(data.index, pd.DatetimeIndex):
         # Проверка и установка временного индекса
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')  # Преобразуем в datetime
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")

        # Убедитесь, что индекс является DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Индекс данных не является DatetimeIndex даже после преобразования.")
        
    
    # Агрегация данных
    data = data.resample('2T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    logging.info(f"Агрегация завершена, размер данных: {len(data)} строк")
    logging.info(f"После агрегации данных на 2 минуты: NaN = {data.isna().sum().sum()}")
    return data'''


# Извлечение признаков
def extract_features(data):
    """
    Извлечение признаков для каждого символа отдельно.
    """
    if 'symbol' in data.columns:
        grouped = data.groupby('symbol')
        features = grouped.apply(_extract_features_per_symbol).reset_index(drop=True)
    else:
        features = _extract_features_per_symbol(data)
    
    return features


def _extract_features_per_symbol(data):
    """
    Извлечение признаков для одного символа.
    """
    data['market_type'] = 0
    logging.info("Извлечение признаков для бычьего рынка")
    data = data.copy()
    
    # 1. Базовые метрики доходности и импульса
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['momentum_1m'] = data['close'].diff(1)
    data['momentum_3m'] = data['close'].diff(3)
    data['acceleration'] = data['momentum_1m'].diff()
    
    # 2. Расчет динамических порогов на основе волатильности
    volatility = data['returns'].rolling(20).std()
    volume_volatility = data['volume'].pct_change().rolling(20).std()
    
    # Базовые пороги для бычьего рынка
    base_strong = 0.001  # 0.1%
    base_medium = 0.0005  # 0.05%
    
    # Корректировка порогов
    strong_threshold = base_strong * (1 + volatility/volatility.mean())
    medium_threshold = base_medium * (1 + volatility/volatility.mean())
    volume_factor = 1 + (volume_volatility/volume_volatility.mean())
    
    # 3. Объемные показатели для определения силы движения
    data['volume_delta'] = data['volume'].diff()
    data['volume_momentum'] = data['volume'].diff().rolling(3).sum()
    volume_ratio = data['volume'] / data['volume'].rolling(5).mean() * volume_factor
    
    # 4. Быстрые трендовые индикаторы для HFT
    data['sma_2'] = SMAIndicator(data['close'], window=2).sma_indicator()
    data['sma_3'] = SMAIndicator(data['close'], window=3).sma_indicator()
    data['ema_3'] = data['close'].ewm(span=3, adjust=False).mean()
    data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
    
    # 5. Микро-тренды для HFT
    data['micro_trend'] = np.where(
        data['close'] > data['close'].shift(1), 1,
        np.where(data['close'] < data['close'].shift(1), -1, 0)
    )
    data['micro_trend_strength'] = data['micro_trend'].rolling(3).sum()
    
    # 6. Быстрый MACD для 1-минутных свечей
    macd = MACD(data['close'], window_slow=12, window_fast=6, window_sign=3)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    data['macd_acceleration'] = data['macd_diff'].diff()
    
    # 7. Короткие осцилляторы
    data['rsi_5'] = RSIIndicator(data['close'], window=5).rsi()
    data['rsi_3'] = RSIIndicator(data['close'], window=3).rsi()
    stoch = StochasticOscillator(data['high'], data['low'], data['close'], window=5)
    data['stoch_k'] = stoch.stoch()
    data['stoch_d'] = stoch.stoch_signal()
    
    # 8. Волатильность для оценки рисков
    bb = BollingerBands(data['close'], window=10)
    data['bb_width'] = bb.bollinger_wband()
    data['atr_3'] = AverageTrueRange(data['high'], data['low'], data['close'], window=3).average_true_range()
    
    # 9. Свечной анализ
    data['candle_body'] = data['close'] - data['open']
    data['body_ratio'] = abs(data['candle_body']) / (data['high'] - data['low'])
    data['upper_wick'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_wick'] = np.minimum(data['open'], data['close']) - data['low']
    
    # 10. Объемные индикаторы
    data['obv'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    data['volume_trend'] = data['volume'].diff() / data['volume'].shift(1)
    
    # 11. Индикаторы ускорения для HFT
    data['price_acceleration'] = data['returns'].diff()
    data['volume_acceleration'] = data['volume_delta'].diff()
    
    # 12. Многоуровневая целевая переменная для бычьего рынка
    data['target'] = np.where(
        # Сильный сигнал: разворот вверх после коррекции
        (data['returns'].shift(-1) > 0.001) & 
        (data['close'] < data['sma_10']) &  # Цена ниже 10-периодной SMA
        (volume_ratio > 1.2) & 
        (data['rsi_5'] < 40),  # Перепроданность на RSI
        2,
        np.where(
            # Умеренный сигнал: локальный отскок
            (data['returns'].shift(-1) > 0.0005) & 
            (data['micro_trend_strength'] < 0) &  # Предшествующее падение
            (data['volume_trend_conf'] > 0),  # Подтверждение объемом
            1,
            0
        )
    )
    
     # Создаем словарь признаков
    features = {}
    
    # Базовые признаки (все, что уже рассчитано)
    for col in data.columns:
        if col not in ['target', 'market_type']:
            features[col] = data[col]
    
    # Добавляем межмонетные признаки, если они есть
    if 'btc_corr' in data.columns:
        features['btc_corr'] = data['btc_corr']
    if 'rel_strength_btc' in data.columns:
        features['rel_strength_btc'] = data['rel_strength_btc']
    if 'beta_btc' in data.columns:
        features['beta_btc'] = data['beta_btc']
            
    # Добавляем признаки подтверждения объемом, если они есть
    if 'volume_strength' in data.columns:
        features['volume_strength'] = data['volume_strength']
    if 'volume_accumulation' in data.columns:
        features['volume_accumulation'] = data['volume_accumulation']
    
    # Добавляем очищенные от шума признаки, если они есть
    if 'clean_returns' in data.columns:
        features['clean_returns'] = data['clean_returns']
        
    # Преобразуем признаки в DataFrame
    features_df = pd.DataFrame(features)
    
    return data.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]


def add_clustering_feature(data):
    features_for_clustering = [
        'close', 'volume', 'rsi', 'macd', 'atr', 'sma_10', 'sma_30', 'ema_50', 'ema_200',
        'bb_width', 'macd_diff', 'obv', 'returns', 'log_returns'
    ]
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[features_for_clustering])
    return data

def prepare_data(data):
    logging.info("Начало подготовки данных")

    # Проверка на пустые данные
    if data.empty:
        raise ValueError("Входные данные пусты")

    logging.info(f"Исходная форма данных: {data.shape}")

    # Убедимся, что временной индекс установлен
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Данные не содержат временного индекса или колонки 'timestamp'.")

    # Извлечение признаков
    data = extract_features(data)
    logging.info(f"После извлечения признаков: {data.shape}")

    # Удаление выбросов
    data = remove_outliers(data)
    logging.info(f"После удаления выбросов: {data.shape}")

    # Добавление кластеризационного признака
    data = add_clustering_feature(data)
    logging.info(f"После кластеризации: {data.shape}")

    # Список признаков
    features = [col for col in data.columns if col != 'target']
    logging.info(f"Количество признаков: {len(features)}")
    logging.info(f"Список признаков: {features}")
    logging.info(f"Распределение target:\n{data['target'].value_counts()}")

    return data, features


def clean_data(X, y):
    logging.info("Очистка данных от бесконечных значений и NaN")
    mask = np.isfinite(X).all(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    return X_clean, y_clean


def load_last_saved_model(model_filename):
    try:
        models = sorted(
            [f for f in os.listdir() if f.startswith(model_filename)],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        last_model = models[-1] if models else None
        if last_model:
            logging.info(f"Загрузка последней модели: {last_model}")
            return load_model(last_model)
        else:
            return None
    except Exception as e:
        logging.error(f"Ошибка при загрузке последней модели: {e}")
        return None



def balance_classes(X, y):
    logging.info("Начало балансировки классов")
    logging.info(f"Размеры данных до балансировки: X={X.shape}, y={y.shape}")
    logging.info(f"Уникальные классы в y: {np.unique(y, return_counts=True)}")

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Данные для балансировки пусты. Проверьте исходные данные и фильтры.")

    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    logging.info(f"Размеры данных после балансировки: X={X_resampled.shape}, y={y_resampled.shape}")
    return X_resampled, y_resampled


# Обучение нейронной сети
def build_bullish_neural_network(data):
    
    # Создаем директории для чекпоинтов
    os.makedirs("checkpoints", exist_ok=True)
    
    network_name = "bullish_neural_network"
    checkpoint_path_regular = f"checkpoints/{network_name}_checkpoint_epoch_{{epoch:02d}}.h5"
    checkpoint_path_best = f"checkpoints/{network_name}_best_model.h5"
    
    if os.path.exists("bullish_neural_network.h5"):
        try:
            model = load_model("bullish_neural_network.h5", custom_objects={"custom_profit_loss": custom_profit_loss})
            logging.info("Обученная модель загружена из 'bullish_neural_network.h5'. Пропускаем обучение.")
            return model
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")
            logging.info("Начинаем обучение с нуля.")
    else:
        logging.info("Сохраненная модель не найдена. Начинаем обучение с нуля.")
    
        
    logging.info("Начало обучения модели для бычьего рынка")

    # Подготовка данных
    features = [col for col in data.columns if col not in ['target']]
    X = data[features].values
    y = data['target'].values

    # Масштабирование данных
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Балансировка данных
    X_resampled, y_resampled = balance_classes(X_scaled, y)

    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Формирование последовательностей
    def create_sequences(X, y, timesteps=10):
        Xs, ys = [], []
        for i in range(len(X) - timesteps):
            Xs.append(X[i:(i + timesteps)])
            ys.append(y[i + timesteps])
        return np.array(Xs), np.array(ys)

    timesteps = 10  # Количество временных шагов
    time_weights = np.exp(np.linspace(-1, 0, timesteps))  # Экспоненциальные веса

    # Применение весов к последовательностям
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, timesteps)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, timesteps)

    # Применяем веса по временным шагам
    X_train_seq_weighted = X_train_seq * time_weights.reshape(1, -1, 1)
    X_val_seq_weighted = X_val_seq * time_weights.reshape(1, -1, 1)

    # Создание tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_seq, y_val_seq))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    # Использование стратегии
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        logging.info('TPU инициализирована')
    except ValueError:
        strategy = tf.distribute.get_strategy()
        logging.info('TPU недоступна, используем стратегию по умолчанию')

    with strategy.scope():
        # Создание модели
        inputs = Input(shape=(timesteps, X_train_seq.shape[2]))

        x = LSTM(256, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = LSTM(128, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = LSTM(64, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs, outputs)
        
        def bull_profit_metric(y_true, y_pred):
            # Метрика для оценки упущенной прибыли
            missed_gains = tf.where(y_true > y_pred, y_true - y_pred, 0)
            return tf.reduce_mean(missed_gains)

        model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_profit_loss, metrics=[bull_profit_metric])

        # Загрузка контрольной точки
        try:
            model.load_weights(checkpoint_path_regular.format(epoch=0))  # Используем определенный путь
            logging.info(f"Загружены веса модели из {checkpoint_path_regular.format(epoch=0)}")
        except FileNotFoundError:
            logging.info(f"Контрольная точка не найдена. Начало обучения с нуля.")
            
    logging.info("Проверка наличия чекпоинтов...")

    # Попытка загрузить последний сохранённый промежуточный чекпоинт
    regular_checkpoints = sorted(glob.glob(f"checkpoints/{network_name}_checkpoint_epoch_*.h5"))
    
    if regular_checkpoints:
        latest_checkpoint = regular_checkpoints[-1]
        try:
            model.load_weights(latest_checkpoint)
            logging.info(f"Загружены веса из последнего регулярного чекпоинта: {latest_checkpoint}")
        except Exception as e:
            logging.error(f"Ошибка загрузки регулярного чекпоинта: {e}")

    # Проверяем наличие лучшего чекпоинта
    if os.path.exists(checkpoint_path_best):
        try:
            model.load_weights(checkpoint_path_best)
            logging.info(f"Лучший чекпоинт найден: {checkpoint_path_best}. После обучения промежуточные чекпоинты будут удалены.")
        except:
            logging.info("Лучший чекпоинт пока не создан. Это ожидаемо, если обучение ещё не завершено.")
        
    # Настройка коллбеков
    
    # Чекпоинт для регулярного сохранения прогресса
    checkpoint_every_epoch = ModelCheckpoint(
        filepath=checkpoint_path_regular,
        save_weights_only=True,
        save_best_only=False,
        verbose=1
    )

    # Чекпоинт для сохранения только лучшей модели
    checkpoint_best_model = ModelCheckpoint(
        filepath=checkpoint_path_best,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(log_dir=f"logs/{time.time()}")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(
        monitor='val_bull_profit_metric',  # Мониторим упущенную прибыль
        patience=10,                       # Меньше терпение для быстрой реакции
        restore_best_weights=True
    )

    # Настройка весов классов для несбалансированных данных
    class_weights = {
        0: 1.0,  # Вес для бычьего рынка
        1: 3.0,  # Вес для медвежьего рынка
        2: 2.5   # Вес для флэтового рынка
    }

    # Обучение модели
    history = model.fit(
        train_dataset,
        epochs=500,  # Увеличено число эпох для более тщательного обучения
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=[
        early_stopping,
        checkpoint_every_epoch,
        checkpoint_best_model,
        tensorboard_callback,
        reduce_lr
    ]
)
    
    logging.info("Очистка промежуточных чекпоинтов...")
    for checkpoint in glob.glob(f"checkpoints/{network_name}_checkpoint_epoch_*.h5"):
        if checkpoint != checkpoint_path_best:  # Сохраняем лучшую модель
            os.remove(checkpoint)
            logging.info(f"Удалён чекпоинт: {checkpoint}")
    logging.info("Очистка завершена. Сохранена только лучшая модель.")

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.legend()
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    try:
        model.save(nn_model_filename)
        logging.info(f"Модель успешно сохранена в {nn_model_filename}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении модели: {e}")


    return model


# Основной процесс обучения
if __name__ == "__main__":
    try:
        # Инициализация стратегии (TPU или CPU/GPU)
        strategy = initialize_strategy()

        # Инициализация клиента Binance
        proxies = {
            'http': 'http://your-proxy.com:port',
            'https': 'http://your-proxy.com:port',
        }

        client = Client(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET", requests_params={"proxies": proxies})


        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT','XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'TONUSDT', 'NEARUSDT']
        
        bullish_periods = [
        {"start": "2017-12-01", "end": "2018-01-16"},  # Пик криптовалютного бума
        {"start": "2020-03-01", "end": "2021-01-31"},  # Пандемический рост
        {"start": "2021-01-01", "end": "2021-05-01"},  # Активный рост 2021
        {"start": "2023-01-01", "end": "2023-06-30"}   # Последний сильный рост
    ]


        # Интервал 1 минута
        interval = Client.KLINE_INTERVAL_1MINUTE

        # Загрузка данных для бычьих периодов
        logging.info("Загрузка данных для бычьих периодов")
        data = load_bullish_data(symbols, bullish_periods, interval=interval)

        # Проверка наличия колонки `timestamp`
        if 'timestamp' not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Загруженные данные не содержат колонку 'timestamp'. Проверьте этап загрузки.")

        # Извлечение признаков
        logging.info("Извлечение признаков из данных")
        data = extract_features(data)

        # Очистка данных
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Проверка наличия данных после очистки
        if data.empty:
            logging.error("После очистки данные отсутствуют. Проверьте входные данные.")
            raise ValueError("После очистки данные отсутствуют.")

        # Убедитесь, что пути к чекпоинтам определены
        checkpoint_path_regular = "checkpoints/tpu_checkpoint_epoch_{epoch:02d}.h5"
        checkpoint_path_best = "checkpoints/tpu_best_model.h5"

        # Обучение модели
        logging.info("Начало обучения модели для бычьего рынка")
        build_bullish_neural_network(data)

    except Exception as e:
        logging.error(f"Ошибка во время выполнения программы: {e}")
    finally:
        # Очистка сессии TensorFlow
        logging.info("Очистка сессии TensorFlow...")
        clear_session()  # Закрывает все графы и фоновые процессы TensorFlow

        # Корректное закрытие клиента Binance
        if client:
            try:
                client.close_connection()
                logging.info("Соединение с Binance успешно закрыто.")
            except Exception as close_error:
                logging.warning(f"Ошибка при закрытии соединения с Binance: {close_error}")

        logging.info("Программа завершена.")
