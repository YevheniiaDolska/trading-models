import pandas as pd
import numpy as np
import tensorflow as tf
import time
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
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
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
import glob
from filterpy.kalman import KalmanFilter


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
nn_model_filename = 'flat_nn_model.h5'
log_file = 'training_log_flat_nn.txt'

network_name = "flat_neural_network"  # Имя модели
checkpoint_path_regular = f"checkpoints/{network_name}_checkpoint_epoch_{{epoch:02d}}.h5"
checkpoint_path_best = f"checkpoints/{network_name}_best_model.h5"



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

def validate_volume_confirmation(data):
    """
    Добавляет признаки подтверждения движений объемом.
    """
    # Объемное подтверждение тренда
    data['volume_trend_conf'] = np.where(
        (data['close'] > data['close'].shift(1)) & 
        (data['volume'] > data['volume'].rolling(20).mean()),
        1,
        np.where(
            (data['close'] < data['close'].shift(1)) & 
            (data['volume'] > data['volume'].rolling(20).mean()),
            -1,
            0
        )
    )
    
    # Сила объемного подтверждения
    data['volume_strength'] = (data['volume'] / 
                             data['volume'].rolling(20).mean() * 
                             data['volume_trend_conf'])
    
    # Накопление объема
    data['volume_accumulation'] = data['volume_trend_conf'].rolling(5).sum()
    
    return data

def remove_noise(data):
    """
    Улучшенная фильтрация шума с использованием фильтра Калмана.
    """
    # Kalman filter для сглаживания цены
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data['close'].iloc[0]], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 10
    kf.R = 5  # Настройка шума измерений
    kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])  # Настройка шума процесса
    
    smoothed_prices = []
    for price in data['close']:
        kf.predict()
        kf.update(price)
        smoothed_prices.append(float(kf.x[0]))
    
    # Используем сглаженные цены Калмана вместо простого скользящего среднего
    data['smoothed_close'] = smoothed_prices
    
    # Дополнительные фильтры для флэтового рынка
    data['price_volatility'] = data['close'].rolling(20).std()
    data['is_significant_move'] = (data['close'].pct_change().abs() > 
                                 data['price_volatility'] * 2)
    
    # Вычисляем "чистые" движения с учетом значимости
    data['clean_returns'] = np.where(
        data['is_significant_move'] & (data['is_anomaly'] == 0),
        data['smoothed_close'].pct_change(),
        0
    )
    
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

# Кастомная функция потерь для флэтового рынка, ориентированная на минимизацию волатильности и частоту сделок
def custom_profit_loss(y_true, y_pred):
    """
    Функция потерь для максимизации прибыли во флэтовом рынке.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    
    # Базовая разница
    diff = y_pred - y_true
    
    # Логарифмический фактор
    log_factor = tf.math.log1p(tf.abs(diff) + 1e-7)
    
    # Штраф за ложные сигналы во флэте
    false_signal_penalty = tf.where(
        y_true == 0,  # Реальный флэт
        tf.abs(y_pred) * 3.0,  # Сильный штраф за любой сигнал
        0.0
    )
    
    # Штраф за пропуск выхода из флэта
    missed_breakout_penalty = tf.where(
        y_true == 1,  # Реальный выход из флэта
        tf.abs(1 - y_pred) * 2.0,  # Штраф за пропуск
        0.0
    )
    
    # Прибыль и убыток
    gain = tf.math.maximum(diff, 0)
    loss = tf.math.abs(tf.math.minimum(diff, 0))
    
    # Итоговая функция потерь
    total_loss = tf.reduce_mean(
        loss * 2 + 
        log_factor * 1.5 + 
        false_signal_penalty + 
        missed_breakout_penalty - 
        gain * 0.8
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
        e = tf.math.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * tf.expand_dims(a, -1)
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


def get_historical_data(symbol, interval, start_date, end_date):
    logging.info(f"Начало загрузки данных для {symbol}, период: с {start_date} по {end_date}, интервал: {interval}")
    data = pd.DataFrame()
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


'''def aggregate_to_2min(data):
    """
    Агрегация данных с интервала 1 минута до 2 минут.
    """
    logging.info("Агрегация данных с интервала 1 минута до 2 минут")
    
    # Убедитесь, что временной индекс установлен
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data.set_index('timestamp', inplace=True)
        else:
            raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")

    # Агрегация данных
    aggregated_data = data.resample('2T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logging.info(f"Агрегация завершена, размер данных: {len(aggregated_data)} строк")
    return aggregated_data'''



def smooth_data(data, window=5):
    """
    Сглаживание временного ряда с помощью скользящего среднего.
    
    Parameters:
        data (pd.Series): Исходный временной ряд.
        window (int): Размер окна сглаживания.
        
    Returns:
        pd.Series: Сглаженные данные.
    """
    return data.rolling(window=window, min_periods=1).mean()



def extract_features(data):
    logging.info("Извлечение признаков для флэтового рынка")
    data = data.copy()
    
    # 1. Базовые метрики
    data['returns'] = data['close'].pct_change()
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # 2. Метрики диапазона для флэта
    data['range_width'] = data['high'] - data['low']
    data['range_stability'] = data['range_width'].rolling(10).std()
    data['range_ratio'] = data['range_width'] / data['range_width'].rolling(20).mean()
    data['price_in_range'] = (data['close'] - data['low']) / data['range_width']
    
    # 3. Быстрые индикаторы для HFT
    data['sma_3'] = SMAIndicator(data['close'], window=3).sma_indicator()
    data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema_8'] = data['close'].ewm(span=8, adjust=False).mean()
    data['clean_volatility'] = data['clean_returns'].rolling(20).std()
    
    # 4. Короткие осцилляторы
    data['rsi_3'] = RSIIndicator(data['close'], window=3).rsi()
    data['rsi_5'] = RSIIndicator(data['close'], window=5).rsi()
    stoch = StochasticOscillator(data['high'], data['low'], data['close'], window=5)
    data['stoch_k'] = stoch.stoch()
    data['stoch_d'] = stoch.stoch_signal()
    
    # 5. Волатильность малых периодов
    bb = BollingerBands(data['close'], window=10)
    data['bb_width'] = bb.bollinger_wband()
    data['bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    data['atr_5'] = AverageTrueRange(data['high'], data['low'], data['close'], window=5).average_true_range()
    
    # 6. Объемные показатели
    data['volume_ma'] = data['volume'].rolling(10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['volume_stability'] = data['volume'].rolling(10).std() / data['volume_ma']
    
    # 7. Индикаторы пробоя
    data['breakout_intensity'] = abs(data['close'] - data['close'].shift(1)) / data['range_width']
    data['false_breakout'] = (data['high'] > data['high'].shift(1)) & (data['close'] < data['close'].shift(1))
    
    # 8. Микро-паттерны
    data['micro_trend'] = np.where(
        data['close'] > data['close'].shift(1), 1,
        np.where(data['close'] < data['close'].shift(1), -1, 0)
    )
    data['micro_trend_change'] = (data['micro_trend'] != data['micro_trend'].shift(1)).astype(int)
    
    # 9. Целевая переменная для флэтового рынка
    volatility = data['returns'].rolling(20).std()
    avg_volatility = volatility.rolling(100).mean()
    
    data['target'] = np.where(
        (abs(data['returns'].shift(-1)) < 0.0002) &  # Малое изменение цены
        (data['volume_ratio'] < 1.2) &  # Нет всплесков объема
        (volatility < avg_volatility) &  # Низкая волатильность
        (data['range_ratio'] < 1.1) &   # Стабильный диапазон
        (data['breakout_intensity'] < 0.3), # Нет сильных пробоев
        1,  # Флэтовый сигнал
        0   # Не флэт
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


# Обучение нейронной сети для флэтового рынка

def build_flat_neural_network(data, model_filename):
    """
    Обучение нейросети для флэтового рынка.

    Parameters:
        data (pd.DataFrame): Данные для обучения.
        model_filename (str): Имя файла для сохранения модели.
        checkpoint_path (str): Путь к контрольным точкам.

    Returns:
        model: Обученная модель.
        scaler: Масштабатор данных.
    """
    # Создаем директории для чекпоинтов
    os.makedirs("checkpoints", exist_ok=True)
    
    network_name = "flat_neural_network"
    checkpoint_path_regular = f"checkpoints/{network_name}_checkpoint_epoch_{{epoch:02d}}.h5"
    checkpoint_path_best = f"checkpoints/{network_name}_best_model.h5"
    
    if os.path.exists("flat_neural_network.h5"):
        try:
            model = load_model("flat_neural_network.h5", custom_objects={"custom_profit_loss": custom_profit_loss})
            logging.info("Обученная модель загружена из 'flat_neural_network.h5'. Пропускаем обучение.")
            return model
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")
            logging.info("Начинаем обучение с нуля.")
    else:
        logging.info("Сохраненная модель не найдена. Начинаем обучение с нуля.")
    
    
    logging.info("Начало обучения нейросети для флэтового рынка")

    # Выделяем признаки и целевые переменные
    features = [col for col in data.columns if col not in ['target', 'symbol', 'timestamp'] 
               and pd.api.types.is_numeric_dtype(data[col])]
    
    # Логирование для отладки
    logging.info(f"Выбранные признаки: {features}")
    logging.info(f"Типы данных признаков:\n{data[features].dtypes}")
    
    try:
        X = data[features].astype(float).values
        y = data['target'].astype(float).values
        
        logging.info(f"Форма X: {X.shape}, тип данных X: {X.dtype}")
        logging.info(f"Форма y: {y.shape}, тип данных y: {y.dtype}")
        
        # Проверка на наличие невалидных значений
        X_isvalid = np.isfinite(X)
        if not X_isvalid.all():
            invalid_count = np.sum(~X_isvalid)
            logging.warning(f"Найдено {invalid_count} невалидных значений в данных")
            
        # Удаление NaN и бесконечностей
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        y = y[mask]
        
        logging.info(f"После очистки - форма X: {X.shape}, форма y: {y.shape}")
        
    except Exception as e:
        logging.error(f"Ошибка при подготовке данных: {e}")
        raise

    # Балансировка классов
    X_resampled, y_resampled = balance_classes(X, y)

    # Разделение на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Масштабирование данных
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_scaled = scaler.transform(X_val).reshape((X_val.shape[0], X_val.shape[1], 1))

    # Создание tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

    # Создание модели
    with strategy.scope():
        with strategy.scope():
            inputs = Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))
            
            # Первая ветвь - анализ ценовых микродвижений
            x1 = LSTM(256, return_sequences=True)(inputs)
            x1 = BatchNormalization()(x1)
            x1 = Dropout(0.3)(x1)
            
            # Вторая ветвь - анализ объемов и волатильности
            x2 = LSTM(256, return_sequences=True)(inputs)
            x2 = BatchNormalization()(x2)
            x2 = Dropout(0.3)(x2)
            
            # Третья ветвь - анализ рыночного контекста
            x3 = LSTM(256, return_sequences=True)(inputs)
            x3 = BatchNormalization()(x3)
            x3 = Dropout(0.3)(x3)
            
            # Объединение ветвей
            x = Add()([x1, x2, x3])
            
            # Основной LSTM для принятия решений
            x = LSTM(256, return_sequences=False)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            # Dense слои для финального анализа
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            
            outputs = Dense(1, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs, outputs)
        
        def flat_trading_metric(y_true, y_pred):
            true_range = tf.reduce_max(y_true) - tf.reduce_min(y_true)
            pred_range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
            range_error = tf.abs(true_range - pred_range)
            return range_error

        model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_profit_loss, metrics=[flat_trading_metric])

    # Попытка загрузить последний сохранённый промежуточный чекпоинт
    # Попытка загрузить последний сохранённый промежуточный чекпоинт
    latest_checkpoint = glob.glob(f"checkpoints/{network_name}_checkpoint_epoch_*.h5")
    
    if latest_checkpoint:
        try:
            model.load_weights(latest_checkpoint[-1])
            logging.info(f"Загружены веса из последнего регулярного чекпоинта: {latest_checkpoint[-1]}")
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
        monitor='flat_trading_metric',
        mode='min',
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(log_dir=f"logs/{time.time()}")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, mode='min')
    early_stopping = EarlyStopping(
        monitor='val_flat_trading_metric',  # Мониторим упущенную прибыль
        patience=10,                       # Меньше терпение для быстрой реакции
        restore_best_weights=True,
        mode='min'
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

    # Построение графика обучения
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.legend()
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Сохранение модели
    model.save(model_filename)
    logging.info(f"Модель сохранена в {model_filename}")
    return model, scaler


if __name__ == "__main__":
    # Инициализация стратегии (TPU или CPU/GPU)
    strategy = initialize_strategy()

    client = Client(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT','XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'TONUSDT', 'NEARUSDT']

    # Периоды флэтового рынка
    flat_periods = [
        {"start": "2019-01-01", "end": "2019-12-31"},  # Полный период консолидации 2019
        {"start": "2020-09-01", "end": "2020-12-31"}   # Предбычий флэт конца 2020
    ]

    # Преобразуем даты из строк в datetime объекты
    start_date = datetime.strptime(flat_periods[0]["start"], "%Y-%m-%d")
    end_date = datetime.strptime(flat_periods[0]["end"], "%Y-%m-%d")

    # Загрузка данных используя load_all_data вместо load_flat_data
    data = load_all_data(symbols, start_date, end_date, interval="1m")
    
    logging.info("Проверка наличия 'timestamp' в данных перед агрегацией")
    if 'timestamp' not in data.columns:
        logging.warning("'timestamp' отсутствует, проверяем индекс.")
        if isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = data.index
            logging.info("Индекс преобразован в колонку 'timestamp'.")
        else:
            raise ValueError("Колонка 'timestamp' отсутствует, и индекс не является DatetimeIndex.")

    # Извлечение признаков
    data = extract_features(data)

    # Удаление NaN
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Обучение модели
    model, scaler = build_flat_neural_network(
        data, 
        model_filename="flat_nn_model.h5"
    )
