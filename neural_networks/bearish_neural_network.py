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
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
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
nn_model_filename = os.path.join(os.getcwd(),'bearish_nn_model.h5')
log_file = 'training_log_bearish_nn.txt'

network_name = "bearish_neural_network"  # Имя модели
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
    data['volume_zscore'] = ((data['volume'] - data['volume'].rolling(50).mean()) / 
                            data['volume'].rolling(50).std())
    data['price_zscore'] = ((data['close'] - data['close'].rolling(50).mean()) / 
                           data['close'].rolling(50).std())
    
    # Более строгие критерии для падений
    data['is_anomaly'] = ((abs(data['volume_zscore']) > 3) & (data['close'] < data['close'].shift(1)) | 
                         (abs(data['price_zscore']) > 3))
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
    Улучшенная фильтрация шума с комбинацией фильтра Калмана и экспоненциального сглаживания.
    """
    # Создаем копию данных
    data = data.copy()
    
    # Kalman filter для сглаживания цены
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[data['close'].iloc[0]], [0.]])
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 10
    kf.R = 5
    kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])
    
    # Применяем фильтр Калмана
    smoothed_prices = []
    for price in data['close']:
        kf.predict()
        kf.update(price)
        smoothed_prices.append(float(kf.x[0]))
    
    # Используем фильтр Калмана для основного сглаживания
    data['smoothed_close'] = smoothed_prices
    
    # Дополнительное сглаживание EMA для определения выбросов
    ema_smooth = data['close'].ewm(span=10, min_periods=1, adjust=False).mean()
    
    # Вычисляем стандартное отклонение для определения выбросов
    rolling_std = data['close'].rolling(window=20).std()
    rolling_mean = data['close'].rolling(window=20).mean()
    
    # Определяем выбросы (z-score > 3)
    data['is_anomaly'] = abs(data['close'] - rolling_mean) > (3 * rolling_std)
    data['is_anomaly'] = data['is_anomaly'].astype(int)  # Преобразуем в 0 и 1
    
    # Вычисляем "чистые" движения с учетом аномалий
    data['clean_returns'] = data['smoothed_close'].pct_change() * (1 - data['is_anomaly'])
    
    # Заполняем NaN значения
    data = data.ffill().bfill()
    
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
    

# Кастомная функция потерь для медвежьего рынка, ориентированная на минимизацию убытков
def custom_profit_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    
    # Основная разница
    diff = y_pred - y_true
    
    # Логарифмический фактор для усиления больших ошибок (оставляем эту важную часть)
    log_factor = tf.math.log1p(tf.abs(diff) + 1e-7)
    
    # Штрафы для медвежьего рынка на минутном таймфрейме
    false_long_penalty = 2.5  # Сильный штраф за ложные сигналы покупки
    false_short_penalty = 1.5  # Умеренный штраф за ложные сигналы продажи
    missed_drop_penalty = 2.0  # Штраф за пропуск сильного падения
    
    # Расчет потерь с учетом специфики медвежьего рынка
    loss = tf.where(
        tf.logical_and(y_true == 0, y_pred > 0.5),  # Ложный сигнал на покупку
        false_long_penalty * tf.abs(diff) * log_factor,  # Умножаем на log_factor
        tf.where(
            tf.logical_and(y_true == 2, y_pred < 0.5),  # Пропуск сильного падения
            missed_drop_penalty * tf.abs(diff) * log_factor,  # Умножаем на log_factor
            tf.where(
                tf.logical_and(y_true == 0, y_pred < 0.2),  # Ложный сигнал на продажу
                false_short_penalty * tf.abs(diff) * log_factor,  # Умножаем на log_factor
                tf.abs(diff) * log_factor  # Умножаем на log_factor
            )
        )
    )
    
    # Добавляем штраф за неуверенные предсказания
    uncertainty_penalty = tf.where(
        tf.logical_and(y_pred > 0.3, y_pred < 0.7),
        0.5 * tf.abs(diff) * log_factor,  # Умножаем на log_factor
        0.0
    )
    
    # Добавить штраф за задержку реакции
    time_penalty = 0.1 * tf.abs(diff) * tf.cast(tf.range(tf.shape(diff)[0]), tf.float32) / tf.cast(tf.shape(diff)[0], tf.float32)
    
    # Добавить компонент для учета транзакционных издержек
    transaction_cost = 0.001 * tf.reduce_sum(tf.abs(y_pred[1:] - y_pred[:-1]))
    
    total_loss = tf.reduce_mean(loss + uncertainty_penalty)
    return total_loss

# Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        return K.sum(output, axis=1)


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
            klines = client.get_historical_klines(symbol, interval, current_date.strftime('%d %b %Y %H:%M:%S'), next_date.strftime('%d %b %Y %H:%M:%S'), limit=1000)
            if not klines:
                logging.warning(f"Нет данных для {symbol} за период с {current_date} по {next_date}")
                continue
            temp_data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'], unit='ms')
            temp_data.set_index('timestamp', inplace=True)
            data = pd.concat([data, temp_data[['open', 'high', 'low', 'close', 'volume']].astype(float)])
            logging.info(f"Данные успешно добавлены для {symbol}. Размер текущих данных: {len(data)} строк.")
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных для {symbol} за период {current_date} - {next_date}: {e}")
            save_logs_to_file(f"Ошибка при загрузке данных для {symbol} за период {current_date} - {next_date}: {e}")
        current_date = next_date

    if data.empty:
        logging.warning(f"Данные для {symbol} пусты.")
    return data

def load_bearish_data(symbols, bearish_periods, interval="1m"):
    all_data = []
    logging.info(f"Начало загрузки данных за медвежьи периоды для символов: {symbols}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for period in bearish_periods:
            start_date = datetime.strptime(period["start"], "%Y-%m-%d")
            end_date = datetime.strptime(period["end"], "%Y-%m-%d")
            for symbol in symbols:
                futures.append(executor.submit(get_historical_data, symbol, interval, start_date, end_date))
        
        for future in futures:
            try:
                symbol_data = future.result()
                if symbol_data is not None:
                    all_data.append(symbol_data)
            except Exception as e:
                logging.error(f"Ошибка загрузки данных: {e}")

    if all_data:
        data = pd.concat(all_data, ignore_index=False)
        logging.info(f"Всего загружено {len(data)} строк данных")
        return data
    else:
        logging.error("Не удалось загрузить данные.")
        raise ValueError("Не удалось загрузить данные.")

'''def aggregate_to_2min(data):
    """
    Агрегация данных с интервала 1 минута до 2 минут.
    
    Parameters:
        data (pd.DataFrame): Исходные данные с временной меткой в колонке 'timestamp'.
    
    Returns:
        pd.DataFrame: Агрегированные данные.
    """
    # Проверяем наличие 'timestamp'
    if 'timestamp' not in data.columns:
        logging.error("Колонка 'timestamp' отсутствует в данных для агрегации.")
        if isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = data.index
            logging.info("Индекс преобразован в колонку 'timestamp'.")
        else:
            raise ValueError("Колонка 'timestamp' отсутствует в данных.")

    # Убедитесь, что временной индекс установлен
    data['timestamp'] = pd.to_datetime(data['timestamp'])  # Убедитесь, что временные метки в формате datetime
    data = data.set_index('timestamp')

    # Агрегация данных
    data = data.resample('2T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()  # Удаление строк с пропущенными значениями

    logging.info(f"Агрегация завершена. Размер данных: {len(data)} строк.")
    return data'''


def adjust_target(data, threshold=-0.0005, trend_window=50):
    """
    Изменение целевой переменной для акцентирования на резких падениях.
    
    Parameters:
        data (pd.DataFrame): Данные с колонкой 'returns'.
        threshold (float): Порог падения (например, -0.05 для падений > 5%).
        
    Returns:
        pd.DataFrame: Обновленные данные с колонкой 'target'.
    """
    data['target'] = (data['returns'] < threshold).astype(int)
    data['trend'] = (data['close'] < data['close'].rolling(trend_window).mean()).astype(int)
    data['target'] = np.where(data['target'] + data['trend'] > 0, 1, 0)
    logging.info(f"Целевая переменная обновлена: {data['target'].value_counts().to_dict()}")
    return data

# Извлечение признаков
def extract_features(data):
    logging.info("Извлечение признаков для медвежьего рынка")
    data = data.copy()
    data = remove_noise(data)

    # 1. Улучшенная целевая переменная с градацией силы сигналов
    returns = data['close'].pct_change()
    volume_ratio = data['volume'] / data['volume'].rolling(10).mean()
    price_acceleration = returns.diff()  # Скорость изменения цены
    
    # 2. Динамические пороги на основе волатильности
    def calculate_dynamic_thresholds(window=10):
        volatility = returns.rolling(window).std()
        avg_volatility = volatility.rolling(100).mean()  # Долгосрочная средняя волатильность
        volatility_ratio = volatility / avg_volatility
        
        # Базовые пороги для 1-минутных свечей на медвежьем рынке
        base_strong = -0.001  # 0.1%
        base_medium = -0.0005  # 0.05%
        
        # Адаптация порогов
        strong_threshold = base_strong * np.where(
            volatility_ratio > 1.5, 1.5,  # Ограничиваем максимальную адаптацию
            np.where(volatility_ratio < 0.5, 0.5, volatility_ratio)
        )
        medium_threshold = base_medium * np.where(
            volatility_ratio > 1.5, 1.5,
            np.where(volatility_ratio < 0.5, 0.5, volatility_ratio)
        )
        
        return strong_threshold, medium_threshold

    # 3. Создание улучшенной целевой переменной
    strong_threshold, medium_threshold = calculate_dynamic_thresholds()
    
    # Учитываем не только цену, но и объем и скорость падения
    data['target'] = np.where(
        (returns.shift(-1) < strong_threshold) & 
        (volume_ratio > 1.2) & 
        (price_acceleration < 0) &
        (data['volume'] > data['volume'].rolling(20).mean()), 
        2,  # Сильный сигнал
        np.where(
            (returns.shift(-1) < medium_threshold) & 
            (volume_ratio > 1) & 
            (price_acceleration < 0),
            1,  # Средний сигнал
            0   # Нет сигнала
        )
    )

    # 2. Базовые характеристики
    data['returns'] = returns
    data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

    # 3. Анализ объемов и давления продаж (специфика медвежьего рынка)
    data['volume_ma'] = data['volume'].rolling(10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['selling_pressure'] = data['volume'] * (data['close'] - data['open']).abs() * \
                              np.where(data['close'] < data['open'], 1, 0)
    data['buying_pressure'] = data['volume'] * (data['close'] - data['open']).abs() * \
                             np.where(data['close'] > data['open'], 1, 0)
    data['pressure_ratio'] = data['selling_pressure'] / data['buying_pressure'].replace(0, 1)

    # 4. Динамические индикаторы волатильности
    data['volatility'] = returns.rolling(10).std()
    data['volatility_ma'] = data['volatility'].rolling(20).mean()
    data['volatility_ratio'] = data['volatility'] / data['volatility_ma']

    # 5. Трендовые индикаторы с адаптивными периодами
    for period in [3, 5, 8, 13, 21]:  # Числа Фибоначчи для лучшего охвата
        data[f'sma_{period}'] = SMAIndicator(data['smoothed_close'], window=period).sma_indicator()
        data[f'ema_{period}'] = data['smoothed_close'].ewm(span=period, adjust=False).mean()

    # 6. MACD с настройкой под минутные свечи
    macd = MACD(data['smoothed_close'], window_slow=26, window_fast=12, window_sign=9)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    data['macd_slope'] = data['macd_diff'].diff()  # Скорость изменения MACD

    # 7. Объемные индикаторы с фокусом на медвежий рынок
    data['obv'] = OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    data['cmf'] = ChaikinMoneyFlowIndicator(data['high'], data['low'], data['close'], data['volume']).chaikin_money_flow()
    data['volume_change'] = data['volume'].pct_change()
    data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

    # 8. Осцилляторы с адаптивными периодами
    for period in [7, 14, 21]:
        data[f'rsi_{period}'] = RSIIndicator(data['close'], window=period).rsi()
    data['stoch_k'] = StochasticOscillator(data['high'], data['low'], data['close'], window=7).stoch()
    data['stoch_d'] = StochasticOscillator(data['high'], data['low'], data['close'], window=7).stoch_signal()
    
    # 9. Индикаторы уровней поддержки/сопротивления
    data['support_level'] = data['low'].rolling(20).min()
    data['resistance_level'] = data['high'].rolling(20).max()
    data['price_to_support'] = data['close'] / data['support_level']
    
    # 10. Паттерны свечей и их силы
    data['candle_body'] = abs(data['close'] - data['open'])
    data['upper_shadow'] = data['high'] - np.maximum(data['close'], data['open'])
    data['lower_shadow'] = np.minimum(data['close'], data['open']) - data['low']
    data['body_to_shadow_ratio'] = data['candle_body'] / (data['upper_shadow'] + data['lower_shadow']).replace(0, 0.001)

    # 11. Ценовые уровни и их прорывы
    data['price_level_breach'] = np.where(
        data['close'] < data['support_level'].shift(1), -1,
        np.where(data['close'] > data['resistance_level'].shift(1), 1, 0)
    )

    # 12. Индикаторы скорости движения
    data['price_acceleration'] = returns.diff()
    data['volume_acceleration'] = data['volume_change'].diff()
    
    # 13. Волатильность (с адаптивными периодами)
    bb = BollingerBands(data['smoothed_close'], window=20)
    data['bb_high'] = bb.bollinger_hband()
    data['bb_low'] = bb.bollinger_lband()
    data['bb_width'] = bb.bollinger_wband()
    data['bb_position'] = (data['close'] - data['bb_low']) / (data['bb_high'] - data['bb_low'])
    
    # 14. ATR с разными периодами для оценки волатильности
    for period in [5, 10, 20]:
        data[f'atr_{period}'] = AverageTrueRange(
            data['high'], data['low'], data['close'], window=period
        ).average_true_range()

    # 4. Добавляем специальные признаки для высокочастотной торговли
    data['micro_trend'] = np.where(
        data['smoothed_close'] > data['smoothed_close'].shift(1), 1,
        np.where(data['smoothed_close'] < data['smoothed_close'].shift(1), -1, 0)
    )
    
    # Считаем микро-тренды последних 5 минут
    data['micro_trend_sum'] = data['micro_trend'].rolling(5).sum()
    
    # Добавляем признак ускорения объемов
    data['volume_acceleration_5m'] = (
        data['volume'].diff() / data['volume'].rolling(5).mean()
    ).fillna(0)

    # 5. Признаки для определения силы медвежьего движения
    data['bearish_strength'] = np.where(
        (data['close'] < data['open']) &  # Медвежья свеча
        (data['volume'] > data['volume'].rolling(20).mean() * 1.5) &  # Объем выше среднего
        (data['close'] == data['low']) &  # Закрытие на минимуме
        (data['clean_returns'] < 0),  # Используем очищенные returns
        3,  # Сильное медвежье движение
        np.where(
            (data['close'] < data['open']) &
            (data['volume'] > data['volume'].rolling(20).mean()) &
            (data['clean_returns'] < 0),
            2,  # Среднее медвежье движение
            np.where(data['close'] < data['open'], 1, 0)  # Слабое/нет движения
        )
    )
    
    # Создаем словарь признаков
    features = {}
    
    features['target'] = data['target']  # Добавляем целевую переменную
    
    # Базовые признаки (все, что уже рассчитано)
    for col in data.columns:
        if col not in ['market_type']:
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
    
    # Добавить в конец функции перед return
    logging.info(f"Количество признаков: {len(features_df.columns)}")
    logging.info(f"Проверка на NaN: {features_df.isna().sum().sum()}")
    logging.info(f"Распределение целевой переменной:\n{features_df['target'].value_counts()}")


    return features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()


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
    
    # Список признаков - ИСКЛЮЧАЕМ timestamp и другие нечисловые колонки
    features = [col for col in data.columns if col not in ['target', 'timestamp'] and pd.api.types.is_numeric_dtype(data[col])]
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



def build_bearish_neural_network(data):
    """
    Обучение нейронной сети для медвежьего рынка.

    Аргументы:
        data (pd.DataFrame): Данные с признаками и целевым значением ('target').

    Возвращает:
        model: Обученная модель.
    """
    # Создаем директории для чекпоинтов
    os.makedirs("checkpoints", exist_ok=True)
    
    network_name = "bearish_neural_network"
    checkpoint_path_regular = f"checkpoints/{network_name}_checkpoint_epoch_{{epoch:02d}}.h5"
    checkpoint_path_best = f"checkpoints/{network_name}_best_model.h5"
    
    if os.path.exists("bearish_neural_network.h5"):
        try:
            model = load_model("bearish_neural_network.h5", custom_objects={"custom_profit_loss": custom_profit_loss})
            logging.info("Обученная модель загружена из 'bearish_neural_network.h5'. Пропускаем обучение.")
            return model
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {e}")
            logging.info("Начинаем обучение с нуля.")
    else:
        logging.info("Сохраненная модель не найдена. Начинаем обучение с нуля.")
    
        
    logging.info("Начало подготовки данных для обучения нейронной сети.")

    # Выделяем признаки и целевое значение
    features = [col for col in data.columns if col not in ['target', 'timestamp'] and pd.api.types.is_numeric_dtype(data[col])]
    X = data[features].values
    y = data['target'].values

    # Логируем начальные размеры данных
    logging.info(f"Размер X до фильтрации: {X.shape}")
    logging.info(f"Размер y до фильтрации: {y.shape}")
    logging.info(f"Уникальные значения y: {np.unique(y, return_counts=True)}")
    
    # Убеждаемся, что все данные числовые
    X = X.astype(float)
    y = y.astype(int)

    # Проверяем, есть ли NaN и удаляем их
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    if X.size == 0 or y.size == 0:
        logging.error("X или y пусты после удаления NaN. Проверьте обработку данных.")
        raise ValueError("X или y пусты после удаления NaN.")

    logging.info(f"Размер X после фильтрации: {X.shape}")
    logging.info(f"Размер y после фильтрации: {y.shape}")

    # Балансировка классов
    try:
        X_resampled, y_resampled = balance_classes(X, y)
        logging.info(f"Размеры после балансировки: X_resampled={X_resampled.shape}, y_resampled={y_resampled.shape}")
        logging.info(f"Уникальные значения в y_resampled: {np.unique(y_resampled, return_counts=True)}")
    except ValueError as e:
        logging.error(f"Ошибка при балансировке классов: {e}")
        raise

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    logging.info(f"Размеры тренировочных данных: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Размеры валидационных данных: X_val={X_val.shape}, y_val={y_val.shape}")

    # Масштабирование данных
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Добавление временного измерения для LSTM
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))

    # Создание tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)
    
    
    def hft_metrics(y_true, y_pred):
        # Метрика для оценки скорости реакции
        reaction_time = tf.reduce_mean(tf.abs(y_pred[1:] - y_pred[:-1]))
        # Метрика для оценки стабильности сигналов
        signal_stability = tf.reduce_mean(tf.abs(y_pred[2:] - 2*y_pred[1:-1] + y_pred[:-2]))
        return reaction_time, signal_stability
    
    # Инициализация модели
    with strategy.scope():
        inputs = Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))
        
        # Первая ветвь - анализ паттернов цены
        x1 = LSTM(256, return_sequences=True)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)
        
        # Вторая ветвь - анализ объемов и волатильности
        x2 = LSTM(256, return_sequences=True)(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Третья ветвь - анализ рыночного контекста
        x3 = LSTM(256, return_sequences=True, name='market_context')(inputs)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.3)(x3)
        
        # Объединение всех трех ветвей
        x = Add()([x1, x2, x3])
        
        # Основной LSTM для принятия решений
        x = LSTM(256, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Dense слои для финального анализа
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Выходной слой с сигмоидой для вероятности падения
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.models.Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=custom_profit_loss,
            metrics=[hft_metrics]  # Добавляем новые метрики
        )

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
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_hft_metrics',  # Мониторим упущенную прибыль
        patience=15,                       # Меньше терпение для быстрой реакции
        restore_best_weights=True,
        mode='min'
    )
    

    # Настройка весов классов для несбалансированных данных
    class_weights = {
        0: 1.0,    # Нет сигнала
        1: 2.0,    # Умеренное падение
        2: 3.0     # Сильное падение
    }
    
    # Добавляем специальные метрики для оценки во время обучения
    def profit_ratio(y_true, y_pred):
        successful_shorts = tf.reduce_sum(tf.where(
            tf.logical_and(y_true >= 1, y_pred >= 0.5),
            1.0, 0.0
        ))
        false_signals = tf.reduce_sum(tf.where(
            tf.logical_and(y_true == 0, y_pred >= 0.5),
            1.0, 0.0
        ))
        return successful_shorts / (false_signals + K.epsilon())

    
    # Обучение модели
    history = model.fit(
        train_dataset,
        epochs=500,
        validation_data=val_dataset,
        class_weight=class_weights,
        verbose=1,
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
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


    # Сохранение модели
    model.save(nn_model_filename)
    logging.info(f"Модель сохранена в {nn_model_filename}")
    save_logs_to_file(f"Модель сохранена в {nn_model_filename}")

    return model


# Основной процесс обучения
if __name__ == "__main__":
    
    # Инициализация стратегии (TPU или CPU/GPU)
    strategy = initialize_strategy()

    # Инициализация клиента Binance
    proxies = {
        'http': 'http://your-proxy.com:port',
        'https': 'http://your-proxy.com:port',
    }

    client = Client(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET", requests_params={"proxies": proxies})

    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT','XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'TONUSDT', 'NEARUSDT']
    
    bearish_periods = [
    {"start": "2018-01-17", "end": "2018-12-31"},  # Пост-пик 2018 года, затяжной спад.
    {"start": "2022-01-01", "end": "2022-12-31"},  # Годичный медвежий рынок 2022 года.
]

    # Загружаем данные и агрегируем их
    data = load_bearish_data(symbols, bearish_periods, interval="1m")
    
    # Логирование проверки колонок
    logging.info(f"Колонки в данных после загрузки: {data.columns.tolist()}")
    logging.info(f"Размер данных после загрузки: {data.shape}")

    # Проверка наличия timestamp
    if 'timestamp' not in data.columns:
        logging.error("Колонка 'timestamp' отсутствует после загрузки данных.")
        # Если timestamp в индексе, переносим его в колонку
        if isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = data.index
            logging.info("Индекс был преобразован в колонку 'timestamp'.")
        else:
            raise ValueError("Колонка 'timestamp' отсутствует в данных.")
        
    logging.info(f"Перед агрегацией данные имеют размер: {data.shape}")
    logging.info(f"Колонки перед агрегацией: {data.columns.tolist()}")

        
    #data = aggregate_to_2min(data)  # Преобразование в 2-минутный интервал
    
    data = extract_features(data)
    
    build_bearish_neural_network(data)
    
    logging.info("Очистка сессии TensorFlow...")
    clear_session()  # Закрывает все графы и фоновые процессы TensorFlow
    logging.info("Программа завершена.")
    sys.exit(0)
