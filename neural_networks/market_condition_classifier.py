import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime, timedelta
from binance.client import Client
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
import pandas_ta as ta
import requests
from scipy.stats import zscore
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
import time
from tensorflow.keras.metrics import AUC, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from binance.exceptions import BinanceAPIException


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

class MarketClassifier:
    def __init__(self, client=None, model_path="market_condition_classifier.h5", scaler_path="scaler.pkl"):
        self.client = client if client else Client(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")
        self.model_path = model_path
        self.scaler_path = scaler_path

        
    def fetch_binance_data(self, symbol, interval, start_date, end_date):
        """
        Скачивает исторические данные с Binance.
        """
        data = []
        while start_date < end_date:
            try:
                # Ограничиваем порции данных на 2 дня
                try:
                    next_date = min(start_date + timedelta(days=2), end_date)
                    logging.info(f"Запрос данных: {symbol}, {start_date}, {next_date}")
                    klines = self.client.get_historical_klines(
                        symbol, interval,
                        start_date.strftime("%d %b %Y %H:%M:%S"),
                        next_date.strftime("%d %b %Y %H:%M:%S")
                    )
                    logging.info(f"Ответ от Binance API: {klines[:5]}...")  # Лог первых 5 строк
                except Exception as e:
                    logging.error(f"Ошибка API: {e}")

                if not klines:
                    logging.warning(f"Нет данных для {symbol} с {start_date} по {next_date}.")
                    start_date = next_date
                    continue

                temp_data = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'], unit='ms')
                temp_data.set_index('timestamp', inplace=True)
                data.append(temp_data[['open', 'high', 'low', 'close', 'volume']].astype(float))
                start_date = datetime.fromtimestamp(klines[-1][0] / 1000) + timedelta(minutes=1)
                
                # Задержка в 1 секунду между запросами
                time.sleep(1)  
            except Exception as e:
                logging.error(f"Ошибка при загрузке данных {symbol}: {e}")
                break
        if not data:
            raise ValueError(f"Нет данных для {symbol} за указанный период.")
        df = pd.concat(data)
        # Проверка интервала на минутные данные
        assert (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all(), "Данные не минутные!"
        return df



    def add_indicators(self, data):
        """Расширенный набор индикаторов для точной классификации с использованием pandas_ta"""
        # Базовые индикаторы
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
        data['adx'] = ta.adx(data['high'], data['low'], data['close'], length=14)['ADX_14']
        
        # Множественные MA для определения силы тренда
        for period in [10, 20, 50, 100]:
            data[f'sma_{period}'] = ta.sma(data['close'], length=period)
            data[f'ema_{period}'] = ta.ema(data['close'], length=period)
        
        # Импульсные индикаторы
        data['rsi'] = ta.rsi(data['close'], length=14)
        macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
        data['macd'], data['macd_signal'], data['macd_hist'] = macd['MACD_12_26_9'], macd['MACDs_12_26_9'], macd['MACDh_12_26_9']
        data['willr'] = ta.willr(data['high'], data['low'], data['close'], length=14)
        
        # Волатильность
        bb = ta.bbands(data['close'], length=20, std=2)
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = bb['BBU_20_2.0'], bb['BBM_20_2.0'], bb['BBL_20_2.0']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Объемные индикаторы
        data['obv'] = ta.obv(data['close'], data['volume'])
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        data['volume_std'] = data['volume'].rolling(window=20).std()
        
        # Трендовые индикаторы
        data['trend_strength'] = abs(data['close'].pct_change().rolling(window=20).sum())
        data['price_momentum'] = data['close'].diff(periods=10) / data['close'].shift(10)
        
        # Динамические уровни поддержки/сопротивления
        data['support_level'] = data['low'].rolling(window=20).min()
        data['resistance_level'] = data['high'].rolling(window=20).max()
        
        return data

    
    
    def validate_predictions(self, data, prediction, window=5):
        """
        Валидация предсказаний классификатора с помощью мультипериодного анализа
        """
        # Проверяем последние N свечей для подтверждения сигнала
        recent_data = data.tail(window)
        
        # 1. Проверка консистентности тренда
        price_direction = recent_data['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        trend_consistency = abs(price_direction.sum()) / window
        
        # 2. Проверка объемного подтверждения
        volume_trend = recent_data['volume'] > recent_data['volume_sma']
        volume_confirmation = volume_trend.mean()
        
        # 3. Проверка импульса
        momentum_confirmation = (
            (recent_data['rsi'] > 60).all() or  # Сильный бычий импульс
            (recent_data['rsi'] < 40).all() or  # Сильный медвежий импульс
            (recent_data['rsi'].between(45, 55)).all()  # Стабильный флэт
        )
        
        # 4. Подтверждение через множественные таймфреймы
        mtf_confirmation = (
            recent_data['adx'].mean() > 25 and  # Сильный тренд
            abs(recent_data['macd_hist']).mean() > recent_data['macd_hist'].std()  # Сильный MACD
        )
        
        # Вычисление общего скора достоверности
        confidence_score = (
            0.3 * trend_consistency +
            0.3 * volume_confirmation +
            0.2 * momentum_confirmation +
            0.2 * mtf_confirmation
        )
        
        # Проверка соответствия предсказания и подтверждений
        if prediction == 'bullish':
            prediction_valid = (
                trend_consistency > 0.6 and
                volume_confirmation > 0.5 and
                recent_data['rsi'].mean() > 55
            )
        elif prediction == 'bearish':
            prediction_valid = (
                trend_consistency > 0.6 and
                volume_confirmation > 0.5 and
                recent_data['rsi'].mean() < 45
            )
        else:  # flat
            prediction_valid = (
                trend_consistency < 0.4 and
                0.3 < volume_confirmation < 0.7 and
                40 < recent_data['rsi'].mean() < 60
            )
        
        return {
            'prediction': prediction,
            'is_valid': prediction_valid,
            'confidence': confidence_score,
            'confirmations': {
                'trend_consistency': trend_consistency,
                'volume_confirmation': volume_confirmation,
                'momentum_confirmation': momentum_confirmation,
                'mtf_confirmation': mtf_confirmation
            }
        }
    
    
    def classify_market_conditions(self, data, window=20):
        """
        Улучшенная классификация рынка с валидацией и множественными подтверждениями
        """
        if len(data) < window:
            logging.warning(f"Недостаточно данных для классификации: {len(data)} < {window}")
            return 'flat'  # Возвращаем flat вместо uncertain как безопасное значение
            
        if not hasattr(self, 'previous_market_type'):
            self.previous_market_type = 'flat'  # Инициализация значением flat

        # Базовые сигналы с подробным логированием
        adx = data['adx'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        macd_hist = data['macd_hist'].iloc[-1]
        willr = data['willr'].iloc[-1]
        volume_ratio = data['volume_ratio'].iloc[-1]
        price = data['close'].iloc[-1]
        support = data['support_level'].iloc[-1]
        resistance = data['resistance_level'].iloc[-1]
        
        # Расчет расстояния до уровней в процентах
        distance_to_support = ((price - support) / price) * 100
        distance_to_resistance = ((resistance - price) / price) * 100
        
        logging.info(f"""
        Текущие значения индикаторов:
        ADX: {adx:.2f}
        RSI: {rsi:.2f}
        MACD Histogram: {macd_hist:.2f}
        Williams %R: {willr:.2f}
        Volume Ratio: {volume_ratio:.2f}
        Цена: {price:.2f}
        Поддержка: {support:.2f} (расстояние: {distance_to_support:.2f}%)
        Сопротивление: {resistance:.2f} (расстояние: {distance_to_resistance:.2f}%)
        """)

        # Подтверждение тренда через MA с логированием
        ma_trends = []
        for period in [10, 20, 50]:
            is_above = price > data[f'sma_{period}'].iloc[-1]
            ma_trends.append(is_above)
            logging.info(f"Цена выше SMA{period}: {is_above}")
        trend_confirmation = sum(ma_trends)
        logging.info(f"Общее подтверждение тренда: {trend_confirmation}/3")

        # Валидационные метрики на последних свечах
        recent_data = data.tail(window)
        price_direction = recent_data['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        trend_consistency = abs(price_direction.sum()) / window
        volume_confirmation = (recent_data['volume'] > recent_data['volume_sma']).mean()
        momentum_strength = abs(recent_data['rsi'] - 50).mean() / 50

        logging.info(f"""
        Метрики тренда:
        Консистентность тренда: {trend_consistency:.2f}
        Подтверждение объёмом: {volume_confirmation:.2f}
        Сила импульса: {momentum_strength:.2f}
        """)

        # Определение типа рынка с более мягкими условиями
        market_type = 'uncertain'
        confidence_score = 0

        # Проверка бычьего рынка
        if (adx > 20 and rsi > 45 and volume_ratio > 1.0 and
            trend_confirmation >= 1 and macd_hist > 0 and
            distance_to_resistance > 0.5):  # Есть пространство для роста
            
            bullish_confidence = (
                0.25 * (trend_consistency if price_direction.sum() > 0 else 0) +
                0.25 * volume_confirmation +
                0.25 * (momentum_strength if rsi > 45 else 0) +
                0.25 * (distance_to_resistance / 5)  # Нормализуем расстояние до сопротивления
            )
            
            logging.info(f"Проверка бычьего рынка. Уверенность: {bullish_confidence:.2f}")
            
            if bullish_confidence > 0.5:
                market_type = 'bullish'
                confidence_score = bullish_confidence

        # Проверка медвежьего рынка
        elif (adx > 20 and rsi < 55 and volume_ratio > 1.0 and
              trend_confirmation <= 2 and macd_hist < 0 and
              distance_to_support > 0.5):  # Есть пространство для падения
            
            bearish_confidence = (
                0.25 * (trend_consistency if price_direction.sum() < 0 else 0) +
                0.25 * volume_confirmation +
                0.25 * (momentum_strength if rsi < 55 else 0) +
                0.25 * (distance_to_support / 5)  # Нормализуем расстояние до поддержки
            )
            
            logging.info(f"Проверка медвежьего рынка. Уверенность: {bearish_confidence:.2f}")
            
            if bearish_confidence > 0.5:
                market_type = 'bearish'
                confidence_score = bearish_confidence

        # Проверка флэта
        elif (adx < 25 and 35 < rsi < 65 and
              abs(macd_hist) < 0.2 * data['macd_hist'].std() and
              0.7 < volume_ratio < 1.3 and
              support < price < resistance and
              max(distance_to_support, distance_to_resistance) < 2):  # Цена между уровнями
            
            flat_confidence = (
                0.25 * (1 - trend_consistency) +
                0.25 * (1 - abs(volume_ratio - 1)) +
                0.25 * (1 - momentum_strength) +
                0.25 * (1 - max(distance_to_support, distance_to_resistance) / 2)
            )
            
            logging.info(f"Проверка флэта. Уверенность: {flat_confidence:.2f}")
            
            if flat_confidence > 0.5:
                market_type = 'flat'
                confidence_score = flat_confidence

        # Если сложная логика не сработала, используем упрощённую классификацию
        if market_type == 'uncertain':
            # Учитываем уровни поддержки и сопротивления в упрощённой логике
            if rsi > 60 and distance_to_resistance > 0.5:
                market_type = 'bullish'
            elif rsi < 40 and distance_to_support > 0.5:
                market_type = 'bearish'
            else:
                market_type = 'flat'
            
            logging.info(f"Используем упрощённую классификацию: {market_type} (RSI: {rsi:.2f}, " +
                        f"Расст. до сопр.: {distance_to_resistance:.2f}%, " +
                        f"Расст. до подд.: {distance_to_support:.2f}%)")
        
        self.previous_market_type = market_type
        return market_type
    

    def remove_outliers(self, data, z_threshold=3):
        """
        Удаляет выбросы на основе метода Z-score.
        """
        from scipy.stats import zscore  # Импортируем stats, если он отсутствует
        z_scores = zscore(data[['open', 'high', 'low', 'close', 'volume']])
        mask = (np.abs(z_scores) < z_threshold).all(axis=1)
        filtered_data = data[mask]
        removed_count = len(data) - len(filtered_data)
        logging.info(f"Удалено выбросов: {removed_count}")
        return filtered_data

    def fetch_market_events(self, api_key, start_date, end_date):
        """
        Получает форс-мажорные события через API.
        """
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "market crash OR economic crisis OR volatility",
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": api_key,
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                events = []
                for article in articles:
                    published_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                    events.append({
                        "start": published_date - timedelta(days=1),
                        "end": published_date + timedelta(days=1),
                        "event": article['title']
                    })
                return events
            else:
                logging.error(f"Ошибка API: {response.status_code} {response.text}")
                return []
        except Exception as e:
            logging.error(f"Ошибка получения событий: {e}")
            return []

    def flag_market_events(self, data, events):
        """
        Добавляет флаги для форс-мажорных рыночных событий.
        """
        data['market_event'] = 'None'
        for event in events:
            mask = (data.index >= event['start']) & (data.index <= event['end'])
            data.loc[mask, 'market_event'] = event['event']
        logging.info(f"Автоматические флаги рыночных событий добавлены.")
        return data

    def fetch_and_label_all(self, symbols, start_date, end_date, save_path="labeled_data", api_key=None):
        os.makedirs(save_path, exist_ok=True)
        all_data = []
        events = self.fetch_market_events(api_key, start_date, end_date) if api_key else []

        for symbol in symbols:
            try:
                logging.info(f"Загрузка данных для {symbol}")
                df = self.fetch_binance_data(symbol, Client.KLINE_INTERVAL_1MINUTE, start_date, end_date)
                # Исправлено data на df
                logging.info(f"Тип данных перед добавлением индикаторов: {type(df)}")
                df = self.add_indicators(df)
                logging.info(f"Тип данных после добавления индикаторов: {type(df)}")
                if events:
                    df = self.flag_market_events(df, events)
                df['market_type'] = self.classify_market_conditions(df)
                df['symbol'] = symbol
                file_path = os.path.join(save_path, f"{symbol}_data.csv")
                df.to_csv(file_path)
                logging.info(f"Данные для {symbol} сохранены в {file_path}")
                all_data.append(df)
            except Exception as e:
                logging.error(f"Ошибка при обработке {symbol}: {e}")
                
        if not all_data:
            raise ValueError("Не удалось собрать данные ни для одного символа.")
        return pd.concat(all_data, ignore_index=True)

    
    def prepare_training_data(self, data_path):
        """
        Загружает и обрабатывает данные для обучения.
        """
        logging.info(f"Загрузка данных из файла {data_path}")
        try:
            data = pd.read_csv(data_path, index_col=0)
        except FileNotFoundError:
            logging.error(f"Файл {data_path} не найден.")
            raise

        # Проверка наличия необходимых столбцов
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'atr', 'adx', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'willr', 'bb_width', 'volume_ratio', 'trend_strength',
            'market_type'
        ]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Отсутствует необходимый столбец: {col}")

        # Удаление строк с пропущенными значениями
        data.dropna(inplace=True)

        # Проверка значений market_type до обработки
        logging.info(f"Уникальные значения market_type до преобразования: {data['market_type'].unique()}")

        # Удаление выбросов
        logging.info("Удаление выбросов из данных...")
        data = self.remove_outliers(data)
        logging.info(f"Размер данных после удаления выбросов: {data.shape}")

        # Преобразование меток рынка в числовые значения
        label_mapping = {'bullish': 0, 'bearish': 1, 'flat': 2}
        logging.info(f"До преобразования меток, уникальные значения: {data['market_type'].unique()}")
        data['market_type'] = data['market_type'].map(label_mapping)
        logging.info(f"После преобразования меток, уникальные значения: {data['market_type'].unique()}")
        logging.info(f"Количество примеров каждого класса:\n{data['market_type'].value_counts()}")
        
        # Проверка после преобразования
        if data['market_type'].isna().any():
            bad_values = data[data['market_type'].isna()]['market_type'].unique()
            logging.error(f"Обнаружены NaN значения в market_type! Исходные значения: {bad_values}")
            data = data.dropna(subset=['market_type'])
        
        features = [col for col in data.columns if col not in ['market_type', 'symbol', 'timestamp']]
        X = data[features].values
        y = data['market_type'].values.astype(int)
        
        logging.info(f"Форма X: {X.shape}")
        logging.info(f"Форма y: {y.shape}")
        logging.info(f"Уникальные значения в y: {np.unique(y)}")
        
        return X, y


    def balance_classes(self, y):
        """
        Рассчитывает веса классов для балансировки.
        """
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        return {i: weight for i, weight in enumerate(class_weights)}

    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(256, input_shape=input_shape, return_sequences=True, 
                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
            Dropout(0.3),
            LSTM(128, return_sequences=True,
                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(3, activation='softmax')  # 3 класса: bullish, bearish, flat
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


    # Обучение классификатора с автоматическим восстановлением прогресса
    def train_market_condition_classifier(self, data_path, model_path='market_condition_classifier.h5', scaler_path='scaler.pkl', checkpoint_path='market_condition_checkpoint.h5'):
        """
        Обучает и сохраняет LSTM модель классификации рыночных условий с кросс-валидацией
        """
        # Подготовка данных
        X, y = self.prepare_training_data(data_path)
        
        # Масштабирование данных
        scaler = StandardScaler()
        if os.path.exists(scaler_path):
            logging.info(f"Загружается существующий масштабировщик из {scaler_path}.")
            scaler = joblib.load(scaler_path)
        else:
            logging.info("Создаётся новый масштабировщик.")
            scaler.fit(X)
            joblib.dump(scaler, scaler_path)
            logging.info(f"Масштабировщик сохранён в {scaler_path}.")
        
        X_scaled = scaler.transform(X)

        # Кросс-валидация для оценки качества модели
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_val_fold = y[val_idx]
            
            # Подготовка данных для LSTM
            X_train_fold = np.expand_dims(X_train_fold, axis=1)
            X_val_fold = np.expand_dims(X_val_fold, axis=1)
            
            logging.info(f"Shape X_train_fold: {X_train_fold.shape}")
            logging.info(f"Shape y_train_fold: {y_train_fold.shape}")
            logging.info(f"Уникальные значения в y_train_fold: {np.unique(y_train_fold)}")  # Добавляем проверку
            
            # Создание и обучение модели для текущего фолда
            with strategy.scope():
                model_fold = self.build_lstm_model(input_shape=(X_train_fold.shape[1], X_train_fold.shape[2]))
                history_fold = model_fold.fit(
                    X_train_fold, y_train_fold,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=1
                )

        # Важно! Тут нужно так же подготовить данные для финальной модели
        X_scaled = np.expand_dims(X_scaled, axis=1)  # Добавляем эту строку
        logging.info(f"Shape X_scaled для финальной модели: {X_scaled.shape}")
        logging.info(f"Уникальные значения в y для финальной модели: {np.unique(y)}")

        # Если кросс-валидация успешна, обучаем финальную модель
        if np.mean(fold_scores) >= 0.7:
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Балансировка классов
            class_weights = self.balance_classes(y_train)

            # Создание финальной модели
            with strategy.scope():
                model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
                ]

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=500,
                    batch_size=64,
                    class_weight=class_weights,
                    callbacks=callbacks
                )
                
                # Оценка финальной модели
                y_pred = model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)

                accuracy = accuracy_score(y_test, y_pred_classes)
                precision = precision_score(y_test, y_pred_classes, average='weighted')
                recall = recall_score(y_test, y_pred_classes, average='weighted')
                f1 = f1_score(y_test, y_pred_classes, average='weighted')

                logging.info(f"""
                   Метрики финальной модели:
                   Accuracy: {accuracy:.4f}
                   Precision: {precision:.4f}
                   Recall: {recall:.4f}
                   F1-Score: {f1:.4f}
               """)

               # Сохранение только если финальные метрики хорошие
                if f1 >= 0.7:
                   model.save(model_path)
                   logging.info(f"Финальная модель сохранена в {model_path}")
                   
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
                   
                   return model
                else:
                   logging.warning("Финальное качество модели ниже порогового (0.7). Модель не сохранена.")
                   return None
        else:
           logging.warning("Кросс-валидация показала низкое качество модели. Требуется доработка архитектуры.")
           return None


if __name__ == "__main__":
    # Инициализация стратегии (TPU или CPU/GPU)
    strategy = initialize_strategy()
    
    client = Client(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT','XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'LINKUSDT', 'TONUSDT', 'NEARUSDT']
    
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    data_path = "labeled_market_data.csv"  # Путь к размеченным данным
    model_path = "market_condition_classifier.h5"  # Путь для сохранения модели
    scaler_path = "scaler.pkl"  # Путь для сохранения масштабировщика

    # Создание экземпляра классификатора
    classifier = MarketClassifier(client)

    # Загрузка и разметка данных
    try:
        logging.info("Начало загрузки и разметки данных.")
        labeled_data = classifier.fetch_and_label_all(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            save_path="labeled_data"
        )
        labeled_data.to_csv(data_path, index=True)
        logging.info(f"Данные успешно сохранены в {data_path}.")
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        exit(1)

    # Обучение классификатора
    try:
        logging.info("Начало обучения классификатора.")
        classifier.train_market_condition_classifier(
            data_path=data_path,
            model_path=model_path,
            scaler_path=scaler_path
        )
        logging.info("Обучение завершено успешно.")
    except Exception as e:
        logging.error(f"Ошибка в процессе обучения: {e}")
        exit(1)


