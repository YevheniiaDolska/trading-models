#!/bin/bash
set -e

# Создаем нужные директории
RUN mkdir -p /root/output/neural_networks /root/output/ensembles

echo "Проверка GPU..."
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "GPU доступен:"
    nvidia-smi
else
    echo "GPU не найден, обучение будет запущено на CPU"
fi

# Запуск обучения всех моделей
declare -a models=(
    "market_condition_classifier.py"
    "bullish_neural_network.py"
    "bullish_ensemble.py"
    "flat_neural_network.py"
    "flat_ensemble.py"
    "bearish_neural_network.py"
    "bearish_ensemble.py"
)

for model in "${models[@]}"; do
    echo "Запускаем обучение: $model"
    python3 "/root/neural_networks/$model"
done

# Копирование результатов
echo "Сохраняем результаты..."
mkdir -p /runpod-volume/{neural_networks,ensembles}
cp -r /root/output/neural_networks/* /runpod-volume/neural_networks/
cp -r /root/output/ensembles/* /runpod-volume/ensembles/

echo "Обучение завершено!"