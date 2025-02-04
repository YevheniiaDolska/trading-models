#!/bin/bash
set -e

# Проверка API-ключа
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Ошибка: Переменная окружения RUNPOD_API_KEY не установлена!"
    exit 1
fi

# Получение ID пода
if [ -z "$1" ]; then
    POD_ID=$(runpod pod list | awk 'NR>1 {print $1; exit}')
    if [ -z "$POD_ID" ]; then
        echo "Ошибка: Не удалось получить POD_ID!"
        exit 1
    fi
else
    POD_ID=$1
fi

echo "Работаем с подом с ID: ${POD_ID}"

# Запуск обучения моделей
echo "Запускаем обучение моделей..."
bash /root/run_all_models.sh

# Мониторинг процесса
echo "Ожидание завершения обучения..."
while true; do
    STATUS=$(RUNPOD_API_KEY=$RUNPOD_API_KEY runpod pod list | grep "$POD_ID" | awk '{print $3}')
    echo "Текущий статус: $STATUS"

    if [ "$STATUS" == "Completed" ]; then
        echo "Обучение завершено!"
        break
    fi

    echo "Проверка новых моделей..."
    NEW_MODELS=$(RUNPOD_API_KEY=$RUNPOD_API_KEY runpod pod download ${POD_ID} --source "/root/models" --dest "./local_models" 2>&1)

    if [[ ! -z "$NEW_MODELS" && "$NEW_MODELS" != *"No such file or directory"* ]]; then
        echo "Загружаем новые модели..."
    fi

    sleep 3600
done

# Скачивание финальных файлов
for DIR in logs checkpoints models; do
    echo "Скачивание ${DIR}..."
    RUNPOD_API_KEY=$RUNPOD_API_KEY runpod pod download ${POD_ID} --source "/root/${DIR}" --dest "./local_${DIR}"
done

# Остановка и удаление пода
echo "Завершаем под..."
RUNPOD_API_KEY=$RUNPOD_API_KEY runpod pod stop ${POD_ID}
RUNPOD_API_KEY=$RUNPOD_API_KEY runpod pod remove ${POD_ID}

echo "Готово!"