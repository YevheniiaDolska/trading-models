# Используем CUDA-совместимый базовый образ
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Создаем рабочие директории
RUN mkdir -p /root/models /root/logs /root/checkpoints /root/output/neural_networks /root/output/ensembles
WORKDIR /root

# Устанавливаем gitattributes (LF окончания строк)
RUN echo "* text=auto eol=lf" > /root/.gitattributes

# Устанавливаем необходимые пакеты
RUN apt-get update && apt-get install -y \
    bash python3.9 python3.9-venv python3.9-dev python3-pip curl jq dos2unix docker.io \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python 3.9 и обновляем pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN python3.9 -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir runpod

# Копируем requirements.txt до установки зависимостей
COPY requirements.txt /root/

# **Используем альтернативные зеркала PyPI для ускорения загрузки**
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --timeout=200 --retries=20 --prefer-binary \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.org/simple \
    --extra-index-url https://pypi.debian.net \
    -r requirements.txt

# Очищаем кеш pip перед установкой TensorFlow
RUN pip cache purge

# Устанавливаем TensorFlow вручную (если он не указан в requirements.txt)
RUN pip install --no-cache-dir tensorflow==2.11.0 tensorflow-gpu==2.11.0

# Копируем файлы проекта
COPY neural_networks/ /root/neural_networks/
COPY ensemble_models/ /root/ensemble_models/
COPY --chmod=755 run_all_models.sh runpod_destroy.sh /root/

# Конвертируем окончания строк (если вдруг файлы были созданы в Windows)
RUN find /root -type f -name "*.sh" -exec dos2unix {} + && chmod +x /root/*.sh && ls -la /root/

# Добавляем /root в PATH
ENV PATH="/root:${PATH}"

# 🔥 Жестко задаем переменную API-ключа (ОПАСНО для публичных образов!)
ARG RUNPOD_API_KEY
ENV RUNPOD_API_KEY=${RUNPOD_API_KEY}

# Стартовый командный процесс
WORKDIR /root
CMD ["sh", "-c", "ls -la /root && cat /root/runpod_destroy.sh && sh /root/runpod_destroy.sh"]

