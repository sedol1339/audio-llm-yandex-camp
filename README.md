# API vllm

## Настройка среды для запуска Voxtral-Small-24B-2507 через vllm

### 1. Создание виртуального окружения Python 3.12

**Используя `venv`:**

```bash
python3.12 -m venv vllm-env
source vllm-env/bin/activate
```

**Или используя `conda`:**

```bash
conda create -n vllm-env python=3.12 -y
conda activate vllm-env
```

### 2. Установка менеджера пакетов uv

```bash
pip install uv
```

### 3. Установка vllm с поддержкой аудио

```bash
uv pip install -U "vllm[audio]" --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```


### 4. Запуск Voxtral-Small-24B-2507

```bash
vllm serve mistralai/Voxtral-Small-24B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tensor-parallel-size 1 --tool-call-parser mistral --enable-auto-tool-choice --gpu-memory-utilization 0.75
```