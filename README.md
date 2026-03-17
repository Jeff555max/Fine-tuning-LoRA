# Fine-tuning и запуск языковых моделей с LoRA

Этот проект содержит код для дообучения языковых моделей с использованием LoRA (Low-Rank Adaptation) и их запуска для интерактивного общения.

**Используемая модель:** [sberbank-ai/rugpt3large_based_on_gpt2](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2) (837.5M параметров) - русскоязычная GPT-3 Large от Сбербанка.

## Структура проекта

```
.
├── fine_tuning/          # Код для дообучения модели
│   ├── train.py         # Основной скрипт обучения
│   └── README.md        # Документация по обучению
├── inference/           # Код для запуска модели
│   ├── chat.py         # Скрипт для интерактивного чата
│   └── README.md       # Документация по запуску
├── models/             # Локальное хранилище моделей (создается автоматически)
├── venv/               # Виртуальное окружение (создается вами)
├── download_model.py   # Скрипт для скачивания моделей
├── hf_login.py         # Скрипт для авторизации в Hugging Face
├── install_pytorch_cuda.bat  # Установка PyTorch с CUDA (Windows)
├── INSTALL_CUDA.md     # Подробная инструкция по установке CUDA
├── HUGGINGFACE_SETUP.md # Настройка Hugging Face Hub
├── requirements.txt    # Зависимости Python
├── example_dataset.json # Пример датасета
└── README.md          # Этот файл
```

## Установка

### 1. Создайте виртуальное окружение

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Установите зависимости

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Установите PyTorch с CUDA (для GPU)

**⚠ ВАЖНО:** По умолчанию устанавливается CPU версия PyTorch. Для использования GPU:

**Windows:**
```bash
# Запустите готовый скрипт
install_pytorch_cuda.bat
```

**Linux/Mac:**
```bash
# Удалите CPU версию
pip uninstall torch torchvision torchaudio

# Установите версию с CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Проверка установки:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Подробные инструкции в файле `INSTALL_CUDA.md`

### 4. (Опционально) Авторизация в Hugging Face

Для большинства публичных моделей авторизация не требуется. Нужна только для приватных моделей:

```bash
python hf_login.py
```

Подробнее в файле `HUGGINGFACE_SETUP.md`

## Скачивание модели

Модели автоматически скачиваются при первом запуске и сохраняются в папку `models/` проекта.

### Автоматическое скачивание

Модели скачаются автоматически при первом запуске обучения или инференса:

```bash
# При обучении
python fine_tuning/train.py --model_name "sberbank-ai/rugpt3large_based_on_gpt2" --dataset_path "dataset.json"

# При запуске чата
python inference/chat.py --base_model "sberbank-ai/rugpt3large_based_on_gpt2"
```

### Ручное скачивание (опционально)

Используйте готовый скрипт для предварительного скачивания:

```bash
# Скачать основную модель проекта (русскоязычная GPT-3 Large)
python download_model.py --model_name sberbank-ai/rugpt3large_based_on_gpt2

# Или другие модели для тестирования
python download_model.py --model_name microsoft/DialoGPT-small
```

### Рекомендуемые модели:

#### **sberbank-ai/rugpt3large_based_on_gpt2** (837.5M параметров) - Основная модель проекта
Русскоязычная GPT-3 Large от Сбербанка, оптимизирована для русского языка.
```bash
python download_model.py --model_name sberbank-ai/rugpt3large_based_on_gpt2
```

#### Альтернативные модели для тестирования:

**1. Microsoft DialoGPT-small** (117M параметров) - Для быстрого тестирования
```bash
python download_model.py --model_name microsoft/DialoGPT-small
```

**2. GPT-2 small** (124M параметров)
```bash
python download_model.py --model_name gpt2
```

**3. TinyLlama-1.1B** (1.1B параметров)
```bash
python download_model.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**4. DistilGPT-2** (82M параметров) - Самый легкий
```bash
python download_model.py --model_name distilgpt2
```

## Быстрый старт

### 1. Подготовка окружения

```bash
# Создайте и активируйте виртуальное окружение
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Установите зависимости
pip install -r requirements.txt

# Установите PyTorch с CUDA (для GPU)
install_pytorch_cuda.bat  # Windows
```

### 2. Подготовка датасета

Создайте файл с вашим датасетом в формате JSON или JSONL. Пример (`dataset.json`):

```json
[
    {
        "text": "Привет! Как дела?"
    },
    {
        "instruction": "Расскажи анекдот",
        "output": "Почему программисты не любят природу? Там слишком много багов!"
    }
]
```

### 3. Дообучение модели

```bash
cd fine_tuning
python train.py \
    --model_name "sberbank-ai/rugpt3large_based_on_gpt2" \
    --dataset_path "../example_dataset.json" \
    --output_dir "./lora_model" \
    --use_4bit \
    --num_train_epochs 3
```

### 4. Запуск модели для общения

```bash
cd inference
python chat.py \
    --base_model "sberbank-ai/rugpt3large_based_on_gpt2" \
    --lora_model "../fine_tuning/lora_model"
```

## Рекомендации по выбору модели

### Основная модель проекта:
- **sberbank-ai/rugpt3large_based_on_gpt2** (837.5M) - Русскоязычная GPT-3 Large от Сбербанка
  - Оптимизирована для русского языка
  - Требует GPU с 8GB+ VRAM (с quantization)
  - Рекомендуется использовать `--use_4bit` для экономии памяти

### Для быстрого тестирования (мало памяти, быстрое обучение):
- **DistilGPT-2** (82M) - самый легкий
- **DialoGPT-small** (117M) - хороший баланс
- **GPT-2 small** (124M) - стандартный выбор

### Для лучшего качества (больше памяти, дольше обучение):
- **sberbank-ai/rugpt3large_based_on_gpt2** (837.5M) - основная модель проекта
- **TinyLlama-1.1B** (1.1B) - хорошее качество при разумном размере
- **GPT-2 medium** (355M)
- **GPT-2 large** (774M)

### Другие русскоязычные модели:
- **sberbank-ai/rugpt3small_based_on_gpt2** (125M) - меньшая версия
- **sberbank-ai/rugpt3medium_based_on_gpt2** (355M) - средняя версия
- **ai-forever/rugpt3small_based_on_gpt2** - альтернативная реализация

## Требования к системе

### Для основной модели (sberbank-ai/rugpt3large_based_on_gpt2):
- **Минимум**: 16GB RAM, GPU с 8GB+ VRAM (с `--use_4bit`)
- **Рекомендуется**: 32GB RAM, GPU с 12GB+ VRAM
- **Без quantization**: GPU с 16GB+ VRAM

### Для легких моделей (DialoGPT-small, GPT-2):
- **Минимум**: 4GB RAM, без GPU (будет медленно)
- **Рекомендуется**: 8GB+ RAM, GPU с 4GB+ VRAM

## Полезные ссылки

- [HuggingFace Models](https://huggingface.co/models) - каталог моделей
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - оригинальная статья о LoRA
- [PEFT Documentation](https://huggingface.co/docs/peft) - документация по PEFT/LoRA
- [PyTorch Installation](https://pytorch.org/get-started/locally/) - официальная установка PyTorch

## Файлы проекта

- `download_model.py` - Скрипт для предварительного скачивания моделей
- `hf_login.py` - Скрипт для авторизации в Hugging Face Hub
- `install_pytorch_cuda.bat` - Автоматическая установка PyTorch с CUDA (Windows)
- `INSTALL_CUDA.md` - Подробная инструкция по установке и настройке CUDA
- `HUGGINGFACE_SETUP.md` - Настройка и использование Hugging Face Hub
- `example_dataset.json` - Пример датасета с 40+ примерами на русском языке

## Решение проблем

### Нехватка памяти GPU:
- Используйте `--use_4bit` при обучении
- Уменьшите `--per_device_train_batch_size`
- Увеличьте `--gradient_accumulation_steps`

### Медленная генерация:
- Уменьшите `--max_length`
- Используйте GPU вместо CPU
- Выберите более легкую модель

### Ошибки при загрузке модели:
- Проверьте подключение к интернету
- Убедитесь, что имя модели корректно
- Попробуйте другую модель

## Лицензия

Этот код предоставляется "как есть" для образовательных целей.

---

## Авторы

Проект создан для демонстрации fine-tuning языковых моделей с использованием LoRA.
