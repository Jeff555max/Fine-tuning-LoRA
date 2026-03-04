# Fine-tuning и запуск языковых моделей с LoRA

Этот проект содержит код для дообучения языковых моделей с использованием LoRA (Low-Rank Adaptation) и их запуска для интерактивного общения.

## Структура проекта

```
.
├── fine_tuning/          # Код для дообучения модели
│   ├── train.py         # Основной скрипт обучения
│   └── README.md        # Документация по обучению
├── inference/           # Код для запуска модели
│   ├── chat.py         # Скрипт для интерактивного чата
│   └── README.md       # Документация по запуску
├── requirements.txt     # Зависимости Python
└── README.md           # Этот файл
```

## Установка

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Убедитесь, что у вас установлен CUDA (для GPU ускорения):

```bash
# Проверка CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**⚠ ВАЖНО для RTX 5060:** Если `torch.cuda.is_available()` возвращает `False`, установите PyTorch с CUDA поддержкой:

```bash
# Удалите текущий PyTorch
pip uninstall torch torchvision torchaudio

# Установите PyTorch с CUDA 12.1 (для RTX 5060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Подробные инструкции в файле `INSTALL_CUDA.md`

## Скачивание модели

### Рекомендуемые легкие модели для быстрого тестирования:

#### 1. **Microsoft DialoGPT-small** (117M параметров) - Рекомендуется для начала
```bash
# Модель автоматически скачается при первом запуске
# Или можно скачать вручную:
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small'); AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')"
```

#### 2. **GPT-2 small** (124M параметров)
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2'); AutoTokenizer.from_pretrained('gpt2')"
```

#### 3. **TinyLlama-1.1B** (1.1B параметров) - Больше, но все еще быстро
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
```

#### 4. **DistilGPT-2** (82M параметров) - Самый легкий
```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('distilgpt2'); AutoTokenizer.from_pretrained('distilgpt2')"
```

### Автоматическое скачивание

Модели автоматически скачиваются при первом запуске скриптов обучения или инференса. Они сохраняются в кэш HuggingFace (обычно `~/.cache/huggingface/`).

### Ручное скачивание через Python

Создайте файл `download_model.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/DialoGPT-small"  # Замените на нужную модель

print(f"Скачивание модели {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Модель успешно скачана!")
```

Запустите:
```bash
python download_model.py
```

## Быстрый старт

### 1. Подготовка датасета

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

### 2. Дообучение модели

```bash
cd fine_tuning
python train.py \
    --model_name "microsoft/DialoGPT-small" \
    --dataset_path "../dataset.json" \
    --output_dir "./lora_model" \
    --use_4bit \
    --num_train_epochs 3
```

### 3. Запуск модели для общения

```bash
cd inference
python chat.py \
    --base_model "microsoft/DialoGPT-small" \
    --lora_model "../fine_tuning/lora_model"
```

## Рекомендации по выбору модели

### Для быстрого тестирования (мало памяти, быстрое обучение):
- **DistilGPT-2** (82M) - самый легкий
- **DialoGPT-small** (117M) - хороший баланс
- **GPT-2 small** (124M) - стандартный выбор

### Для лучшего качества (больше памяти, дольше обучение):
- **TinyLlama-1.1B** (1.1B) - хорошее качество при разумном размере
- **GPT-2 medium** (355M)
- **GPT-2 large** (774M)

### Для русского языка:
- **rugpt3small_based_on_gpt2** - русскоязычная модель на базе GPT-2
- **ai-forever/rugpt3small_based_on_gpt2**

## Требования к системе

- **Минимум**: 4GB RAM, без GPU (будет медленно)
- **Рекомендуется**: 8GB+ RAM, GPU с 4GB+ VRAM
- **Для больших моделей**: 16GB+ RAM, GPU с 8GB+ VRAM

## Полезные ссылки

- [HuggingFace Models](https://huggingface.co/models) - каталог моделей
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - оригинальная статья о LoRA
- [PEFT Documentation](https://huggingface.co/docs/peft) - документация по PEFT/LoRA

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

