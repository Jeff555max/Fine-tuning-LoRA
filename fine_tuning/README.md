# Fine-tuning модели с LoRA

Эта папка содержит код для дообучения модели с использованием LoRA (Low-Rank Adaptation).

## Оптимизация для RTX 5060

Код автоматически определяет RTX 5060 и оптимизирует настройки:
- ✅ FP16 включен для ускорения
- ✅ Gradient checkpointing для экономии памяти (8GB VRAM)
- ✅ Оптимизированные настройки DataLoader
- ✅ Рекомендуемый batch_size: 2-4 для моделей ~120M параметров

## Использование

### Базовый запуск:

```bash
python train.py \
    --model_name "microsoft/DialoGPT-small" \
    --dataset_path "your_dataset.json" \
    --output_dir "./lora_model"
```

### С дополнительными параметрами:

```bash
python train.py \
    --model_name "microsoft/DialoGPT-small" \
    --dataset_path "your_dataset.json" \
    --output_dir "./lora_model" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --use_4bit \
    --lora_r 16 \
    --lora_alpha 32
```

## Параметры

- `--model_name`: Имя модели с HuggingFace (обязательно)
- `--dataset_path`: Путь к датасету в формате .json или .jsonl (обязательно)
- `--output_dir`: Директория для сохранения обученной модели (по умолчанию: ./lora_model)
- `--num_train_epochs`: Количество эпох обучения (по умолчанию: 3)
- `--per_device_train_batch_size`: Размер батча на устройство (по умолчанию: 4)
- `--gradient_accumulation_steps`: Шаги накопления градиента (по умолчанию: 4)
- `--learning_rate`: Скорость обучения (по умолчанию: 2e-4)
- `--max_length`: Максимальная длина последовательности (по умолчанию: 512)
- `--use_4bit`: Использовать 4-bit quantization для экономии памяти
- `--lora_r`: Rank LoRA (по умолчанию: 16)
- `--lora_alpha`: Alpha параметр LoRA (по умолчанию: 32)
- `--lora_dropout`: Dropout для LoRA (по умолчанию: 0.05)

## Формат датасета

Скрипт поддерживает несколько форматов датасета:

1. **Простой текст**:
```json
[
    {"text": "Ваш текст здесь"},
    {"text": "Еще один текст"}
]
```

2. **Instruction-Output**:
```json
[
    {
        "instruction": "Вопрос или инструкция",
        "output": "Ответ модели"
    }
]
```

3. **Prompt-Completion**:
```json
[
    {
        "prompt": "Промпт",
        "completion": "Завершение"
    }
]
```

4. **Input-Output**:
```json
[
    {
        "input": "Входные данные",
        "output": "Выходные данные"
    }
]
```

## Примечания

- Используйте `--use_4bit` для экономии памяти на GPU
- Уменьшите `per_device_train_batch_size` если возникают проблемы с памятью
- Увеличьте `gradient_accumulation_steps` для эффективного обучения с маленькими батчами

