"""
Скрипт для fine-tuning модели с использованием LoRA
"""
import os
import json
import time
import sys
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import torch
from transformers import BitsAndBytesConfig

class DetailedLoggingCallback(TrainerCallback):
    """Callback для детального логирования процесса обучения"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Вызывается в начале обучения"""
        self.start_time = time.time()
        effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        print(f"\n{'='*60}")
        print(f"ОБУЧЕНИЕ | Эпох: {args.num_train_epochs} | Батч: {effective_batch} | LR: {args.learning_rate}")
        print(f"{'='*60}\n")
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Вызывается в начале каждой эпохи"""
        self.epoch_start_time = time.time()
        steps = state.max_steps // int(args.num_train_epochs) if state.max_steps else '?'
        print(f"\nЭпоха {state.epoch}/{int(args.num_train_epochs)} | Шагов: {steps}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при каждом логировании"""
        if logs is None:
            return
        
        step = state.global_step
        loss = logs.get('loss', None)
        lr = logs.get('learning_rate', None)

        loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss or 'N/A')
        lr_str = f"{lr:.2e}" if isinstance(lr, float) else str(lr or 'N/A')

        # Компактный вывод
        if state.max_steps:
            progress = (step / state.max_steps) * 100
            print(f"Шаг {step}/{state.max_steps} ({progress:.1f}%) | Loss: {loss_str} | LR: {lr_str}", end='')
        else:
            print(f"Шаг {step} | Loss: {loss_str} | LR: {lr_str}", end='')
        
        # Память GPU (только если используется)
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f" | GPU: {mem:.1f}GB")
        else:
            print()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Вызывается в конце каждой эпохи"""
        epoch_time = time.time() - self.epoch_start_time
        loss_val = state.log_history[-1].get('loss', None) if state.log_history else None
        loss_str = f"{loss_val:.4f}" if isinstance(loss_val, float) else 'N/A'
        print(f"Эпоха {state.epoch} завершена | Время: {epoch_time/60:.1f}мин | Loss: {loss_str}\n")
        
    def on_train_end(self, args, state, control, **kwargs):
        """Вызывается в конце обучения"""
        total_time = time.time() - self.start_time
        loss_val = state.log_history[-1].get('loss', None) if state.log_history else None
        loss_str = f"{loss_val:.4f}" if isinstance(loss_val, float) else 'N/A'
        print(f"\n{'='*60}")
        print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"Время: {total_time/60:.1f}мин | Шагов: {state.global_step} | Loss: {loss_str}")
        print(f"{'='*60}\n")

def print_system_info():
    """Выводит информацию о системе"""
    print("\n" + "="*60)
    print("СИСТЕМА")
    print("="*60)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        compute_capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"CUDA: {cuda_version}")
        print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        
        # Проверка совместимости для RTX 5060 (sm_120)
        if compute_capability[0] >= 12:
            print("[!] RTX 5060 (sm_120) требует PyTorch 2.5+ или nightly build")
            print("Если возникают ошибки, установите:")
            print("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
    else:
        print("[!] ВНИМАНИЕ: CUDA недоступна!")
        print("PyTorch установлен без поддержки GPU")
        print("\nДля RTX 5060 установите PyTorch Nightly:")
        print("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
        sys.exit(1)
    print("="*60)

def load_model_and_tokenizer(model_name, use_4bit=True, cache_dir=None):
    """
    Загружает модель и токенизатор
    
    Args:
        model_name: имя модели с HuggingFace
        use_4bit: использовать ли 4-bit quantization для экономии памяти
        cache_dir: директория для сохранения модели (если None, используется локальная директория)
    """
    print(f"\n{'='*60}")
    print(f"ЗАГРУЗКА МОДЕЛИ: {model_name}")
    print(f"{'='*60}")
    if use_4bit:
        print("Quantization: 4-bit")
    
    # Определяем директорию для сохранения модели
    if cache_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "models", model_name.replace("/", "_"))
        os.makedirs(cache_dir, exist_ok=True)
    else:
        os.makedirs(cache_dir, exist_ok=True)
    
    load_start = time.time()
    
    # Настройка quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Загрузка токенизатора
    if os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        print("Токенизатор: локальный")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.save_pretrained(cache_dir)
        print("Токенизатор: скачан")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Загрузка модели - ПРИНУДИТЕЛЬНО на GPU
    model_start = time.time()
    if not torch.cuda.is_available():
        print("ОШИБКА: CUDA недоступна! Установите PyTorch с CUDA поддержкой")
        sys.exit(1)
    
    # Принудительно используем GPU
    device = "cuda:0"
    device_map = "auto"
    torch_dtype = torch.float16 if not use_4bit else None
    
    # Устанавливаем текущее устройство
    torch.cuda.set_device(0)
    
    # Проверка compute capability для RTX 5060
    compute_cap = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Устройство: GPU ({gpu_name})")
    
    if compute_cap[0] >= 12:
        print(f"[!] Обнаружена архитектура sm_{compute_cap[0]}{compute_cap[1]} (Blackwell)")
        print("  Для RTX 5060 требуется PyTorch 2.5+ или nightly build")
        if use_4bit:
            print("  [!] Quantization может не работать с sm_120")
            print("  Если возникнут ошибки, запустите БЕЗ --use_4bit")
    
    # Проверяем, есть ли уже сохраненная модель локально
    config_path = os.path.join(cache_dir, "config.json")
    model_files = [
        os.path.join(cache_dir, "model.safetensors"),
        os.path.join(cache_dir, "pytorch_model.bin"),
        os.path.join(cache_dir, "model.safetensors.index.json"),
    ]
    
    has_local_model = os.path.exists(config_path) and any(os.path.exists(f) for f in model_files)
    
    # Загрузка модели с обработкой ошибок
    def _handle_cuda_error(e):
        err = str(e)
        if "no kernel image is available" in err or "CUDA capability" in err or "sm_120" in err:
            print("\n" + "="*60)
            print("ОШИБКА: Несовместимость CUDA capability")
            print("="*60)
            print("Решение 1: Установите PyTorch Nightly")
            print("  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
            print("Решение 2: Запустите БЕЗ --use_4bit")
            print("="*60)
            sys.exit(1)
        raise e

    if has_local_model:
        try:
            if use_4bit:
                quantization_config_path = os.path.join(cache_dir, "quantization_config.json")
                src = model_name if not os.path.exists(quantization_config_path) else cache_dir
                model = AutoModelForCausalLM.from_pretrained(
                    src,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
                if hasattr(model, 'config'):
                    model.config.save_pretrained(cache_dir)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    cache_dir,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
        except RuntimeError as e:
            _handle_cuda_error(e)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config if use_4bit else None,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            if not use_4bit:
                model.save_pretrained(cache_dir)
            elif hasattr(model, 'config'):
                model.config.save_pretrained(cache_dir)
        except RuntimeError as e:
            _handle_cuda_error(e)
    
    model_time = time.time() - model_start
    
    # Информация о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметры: {total_params/1e6:.1f}M всего, {trainable_params/1e6:.1f}M обучаемых")
    
    # Память GPU
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated(0) / 1024**3
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU память: {mem:.1f}/{total_mem:.1f}GB")
    
    # Подготовка модели для обучения с quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    total_load_time = time.time() - load_start
    print(f"Загружено за {total_load_time:.1f}с\n")
    
    return model, tokenizer

def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Настраивает LoRA для модели
    
    Args:
        model: модель для настройки
        r: rank LoRA
        lora_alpha: alpha параметр LoRA
        lora_dropout: dropout для LoRA
    """
    print(f"{'='*60}")
    print(f"НАСТРОЙКА LoRA | r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"{'='*60}")
    
    # Определяем target_modules в зависимости от архитектуры модели
    model_type = model.config.model_type.lower() if hasattr(model.config, 'model_type') else 'unknown'
    
    # Определяем target_modules в зависимости от архитектуры
    if model_type in ['gpt2', 'gpt_neo', 'gpt_neo_x']:
        # GPT-2, DialoGPT, GPT-Neo используют c_attn и c_proj
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif model_type in ['llama', 'mistral', 'mixtral']:
        # LLaMA модели используют q_proj, v_proj и т.д.
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif model_type in ['bloom', 'bloomz']:
        # BLOOM модели
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif model_type in ['opt']:
        # OPT модели
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    else:
        # Пытаемся автоматически найти подходящие модули
        print("  Автоматическое определение модулей...")
        all_module_names = set()
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Листовые модули
                module_name = name.split('.')[-1]
                all_module_names.add(module_name)
        
        # Ищем распространенные паттерны
        target_modules = []
        common_patterns = ['attn', 'proj', 'fc', 'dense', 'query', 'key', 'value']
        for pattern in common_patterns:
            matching = [name for name in all_module_names if pattern.lower() in name.lower()]
            target_modules.extend(matching)
        
        if not target_modules:
            # Если ничего не найдено, используем все линейные слои
            target_modules = [name for name in all_module_names if 'linear' in name.lower() or 'weight' in name.lower()]
        
        if not target_modules:
            # Последняя попытка - используем стандартные для GPT-2
            print("  [!] Не удалось определить модули, используем стандартные для GPT-2")
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            target_modules = list(set(target_modules))  # Убираем дубликаты
    
    print(f"Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Информация о параметрах
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = (trainable_params / total_params) * 100
    print(f"Обучаемых параметров: {trainable_params/1e6:.2f}M ({trainable_percent:.2f}%)\n")
    
    return model

def load_dataset_from_file(dataset_path):
    """
    Загружает датасет из файла
    
    Поддерживаемые форматы:
    - JSON файл с полем 'text' или 'instruction'/'output'
    - JSONL файл (каждая строка - JSON объект)
    """
    print(f"{'='*60}")
    print(f"ЗАГРУЗКА ДАТАСЕТА: {os.path.basename(dataset_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл датасета не найден: {dataset_path}")
    
    load_start = time.time()
    
    if dataset_path.endswith('.jsonl'):
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    elif dataset_path.endswith('.json'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())[0] if data else []
    else:
        raise ValueError("Поддерживаются только .json и .jsonl файлы")
    
    load_time = time.time() - load_start
    file_size = os.path.getsize(dataset_path) / 1024**2
    print(f"Примеров: {len(data)} | Размер: {file_size:.1f}MB | Время: {load_time:.1f}с\n")
    
    return data

def preprocess_dataset(data, tokenizer, max_length=512):
    """
    Предобрабатывает датасет для обучения
    
    Args:
        data: список словарей с данными
        tokenizer: токенизатор
        max_length: максимальная длина последовательности
    """
    print(f"{'='*60}")
    print(f"ПРЕДОБРАБОТКА | max_length={max_length}")
    print(f"{'='*60}")
    
    preprocess_start = time.time()
    
    def format_prompt(example):
        """
        Форматирует пример в промпт
        Поддерживает разные форматы датасета
        """
        if 'text' in example:
            # Простой текст
            return example['text']
        elif 'instruction' in example and 'output' in example:
            # Формат instruction-output
            return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        elif 'prompt' in example and 'completion' in example:
            # Формат prompt-completion
            return f"{example['prompt']}\n\n{example['completion']}"
        elif 'input' in example and 'output' in example:
            # Формат input-output
            return f"Input: {example['input']}\nOutput: {example['output']}"
        else:
            # Пытаемся найти любой текстовый ключ
            text_keys = [k for k in example.keys() if 'text' in k.lower() or 'content' in k.lower()]
            if text_keys:
                return example[text_keys[0]]
            else:
                return str(example)
    
    # Анализ длины текстов (только статистика)
    text_lengths = []
    for example in data[:min(100, len(data))]:
        text = format_prompt(example)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        text_lengths.append(len(tokens))
    
    if text_lengths:
        avg_tokens = sum(text_lengths) / len(text_lengths)
        truncated = sum(1 for t in text_lengths if t > max_length)
        print(f"Средняя длина: {avg_tokens:.0f} токенов | Обрезано: {truncated}/{len(text_lengths)}")
    
    def tokenize_function(examples):
        """
        Токенизирует примеры
        При batched=True examples - это словарь со списками значений
        """
        # Преобразуем batched формат в список словарей
        # examples это словарь вида {'instruction': [list], 'output': [list], ...}
        batch_size = len(list(examples.values())[0])
        examples_list = []
        for i in range(batch_size):
            example_dict = {key: examples[key][i] for key in examples.keys()}
            examples_list.append(example_dict)
        
        # Форматируем каждый пример
        texts = [format_prompt(ex) for ex in examples_list]
        
        # Токенизация (без return_tensors, чтобы вернуть списки)
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
        # Добавляем labels (копия input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Преобразуем в формат для datasets
    from datasets import Dataset
    dataset = Dataset.from_list(data)
    
    # Токенизация
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Токенизация"
    )
    
    preprocess_time = time.time() - preprocess_start
    total_tokens = sum(len(ids) for ids in tokenized_dataset['input_ids'])
    print(f"Токенизировано: {len(tokenized_dataset)} примеров, {total_tokens/1e3:.0f}K токенов | Время: {preprocess_time:.1f}с\n")
    
    return tokenized_dataset

def train(
    model_name,
    dataset_path,
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=512,
    use_4bit=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    save_steps=500,
    logging_steps=10,
    warmup_steps=100
):
    """
    Основная функция обучения
    
    Args:
        model_name: имя модели с HuggingFace
        dataset_path: путь к датасету
        output_dir: директория для сохранения модели
        num_train_epochs: количество эпох
        per_device_train_batch_size: размер батча на устройство
        gradient_accumulation_steps: шаги накопления градиента
        learning_rate: скорость обучения
        max_length: максимальная длина последовательности
        use_4bit: использовать ли 4-bit quantization
        lora_r: rank LoRA
        lora_alpha: alpha параметр LoRA
        lora_dropout: dropout для LoRA
        save_steps: шаги сохранения
        logging_steps: шаги логирования
        warmup_steps: шаги warmup
    """
    # Вывод информации о системе
    print_system_info()
    
    # Определяем директорию для сохранения обученной модели (в проекте)
    if not os.path.isabs(output_dir):
        # Если путь относительный, делаем его относительно корня проекта
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, output_dir)
    
    # Создаем директорию для сохранения модели
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nДиректория для сохранения обученной модели: {os.path.abspath(output_dir)}")
    
    # Определяем директорию для базовой модели (в проекте)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_cache_dir = os.path.join(project_root, "models", model_name.replace("/", "_"))
    
    # Загрузка модели и токенизатора
    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit=use_4bit, cache_dir=base_model_cache_dir)
    
    # Настройка LoRA
    model = setup_lora(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    # Проверка и принудительное перемещение на GPU
    if not torch.cuda.is_available():
        print("ОШИБКА: CUDA недоступна! Установите PyTorch с CUDA")
        sys.exit(1)
    
    try:
        model_device = next(model.parameters()).device
        if model_device.type == 'cpu':
            print("[!] Модель на CPU, перемещаем на GPU...")
            model = model.to("cuda:0")
        else:
            print(f"[OK] Модель на GPU: {model_device}")
    except:
        model = model.to("cuda:0")
        print("[OK] Модель перемещена на GPU")
    
    # Загрузка датасета
    data = load_dataset_from_file(dataset_path)
    train_dataset = preprocess_dataset(data, tokenizer, max_length=max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, не masked LM
    )
    
    # Вычисляем общее количество шагов
    total_steps = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps) * num_train_epochs
    steps_per_epoch = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
    
    print("\n" + "="*80)
    print("НАСТРОЙКА ПАРАМЕТРОВ ОБУЧЕНИЯ")
    print("="*80)
    print(f"Размер датасета: {len(train_dataset)} примеров")
    print(f"Размер батча на устройство: {per_device_train_batch_size}")
    print(f"Шаги накопления градиента: {gradient_accumulation_steps}")
    print(f"Эффективный размер батча: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"Шагов в эпохе: {steps_per_epoch}")
    print(f"Всего шагов: {total_steps}")
    print(f"Количество эпох: {num_train_epochs}")
    print(f"Скорость обучения: {learning_rate}")
    print(f"Warmup шагов: {warmup_steps}")
    print(f"Шаги логирования: {logging_steps}")
    print(f"Шаги сохранения: {save_steps}")
    # ПРИНУДИТЕЛЬНОЕ использование GPU
    if not torch.cuda.is_available():
        print("ОШИБКА: CUDA недоступна! Обучение невозможно без GPU")
        sys.exit(1)
    
    use_gpu = True  # Принудительно используем GPU
    use_gradient_checkpointing = False  # Инициализация
    
    # Проверка конкретной GPU модели
    gpu_name = None
    gpu_memory_gb = None
    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # RTX 5060 имеет ~8GB VRAM, оптимизируем настройки
        if "RTX 5060" in gpu_name or gpu_memory_gb < 10:
            use_gradient_checkpointing = True
            if per_device_train_batch_size > 4:
                print(f"[!] Батч {per_device_train_batch_size} может быть слишком большим для 8GB GPU")
        else:
            use_gradient_checkpointing = False
    
    if use_4bit and use_gpu:
        optim_name = "paged_adamw_8bit"
    elif use_gpu:
        optim_name = "adamw_torch"
    else:
        optim_name = "adamw_torch"
    
    # Настройки для GPU
    if use_gpu:
        fp16 = True
        bf16 = False
        dataloader_pin_memory = True
        dataloader_num_workers = 4
    else:
        fp16 = False
        bf16 = False
        dataloader_pin_memory = False
        dataloader_num_workers = 0
        use_gradient_checkpointing = False
        print("[!] Обучение на CPU будет очень медленным!\n")
    
    # Включаем gradient checkpointing если нужно
    if use_gpu and use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=fp16,  # FP16 для RTX 5060
        bf16=bf16,  # BF16 не поддерживается RTX 5060
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_num_workers=dataloader_num_workers,
        logging_steps=logging_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        optim=optim_name,
        logging_first_step=True,
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,  # Сохраняем все колонки
        ddp_find_unused_parameters=False if use_gpu else None,  # Оптимизация для multi-GPU
        gradient_checkpointing=use_gradient_checkpointing if use_gpu else False,  # Экономия памяти
    )
    
    # Создаем директорию для логов
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Trainer с callback для детального логирования
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[DetailedLoggingCallback()],
    )
    
    # Убеждаемся что модель на GPU перед обучением
    try:
        model_device = next(model.parameters()).device
        if model_device.type != 'cuda':
            print("Перемещение модели на GPU перед обучением...")
            model = model.to("cuda:0")
            # Обновляем trainer с моделью на GPU
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                callbacks=[DetailedLoggingCallback()],
            )
    except:
        model = model.to("cuda:0")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[DetailedLoggingCallback()],
        )
    
    # Информация о памяти перед обучением
    mem = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - mem
    print(f"GPU память: {mem:.1f}/{total:.1f}GB (свободно: {free:.1f}GB)")
    if free < 1.0:
        print("[!] Мало памяти! Уменьшите batch_size")
    torch.cuda.empty_cache()
    print()
    
    # Обучение
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    
    # Финальная статистика
    loss_val = trainer.state.log_history[-1].get('loss', None) if trainer.state.log_history else None
    loss_str = f"{loss_val:.4f}" if isinstance(loss_val, float) else 'N/A'
    speed = len(train_dataset) * num_train_epochs / train_time if train_time > 0 else 0
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Время: {train_time/60:.1f}мин | Скорость: {speed:.1f} примеров/сек | Loss: {loss_str}")
    print(f"{'='*60}\n")
    
    # Сохранение модели
    print(f"Сохранение модели в {output_dir}...")
    save_start = time.time()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_time = time.time() - save_start
    print(f"[OK] Сохранено за {save_time:.1f}с | Путь: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning модели с LoRA")
    parser.add_argument("--model_name", type=str, required=True, help="Имя модели с HuggingFace")
    parser.add_argument("--dataset_path", type=str, required=True, help="Путь к датасету (.json или .jsonl)")
    parser.add_argument("--output_dir", type=str, default="./lora_model", help="Директория для сохранения")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Размер батча")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Шаги накопления градиента")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Скорость обучения")
    parser.add_argument("--max_length", type=int, default=512, help="Максимальная длина последовательности")
    parser.add_argument("--use_4bit", action="store_true", help="Использовать 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16, help="Rank LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout LoRA")
    parser.add_argument("--save_steps", type=int, default=500, help="Шаги сохранения")
    parser.add_argument("--logging_steps", type=int, default=10, help="Шаги логирования")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Шаги warmup")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
    )

