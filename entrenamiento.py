from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
import json
import os
from sklearn.metrics import classification_report

# === 1. Etiquetas a usar ===
labels = [
    "B-MUNICIPIO",
    "B-COLONIA",
    "B-CIUDAD",
    "B-CODIGO_POSTAL",
    "I-MUNICIPIO",
    "I-COLONIA",
    "I-CIUDAD",
    "I-CODIGO_POSTAL"
]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# === 2. Cargar el dataset ===
dataset = load_dataset("json", data_files={
    "train": "data/train.json",
    "validation": "data/val.json"
})

# === 3. Tokenizador y modelo base ===
modelName = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForTokenClassification.from_pretrained(
    modelName,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# === 4. Función para alinear tokens y etiquetas ===
def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    word_ids = tokenized.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(example["ner_tags"][word_idx])
        else:
            aligned_labels.append(example["ner_tags"][word_idx])
        previous_word_idx = word_idx
    tokenized["labels"] = aligned_labels
    return tokenized

# === 5. Aplicar tokenización al dataset ===
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# === 6. Métricas para evaluación ===
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(preds, labels)]
    report = classification_report(true_labels, true_preds, output_dict=True, zero_division=0)
    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

# === 7. Argumentos de entrenamiento ===
training_args = TrainingArguments(
    output_dir="./piiranha-reduced-8labels",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# === 8. Entrenador ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

# === 9. Entrenamiento ===
trainer.train()

# === 10. Guardado final ===
trainer.save_model("./piiranha-reduced-8labels")
tokenizer.save_pretrained("./piiranha-reduced-8labels")
