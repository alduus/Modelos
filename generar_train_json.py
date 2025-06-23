import json
import random

# === 1. Cargar archivo original ===
with open("train.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# === 2. Mezclar aleatoriamente
random.seed(42)  # para reproducibilidad
random.shuffle(data)

# === 3. Dividir 80/20
split_index = int(0.8 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# === 4. Guardar archivos
with open("data/train.json", "w", encoding="utf-8") as f_train:
    for item in train_data:
        f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("data/val.json", "w", encoding="utf-8") as f_val:
    for item in val_data:
        f_val.write(json.dumps(item, ensure_ascii=False) + "\n")
------------------------------------------------------------------
import json

# === 1. Detectar todas las etiquetas únicas ===
with open("dataset_etiquetado_SEPOMEX.json", "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

# Extraer etiquetas únicas del campo 'tags'
unique_tags = sorted({tag for row in raw_data for tag in row["tags"]})
label2id = {label: i for i, label in enumerate(unique_tags)}

print("Etiquetas detectadas:", label2id)

# === 2. Convertir etiquetas string a índices ===
converted = []
for row in raw_data:
    tokens = row["tokens"]
    tag_ids = [label2id[tag] for tag in row["tags"]]
    converted.append({"tokens": tokens, "ner_tags": tag_ids})

# === 3. Guardar como train.json
with open("train.json", "w", encoding="utf-8") as f:
    for item in converted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
