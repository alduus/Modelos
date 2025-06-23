import json

# === 1. Definir etiquetas válidas en orden ===
labels = [
    "I-ESTADO", "B-MUNICIPIO", "B-EXTERIOR", "I-CODIGO_POSTAL", "O",
    "I-INTERIOR", "B-COLONIA", "B-INTERIOR", "I-CIUDAD", "TIPO_VIA",
    "B-CODIGO_POSTAL", "B-CALLE", "I-COLONIA", "B-CIUDAD", "B-TIPO_VIA",
    "I-CALLE", "B-ESTADO", "I-MUNICIPIO"
]
label2id = {label: i for i, label in enumerate(labels)}

# === 2. Leer archivo original ===
with open("dataset_etiquetado_SEPOMEX.json", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# === 3. Convertir tags a ids ===
converted = []
for row in data:
    tokenList = row["tokens"]
    tagList = row["tags"]
    tagIds = [label2id.get(tag, label2id["O"]) for tag in tagList]  # fallback seguro
    converted.append({"tokens": tokenList, "ner_tags": tagIds})

# === 4. Guardar dataset nuevo ===
with open("train.json", "w", encoding="utf-8") as f:
    for item in converted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
