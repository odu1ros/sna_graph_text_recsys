#!/bin/bash

# Проверка наличия аргумента
if [ $# -ne 1 ]; then
    echo "❗ Использование: $0 путь_к_input.jsonl"
    exit 1
fi

INPUT="$1"
BATCH_SIZE=700000
REVIEW_FILE="review.json"
LINES=$(wc -l < "$INPUT")
START=1
CHUNK=0

# Создание папки для сохранения результатов
mkdir -p storage

echo "📦 Всего строк в файле: $LINES"
echo "🔄 Начинаем обработку батчами по $BATCH_SIZE строк..."

while [ $START -le $LINES ]; do
    END=$((START + BATCH_SIZE - 1))
    if [ $END -gt $LINES ]; then
        END=$LINES
    fi

    echo "📂 Обработка строк $START до $END..."

    sed -n "${START},${END}p" "$INPUT" > "$REVIEW_FILE"

    OUTPUT_PATH="storage/embeddings_chunk_$(printf "%02d" $CHUNK).hdf5"

    python prepare_embeddings_enriched.py \
        --input_json "$REVIEW_FILE" \
        --output_hdf5 "$OUTPUT_PATH" \
        --batch_size 512 \
        --num_workers 16

    START=$((END + 1))
    CHUNK=$((CHUNK + 1))
done

echo "✅ Обработка завершена. Все результаты сохранены в папке storage/"
