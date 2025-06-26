#!/bin/bash

# Проверка аргументов
if [ $# -ne 1 ]; then
  echo "Использование: $0 путь_к_reviews.json"
  exit 1
fi

INPUT_FILE=$1
TRAIN_FILE="train.json"
TEST_FILE="test.json"

# Проверка, существует ли входной файл
if [ ! -f "$INPUT_FILE" ]; then
  echo "Файл $INPUT_FILE не найден!"
  exit 2
fi

echo "Формируем train.json (2021 год)..."
jq -c 'select(.date | startswith("2021"))' "$INPUT_FILE" > "$TRAIN_FILE"

echo "Формируем test.json (2022 год)..."
jq -c 'select(.date | startswith("2022"))' "$INPUT_FILE" > "$TEST_FILE"

echo "Готово!"
echo "Файл для обучения: $TRAIN_FILE"
echo "Файл для теста: $TEST_FILE"