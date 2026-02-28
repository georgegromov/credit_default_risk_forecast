# Credit Card Default Prediction

Нейросетевое приложение для прогнозирования риска дефолта по кредитным картам.

## Структура проекта

```
credit_default_project/
├── app/
│   ├── main.py          # FastAPI приложение
│   └── static/
│       └── index.html   # Веб-интерфейс
├── model/               # Сюда положить файлы модели
│   ├── credit_default_model.keras
│   ├── scaler.pkl
│   └── feature_names.json
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── README.md
```

## Запуск через Docker

1. Положить файлы модели в папку `model/`
2. Собрать и запустить:

```bash
docker compose up --build
```

3. Открыть браузер: http://localhost:8000

## API

POST /predict — предсказание по данным клиента
GET /health  — проверка работоспособности