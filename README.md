# Сервис рекомендаций фильмов на основе данных с сайта Kion

## Подготовка

## Установка окружения
```bash
python3 -m pip install poetry
poetry install
```

## Установить CloudFlare
```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000/
```

## Установка данных с рекомендациями
```bash
pip install -U --no-cache-dir gdown --pre
gdown --id 1ZK_iUE1U9WhD4t2e4jXZdQ9dAbSYcahV
```


## Contributors
1. Renat Shakirov
2. Vladislav Mostovik
3. Robert Zaraev
