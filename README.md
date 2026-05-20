# Alina Bot — Deploy Guide

## Быстрый старт на Railway

### 1. Подготовка GitHub репозитория

1. Создай новый репозиторий на github.com
2. Загрузи все файлы этого проекта
3. Убедись что `.env` НЕ попал в репозиторий (он в .gitignore)

### 2. Деплой на Railway

1. Зайди на railway.app
2. New Project → Deploy from GitHub repo
3. Выбери свой репозиторий
4. Railway автоматически определит Python проект

### 3. База данных

1. В Railway проекте: New → Database → PostgreSQL
2. Нажми на базу данных → Variables
3. Скопируй `DATABASE_URL`

### 4. Переменные окружения

В Railway → твой сервис → Variables добавь:

```
BOT_TOKEN=         (от @BotFather)
DEEPSEEK_API_KEY=  (от platform.deepseek.com)
DATABASE_URL=      (от Railway PostgreSQL)
```

### 5. Деплой

Railway автоматически задеплоит после добавления переменных.
Смотри логи в разделе Deployments.

## Проверка

Напиши своему боту `/start` в Telegram.
Алина должна ответить: "привет) как тебя зовут?"

## Структура файлов

```
main.py       — точка входа, хендлеры Telegram
ai.py         — генерация ответов через DeepSeek
memory.py     — извлечение и инжект памяти
database.py   — все операции с БД
persona.py    — характер и настройки Алины
railway.toml  — конфиг деплоя
requirements.txt
```

## Лимиты

- Бесплатные пользователи: 20 сообщений/день
- Premium: безлимит
- Оплата через Telegram Stars (/pay_week, /pay_month)
