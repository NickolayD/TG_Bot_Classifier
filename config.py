import pickle
import gdown
import os
from dotenv import load_dotenv, dotenv_values


load_dotenv()
config = dotenv_values(".env")

TELEGRAM_BOT_TOKEN= config['BOT_TOKEN'] 

WEB_SERVICE_NAME = 'https://f2dc-95-73-19-223.ngrok-free.app'

START_BOT_TEXT = "Привет! Я - бот для распознавания различных овощей по фотографии.\n\n"\
                 "Если хочешь, чтобы я определил овощ по фото - просто отправь мне его.\n\n"\
                 "Также доступные следующие команды:\n"\
                 "- /rate - вы можете оценить качество работы данного tg-бота оценкой от 0 до 5;\n"\
                 "- /stat - вы можете просмотреть некоторую статистику по данному tg-боту."
                 
STATISTIC_TEXT = "Количество уникальных посетителей: {}\n"\
                  "Пользовательский рейтинг бота: {}\n"\
                  "Количество обработанных фотографий: {}\n"\
                  "Процент правильно сделанных предсказаний: {}%"
    
ANSWER_ON_TEXT = "Извини, я пока не распознаю текстовые сообщения. Вышли фото, если хочешь проверить мою работу."

VEG_DICT = {
    0: 'Бобы (Bean)',
    1: 'Горькая тыква (Bitter Gourd)',
    2: 'Бутылочная тыква (Botter Gourd)',
    3: 'Баклажан (Brinjal)',
    4: 'Брокколи (Broccoli)',
    5: 'Капуста (Cabbage)',
    6: 'Стручковый перец (Capsicum)',
    7: 'Морковь (Carrot)',
    8: 'Цветная капуста (Cauliflower)',
    9: 'Огурец (Cucumber)',
    10: 'Папайя (Papaya)',
    11: 'Картофель (Potato)',
    12: 'Тыква (Pumpkin)',
    13: 'Редька (Radish)',
    14: 'Томат (Tomato)',
}

# для загрузки pickle-файла с обученной ML моделью
url = 'https://drive.google.com/uc?id=14e4XLC96dYte0cF2XT8l9YMBSGb0JrXd'
filename = 'LinearSVCBest.pkl'
if filename not in os.listdir():
	gdown.download(url, filename, quiet=False)
with open(filename, "rb") as file:
    MODEL = pickle.load(file)

statistics = {
    'unique_users': [],
    'users_score': {},
    'amount_of_predictions': 0,
    'accuracy': [0, 0]
}
