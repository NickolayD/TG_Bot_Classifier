import config 
import logging
import numpy as np
import time
from aiogram import Dispatcher, Bot, types, executor
from io import BytesIO
from PIL import Image
from skimage import color
from skimage.feature import hog


# логирование
logging.basicConfig(level=logging.INFO)
# бот
bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
# диспетчер
dp = Dispatcher(bot)
# флаг для обработки сообщений во время пользовательской оценки
dp["eval_standby_mode"] = False

# хендлер на команду /start
@dp.message_handler(commands=["start"])
async def start(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    # отслеживание уникальных пользователей тг-бота
    if user_id not in config.statistics["unique_users"]:
        config.statistics["unique_users"].append(user_id)
    logging.info(f'Start: {user_id} {user_full_name} {time.asctime()}. Message: {message}')
    await message.answer(config.START_BOT_TEXT)
    
# хендлер на команду /rate
@dp.message_handler(commands=["rate"])
async def rate(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    dp["eval_standby_mode"] = True
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(*["1", "2", "3", "4", "5"])
    logging.info(f'Rate: {user_id} {user_full_name} {time.asctime()}. User was asked to rate bot.')
    await message.answer("Оцени, насколько ты доволен моей работой, по шкале от 0 до 5.", reply_markup=keyboard)

# хендлер на команду /stat
@dp.message_handler(commands=["stat"])
async def stat(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    amount = len(config.statistics['unique_users'])
    if len(config.statistics["users_score"]): 
        rating = round(np.array(list(config.statistics["users_score"].values()), dtype=int).mean(), 2)
    else:
        rating = "Оценок нет."
    predicts = config.statistics['amount_of_predictions']
    if config.statistics["accuracy"][1] == 0:
        accuracy = 0
    else:
        accuracy = round(config.statistics["accuracy"][0] / config.statistics["accuracy"][1] * 100, 2)
    logging.info(f'Statistics: {user_id} {user_full_name} {time.asctime()}. Bot statistics was shown.')
    await message.answer(config.STATISTIC_TEXT.format(amount, rating, predicts, accuracy))

# хендлер на текстовое сообщение от пользователя    
@dp.message_handler(content_types=["text"])
async def answer_on_text(message: types.Message):
    try:
        user_id = message.from_user.id
        user_full_name = message.from_user.full_name
        logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Text has been sent. Message: {message["text"]}.')
        if dp["eval_standby_mode"]:
            try:
                score = message["text"]
                assert (score.isdigit() and int(score) in [0, 1, 2, 3, 4, 5])
                config.statistics["users_score"][user_id] = int(score)
                dp["eval_standby_mode"] = False
                logging.info(f'Rate: {user_id} {user_full_name} {time.asctime()}. Bot was rated by user.')
                await message.answer("Спасибо за вашу оценку!", reply_markup=types.ReplyKeyboardRemove())
            except:
                if message["text"].lower() == "quit":
                    dp["eval_standby_mode"] = False
                    logging.info(f'Rate: {user_id} {user_full_name} {time.asctime()}. User left rating mode.')
                    await message.answer("Вы вышли из режима оценивания.", reply_markup=types.ReplyKeyboardRemove())
                else:
                    logging.info(f'Rate: {user_id} {user_full_name} {time.asctime()}. Sth wrong while rating.')
                    await message.answer("Что-то пошло не так. Попробуйте снова отправить целое число от 0 до 5."\
                                         "\nЧтобы выйти из режима оценивания, введите 'quit'.")
        else:
            await message.reply(config.ANSWER_ON_TEXT)
    except:
        logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Error in main_handler. Message: {message["text"]}.')
        await message.reply("Something went wrong...")  

# хендлер на фото   
@dp.message_handler(content_types=['photo'])
async def photo_id(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'Photo Handler: {user_id} {user_full_name} {time.asctime()}. Photo was received.')
    try:
        # загрузка фото и классификация изображенного на нем объекта
        io = BytesIO()
        await message.photo[-1].download(destination_file=io)
        image = Image.open(io)
        fd = hog(color.rgb2gray(image), 
            orientations=8, 
            pixels_per_cell=(16,16), 
            cells_per_block=(4, 4), 
            block_norm= 'L2'
        )
        fd = fd.reshape((1, fd.shape[0]))
        prediction = config.MODEL.predict(fd)
        config.statistics['amount_of_predictions'] += 1
        logging.info(f'Photo Handler: {user_id} {user_full_name} {time.asctime()}. Photo has been processed.')
        # добавление inline-клавиатуры
        keyboard = types.InlineKeyboardMarkup(resize_keyboard=True)
        keyboard.add(types.InlineKeyboardButton(text="Предсказание верно", callback_data="right_predict"))
        keyboard.add(types.InlineKeyboardButton(text="Предсказание неверно", callback_data="wrong_predict"))
        await message.answer('Овощ на фото - {}'.format(config.VEG_DICT[int(prediction[0])]), reply_markup=keyboard)
    except:
        logging.info(f'Photo Handler: {user_id} {user_full_name} {time.asctime()}. Exception while editing photo.')
        await message.answer('Something went wrong while editing photo.')
        
@dp.callback_query_handler(text=["right_predict", "wrong_predict"])
async def right_predict(call: types.CallbackQuery):
    user_id = call.from_user.id
    user_full_name = call.from_user.full_name
    config.statistics["accuracy"][0] += int(call.data == "right_predict")
    config.statistics["accuracy"][1] += 1
    await bot.edit_message_reply_markup(chat_id=call.from_user.id, message_id=call.message.message_id, reply_markup=None)
    logging.info(f'Callback Handler: {user_id} {user_full_name} {time.asctime()}. Callback was received.')
    await call.answer(text="Спасибо, что проголосовали!", show_alert=True)

