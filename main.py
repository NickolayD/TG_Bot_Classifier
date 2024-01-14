from fastapi import FastAPI
from aiogram import types, Dispatcher, Bot
from bot import dp, bot
from config import TELEGRAM_BOT_TOKEN, WEB_SERVICE_NAME


WEBHOOK_PATH = f"/bot/{TELEGRAM_BOT_TOKEN}"
WEBHOOK_URL = f"{WEB_SERVICE_NAME}{WEBHOOK_PATH}"

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    webhook_info = await bot.get_webhook_info()
    if webhook_info.url != WEBHOOK_URL:
        await bot.set_webhook(url=WEBHOOK_URL)

@app.post(WEBHOOK_PATH)
async def bot_webhook(update: dict):
    telegram_update = types.Update(**update)
    Dispatcher.set_current(dp)
    Bot.set_current(bot)
    await dp.process_update(telegram_update)

@app.on_event("shutdown")
async def on_shutdown():
    session = await bot.get_session()
    await session.close()
    
@app.get("/")
async def main_web_handler():
    return "Everything is OK!"
