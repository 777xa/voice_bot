import os
import json
import asyncio
import uuid
from telegram.ext import Application, MessageHandler, filters
from vosk import Model, KaldiRecognizer

# Загружаем модель Vosk (укажите корректный путь)
model = Model("models/vosk-model-small-ru")
SAMPLE_RATE = 16000


async def transcribe_audio(file_path: str) -> str:
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    try:
        # Асинхронно запускаем ffmpeg для конвертации в сырые PCM данные
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i", file_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    except Exception as e:
        print(f"Ошибка при запуске ffmpeg: {e}")
        return "Ошибка конвертации"

    try:
        # Читаем данные блоками (по 4000 байт)
        while True:
            data = await process.stdout.read(4000)
            if not data:
                break
            rec.AcceptWaveform(data)
        # Ждём завершения процесса
        await process.wait()
        result_json = rec.FinalResult()
        result_data = json.loads(result_json)
        text = result_data.get("text", "").strip()
        if not text:
            text = "Не удалось распознать речь."
        return text
    except Exception as e:
        print(f"Ошибка во время распознавания: {e}")
        return "Ошибка распознавания"

async def voice_handler(update, context):

    temp_file = f'voice{uuid.uuid4().hex}.ogg'

    try:
        # Скачиваем голосовое сообщение (формат OGG)
        voice_file = await update.message.voice.get_file()
        await voice_file.download_to_drive(temp_file)
        # Уведомляем пользователя
        await update.message.reply_text("Файл получен. Начинается распознавание...")
        # Асинхронно распознаём аудио
        transcribed_text = await transcribe_audio(temp_file)
        print(transcribed_text)
        await update.message.reply_text(transcribed_text)
    except Exception as e:
        print(f"Исключение в voice_handler: {e}")
        await update.message.reply_text("Произошла ошибка при обработке голосового сообщения.")
    finally:
        # Удаляем временный файл, если он существует
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"Ошибка при удалении файла: {e}")

# Пример регистрации обработчика в боте
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Пожалуйста, задайте токен в переменной окружения TELEGRAM_BOT_TOKEN")

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    app.run_polling()

if __name__ == '__main__':
    main()