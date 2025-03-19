import os
import json
import asyncio
import uuid
from telegram.ext import Application, MessageHandler, filters
from vosk import Model, KaldiRecognizer
from punctuators.models import PunctCapSegModelONNX

# Убираем некритичное предупреждение huggingface_hub
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Загружаем модель Vosk
model_vosk = Model(os.path.join("models", "vosk-model-ru"))
SAMPLE_RATE = 16000
print('Голосовая модель загружена')

# Загружаем пунктуатор
punctuator = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    ort_providers=["CPUExecutionProvider"]

)
print('Пунктуатор загружен')

async def transcribe_audio(file_path: str) -> str:
    rec = KaldiRecognizer(model_vosk, SAMPLE_RATE)
    try:
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
        while True:
            data = await process.stdout.read(4000)
            if not data:
                break
            rec.AcceptWaveform(data)

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
        voice_file = await update.message.voice.get_file()
        await voice_file.download_to_drive(temp_file)

        await update.message.reply_text("Файл получен. Начинается распознавание...")

        # Получаем текст с помощью Vosk
        raw_text = await transcribe_audio(temp_file)

        # Улучшаем текст с помощью пунктуатора
        punctuated_result = punctuator.infer([raw_text], apply_sbd=True)

        # punctuator.infer возвращает список списков, берём первый элемент
        enhanced_text = " ".join(punctuated_result[0])

        print(f"Исходный текст: {raw_text}")
        print(f"Текст после пунктуации: {enhanced_text}")

        await update.message.reply_text(enhanced_text)

    except Exception as e:
        print(f"Исключение в voice_handler: {e}")
        await update.message.reply_text("Произошла ошибка при обработке голосового сообщения.")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Пожалуйста, задайте токен в переменной окружения TELEGRAM_BOT_TOKEN")

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    app.run_polling()


if __name__ == '__main__':
    main()
