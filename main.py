#!/usr/bin/env python

import os
import random

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
)
from telegram.ext import filters


def getenv_or_die(name):
    value = os.getenv(name)
    if value is None:
        print(f"Environment variable {name} is not set.")
        exit(1)
    return value


OPENAPI_KEY = getenv_or_die("OPENAPI_KEY")
TG_TOKEN = getenv_or_die("TG_TOKEN")
TG_MENTION = getenv_or_die("TG_MENTION")

client = OpenAI(api_key=OPENAPI_KEY)

# transformer = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading model...")
transformer = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Load user messages from file
print("Loading messages...")
with open("./maverik.txt", "r") as file:
    messages = [x.strip() for x in file.readlines()]


if not os.path.exists("./embeddings.txt"):
    # Encode user messages
    print("Encoding messages...")
    embeddings = transformer.encode(messages)

    # Save embeddings to file
    print("Saving embeddings...")
    with open("./embeddings.txt", "w") as file:
        for emb in embeddings:
            file.write(" ".join([str(x) for x in emb]) + "\n")
else:
    # Load embeddings from file
    print("Loading embeddings...")
    with open("./embeddings.txt", "r") as file:
        embeddings = []
        for line in file:
            embeddings.append([float(x) for x in line.strip().split()])


def prompt(message):
    # Read input from terminal
    # message = input("> ")

    print("Finding similar messages...")

    # Find 20 most similar user messages across 1000 random samples
    message_embedding = transformer.encode([message])[0]
    similarities = []
    for i, emb in enumerate(random.sample(embeddings, 1000)):
        similarity = transformer.similarity(message_embedding, emb)
        similarities.append((i, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    # for i in range(20):
    #     print(messages[similarities[i][0]].strip(), similarities[i][1])

    texts = [messages[similarities[i][0]].strip() for i in range(20)]
    # Add 20 more random messages
    texts += random.sample(messages, 20)

    print("Similar messages:")
    for text in texts:
        print(text)

    print("Generating response...")

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Тебе звати Маверік, ти - людина, що спілкується у чаті."},
            {"role": "system", "content": "Ти зазвичай говориш в такому стилі:"},
            {"role": "system", "content": "\n".join([f"{text}" for text in texts])},
            {
                "role": "system",
                "content": "Тобі пишуть повідомлення і ти маєш відповісти на нього в стилі Маверіка, використовуючи вказану лексику.",
            },
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Маверік: "},
        ],
    )

    result = completion.choices[0].message.content

    print("Response:", result)

    return result


app = ApplicationBuilder().token(TG_TOKEN).build()


# React to user messages that mention the bot
async def mentioned(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    replied_to = update.message.reply_to_message
    if (
        f"@{TG_MENTION}" in message
        or "маверік" in message.lower()
        or (replied_to and replied_to.from_user.username == TG_MENTION)
    ):
        await update.get_bot().send_chat_action(
            update.message.chat.id, ChatAction.TYPING
        )
        message = message.replace(f"@{TG_MENTION}", "").strip()
        response = prompt(message)
        # response = "You said: " + message
        await update.message.reply_text(response)


app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), mentioned))

print("Starting polling...")

app.run_polling()
