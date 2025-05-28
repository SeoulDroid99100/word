import pyrogram
from pyrogram import Client, filters
from pyrogram.enums import ChatAction
import wordfreq
import nltk
from nltk.corpus import words
import re
import random
import asyncio
import json
import os
import aiofiles
from typing import Dict, Set, Optional
from dotenv import load_dotenv
from pyrogram.errors import FloodWait

# Download NLTK words corpus if not already present
try:
    nltk.data.find('corpora/words')
except LookupError:
    try:
        nltk.download('words')
    except Exception as e:
        print(f"Failed to download NLTK words corpus: {e}")

# Load environment variables
load_dotenv()
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
SESSION_STRING = os.getenv("SESSION_STRING", "")
LOG_CHAT_ID = int(os.getenv("LOG_CHAT_ID", "0"))

# Authorized user IDs
ADMIN_IDS = {6783092268, 7360592638}

# Initialize Pyrogram client
app = Client(
    "word_game_userbot",
    api_id=API_ID,
    api_hash=API_HASH,
    session_string=SESSION_STRING
)

# Data structures
enabled_chats: Dict[int, Dict[str, str]] = {}  # chat_id -> {alias, name, case}
used_words: Dict[int, Set[str]] = {}  # chat_id -> set of used words
CONFIG_FILE = "chat_config.json"
LETTER_FREQUENCY = None  # Cache for letter frequency
INITIALIZED = False  # Flag to ensure load_config runs only once
last_bot_message_id: Dict[int, int] = {}  # Track last bot message ID per chat

# Function to load chat config
async def load_config():
    global enabled_chats, used_words
    if os.path.exists(CONFIG_FILE):
        try:
            async with aiofiles.open(CONFIG_FILE, 'r') as f:
                data = json.loads(await f.read())
                enabled_chats = {int(k): v for k, v in data.get('enabled_chats', {}).items()}
                used_words = {int(k): set(v) for k, v in data.get('used_words', {}).items()}
        except Exception as e:
            await safe_send_message(LOG_CHAT_ID, f"Failed to load config: {e}")
            enabled_chats = {}
            used_words = {}
    else:
        enabled_chats = {}
        used_words = {}

# Function to save chat config
async def save_config():
    try:
        async with aiofiles.open(CONFIG_FILE, 'w') as f:
            await f.write(json.dumps({
                'enabled_chats': enabled_chats,
                'used_words': {k: list(v) for k, v in used_words.items()}
            }))
    except Exception as e:
        await safe_send_message(LOG_CHAT_ID, f"Failed to save config: {e}")

# Function to generate 4-digit alias
def generate_alias() -> str:
    return str(random.randint(1000, 9999))

# Function to count words starting with each letter in wordfreq
def get_letter_frequency() -> Dict[str, int]:
    global LETTER_FREQUENCY
    if LETTER_FREQUENCY is None:
        LETTER_FREQUENCY = {chr(i): 0 for i in range(ord('a'), ord('z') + 1)}
        wordfreq_words = wordfreq.top_n_list('en', 300000)  # Top 10,000 words
        for word in wordfreq_words:
            if word and re.match(r'^[a-zA-Z]+$', word):
                first_letter = word.lower()[0]
                if first_letter in LETTER_FREQUENCY:
                    LETTER_FREQUENCY[first_letter] += 1
    return LETTER_FREQUENCY

# Function for safe message sending with flood control
async def safe_send_message(chat_id, text, **kwargs):
    try:
        message = await app.send_message(chat_id, text, **kwargs)
        if chat_id in enabled_chats and 'disable_notification' in kwargs and kwargs['disable_notification']:
            last_bot_message_id[chat_id] = message.id  # Track bot's game word message
        return message
    except FloodWait as e:
        await asyncio.sleep(e.x)
        message = await app.send_message(chat_id, text, **kwargs)
        if chat_id in enabled_chats and 'disable_notification' in kwargs and kwargs['disable_notification']:
            last_bot_message_id[chat_id] = message.id
        return message
    except Exception as e:
        print(f"Error sending message to {chat_id}: {e}")
        if chat_id != LOG_CHAT_ID:  # Avoid infinite recursion
            await safe_send_message(LOG_CHAT_ID, f"Error sending message to {chat_id}: {e}")
        return None

# Function to retrieve game word
async def get_game_word(start_letter: str, min_length: int, chat_id: int, case: str) -> Optional[str]:
    """
    Get a word starting with start_letter, at least min_length, for the given chat.
    Case 1: Use wordfreq (highest frequency), then NLTK (alphabetical).
    Case 2: Use wordfreq (top 10,000, prefer least frequent ending letters), then NLTK.
    Avoid chat-specific duplicates. Capitalize first letter for output.
    """
    if chat_id not in used_words:
        used_words[chat_id] = set()
    
    # Step 1: Try wordfreq
    wordfreq_words = wordfreq.top_n_list('en', 321180)  # Top 10,000 words
    matching_words = [
        word for word in wordfreq_words
        if len(word) >= min_length
        and word.lower().startswith(start_letter.lower())
        and re.match(r'^[a-zA-Z]+$', word)
        and word.lower() not in used_words[chat_id]
    ]
    
    if matching_words:
        if case == '1':
            # Case 1: Pick highest frequency word
            word_freq = [(word, wordfreq.word_frequency(word.lower(), 'en')) for word in matching_words if len(word) >= min_length]
            if not word_freq:
                await safe_send_message(LOG_CHAT_ID, f"No wordfreq words meet min_length {min_length} for '{start_letter}' in chat {chat_id}")
            else:
                word_freq.sort(key=lambda x: x[1], reverse=True)
                selected_word = word_freq[0][0]
                frequency = word_freq[0][1]
                if len(selected_word) >= min_length:
                    used_words[chat_id].add(selected_word.lower())
                    await save_config()
                    await safe_send_message(LOG_CHAT_ID, f"Sent word (Case 1): {selected_word} (length={len(selected_word)}, freq={frequency:.6f}) to chat {chat_id} ({enabled_chats[chat_id]['name']})")
                    return selected_word[0].upper() + selected_word[1:].lower()
                else:
                    await safe_send_message(LOG_CHAT_ID, f"Selected word {selected_word} (length={len(selected_word)}) does not meet min_length {min_length} in Case 1")
        elif case == '2':
            # Case 2: Prefer words ending with least frequent letters
            letter_counts = get_letter_frequency()
            min_count = min(letter_counts.values())
            least_frequent_letters = [letter for letter, count in letter_counts.items() if count == min_count]
            default_letters = ['x', 'y', 'z']
            target_end_letters = least_frequent_letters if least_frequent_letters else default_letters
            
            # Filter words ending with target letters, ensuring min_length
            valid_words = [
                (word, wordfreq.word_frequency(word.lower(), 'en'))
                for word in matching_words
                if word.lower()[-1] in target_end_letters and len(word) >= min_length
            ]
            if valid_words:
                # Pick highest frequency among valid words
                valid_words.sort(key=lambda x: x[1])
                selected_word = valid_words[0][0]
                frequency = valid_words[0][1]
                if len(selected_word) >= min_length:
                    used_words[chat_id].add(selected_word.lower())
                    await save_config()
                    await safe_send_message(LOG_CHAT_ID, f"Sent word (Case 2, ends with {selected_word.lower()[-1]}): {selected_word} (length={len(selected_word)}, freq={frequency:.6f}) to chat {chat_id}")
                    return selected_word[0].upper() + selected_word[1:].lower()
                else:
                    await safe_send_message(LOG_CHAT_ID, f"Selected word {selected_word} (length={len(selected_word)}) does not meet min_length {min_length} in Case 2")
            # Fallback to any wordfreq word
            word_freq = [(word, wordfreq.word_frequency(word.lower(), 'en')) for word in matching_words if len(word) >= min_length]
            if word_freq:
                word_freq.sort(key=lambda x: x[1])
                selected_word = word_freq[0][0]
                frequency = word_freq[0][1]
                if len(selected_word) >= min_length:
                    used_words[chat_id].add(selected_word.lower())
                    await save_config()
                    await safe_send_message(LOG_CHAT_ID, f"Sent word (Case 2, no target ending): {selected_word} (length={len(selected_word)}, freq={frequency:.6f}) to chat {chat_id} ({enabled_chats[chat_id]['name']})")
                    return selected_word[0].upper() + selected_word[1:].lower()
                else:
                    await safe_send_message(LOG_CHAT_ID, f"Selected word {selected_word} (length={len(selected_word)}) does not meet min_length {min_length} in Case 2")
    
    # Step 2: Fall back to NLTK
    nltk_word_set = set(words.words())
    nltk_matching_words = [
        word for word in nltk_word_set
        if len(word) >= min_length
        and word.lower().startswith(start_letter.lower())
        and re.match(r'^[a-zA-Z]+$', word)
        and word.lower() not in used_words[chat_id]
    ]
    
    if nltk_matching_words:
        if case == '1':
            # Case 1: Pick first alphabetically
            nltk_matching_words.sort()
            selected_word = nltk_matching_words[0]
            if len(selected_word) >= min_length:
                used_words[chat_id].add(selected_word.lower())
                await save_config()
                await safe_send_message(LOG_CHAT_ID, f"Sent word (Case 1, NLTK): {selected_word} (length={len(selected_word)}) to chat {chat_id} ({enabled_chats[chat_id]['name']})")
                return selected_word[0].upper() + selected_word[1:].lower()
            else:
                await safe_send_message(LOG_CHAT_ID, f"Selected NLTK word {selected_word} (length={len(selected_word)}) does not meet min_length {min_length} in Case 1")
        elif case == '2':
            # Case 2: Try words ending with least frequent letters
            letter_counts = get_letter_frequency()
            min_count = min(letter_counts.values())
            least_frequent_letters = [letter for letter, count in letter_counts.items() if count == min_count]
            default_letters = ['x', 'y', 'z']
            target_end_letters = least_frequent_letters if least_frequent_letters else default_letters
            
            valid_nltk_words = [
                word for word in nltk_matching_words
                if word.lower()[-1] in target_end_letters and len(word) >= min_length
            ]
            if valid_nltk_words:
                valid_nltk_words.sort()
                selected_word = valid_nltk_words[0]
                if len(selected_word) >= min_length:
                    used_words[chat_id].add(selected_word.lower())
                    await save_config()
                    await safe_send_message(LOG_CHAT_ID, f"Sent word (Case 2, NLTK, ends with {selected_word.lower()[-1]}): {selected_word} (length={len(selected_word)}) to chat {chat_id} ({enabled_chats[chat_id]['name']})")
                    return selected_word[0].upper() + selected_word[1:].lower()
                else:
                    await safe_send_message(LOG_CHAT_ID, f"Selected NLTK word {selected_word} (length={len(selected_word)}) does not meet min_length {min_length} in Case 2")
            # Fallback to any NLTK word
            nltk_matching_words.sort()
            selected_word = nltk_matching_words[0]
            if len(selected_word) >= min_length:
                used_words[chat_id].add(selected_word.lower())
                await save_config()
                await safe_send_message(LOG_CHAT_ID, f"Sent word (Case 2, NLTK, no target ending): {selected_word} (length={len(selected_word)}) to chat {chat_id} ({enabled_chats[chat_id]['name']})")
                return selected_word[0].upper() + selected_word[1:].lower()
            else:
                await safe_send_message(LOG_CHAT_ID, f"Selected NLTK word {selected_word} (length={len(selected_word)}) does not meet min_length {min_length} in Case 2")
    
    await safe_send_message(LOG_CHAT_ID, f"No valid word found for '{start_letter}' with min length {min_length} in chat {chat_id} ({enabled_chats[chat_id]['name']})")
    return None

# Command handler: Enable chat
@app.on_message(filters.command("on"))
async def enable_chat(client, message):
    if message.from_user.id not in ADMIN_IDS:
        print(f"Unauthorized /on attempt by user {message.from_user.id}")
        return
    if len(message.command) != 3:
        await safe_send_message(LOG_CHAT_ID, "Usage: /on {chat_id} {case}")
        return
    try:
        chat_id = int(message.command[1])
        case = message.command[2]
        if case not in ['1', '2']:
            await safe_send_message(LOG_CHAT_ID, f"Failed to enable chat {chat_id}: Invalid case {case}")
            return
        if chat_id not in enabled_chats:
            chat = await client.get_chat(chat_id)
            chat_name = chat.title if chat.type in ["group", "supergroup"] else chat.username or f"{chat.first_name or ''} {chat.last_name or ''}".strip()
            alias = generate_alias()
            enabled_chats[chat_id] = {"alias": alias, "name": chat_name, "case": case}
            used_words[chat_id] = set()
            await save_config()
            log_message = f"Enabled chat {chat_id} ({chat_name}) with alias {alias}, case {case}"
            if case == '2':
                log_message += " (Danger Mode)"
            await safe_send_message(LOG_CHAT_ID, log_message)
        else:
            await safe_send_message(LOG_CHAT_ID, f"Chat {chat_id} ({enabled_chats[chat_id]['name']}) is already enabled with alias {enabled_chats[chat_id]['alias']}, case {enabled_chats[chat_id]['case']}")
    except (ValueError, pyrogram.errors.exceptions.bad_request_400.PeerIdInvalid):
        await safe_send_message(LOG_CHAT_ID, f"Failed to enable chat: Invalid chat ID {message.command[1]}")

# Command handler: Disable chat
@app.on_message(filters.command("off"))
async def disable_chat(client, message):
    if message.from_user.id not in ADMIN_IDS:
        print(f"Unauthorized /off attempt by user {message.from_user.id}")
        return
    if len(message.command) != 2:
        await safe_send_message(LOG_CHAT_ID, "Usage: /off {chat_id}")
        return
    try:
        chat_id = int(message.command[1])
        if chat_id in enabled_chats:
            alias = enabled_chats[chat_id]["alias"]
            name = enabled_chats[chat_id]["name"]
            case = enabled_chats[chat_id]["case"]
            enabled_chats.pop(chat_id)
            used_words.pop(chat_id, None)
            await save_config()
            await safe_send_message(LOG_CHAT_ID, f"Disabled chat {chat_id} ({name}) with alias {alias}, case {case}")
        else:
            await safe_send_message(LOG_CHAT_ID, f"Failed to disable chat {chat_id}: Not enabled")
    except ValueError:
        await safe_send_message(LOG_CHAT_ID, f"Failed to disable chat: Invalid chat ID {message.command[1]}")

# Command handler: Clear used words
@app.on_message(filters.command("clear"))
async def clear_words(client, message):
    if message.from_user.id not in ADMIN_IDS:
        print(f"Unauthorized /clear attempt by user {message.from_user.id}")
        return
    if len(message.command) != 2:
        await safe_send_message(LOG_CHAT_ID, "Usage: /clear {chat_id}")
        return
    try:
        chat_id = int(message.command[1])
        if chat_id in enabled_chats:
            used_words[chat_id] = set()
            await save_config()
            await safe_send_message(LOG_CHAT_ID, f"Cleared used words for chat {chat_id} ({enabled_chats[chat_id]['name']}) with alias {enabled_chats[chat_id]['alias']}, case {enabled_chats[chat_id]['case']}")
        else:
            await safe_send_message(LOG_CHAT_ID, f"Failed to clear words for chat {chat_id}: Not enabled")
    except ValueError:
        await safe_send_message(LOG_CHAT_ID, f"Failed to clear words: Invalid chat ID {message.command[1]}")

# Command handler: Show enabled chats
@app.on_message(filters.command("runs"))
async def show_enabled_chats(client, message):
    if message.from_user.id not in ADMIN_IDS:
        print(f"Unauthorized /runs attempt by user {message.from_user.id}")
        return
    if enabled_chats:
        response = "Enabled chats:\n"
        for chat_id, info in enabled_chats.items():
            response += f"Chat ID: {chat_id}, Name: {info['name']}, Alias: {info['alias']}, Case: {info['case']}\n"
        await safe_send_message(LOG_CHAT_ID, f"Listed enabled chats:\n{response}")
    else:
        await safe_send_message(LOG_CHAT_ID, "No chats are enabled")

# Game message handler
@app.on_message(filters.text & filters.group)
async def handle_game_message(client, message):
    chat_id = message.chat.id
    if chat_id not in enabled_chats:
        return
    
    # Pattern for game prompt
    prompt_pattern = r"Turn: X @ja \(Next: .+?\)\nYour word must start with (\w) and include at least (\d+) letters\."
    # Pattern for invalid word reply
    invalid_pattern = r"^(\w+) is not in my list of words\.$"
    
    if re.match(prompt_pattern, message.text, re.MULTILINE):
        match = re.match(prompt_pattern, message.text, re.MULTILINE)
        start_letter = match.group(1)
        min_length = int(match.group(2))
        case = enabled_chats[chat_id]['case']
        
        # Send typing action for 3 seconds
        try:
            await client.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(2)
        except Exception as e:
            await safe_send_message(LOG_CHAT_ID, f"Error sending typing action to {chat_id}: {e}")
        
        # Get and send word
        word = await get_game_word(start_letter, min_length, chat_id, case)
        if word:
            await safe_send_message(chat_id, word, disable_notification=True)
        # No message sent to chat if no word is found
    
    elif message.reply_to_message and message.reply_to_message.id == last_bot_message_id.get(chat_id):
        # Check if message is a reply to bot's last word
        invalid_match = re.match(invalid_pattern, message.text)
        if invalid_match:
            invalid_word = invalid_match.group(1)
            # Add invalid word to used_words to avoid reuse
            used_words[chat_id].add(invalid_word.lower())
            await save_config()
            await safe_send_message(LOG_CHAT_ID, f"Word '{invalid_word}' rejected in chat {chat_id} ({enabled_chats[chat_id]['name']}). Retrying...")
            
            # Get stored start_letter and min_length (assume from last prompt; you may need to store these)
            # For simplicity, we'll need to re-fetch the prompt message to get start_letter and min_length
            try:
                # Fetch the prompt message (assuming it's recent; adjust logic if needed)
                async for msg in app.get_chat_history(chat_id, limit=10):
                    if re.match(prompt_pattern, msg.text, re.MULTILINE):
                        match = re.match(prompt_pattern, msg.text, re.MULTILINE)
                        start_letter = match.group(1)
                        min_length = int(match.group(2))
                        case = enabled_chats[chat_id]['case']
                        break
                else:
                    await safe_send_message(LOG_CHAT_ID, f"Could not find recent prompt for retry in chat {chat_id}")
                    return
            except Exception as e:
                await safe_send_message(LOG_CHAT_ID, f"Error fetching prompt for retry in chat {chat_id}: {e}")
                return
            
            # Send typing action for 1 second
            try:
                await client.send_chat_action(chat_id, ChatAction.TYPING)
                await asyncio.sleep(1.5)
            except Exception as e:
                await safe_send_message(LOG_CHAT_ID, f"Error sending retry typing action to {chat_id}: {e}")
            
            # Retry with same parameters
            word = await get_game_word(start_letter, min_length, chat_id, case)
            if word:
                await safe_send_message(chat_id, word, disable_notification=True)
            else:
                await safe_send_message(LOG_CHAT_ID, f"No valid retry word found for '{start_letter}' with min length {min_length} in chat {chat_id}")

# Startup handler using raw update
@app.on_raw_update()
async def on_startup(client, update, users, chats):
    global INITIALIZED
    if not INITIALIZED:
        try:
            await load_config()
            await safe_send_message(LOG_CHAT_ID, "Bot started successfully")
            INITIALIZED = True
        except Exception as e:
            print(f"Failed to initialize bot: {e}")
            await safe_send_message(LOG_CHAT_ID, f"Failed to initialize bot: {e}")

# Run the bot
try:
    print("Bot is running...")
    app.run()
except Exception as e:
    print(f"Bot failed to start: {e}")
