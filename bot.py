import json
import logging
import os
import re
import sys
import discord
from discord import app_commands
from dotenv import load_dotenv
from groq import AsyncGroq

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VoiceNoted")

# Load environment variables
load_dotenv()

# Global variables for clients
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client: AsyncGroq | None = None


def validate_environment() -> None:
    """Validates required environment variables at startup and exits gracefully if missing."""
    global groq_client, DISCORD_TOKEN, GROQ_API_KEY
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not DISCORD_TOKEN or not GROQ_API_KEY:
        missing_vars = []
        if not DISCORD_TOKEN:
            missing_vars.append("DISCORD_TOKEN")
        if not GROQ_API_KEY:
            missing_vars.append("GROQ_API_KEY")

        logger.critical(
            "ERRO DE CONFIGURAÇÃO: As seguintes variáveis de ambiente obrigatórias não foram encontradas: %s.\n"
            "Crie ou edite o ficheiro .env no diretório raiz do projeto e defina os valores necessários.\n"
            "Consulte o ficheiro .env.example como modelo de referência.",
            ", ".join(missing_vars),
        )
        sys.exit(1)

    groq_client = AsyncGroq(api_key=GROQ_API_KEY)


# Constants
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB Groq Whisper API limit
VOICE_MESSAGE_FLAG = 8192  # Discord's IS_VOICE_MESSAGE bitflag (1 << 13)
AUDIO_EXTENSIONS = (".ogg", ".mp3", ".wav", ".m4a")

# Custom words persistence file for guild vocabularies
CUSTOM_WORDS_FILE = "custom_words.json"


def load_custom_words() -> dict[str, list[str]]:
    """Loads persistent custom words and jargon per guild from custom_words.json."""
    if os.path.exists(CUSTOM_WORDS_FILE):
        try:
            with open(CUSTOM_WORDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Erro ao carregar ficheiro de palavras personalizadas: %s", e)
    return {}


def save_custom_words(data: dict[str, list[str]]) -> None:
    """Saves persistent custom words to custom_words.json."""
    try:
        with open(CUSTOM_WORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Erro ao guardar ficheiro de palavras personalizadas: %s", e)


# Context prompt template for European Portuguese (PT-PT) with WoW/gaming terminology
BASE_TRANSCRIPTION_PROMPT = (
    "Transcrição em português de Portugal (PT-PT) de uma conversa entre gamers sobre World of Warcraft (WoW). "
    "Manter termos em inglês quando ditos: Guild (escrever sempre Guild, nunca guilde), Guild Master, GM, Raid, "
    "Raid Leader, RL, PUG, pugar, loot, loot drama, wipe, boss, aggro, pull, tank, healer, DPS, mythic, M+, "
    "dungeon, addon, Discord, spreadsheet, Paint. Calão português: gajos, malta, cenas, fixe, caralho, foda-se, caraças, acumulado."
)

# Common Whisper hallucinations on silent or background-noise-only audio
HALLUCINATION_PHRASES = {
    "obrigado.",
    "obrigado",
    "obrigada.",
    "obrigada",
    "muito obrigado.",
    "muito obrigado",
    "legendas pela comunidade...",
    "legendas pela comunidade",
    "legendas pela comunidade amara.org",
    "subtítulos realizados pela comunidade...",
    "subtítulos realizados pela comunidade",
    "transcrição pela comunidade...",
    "transcrição pela comunidade",
    "transcrição: ",
    "transcrição:",
    "...",
    ".",
}

# In-memory cache of server member display names per guild
guild_known_members: dict[int, set[str]] = {}


def extract_member_display_names(member: discord.Member | discord.User) -> list[str]:
    """
    Extracts ONLY the real server display name and, if multi-word,
    the primary first word (e.g. 'Bruno Silva' -> ['Bruno Silva', 'Bruno']).
    Strictly avoids artificial suffixes, guessing, or loose reductions.
    """
    if member.bot:
        return []

    display = member.display_name.strip()
    if not display:
        return []

    results: set[str] = {display}
    # If the display name has multiple words (e.g. "Bruno Silva"), also include the first word
    words = display.split()
    if len(words) > 1 and len(words[0]) >= 3:
        results.add(words[0])

    return list(results)


def track_guild_member(guild: discord.Guild | None, member: discord.Member | discord.User) -> None:
    """Caches a member's display name when they speak or join."""
    if not guild or member.bot:
        return
    if guild.id not in guild_known_members:
        guild_known_members[guild.id] = set()

    for name in extract_member_display_names(member):
        guild_known_members[guild.id].add(name)


def update_member_in_cache(guild: discord.Guild | None, before: discord.Member, after: discord.Member) -> None:
    """Updates the cached display names when a member updates their server nickname."""
    if not guild or after.bot:
        return
    cache = guild_known_members.setdefault(guild.id, set())
    for old_name in extract_member_display_names(before):
        cache.discard(old_name)
    for new_name in extract_member_display_names(after):
        cache.add(new_name)


def get_transcription_prompt(guild: discord.Guild | None = None) -> str:
    """
    Constructs a contextual Whisper prompt containing WoW jargon, PT-PT slang,
    real server member display names, and words added via /voicenoted addword.
    """
    names: set[str] = set()

    # 1. Custom names from environment variables (optional)
    custom_names_env = os.getenv("CUSTOM_NAMES", "")
    if custom_names_env:
        for n in custom_names_env.split(","):
            if n.strip():
                names.add(n.strip())

    # 2. Custom words registered via /voicenoted addword for this guild
    if guild:
        custom_words_data = load_custom_words()
        for w in custom_words_data.get(str(guild.id), []):
            if w.strip():
                names.add(w.strip())

    # 3. Add real member display names (from cache and guild.members)
    if guild:
        if guild.id in guild_known_members:
            names.update(guild_known_members[guild.id])
        if guild.members:
            for m in guild.members:
                for n in extract_member_display_names(m):
                    names.add(n)

    if names:
        # Keep top ~30 names/words to stay comfortably within Whisper prompt context limits
        top_names = sorted(names)[:35]
        return f"{BASE_TRANSCRIPTION_PROMPT} Nomes e termos deste servidor: {', '.join(top_names)}."

    return BASE_TRANSCRIPTION_PROMPT


def polish_transcript(text: str) -> str:
    """Post-processes Whisper transcriptions to normalize known gaming terms and phonetic slips."""
    # Ensure 'Guild' is written without an 'e' at the end
    text = re.sub(r'\bguilde\b', 'Guild', text, flags=re.IGNORECASE)
    # Fix 'Redes de 20' -> 'Raids de 20' (common acoustic slip in PT-PT gaming)
    text = re.sub(r'\b[Rr]edes de (\d+)\b', r'Raids de \1', text)
    text = re.sub(r'\b[Rr]ede de (\d+)\b', r'Raid de \1', text)
    return text


def format_paragraphs(text: str, max_sentences_per_para: int = 3, max_chars_per_para: int = 280) -> str:
    """
    Splits continuous transcription text into readable, natural paragraphs
    separated by blank lines for clean Discord presentation.
    """
    if not text:
        return ""

    # Split text into sentences retaining punctuation
    sentence_pattern = re.compile(r'([.!?]+[\s]+)')
    parts = sentence_pattern.split(text.strip())

    sentences: list[str] = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i] + parts[i + 1]
        if s.strip():
            sentences.append(s.strip())
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    if not sentences:
        return text

    paragraphs: list[str] = []
    current_para: list[str] = []
    current_len = 0

    for sentence in sentences:
        current_para.append(sentence)
        current_len += len(sentence)
        if len(current_para) >= max_sentences_per_para or current_len >= max_chars_per_para:
            paragraphs.append(" ".join(current_para))
            current_para = []
            current_len = 0

    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(paragraphs)


def split_text(text: str, max_length: int = 1900) -> list[str]:
    """
    Splits long transcripts cleanly on paragraph boundaries (\\n\\n),
    sentence boundaries (. , ! , ? , \\n), or spaces without cutting words in half.
    """
    if not text:
        return []

    text = text.strip()
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_length:
        segment = remaining[:max_length]
        split_pos = -1

        # 1. Priority 1: Paragraph boundaries (\n\n)
        pos = segment.rfind("\n\n")
        if pos != -1:
            split_pos = pos + 2

        # 2. Priority 2: Sentence boundaries (. , ! , ? , \n)
        if split_pos <= 0:
            for sep in ("\n", ". ", "! ", "? "):
                pos = segment.rfind(sep)
                if pos != -1:
                    candidate_pos = pos + len(sep)
                    if candidate_pos > split_pos:
                        split_pos = candidate_pos

        # 3. Priority 3: Whitespace
        if split_pos <= 0:
            split_pos = segment.rfind(" ")

        # 4. Fallback: hard cut
        if split_pos <= 0:
            split_pos = max_length

        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def is_audio_attachment(attachment: discord.Attachment) -> bool:
    """Checks if an attachment has an audio content type or audio file extension."""
    if attachment.content_type and attachment.content_type.startswith("audio/"):
        return True
    if attachment.filename and attachment.filename.lower().endswith(AUDIO_EXTENSIONS):
        return True
    return False


def get_audio_attachments(message: discord.Message) -> list[discord.Attachment]:
    """
    Identifies audio attachments in a message, checking:
    1. Direct attachments (including Discord voice notes with IS_VOICE_MESSAGE bitflag 8192).
    2. Forwarded message snapshots (Discord Message Forwarding feature).
    3. Referenced messages (replies to older voice notes).
    """
    audio_attachments: list[discord.Attachment] = []

    # 1. Direct attachments
    is_direct_voice = bool(message.flags.value & VOICE_MESSAGE_FLAG)
    for attachment in message.attachments:
        if is_direct_voice or is_audio_attachment(attachment):
            audio_attachments.append(attachment)

    # 2. Forwarded message snapshots (Discord Message Forwarding feature)
    snapshots = getattr(message, "message_snapshots", None)
    if snapshots:
        for snapshot in snapshots:
            snapshot_flags = getattr(snapshot, "flags", None)
            is_snapshot_voice = bool(snapshot_flags.value & VOICE_MESSAGE_FLAG) if snapshot_flags else False
            for attachment in getattr(snapshot, "attachments", []):
                if is_snapshot_voice or is_audio_attachment(attachment):
                    audio_attachments.append(attachment)

    # 3. Referenced messages (if a user replies to an existing voice message)
    if not audio_attachments and message.reference:
        ref_message = message.reference.resolved
        if isinstance(ref_message, discord.Message):
            is_ref_voice = bool(ref_message.flags.value & VOICE_MESSAGE_FLAG)
            for attachment in ref_message.attachments:
                if is_ref_voice or is_audio_attachment(attachment):
                    audio_attachments.append(attachment)

    return audio_attachments


# Configure Discord client intents
# Explicitly disable voice_states so the bot never connects to voice channels
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.voice_states = False

ENABLE_MEMBERS_INTENT = os.getenv("ENABLE_MEMBERS_INTENT", "false").lower() == "true"
if ENABLE_MEMBERS_INTENT:
    intents.members = True


class VoiceNotedClient(discord.Client):
    """Discord client with app_commands and optional background health-check HTTP server for cloud platforms."""

    def __init__(self, *, intents: discord.Intents) -> None:
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        port = os.getenv("PORT")
        if port:
            try:
                from aiohttp import web

                async def handle_health(request: web.Request) -> web.Response:
                    return web.Response(
                        text="VoiceNoted Discord Bot is active!\nStatus: 200 OK",
                        content_type="text/plain",
                    )

                app = web.Application()
                app.router.add_get("/", handle_health)
                app.router.add_get("/health", handle_health)

                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "0.0.0.0", int(port))
                await site.start()
                logger.info("Servidor HTTP de health-check ativo na porta %s (Render / Cloud)", port)
            except Exception as e:
                logger.exception("Erro ao iniciar servidor HTTP de health-check na porta %s: %s", port, e)


client = VoiceNotedClient(intents=intents)

# Slash command group /voicenoted
voicenoted_group = app_commands.Group(
    name="voicenoted",
    description="Comandos para gerir o vocabulário e nomes do VoiceNoted",
)


@voicenoted_group.command(name="addword", description="Adiciona uma palavra, nome ou jargão ao vocabulário de transcrição")
@app_commands.describe(word="A palavra, nome ou jargão a adicionar (ex: Caramelo, Tank, Healer, Ragnaros)")
async def add_word_cmd(interaction: discord.Interaction, word: str) -> None:
    guild_id = str(interaction.guild_id) if interaction.guild_id else "global"
    clean_word = word.strip()
    if not clean_word:
        await interaction.response.send_message("⚠️ Por favor introduz uma palavra válida.", ephemeral=True)
        return

    data = load_custom_words()
    guild_words = data.setdefault(guild_id, [])
    if clean_word not in guild_words:
        guild_words.append(clean_word)
        save_custom_words(data)
        logger.info("Palavra '%s' adicionada ao vocabulário do servidor %s por %s", clean_word, guild_id, interaction.user)
        await interaction.response.send_message(
            f"✅ A palavra **`{clean_word}`** foi adicionada ao vocabulário de transcrição deste servidor!",
        )
    else:
        await interaction.response.send_message(
            f"ℹ️ A palavra **`{clean_word}`** já existia no vocabulário deste servidor.",
            ephemeral=True,
        )


@voicenoted_group.command(name="removeword", description="Remove uma palavra personalizada do vocabulário")
@app_commands.describe(word="A palavra a remover")
async def remove_word_cmd(interaction: discord.Interaction, word: str) -> None:
    guild_id = str(interaction.guild_id) if interaction.guild_id else "global"
    clean_word = word.strip()

    data = load_custom_words()
    guild_words = data.get(guild_id, [])
    if clean_word in guild_words:
        guild_words.remove(clean_word)
        save_custom_words(data)
        logger.info("Palavra '%s' removida do vocabulário do servidor %s por %s", clean_word, guild_id, interaction.user)
        await interaction.response.send_message(f"🗑️ A palavra **`{clean_word}`** foi removida do vocabulário.")
    else:
        await interaction.response.send_message(
            f"⚠️ A palavra **`{clean_word}`** não foi encontrada no vocabulário personalizado.",
            ephemeral=True,
        )


@voicenoted_group.command(name="listwords", description="Lista as palavras e nomes reconhecidos neste servidor")
async def list_words_cmd(interaction: discord.Interaction) -> None:
    guild_id = str(interaction.guild_id) if interaction.guild_id else "global"
    data = load_custom_words()
    custom_list = data.get(guild_id, [])

    cached_members = sorted(guild_known_members.get(interaction.guild_id, set())) if interaction.guild_id else []

    msg_lines = ["📋 **Vocabulário de Transcrição do Servidor:**\n"]
    if custom_list:
        msg_lines.append(f"**Palavras Personalizadas (/voicenoted addword):**\n`{', '.join(custom_list)}`\n")
    else:
        msg_lines.append("*Nenhuma palavra personalizada adicionada ainda (usa `/voicenoted addword <palavra>`).*\n")

    if cached_members:
        msg_lines.append(f"**Nomes de Membros Reconhecidos:**\n`{', '.join(cached_members)}`")
    else:
        msg_lines.append("*Nenhum nome de membro em cache ainda.*")

    await interaction.response.send_message("\n".join(msg_lines), ephemeral=True)


@voicenoted_group.command(name="syncnames", description="Força a sincronização imediata dos nomes de exibição dos membros")
async def sync_names_cmd(interaction: discord.Interaction) -> None:
    if not interaction.guild:
        await interaction.response.send_message("⚠️ Este comando só pode ser usado dentro de um servidor.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    cache = guild_known_members.setdefault(interaction.guild.id, set())

    # Try to chunk all members if intent is enabled
    try:
        await interaction.guild.chunk()
    except Exception as e:
        logger.debug("Chunk de membros ignorado ou não suportado: %s", e)

    # 1. Inspect all members loaded in guild
    for member in interaction.guild.members:
        for n in extract_member_display_names(member):
            cache.add(n)

    # 2. Inspect members visible in the current channel
    if hasattr(interaction.channel, "members"):
        for member in interaction.channel.members:
            for n in extract_member_display_names(member):
                cache.add(n)

    await interaction.followup.send(
        f"🔄 Nomes dos membros sincronizados com sucesso! Total de {len(cache)} nome(s) de exibição registados.",
        ephemeral=True,
    )


client.tree.add_command(voicenoted_group)


@client.event
async def on_member_join(member: discord.Member) -> None:
    """Automatically learns the display name of newly joined members."""
    if member.guild:
        track_guild_member(member.guild, member)
        logger.info("Novo membro no servidor '%s': '%s' adicionado ao vocabulário", member.guild.name, member.display_name)


@client.event
async def on_member_remove(member: discord.Member) -> None:
    """Removes leaving members from dynamic vocabulary."""
    if member.guild and member.guild.id in guild_known_members:
        for name in extract_member_display_names(member):
            guild_known_members[member.guild.id].discard(name)


@client.event
async def on_member_update(before: discord.Member, after: discord.Member) -> None:
    """Automatically updates vocabulary when a member updates their server nickname or display name."""
    if before.display_name != after.display_name and after.guild:
        update_member_in_cache(after.guild, before, after)
        logger.info(
            "Membro atualizou nome em '%s': '%s' -> '%s'",
            after.guild.name,
            before.display_name,
            after.display_name,
        )


@client.event
async def on_ready() -> None:
    """Invoked when the bot has connected to Discord."""
    logger.info("==================================================")
    logger.info("Bot autenticado como: %s (ID: %s)", client.user.name, client.user.id)
    logger.info("Conectado a %d servidor(es)", len(client.guilds))

    for guild in client.guilds:
        # Sync slash commands to each guild immediately for instant appearance
        try:
            client.tree.copy_global_to(guild=guild)
            await client.tree.sync(guild=guild)
            logger.info("Comandos slash /voicenoted sincronizados no servidor '%s'", guild.name)
        except Exception as e:
            logger.warning("Não foi possível sincronizar slash commands no servidor '%s': %s", guild.name, e)

        # Cache member display names
        for member in guild.members:
            track_guild_member(guild, member)
        cached_count = len(guild_known_members.get(guild.id, set()))
        logger.info("Servidor '%s': %d nome(s) de exibição registados no vocabulário", guild.name, cached_count)

    logger.info(
        "Intents configurados - message_content: True, voice_states: False, members: %s",
        ENABLE_MEMBERS_INTENT,
    )
    logger.info("Pronto para transcrever notas de voz (PT-PT)!")
    logger.info("==================================================")


@client.event
async def on_message(message: discord.Message) -> None:
    """Processes incoming messages to detect and transcribe voice notes and audio."""
    # Ignore messages sent by bots (including our own messages)
    if message.author.bot:
        return

    # Dynamically register author names/nicknames into guild vocabulary
    if message.guild:
        track_guild_member(message.guild, message.author)

    # Check for voice messages or audio attachments
    audio_attachments = get_audio_attachments(message)
    if not audio_attachments:
        return

    for attachment in audio_attachments:
        # Check size before downloading to conserve bandwidth and respect Groq's 25MB limit
        if attachment.size and attachment.size > MAX_AUDIO_BYTES:
            logger.warning(
                "Anexo de áudio '%s' na mensagem %s excede 25MB (%d bytes)",
                attachment.filename,
                message.id,
                attachment.size,
            )
            await message.reply(
                "⚠️ O áudio excede o limite de 25MB da API.",
                mention_author=False,
            )
            continue

        try:
            # Show typing indicator in the channel while downloading & transcribing
            async with message.channel.typing():
                audio_bytes = await attachment.read()

                # Guard check against downloaded payload size
                if len(audio_bytes) > MAX_AUDIO_BYTES:
                    await message.reply(
                        "⚠️ O áudio excede o limite de 25MB da API.",
                        mention_author=False,
                    )
                    continue

                filename = attachment.filename or "voice.ogg"
                content_type = attachment.content_type or "audio/ogg"

                logger.info(
                    "A transcrever áudio da mensagem %s (autor: %s, ficheiro: %s, tamanho: %d bytes)",
                    message.id,
                    message.author,
                    filename,
                    len(audio_bytes),
                )

                # Dynamically construct context prompt with gaming terms and member names
                dynamic_prompt = get_transcription_prompt(message.guild)

                # Send in-memory buffer to Groq Whisper (strictly transcriptions, never translations)
                model_name = os.getenv("GROQ_MODEL", "whisper-large-v3")
                temperature = float(os.getenv("WHISPER_TEMPERATURE", "0.2"))

                transcription = await groq_client.audio.transcriptions.create(
                    file=(filename, audio_bytes, content_type),
                    model=model_name,
                    language="pt",
                    temperature=temperature,
                    prompt=dynamic_prompt,
                )

                raw_text = transcription.text.strip() if transcription.text else ""

                # Anti-hallucination guard: silent audio or empty text
                if not raw_text:
                    logger.info("Transcrição vazia na mensagem %s; nenhuma resposta enviada.", message.id)
                    continue

                if raw_text.lower() in HALLUCINATION_PHRASES:
                    logger.info(
                        "Alucinação de silêncio detetada ('%s') na mensagem %s; nenhuma resposta enviada.",
                        raw_text,
                        message.id,
                    )
                    continue

                # Polish gaming terminology and format into readable paragraphs
                polished_text = polish_transcript(raw_text)
                formatted_text = format_paragraphs(polished_text)

                # Split long transcripts cleanly (under Discord 2000-char limit)
                chunks = split_text(formatted_text, max_length=1900)
                if not chunks:
                    continue

                # Determine header based on whether audio was forwarded or recorded directly
                is_forwarded = bool(getattr(message, "message_snapshots", None))
                if is_forwarded:
                    header = f"🎙️ **Transcrição** • *Nota Encaminhada (enviada por **{message.author.display_name}**)*"
                else:
                    header = f"🎙️ **Transcrição** • *(áudio de **{message.author.display_name}**)*"

                # Reply to the original voice message with the first chunk
                first_reply = f"{header}\n\n{chunks[0]}"
                await message.reply(first_reply, mention_author=False)

                # Send subsequent chunks sequentially as follow-ups
                for chunk in chunks[1:]:
                    await message.channel.send(chunk)

                logger.info(
                    "Transcrição enviada com sucesso para a mensagem %s (%d bloco(s))",
                    message.id,
                    len(chunks),
                )

        except Exception as e:
            logger.exception("Erro ao transcrever anexo de áudio na mensagem %s: %s", message.id, e)
            try:
                await message.reply(
                    "⚠️ Ocorreu um erro ao processar a transcrição do áudio.",
                    mention_author=False,
                )
            except Exception:
                logger.exception("Falha ao enviar mensagem de erro no Discord para a mensagem %s", message.id)


def main() -> None:
    """Bot entry point."""
    validate_environment()
    logger.info("A iniciar VoiceNoted Discord Bot...")
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
