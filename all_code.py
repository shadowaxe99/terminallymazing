from realtime_ai_character.models.user import User
from realtime_ai_character.models.interaction import Interaction
from realtime_ai_character.database.base import Base  # import the Base model
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from logging.config import fileConfig
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add the project root to the system path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

# import your models here

# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config
database_url = os.getenv('DATABASE_URL') if os.getenv('DATABASE_URL') else 'sqlite:///./test.db'
config.set_main_option('sqlalchemy.url', database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata  # use your Base metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
"""Added interaction table

Revision ID: 0f355a71adbb
Revises: ead242c61258
Create Date: 2023-06-26 22:07:19.624594

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0f355a71adbb'
down_revision = 'ead242c61258'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('interactions',
                    sa.Column('id', sa.Integer(),
                              primary_key=True, index=True),
                    sa.Column('client_id', sa.Integer(), nullable=True),
                    sa.Column('client_message', sa.String(), nullable=True),
                    sa.Column('server_message', sa.String(), nullable=True),
                    sa.Column('timestamp', sa.DateTime(), nullable=True),
                    sa.PrimaryKeyConstraint('id')
                    )


def downgrade() -> None:
    op.drop_table('interactions')
"""Change message schema to unicode

Revision ID: 27fe156a6d72
Revises: 9ed6d1431c1d
Create Date: 2023-07-18 22:32:03.388403

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '27fe156a6d72'
down_revision = '9ed6d1431c1d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('interactions', sa.Column('client_message_unicode', sa.Unicode(65535)))
    op.add_column('interactions', sa.Column('server_message_unicode', sa.Unicode(65535)))

def downgrade() -> None:
    op.drop_column('interactions', 'client_message_unicode')
    op.drop_column('interactions', 'server_message_unicode')
"""Add session ID

Revision ID: 3821f7adaca9
Revises: 27fe156a6d72
Create Date: 2023-07-18 22:44:33.107380

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3821f7adaca9'
down_revision = '27fe156a6d72'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('interactions', sa.Column('session_id', sa.String(50), nullable=True))


def downgrade() -> None:
    op.drop_column('interactions', 'session_id')
"""add platform and action types

Revision ID: 9ed6d1431c1d
Revises: 0f355a71adbb
Create Date: 2023-07-18 00:31:05.044828

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9ed6d1431c1d'
down_revision = '0f355a71adbb'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('interactions', sa.Column('platform', sa.String(50), nullable=True))
    op.add_column('interactions', sa.Column('action_type', sa.String(50), nullable=True))


def downgrade() -> None:
    op.drop_column('interactions', 'platform')
    op.drop_column('interactions', 'action_type')
"""Added user table

Revision ID: ead242c61258
Revises: 
Create Date: 2023-06-26 16:25:00.614978

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ead242c61258'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('users',
                    sa.Column('id', sa.Integer(), primary_key=True),
                    sa.Column('name', sa.String(), nullable=True),
                    sa.Column('email', sa.String(),
                              nullable=False, unique=True),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
"""Add string user ID

Revision ID: eced1ae3918a
Revises: 3821f7adaca9
Create Date: 2023-07-19 11:02:52.002939

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'eced1ae3918a'
down_revision = '3821f7adaca9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('interactions', sa.Column('user_id', sa.String(50), nullable=True))

    # Populate the new column with the old column's data
    op.execute("""
    UPDATE interactions
    SET user_id = CAST(client_id AS TEXT)
    """)

    # TODO: make the user_id column non-nullable after prod migration.
    # Skip for now given production servers are distributed. Note this is not
    # relevant if you deploy locally.


def downgrade() -> None:
    op.drop_column('interactions', 'user_id')
#!/usr/bin/env python
"""A CLI for building an running RealChar project locally."""
import click
import os
import subprocess


@click.group()
def cli():
    pass


@click.command()
@click.option('--name', default="realtime-ai-character", help='The name to give to your Docker image.')
@click.option('--rebuild', is_flag=True, help='Flag to indicate whether to rebuild the Docker image.')
def docker_build(name, rebuild):
    if rebuild or not image_exists(name):
        click.secho(f"Building Docker image: {name}...", fg='green')
        if (image_exists(name)):
            subprocess.run(["docker", "rmi", "-f", name])
        subprocess.run(["docker", "build", "-t", name, "."])
    else:
        click.secho(
            f"Docker image: {name} already exists. Skipping build. To rebuild, use --rebuild option", fg='yellow')


@click.command()
@click.option('--name', default="realtime-ai-character", help='The name of the Docker image to run.')
@click.option('--db-file', default=None, help='Path to the database file to mount inside the container.')
def docker_run(name, db_file):
    click.secho(f"Running Docker image: {name}...", fg='green')
    if not os.path.isfile('.env'):
        click.secho(
            "Warning: .env file not found. Running without environment variables.", fg='yellow')
    # Remove existing container if it exists
    subprocess.run(["docker", "rm", "-f", name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if db_file:
        subprocess.run(["docker", "run", "--env-file", ".env", "--name", name, "-p", "8000:8000",
                        "-v", f"{os.path.abspath(db_file)}:/realtime_ai_character/test.db", name])
    else:
        subprocess.run(["docker", "run", "--env-file", ".env",
                       "--name", name, "-p", "8000:8000", name])


@click.command()
@click.option('--name', default="realtime-ai-character", help='The name of the Docker image to delete.')
def docker_delete(name):
    if image_exists(name):
        click.secho(f"Deleting Docker image: {name}...", fg='green')
        subprocess.run(["docker", "rmi", "-f", name])
    else:
        click.secho(f"Docker image: {name} does not exist.", fg='yellow')


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def run_uvicorn(args):
    click.secho("Running uvicorn server...", fg='green')
    subprocess.run(["uvicorn", "realtime_ai_character.main:app",
                   "--ws-ping-interval", "60", "--ws-ping-timeout", "60", "--timeout-keep-alive", "60"] + list(args))


def image_exists(name):
    result = subprocess.run(
        ["docker", "image", "inspect", name], capture_output=True, text=True)
    return result.returncode == 0


cli.add_command(docker_build)
cli.add_command(docker_run)
cli.add_command(docker_delete)
cli.add_command(run_uvicorn)


if __name__ == '__main__':
    cli()
import os
import queue
import asyncio
import concurrent.futures
import functools
import io
import sys
import random
from threading import Thread
import time

from dotenv import load_dotenv

import pyaudio
import speech_recognition as sr
import websockets
from aioconsole import ainput  # for async input
from pydub import AudioSegment
from simpleaudio import WaveObject

load_dotenv()

executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


class AudioPlayer:
    def __init__(self):
        self.play_thread = None
        self.stop_flag = False
        self.queue = queue.Queue()

    def play_audio(self):
        while not self.stop_flag or not self.queue.empty():
            try:
                wav_data = self.queue.get_nowait()
            except queue.Empty:
                continue

            wave_obj = WaveObject.from_wave_file(wav_data)
            play_obj = wave_obj.play()

            while play_obj.is_playing() and not self.stop_flag:
                time.sleep(0.1)

            if self.stop_flag:
                play_obj.stop()

    def start_playing(self, wav_data):
        self.stop_flag = False
        self.queue.put(wav_data)

        if self.play_thread is None or not self.play_thread.is_alive():
            self.play_thread = Thread(target=self.play_audio)
            self.play_thread.start()

    def stop_playing(self):
        if self.play_thread and self.play_thread.is_alive():
            self.stop_flag = True
            self.play_thread.join()
            self.play_thread = None

    def add_to_queue(self, wav_data):
        self.queue.put(wav_data)


audio_player = AudioPlayer()


def get_input_device_id():
    p = pyaudio.PyAudio()
    devices = [(i, p.get_device_info_by_index(i)['name'])
               for i in range(p.get_device_count())
               if p.get_device_info_by_index(i).get('maxInputChannels')]

    print('Available devices:')
    for id, name in devices:
        print(f"Device id {id} - {name}")

    return int(input('Please select device id: '))


async def handle_audio(websocket, device_id):
    with sr.Microphone(device_index=device_id, sample_rate=RATE) as source:
        recognizer = sr.Recognizer()
        print('Source sample rate: ', source.SAMPLE_RATE)
        print('Source width: ', source.SAMPLE_WIDTH)
        print('Adjusting for ambient noise...Wait for 2 seconds')
        recognizer.energy_threshold = 5000
        recognizer.dynamic_energy_ratio = 6
        recognizer.dynamic_energy_adjustment_damping = 0.85
        recognizer.non_speaking_duration = 0.5
        recognizer.pause_threshold = 0.8
        recognizer.phrase_threshold = 0.5
        recognizer.adjust_for_ambient_noise(source, duration=2)
        listen_func = functools.partial(
            recognizer.listen, source, phrase_time_limit=30)

        print('Okay, start talking!')
        while True:
            print('[*]', end="")  # indicate that we are listening
            audio = await asyncio.get_event_loop().run_in_executor(executor, listen_func)
            await websocket.send(audio.frame_data)
            print('[-]', end="")  # indicate that we are done listening
            await asyncio.sleep(2)


async def handle_text(websocket):
    print('You: ', end="", flush=True)
    while True:
        message = await ainput()
        await websocket.send(message)


async def receive_message(websocket):
    while True:
        try:
            message = await websocket.recv()
        except websockets.exceptions.ConnectionClosedError as e:
            print("Connection closed unexpectedly: ", e)
            break
        except Exception as e:
            print("An error occurred: ", e)
            break

        if isinstance(message, str):
            if message == '[end]\n':
                print('\nYou: ', end="", flush=True)
            elif message.startswith('[+]'):
                # stop playing audio
                audio_player.stop_playing()
                # indicate the transcription is done
                print(f"\n{message}", end="\n", flush=True)
            elif message.startswith('[=]'):
                # indicate the response is done
                print(f"{message}", end="\n", flush=True)
            else:
                print(f"{message}", end="", flush=True)
        elif isinstance(message, bytes):
            audio_data = io.BytesIO(message)
            audio = AudioSegment.from_mp3(audio_data)
            wav_data = io.BytesIO()
            audio.export(wav_data, format="wav")
            # Start playing audio
            audio_player.start_playing(wav_data)
        else:
            print("Unexpected message")
            break


def select_model():
    llm_model_selection = input(
        '1: gpt-3.5-turbo-16k \n'
        '2: gpt-4 \n'
        '3: claude-2 \n'
        'Select llm model:')
    if llm_model_selection == '1':
        llm_model = 'gpt-3.5-turbo-16k'
    elif llm_model_selection == '2':
        llm_model = 'gpt-4'
    elif llm_model_selection == '3':
        llm_model = 'claude-2'
    return llm_model


async def start_client(client_id, url):
    api_key = os.getenv('AUTH_API_KEY')
    llm_model = select_model()
    uri = f"ws://{url}/ws/{client_id}?api_key={api_key}&llm_model={llm_model}"
    async with websockets.connect(uri) as websocket:
        # send client platform info
        await websocket.send('terminal')
        print(f"Client #{client_id} connected to server")
        welcome_message = await websocket.recv()
        print(f"{welcome_message}")
        character = input('Select character: ')
        await websocket.send(character)

        mode = input('Select mode (1: audio, 2: text): ')
        if mode.lower() == '1':
            device_id = get_input_device_id()
            send_task = asyncio.create_task(handle_audio(websocket, device_id))
        else:
            send_task = asyncio.create_task(handle_text(websocket))

        receive_task = asyncio.create_task(receive_message(websocket))
        await asyncio.gather(receive_task, send_task)


async def main(url):
    client_id = random.randint(0, 1000000)
    task = asyncio.create_task(start_client(client_id, url))
    try:
        await task
    except KeyboardInterrupt:
        task.cancel()
        await asyncio.wait_for(task, timeout=None)
        print("Client stopped by user")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else 'localhost:8000'
    asyncio.run(main(url))
import os

from realtime_ai_character.audio.speech_to_text.base import SpeechToText


def get_speech_to_text() -> SpeechToText:
    use = os.getenv('SPEECH_TO_TEXT_USE', 'LOCAL_WHISPER')
    if use == 'GOOGLE':
        from realtime_ai_character.audio.speech_to_text.google import Google
        Google.initialize()
        return Google.get_instance()
    elif use == 'LOCAL_WHISPER':
        from realtime_ai_character.audio.speech_to_text.whisper import Whisper
        Whisper.initialize(use='local')
        return Whisper.get_instance()
    elif use == 'OPENAI_WHISPER':
        from realtime_ai_character.audio.speech_to_text.whisper import Whisper
        Whisper.initialize(use='api')
        return Whisper.get_instance()
    else:
        raise NotImplementedError(f'Unknown speech to text engine: {use}')
from abc import ABC, abstractmethod


class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes, platform='web', prompt='') -> str:
        # platform: 'web' | 'mobile' | 'terminal'
        pass
from google.cloud import speech
import types

from realtime_ai_character.audio.speech_to_text.base import SpeechToText
from realtime_ai_character.logger import get_logger
from realtime_ai_character.utils import Singleton

logger = get_logger(__name__)
config = types.SimpleNamespace(**{
    'web': {
        'encoding': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        'sample_rate_hertz': 48000,
        'language_code': 'en-US',
        'max_alternatives': 1,
    },
    'terminal': {
        'encoding': speech.RecognitionConfig.AudioEncoding.LINEAR16,
        'sample_rate_hertz': 44100,
        'language_code': 'en-US',
        'max_alternatives': 1,
    },
})


class Google(Singleton, SpeechToText):
    def __init__(self):
        super().__init__()
        logger.info("Setting up [Google Speech to Text]...")
        self.client = speech.SpeechClient()

    def transcribe(self, audio_bytes, platform, prompt='') -> str:
        batch_config = speech.RecognitionConfig({
            'speech_contexts': [speech.SpeechContext(phrases=prompt.split(','))],
            **config.__dict__[platform]})
        response = self.client.recognize(
            config=batch_config,
            audio=speech.RecognitionAudio(content=audio_bytes)
        )
        if not response.results:
            return ''
        result = response.results[0]
        if not result.alternatives:
            return ''
        return result.alternatives[0].transcript
import io
import os
import types
import wave

import speech_recognition as sr
import whisper
from pydub import AudioSegment

from realtime_ai_character.audio.speech_to_text.base import SpeechToText
from realtime_ai_character.logger import get_logger
from realtime_ai_character.utils import Singleton

DEBUG = False
logger = get_logger(__name__)
config = types.SimpleNamespace(**{
    'model': 'tiny',
    'language': 'en',
    'api_key': os.getenv("OPENAI_API_KEY"),
})


class Whisper(Singleton, SpeechToText):
    def __init__(self, use='local'):
        super().__init__()
        if use == 'local':
            logger.info(f"Loading [Local Whisper] model: [{config.model}]...")
            self.model = whisper.load_model(config.model)
        self.recognizer = sr.Recognizer()
        self.use = use
        if DEBUG:
            self.wf = wave.open('output.wav', 'wb')
            self.wf.setnchannels(1)  # Assuming mono audio
            self.wf.setsampwidth(2)  # Assuming 16-bit audio
            self.wf.setframerate(44100)  # Assuming 44100Hz sample rate

    def transcribe(self, audio_bytes, platform, prompt=''):
        logger.info("Transcribing audio...")
        if platform == 'web':
            audio = self._convert_webm_to_wav(audio_bytes)
        else:
            audio = sr.AudioData(audio_bytes, 44100, 2)
        if self.use == 'local':
            return self._transcribe(audio, prompt)
        elif self.use == 'api':
            return self._transcribe_api(audio, prompt)

    def _transcribe(self, audio, prompt=''):
        text = self.recognizer.recognize_whisper(
            audio,
            model=config.model,
            language=config.language,
            show_dict=True,
            initial_prompt=prompt
        )['text']
        return text

    def _transcribe_api(self, audio, prompt=''):
        text = self.recognizer.recognize_whisper_api(
            audio,
            api_key=config.api_key,
        )
        return text

    def _convert_webm_to_wav(self, webm_data):
        webm_audio = AudioSegment.from_file(
            io.BytesIO(webm_data), format="webm")
        wav_data = io.BytesIO()
        webm_audio.export(wav_data, format="wav")
        with sr.AudioFile(wav_data) as source:
            audio = self.recognizer.record(source)
        return audio
import os

from realtime_ai_character.audio.text_to_speech.base import TextToSpeech


def get_text_to_speech() -> TextToSpeech:
    use = os.getenv('TEXT_TO_SPEECH_USE', 'ELEVEN_LABS')
    if use == 'ELEVEN_LABS':
        from realtime_ai_character.audio.text_to_speech.elevenlabs import ElevenLabs
        ElevenLabs.initialize()
        return ElevenLabs.get_instance()
    else:
        raise NotImplementedError(f'Unknown text to speech engine: {use}')
from abc import ABC, abstractmethod


class TextToSpeech(ABC):
    @abstractmethod
    async def stream(self, *args, **kwargs):
        pass
import asyncio
import os
import types
import httpx

from realtime_ai_character.logger import get_logger
from realtime_ai_character.utils import Singleton
from realtime_ai_character.audio.text_to_speech.base import TextToSpeech

logger = get_logger(__name__)
DEBUG = False

config = types.SimpleNamespace(**{
    'default_voice': '21m00Tcm4TlvDq8ikWAM',
    'default_female_voice': 'EXAVITQu4vr4xnSDxMaL',
    'default_male_voice': 'ErXwobaYiN019PkySvjV',
    'chunk_size': 1024,
    'url': 'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream',
    'headers': {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': os.environ['ELEVEN_LABS_API_KEY']
    },
    'data': {
        'model_id': 'eleven_monolingual_v1',
        'voice_settings': {
            'stability': 0.5,
            'similarity_boost': 0.75
        }
    }
})


class ElevenLabs(Singleton, TextToSpeech):
    def __init__(self):
        super().__init__()
        logger.info("Initializing [ElevenLabs Text To Speech] voices...")
        self.voice_ids = {
            "Raiden Shogun And Ei": os.environ.get('RAIDEN_VOICE') or config.default_female_voice,
            "Loki": os.environ.get('LOKI_VOICE') or config.default_male_voice,
            "Reflection Pi": os.environ.get('PI_VOICE') or config.default_female_voice,
            "Elon Musk": os.environ.get('ELON_VOICE') or config.default_male_voice,
            "Bruce Wayne": os.environ.get('BRUCE_VOICE') or config.default_male_voice,
            "Steve Jobs": os.environ.get('JOBS_VOICE') or config.default_male_voice
        }

    def get_voice_id(self, name):
        return self.voice_ids.get(name, config.default_voice)

    async def stream(self, text, websocket, tts_event: asyncio.Event, characater_name="", first_sentence=False) -> None:
        if DEBUG:
            return
        headers = config.headers
        data = {
            "text": text,
            **config.data,
        }
        voice_id = self.get_voice_id(characater_name)
        url = config.url.format(voice_id=voice_id)
        if first_sentence:
            url = url + '?optimize_streaming_latency=4'
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers)
            async for chunk in response.aiter_bytes():
                await asyncio.sleep(0.1)
                if tts_event.is_set():
                    # stop streaming audio
                    break
                await websocket.send_bytes(chunk)
import os
from dotenv import load_dotenv
from pathlib import Path
from contextlib import ExitStack
from realtime_ai_character.logger import get_logger
from realtime_ai_character.utils import Singleton, Character
from realtime_ai_character.database.chroma import get_chroma
from llama_index import SimpleDirectoryReader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
logger = get_logger(__name__)


class CatalogManager(Singleton):
    def __init__(self, overwrite=True):
        super().__init__()
        self.db = get_chroma()
        if overwrite:
            logger.info('Overwriting existing data in the chroma.')
            self.db.delete_collection()
            self.db = get_chroma()

        self.characters = {}
        self.load_characters(overwrite)
        if overwrite:
            logger.info('Persisting data in the chroma.')
            self.db.persist()
        logger.info(
            f"Total document load: {self.db._client.get_collection('llm').count()}")

    def get_character(self, name) -> Character:
        return self.characters.get(name)

    def load_character(self, directory):
        with ExitStack() as stack:
            f_system = stack.enter_context(open(directory / 'system'))
            f_user = stack.enter_context(open(directory / 'user'))
            system_prompt = f_system.read()
            user_prompt = f_user.read()

        name = directory.stem.replace('_', ' ').title()

        self.characters[name] = Character(
            name=name,
            llm_system_prompt=system_prompt,
            llm_user_prompt=user_prompt
        )
        return name

    def load_characters(self, overwrite):
        """
        Load characters from the character_catalog directory. Use /data to create
        documents and add them to the chroma.

        :overwrite: if True, overwrite existing data in the chroma.
        """
        path = Path(__file__).parent
        excluded_dirs = {'__pycache__', 'archive'}

        directories = [d for d in path.iterdir() if d.is_dir()
                       and d.name not in excluded_dirs]

        for directory in directories:
            character_name = self.load_character(directory)
            if overwrite:
                self.load_data(character_name, directory / 'data')
                logger.info('Loaded data for character: ' + character_name)
        logger.info(
            f'Loaded {len(self.characters)} characters: names {list(self.characters.keys())}')

    def load_data(self, character_name: str, data_path: str):
        loader = SimpleDirectoryReader(Path(data_path))
        documents = loader.load_data()
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=500,
            chunk_overlap=100)
        docs = text_splitter.create_documents(
            texts=[d.text for d in documents],
            metadatas=[{
                'character_name': character_name,
                'id': d.id_,
            } for d in documents])
        self.db.add_documents(docs)


def get_catalog_manager():
    return CatalogManager.get_instance()


if __name__ == '__main__':
    manager = CatalogManager.get_instance()
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from realtime_ai_character.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
if os.getenv('OPENAI_API_TYPE') == 'azure':
    embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), deployment=os.getenv("OPENAI_API_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002"), chunk_size=1)

def get_chroma():
    chroma = Chroma(
        collection_name='llm',
        embedding_function=embedding,
        persist_directory='./chroma.db'
    )
    return chroma
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv
import os

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith(
    "sqlite") else {}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args=connect_args
)

SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    print(SQLALCHEMY_DATABASE_URL)
    from realtime_ai_character.models.user import User
    from realtime_ai_character.models.interaction import Interaction
    with SessionLocal() as session:
        print(session.query(User).all())
        session.delete(User(name="Test", email="text@gmail.com"))
        session.commit()

        print(session.query(User).all())
        session.query(User).filter(User.name == "Test").delete()
        session.commit()

        print(session.query(User).all())
from realtime_ai_character.llm.base import AsyncCallbackAudioHandler, AsyncCallbackTextHandler, LLM


def get_llm(model='gpt-3.5-turbo-16k') -> LLM:
    if model.startswith('gpt'):
        from realtime_ai_character.llm.openai_llm import OpenaiLlm
        return OpenaiLlm(model=model)
    elif model.startswith('claude'):
        from realtime_ai_character.llm.anthropic_llm import AnthropicLlm
        return AnthropicLlm(model=model)
    else:
        raise ValueError(f'Invalid llm model: {model}')
import os
from typing import List

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatAnthropic
from langchain.schema import BaseMessage, HumanMessage

from realtime_ai_character.database.chroma import get_chroma
from realtime_ai_character.llm.base import AsyncCallbackAudioHandler, AsyncCallbackTextHandler, LLM
from realtime_ai_character.logger import get_logger
from realtime_ai_character.utils import Character

logger = get_logger(__name__)


class AnthropicLlm(LLM):
    def __init__(self, model):
        self.chat_anthropic = ChatAnthropic(
            model=model,
            temperature=0.5,
            streaming=True
        )
        self.db = get_chroma()

    async def achat(self,
                    history: List[BaseMessage],
                    user_input: str,
                    user_input_template: str,
                    callback: AsyncCallbackTextHandler,
                    audioCallback: AsyncCallbackAudioHandler,
                    character: Character) -> str:
        # 1. Generate context
        context = self._generate_context(user_input, character)

        # 2. Add user input to history
        history.append(HumanMessage(content=user_input_template.format(
            context=context, query=user_input)))

        # 3. Generate response
        response = await self.chat_anthropic.agenerate(
            [history], callbacks=[callback, audioCallback, StreamingStdOutCallbackHandler()])
        logger.info(f'Response: {response}')
        return response.generations[0][0].text

    def _generate_context(self, query, character: Character) -> str:
        docs = self.db.similarity_search(query)
        docs = [d for d in docs if d.metadata['character_name'] == character.name]
        logger.info(f'Found {len(docs)} documents')

        context = '\n'.join([d.page_content for d in docs])
        return context
from abc import ABC, abstractmethod

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

StreamingStdOutCallbackHandler.on_chat_model_start = lambda *args, **kwargs: None


class AsyncCallbackTextHandler(AsyncCallbackHandler):
    def __init__(self, on_new_token=None, token_buffer=None, on_llm_end=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_new_token = on_new_token
        self._on_llm_end = on_llm_end
        self.token_buffer = token_buffer

    async def on_chat_model_start(self, *args, **kwargs):
        pass

    async def on_llm_new_token(self, token: str, *args, **kwargs):
        if self.token_buffer is not None:
            self.token_buffer.append(token)
        await self.on_new_token(token)

    async def on_llm_end(self, *args, **kwargs):
        if self._on_llm_end is not None:
            await self._on_llm_end(''.join(self.token_buffer))
            self.token_buffer.clear()


class AsyncCallbackAudioHandler(AsyncCallbackHandler):
    def __init__(self, text_to_speech=None, websocket=None, tts_event=None, character_name="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if text_to_speech is None:
            def text_to_speech(token): return print(
                f'New audio token: {token}')
        self.text_to_speech = text_to_speech
        self.websocket = websocket
        self.current_sentence = ""
        self.character_name = character_name
        self.is_reply = False  # the start of the reply. i.e. the substring after '>'
        self.tts_event = tts_event
        # optimization: trade off between latency and quality for the first sentence
        self.is_first_sentence = True

    async def on_chat_model_start(self, *args, **kwargs):
        pass

    async def on_llm_new_token(self, token: str, *args, **kwargs):
        if not self.is_reply and token == ">":
            self.is_reply = True
        elif self.is_reply:
            if token != ".":
                self.current_sentence += token
            else:
                await self.text_to_speech.stream(
                    self.current_sentence,
                    self.websocket,
                    self.tts_event,
                    self.character_name,
                    self.is_first_sentence)
                self.current_sentence = ""
                if self.is_first_sentence:
                    self.is_first_sentence = False

    async def on_llm_end(self, *args, **kwargs):
        if self.current_sentence != "":
            await self.text_to_speech.stream(
                self.current_sentence,
                self.websocket, self.tts_event, self.character_name, self.is_first_sentence)


class LLM(ABC):
    @abstractmethod
    async def achat(self, *args, **kwargs):
        pass
import os
from typing import List

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
if os.getenv('OPENAI_API_TYPE') == 'azure':
    from langchain.chat_models import AzureChatOpenAI
else:
    from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage

from realtime_ai_character.database.chroma import get_chroma
from realtime_ai_character.llm.base import AsyncCallbackAudioHandler, AsyncCallbackTextHandler, LLM
from realtime_ai_character.logger import get_logger
from realtime_ai_character.utils import Character

logger = get_logger(__name__)

class OpenaiLlm(LLM):
    def __init__(self, model):
        if os.getenv('OPENAI_API_TYPE') == 'azure':
            self.chat_open_ai = AzureChatOpenAI(
                deployment_name=os.getenv('OPENAI_API_MODEL_DEPLOYMENT_NAME', 'gpt-35-turbo'),
                model=model,
                temperature=0.5,
                streaming=True
            )
        else:
            self.chat_open_ai = ChatOpenAI(
                model=model,
                temperature=0.5,
                streaming=True
            )
        self.db = get_chroma()

    async def achat(self,
                    history: List[BaseMessage],
                    user_input: str,
                    user_input_template: str,
                    callback: AsyncCallbackTextHandler,
                    audioCallback: AsyncCallbackAudioHandler,
                    character: Character) -> str:
        # 1. Generate context
        context = self._generate_context(user_input, character)

        # 2. Add user input to history
        history.append(HumanMessage(content=user_input_template.format(
            context=context, query=user_input)))

        # 3. Generate response
        response = await self.chat_open_ai.agenerate(
            [history], callbacks=[callback, audioCallback, StreamingStdOutCallbackHandler()])
        logger.info(f'Response: {response}')
        return response.generations[0][0].text

    def _generate_context(self, query, character: Character) -> str:
        docs = self.db.similarity_search(query)
        docs = [d for d in docs if d.metadata['character_name'] == character.name]
        logger.info(f'Found {len(docs)} documents')

        context = '\n'.join([d.page_content for d in docs])
        return context
import logging

formatter = '%(asctime)s - %(funcName)s - %(filename)s - %(levelname)s - %(message)s'


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    ch_format = logging.Formatter(formatter)
    console_handler.setFormatter(ch_format)

    logger.addHandler(console_handler)

    return logger
import os
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from realtime_ai_character.audio.speech_to_text import get_speech_to_text
from realtime_ai_character.audio.text_to_speech import get_text_to_speech
from realtime_ai_character.character_catalog.catalog_manager import CatalogManager
from realtime_ai_character.restful_routes import router as restful_router
from realtime_ai_character.utils import ConnectionManager
from realtime_ai_character.websocket_routes import router as websocket_router

load_dotenv()

app = FastAPI()

app.include_router(restful_router)
app.include_router(websocket_router)
app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'static')), name="static")


# initializations
CatalogManager.initialize(overwrite=True)
ConnectionManager.initialize()
get_text_to_speech()
get_speech_to_text()

# suppress deprecation warnings
warnings.filterwarnings("ignore", module="whisper")
from sqlalchemy import Column, Integer, String, DateTime, Unicode
import datetime
from realtime_ai_character.database.base import Base


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True, nullable=False)
    client_id = Column(Integer) # deprecated, use user_id instead
    user_id = Column(String(50))
    session_id = Column(String(50))
    client_message = Column(String) # deprecated, use client_message_unicode instead
    server_message = Column(String) # deprecated, use server_message_unicode instead
    client_message_unicode = Column(Unicode(65535))
    server_message_unicode = Column(Unicode(65535))

    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    platform = Column(String(50))
    action_type = Column(String(50))

    def save(self, db):
        db.add(self)
        db.commit()
from sqlalchemy import Column, Integer, String
from realtime_ai_character.database.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)

    def save(self, db):
        db.add(self)
        db.commit()
import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'static'))


@router.get("/status")
async def status():
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
from dataclasses import field
from typing import List

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic.dataclasses import dataclass
from starlette.websockets import WebSocket, WebSocketState


@dataclass
class Character:
    name: str
    llm_system_prompt: str
    llm_user_prompt: str


@dataclass
class ConversationHistory:
    system_prompt: str = ''
    user: list[str] = field(default_factory=list)
    ai: list[str] = field(default_factory=list)

    def __iter__(self):
        yield self.system_prompt
        for user_message, ai_message in zip(self.user, self.ai):
            yield user_message
            yield ai_message


def build_history(conversation_history: ConversationHistory) -> List[BaseMessage]:
    history = []
    for i, message in enumerate(conversation_history):
        if i == 0:
            history.append(SystemMessage(content=message))
        elif i % 2 == 0:
            history.append(AIMessage(content=message))
        else:
            history.append(HumanMessage(content=message))
    return history


class Singleton:
    _instances = {}

    @classmethod
    def get_instance(cls, *args, **kwargs):
        """ Static access method. """
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args, **kwargs)

        return cls._instances[cls]

    @classmethod
    def initialize(cls, *args, **kwargs):
        """ Static access method. """
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args, **kwargs)


class ConnectionManager(Singleton):
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client #{id(websocket)} left the chat")
        # await self.broadcast_message(f"Client #{id(websocket)} left the chat")

    async def send_message(self, message: str, websocket: WebSocket):
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_text(message)

    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            if connection.application_state == WebSocketState.CONNECTED:
                await connection.send_text(message)


def get_connection_manager():
    return ConnectionManager.get_instance()
import asyncio
import os
import uuid

from fastapi import APIRouter, Depends, Path, WebSocket, WebSocketDisconnect, Query
from requests import Session

from realtime_ai_character.audio.speech_to_text import (SpeechToText,
                                                        get_speech_to_text)
from realtime_ai_character.audio.text_to_speech import (TextToSpeech,
                                                        get_text_to_speech)
from realtime_ai_character.character_catalog.catalog_manager import (
    CatalogManager, get_catalog_manager)
from realtime_ai_character.database.connection import get_db
from realtime_ai_character.llm import (AsyncCallbackAudioHandler,
                                       AsyncCallbackTextHandler, get_llm, LLM)
from realtime_ai_character.logger import get_logger
from realtime_ai_character.models.interaction import Interaction
from realtime_ai_character.utils import (ConversationHistory, build_history,
                                         get_connection_manager)


logger = get_logger(__name__)

router = APIRouter()

manager = get_connection_manager()

GREETING_TXT = 'Hi, my friend, what brings you here today?'


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
        websocket: WebSocket,
        client_id: int = Path(...),
        api_key: str = Query(None),
        llm_model: str = Query(default=os.getenv(
            'LLM_MODEL_USE', 'gpt-3.5-turbo-16k')),
        db: Session = Depends(get_db),
        catalog_manager=Depends(get_catalog_manager),
        speech_to_text=Depends(get_speech_to_text),
        text_to_speech=Depends(get_text_to_speech)):
    # basic authentication
    if os.getenv('USE_AUTH', '') and api_key != os.getenv('AUTH_API_KEY'):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    # TODO: replace client_id with user_id completely.
    user_id = str(client_id)
    llm = get_llm(model=llm_model)
    await manager.connect(websocket)
    try:
        main_task = asyncio.create_task(
            handle_receive(websocket, client_id, db, llm, catalog_manager, speech_to_text, text_to_speech))

        await asyncio.gather(main_task)

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        await manager.broadcast_message(f"User #{user_id} left the chat")


async def handle_receive(
        websocket: WebSocket,
        client_id: int,
        db: Session,
        llm: LLM,
        catalog_manager: CatalogManager,
        speech_to_text: SpeechToText,
        text_to_speech: TextToSpeech):
    try:
        conversation_history = ConversationHistory()
        # TODO: clean up client_id once migration is done.
        user_id = str(client_id)
        session_id = str(uuid.uuid4().hex)

        # 0. Receive client platform info (web, mobile, terminal)
        data = await websocket.receive()
        if data['type'] != 'websocket.receive':
            raise WebSocketDisconnect('disconnected')
        platform = data['text']
        logger.info(f"User #{user_id}:{platform} connected to server with "
                    f"session_id {session_id}")

        # 1. User selected a character
        character = None
        character_list = list(catalog_manager.characters.keys())
        user_input_template = 'Context:{context}\n User:{query}'
        while not character:
            character_message = "\n".join(
                [f"{i+1} - {character}" for i, character in enumerate(character_list)])
            await manager.send_message(
                message=f"Select your character by entering the corresponding number:\n{character_message}\n",
                websocket=websocket)
            data = await websocket.receive()

            if data['type'] != 'websocket.receive':
                raise WebSocketDisconnect('disconnected')

            if not character and 'text' in data:
                selection = int(data['text'])
                if selection > len(character_list) or selection < 1:
                    await manager.send_message(
                        message=f"Invalid selection. Select your character [{', '.join(catalog_manager.characters.keys())}]\n",
                        websocket=websocket)
                    continue
                character = catalog_manager.get_character(
                    character_list[selection - 1])
                conversation_history.system_prompt = character.llm_system_prompt
                user_input_template = character.llm_user_prompt
                logger.info(
                    f"User #{user_id} selected character: {character.name}")

        tts_event = asyncio.Event()
        tts_task = None
        previous_transcript = None
        token_buffer = []

        # Greet the user
        await manager.send_message(message=GREETING_TXT, websocket=websocket)
        tts_task = asyncio.create_task(text_to_speech.stream(
            text=GREETING_TXT,
            websocket=websocket,
            tts_event = tts_event,
            characater_name = character.name,
            first_sentence=True,
        ))

        async def on_new_token(token):
            return await manager.send_message(message=token, websocket=websocket)

        async def stop_audio():
            if tts_task and not tts_task.done():
                tts_event.set()
                tts_task.cancel()
                if previous_transcript:
                    conversation_history.user.append(previous_transcript)
                    conversation_history.ai.append(' '.join(token_buffer))
                    token_buffer.clear()
                try:
                    await tts_task
                except asyncio.CancelledError:
                    pass
                tts_event.clear()

        while True:
            data = await websocket.receive()
            if data['type'] != 'websocket.receive':
                raise WebSocketDisconnect('disconnected')

            # handle text message
            if 'text' in data:
                msg_data = data['text']
                # 0. itermidiate transcript starts with [&]
                if msg_data.startswith('[&]'):
                    logger.info(f'intermediate transcript: {msg_data}')
                    if not os.getenv('EXPERIMENT_CONVERSATION_UTTERANCE', ''):
                        continue
                    asyncio.create_task(stop_audio())
                    asyncio.create_task(llm.achat_utterances(
                        history=build_history(conversation_history),
                        user_input=msg_data,
                        callback=AsyncCallbackTextHandler(on_new_token, []),
                        audioCallback=AsyncCallbackAudioHandler(text_to_speech, websocket, tts_event, character.name)))
                    continue
                # 1. Send message to LLM
                response = await llm.achat(
                    history=build_history(conversation_history),
                    user_input=msg_data,
                    user_input_template=user_input_template,
                    callback=AsyncCallbackTextHandler(
                        on_new_token, token_buffer),
                    audioCallback=AsyncCallbackAudioHandler(
                        text_to_speech, websocket, tts_event, character.name),
                    character=character)

                # 2. Send response to client
                await manager.send_message(message='[end]\n', websocket=websocket)

                # 3. Update conversation history
                conversation_history.user.append(msg_data)
                conversation_history.ai.append(response)
                token_buffer.clear()
                # 4. Persist interaction in the database
                Interaction(
                    client_id=client_id,
                    user_id=user_id,
                    session_id=session_id,
                    client_message_unicode=msg_data,
                    server_message_unicode=response,
                    platform=platform,
                    action_type='text'
                ).save(db)

            # handle binary message(audio)
            elif 'bytes' in data:
                binary_data = data['bytes']
                # 1. Transcribe audio
                transcript: str = speech_to_text.transcribe(
                    binary_data, platform=platform, prompt=character.name).strip()

                # ignore audio that picks up background noise
                if (not transcript or len(transcript) < 2):
                    continue

                # 2. Send transcript to client
                await manager.send_message(message=f'[+]You said: {transcript}', websocket=websocket)

                # 3. stop the previous audio stream, if new transcript is received
                await stop_audio()

                previous_transcript = transcript

                async def tts_task_done_call_back(response):
                    # Send response to client, [=] indicates the response is done
                    await manager.send_message(message='[=]', websocket=websocket)
                    # Update conversation history
                    conversation_history.user.append(transcript)
                    conversation_history.ai.append(response)
                    token_buffer.clear()
                    # Persist interaction in the database
                    Interaction(
                        client_id=client_id,
                        user_id=user_id,
                        session_id=session_id,
                        client_message_unicode=transcript,
                        server_message_unicode=response,
                        platform=platform,
                        action_type='audio'
                    ).save(db)

                # 4. Send message to LLM
                tts_task = asyncio.create_task(llm.achat(
                    history=build_history(conversation_history),
                    user_input=transcript,
                    user_input_template=user_input_template,
                    callback=AsyncCallbackTextHandler(
                        on_new_token, token_buffer, tts_task_done_call_back),
                    audioCallback=AsyncCallbackAudioHandler(
                        text_to_speech, websocket, tts_event, character.name),
                    character=character)
                )

    except WebSocketDisconnect:
        logger.info(f"User #{user_id} closed the connection")
        await manager.disconnect(websocket)
        return
