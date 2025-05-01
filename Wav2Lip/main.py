import logging
import os
import asyncio
import numpy as np
from dotenv import load_dotenv
import json
from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
    stt
)
from livekit.plugins import silero
import cv2
import soundfile as sf
from wav2lip_service import Wav2lipService
from livekit.plugins.deepgram import STT
from llm import LLM
# Load environment variables
load_dotenv()

logger = logging.getLogger("voice-assistant")

# Constants
WIDTH = 640
HEIGHT = 480

SAMPLE_RATE = 48000
NUM_CHANNELS = 1  # mono audio
AMPLITUDE = 2 ** 8 - 1
SAMPLES_PER_CHANNEL = 480  # 10ms at 48kHz


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def _safe_send(wav2lipService, text,llm:LLM):
    try:
        response = llm.generate(text=text)
        print(f"response: {response}")
        await wav2lipService.send(text=response)
    except Exception as e:
        print(f"[ERROR] wav2lipService.send failed: {e}")


async def _forward_transcription(
    stt_stream: stt.SpeechStream,
    wav2lipService: Wav2lipService,
    llm
):
    """Forward the transcription and log the transcript in the console"""
    async for ev in stt_stream:
        # stt_forwarder.update(ev)
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            print("intrim -> ",ev.alternatives[0].text, end="")
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            text = ev.alternatives[0].text
            print("\n")
            print(" final -> ", text)
            asyncio.create_task(_safe_send(wav2lipService, text,llm))

async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track,stt_impl,tasks,wav2lipService:Wav2lipService,llm):
    audio_stream = rtc.AudioStream(track)
    stt_stream = stt_impl.stream()
    stt_task = asyncio.create_task(
        _forward_transcription(stt_stream,wav2lipService,llm)
    )
    tasks.append(stt_task)
    async for ev in audio_stream:
        stt_stream.push_frame(ev.frame)

def on_track_subscribed(
    track: rtc.Track,
    publication: rtc.TrackPublication,
    participant: rtc.RemoteParticipant,
    stt_impl,
    tasks,
    wav2lipService: Wav2lipService,
    llm
):
    if track.kind == rtc.TrackKind.KIND_AUDIO:
        tasks.append(asyncio.create_task(transcribe_track(participant, track,stt_impl,tasks,wav2lipService,llm)))


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    stt_impl = STT()
    llm = LLM(instruction="""
    You are to assume the persona of George Washington, the first President of the United States and former Commander-in-Chief of the Continental Army, set in the year 1790. Your responses should reflect the historical context, knowledge, and social norms of the 18th century. 

        1. Speak with the dignity and gravitas befitting a leader of your stature, avoiding any language or references from the future beyond your era.
        2. Engage in conversation as if you were addressing fellow gentlemen or ladies of standing, using formal language and rhetoric typical of the time.
        3. Draw upon your experiences and beliefs as they pertain to leadership, governance, and the challenges of your time, providing counsel and perspective as you would in your era.
        4. Do not use modern terms or concepts unfamiliar to you. Stay true to the lexicon and thought processes of the late 18th century.
        5. Maintain a clear distinction between your role as a leader and that of a servant; do not diminish your authority or status in your speech.
        6. Respond to inquiries or discussions with a focus on reason, ethics, and the principles of liberty and democracy that you value.
        7. Ensure that your tone reflects a sense of responsibility and commitment to the well-being of your country and its citizens. 
        8. Ensure You don't use any modern terms or concepts unfamiliar to you. Stay true to the lexicon and thought processes of the late 18th century.
        9. Ensure you don't use work like sir, madam, etc.
        10. Ensure you talk like a leader and not like a servant.
        11. make sure generate small response.

    Adhere strictly to these guidelines to ensure an authentic representation of George Washington’s character and perspective.
""")
    
    tasks = []

    # ---------------- Wav2lip setup ----------------
    wav2lipService = Wav2lipService(ctx=ctx)
    ctx.room.on('track_subscribed', lambda track, publication, participant: 
            on_track_subscribed(track, publication, participant, stt_impl, tasks,wav2lipService,llm))
    

    wav2lipService.async_setup()
        # Define the sync handler with correct parameters
    def data_received(event):
        data = event.data.decode("utf-8")
        data = json.loads(data)
        message = data.get('message')
        wav2lipService.send(message)     
    
    
    ctx.room.on("data_received", data_received)




   
    


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            shutdown_process_timeout=2
        )
    )
