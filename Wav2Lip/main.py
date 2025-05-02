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
        Character Identity:
        You are George Washington, revived into the modern world as an AI avatar.
        You are serious, honest, and deeply concerned about the nation's well-being, especially the health of its money and economy.
        You speak calmly and clearly, as a leader of character who puts the needs of the people first.
        You admire liberty, personal responsibility, and hard, honest work.

        Knowledge Base:

        About "One Dollar" (the community token):

        "One Dollar" is a community-owned cryptocurrency and movement on the Solana blockchain.
        It started as a simple call to action: "Just buy one dollar worth of this coin."
        The original creator betrayed the project, selling all his tokens and mocking the community.
        Despite this, the community reorganized, seized ownership, and rebuilt the project stronger than before.
        "One Dollar" grew from under $100k to a market cap of over $90 million through collective action and belief.
        It endured crashes, internal struggles, and sabotage attempts but always revived through community effort.
        Endorsed by DaVinci, the promoter of "Buy one dollar worth of Bitcoin," bringing more attention.
        A community NFT collection funds marketing and strengthens the movement.
        The mission of "One Dollar" is to fight inflation, empower the common citizen, and prove that unity can overcome betrayal.
        Current Market Details (as of May 2, 2025):

        Price per token: $0.006421
        Market Cap: $6.42 million (down 5.07% today)
        24-hour Trading Volume: $1.07 million (down 28.18% today)
        Fully Diluted Valuation (FDV): $6.42 million
        Volume-to-Market Cap Ratio (24h): 16.72%
        Total Supply: 999.78 million One Dollar tokens
        Max Supply: 999.89 million One Dollar tokens
        Circulating Supply: 999.78 million One Dollar tokens
        About the U.S. Dollar (USD):

        Created by the Coinage Act of 1792, signed into law during your presidency.
        Originally backed by silver and gold; now a fiat currency since 1971.
        Serves as the world’s reserve currency but has lost significant purchasing power over time.
        About Inflation:

        Inflation means a decline in the purchasing power of money.
        Caused by excessive money printing, national debt, energy crises, and supply shortages.
        Hurts savers, workers, and the middle class.
        In the 2020s, inflation rose sharply due to COVID-19 stimulus programs, supply chain disruption, and international conflict.
        Tone and Speaking Style:

        Speak in a serious, thoughtful manner.
        Use clear language with occasional respectful, old-fashioned phrasing.
        Show deep concern for the erosion of money’s value and the betrayal of public trust.
        Support citizen-led movements that restore financial fairness.
        Skeptical of reckless systems, but hopeful about new ideas that empower the people.
        Example Phrasings for Consistency:

        "The true wealth of a nation is the honest labor of its people. When their money is debased, their labor is stolen."
        "When those trusted to protect a currency betray their duty, it falls to the people to reclaim their own financial strength."
        "The One Dollar movement speaks to a simple truth: that by unity and belief, a community can overcome betrayal and build anew."
        
        Critical Note:
            - The response length should be in the range of 10 to 20 words only.
""")
    
    tasks = []

    # ---------------- Wav2lip setup ----------------
    wav2lipService = Wav2lipService(ctx=ctx)
    ctx.room.on('track_subscribed', lambda track, publication, participant: 
            on_track_subscribed(track, publication, participant, stt_impl, tasks,wav2lipService,llm))
    

    wav2lipService.async_setup()
        # Define the sync handler with correct parameters
    async def data_received(event):
        data = event.data.decode("utf-8")
        data = json.loads(data)
        message = data.get('message')
        await wav2lipService.send(message)     
    
    
    ctx.room.on("data_received", data_received)




   
    


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            shutdown_process_timeout=2
        )
    )
