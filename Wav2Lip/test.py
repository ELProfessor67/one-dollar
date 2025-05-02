import asyncio
import aiofiles
from smallestai.waves import AsyncWavesClient
import dotenv
import os
dotenv.load_dotenv()


SMALLEST_AI_API_KEY = os.getenv("SMALLEST_AI_API_KEY")

async def main():
    client = AsyncWavesClient(api_key=SMALLEST_AI_API_KEY)
    async with client as tts:
        audio_bytes = await tts.synthesize("Hello, this is a test of the async synthesis function.",voice_id="arman",sample_rate=16000) 
        print(audio_bytes)

if __name__ == "__main__":
    asyncio.run(main())