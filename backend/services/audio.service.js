import dotenv from 'dotenv';
import { createClient } from "@deepgram/sdk";
import fs from "fs";
dotenv.config();
import wav from 'wav'
import path from 'path';
const __dirname = path.resolve()
const deepgram = createClient(process.env.DEEPGRAM_API_KEY);


export const generateAudio = async (text, outputpath) => {
    // STEP 1: Make a request and configure the request with options (such as model choice, audio configuration, etc.)
    const response = await deepgram.speak.request(
        { text },
        {
            model: "aura-zeus-en",
            encoding: "linear16",
            container: "wav",
        }
    );

    // STEP 2: Get the audio stream and headers from the response
    const stream = await response.getStream();
    const headers = await response.getHeaders();
    if (stream) {
        // STEP 3: Convert the stream to an audio buffer
        let buffer = await getAudioBuffer(stream);
        const list = saveAudioInChunks(buffer,'./chunks');
        fs.writeFile(outputpath, buffer, (err) => {
            if (err) {
                console.error("Error writing audio to file:", err);
            } else {
                console.log("Audio file written to output.wav");
            }
        });
        return list
    } else {
        console.error("Error generating audio:", stream);
        return []
    }
};

// Helper function to convert the stream to an audio buffer
const getAudioBuffer = async (response) => {
    const reader = response.getReader();
    const chunks = [];

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
    }

    const dataArray = chunks.reduce(
        (acc, chunk) => Uint8Array.from([...acc, ...chunk]),
        new Uint8Array(0)
    );

    return Buffer.from(dataArray.buffer);
};


async function saveAudioInChunks(buffer, outputDir) {
    const chunkSize = 70000; // 70,000 bytes per chunk
    const chunks_list = []
    // Calculate the number of chunks
    const numChunks = Math.ceil(buffer.length / chunkSize);

    for (let i = 0; i < numChunks; i++) {
        // Get the start and end positions of the chunk
        const start = i * chunkSize;
        let end = start + chunkSize;

        // If it's the last chunk, make sure it's padded to 70,000 bytes
        if (end > buffer.length) {
            end = buffer.length;
            // Add padding if needed
            const padding = Buffer.alloc(chunkSize - (end - start)); // Pad the last chunk to 70,000 bytes
            buffer = Buffer.concat([buffer.slice(0, end), padding]);
        }

        // Slice the buffer for the current chunk
        const chunk = buffer.slice(start, end);

        // Create a writable stream to save the WAV chunk
        const outputPath = `${outputDir}/chunk-${i}.wav`;
        chunks_list.push(path.join(__dirname,outputPath));
        const writer = new wav.Writer({
            sampleRate: 24000, // Sample rate (adjust according to your audio specs)
            channels: 1,       // Mono channel (adjust if stereo)
            bitDepth: 16       // Bit depth (adjust as per your original audio data)
        });

        // Write the chunk to the file with the WAV header
        const writableStream = fs.createWriteStream(outputPath);
        writableStream.on('close', () => {
            console.log(`Chunk ${i} saved as ${outputPath}`);
        });

        writer.pipe(writableStream);
        writer.write(chunk);
        writer.end(); // End the WAV stream
    }

    return chunks_list;
}