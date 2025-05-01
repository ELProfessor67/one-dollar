import express from 'express';
import cors from 'cors';
import { generateVideo } from './services/video.service.js';
import expressWs from "express-ws";
import { TranscriptionService } from './services/transcription.service.js';
import { generateResponse } from './services/openai.service.js';
const app = express();
const PORT = 4000;
expressWs(app);


// Middleware to parse JSON bodies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors({
    origin: '*'
}));


app.get('/', (req, res) => {
    res.send('Hello, World!');
});

app.get('/video', async (req, res) => {
    const text = req.query.text;
    if (!text) {
        return res.status(400).json({ error: 'text is required' });
    }

    try {
        // Simulate video generation process
        const base64 = await generateVideo(text);
        res.status(200).send(base64);
    } catch (error) {
        res.status(500).json({ error: 'Failed to generate video' });
    }
});


app.ws("/transcribtion", async (connection, req) => {
    console.log("WebSocket connection opened");
    const config = {
        stopStream: false,
        assistantSpeaking: false,
    }

    let transcriptionService;
    try {

        transcriptionService = new TranscriptionService(handleIntrupt);
    } catch (error) {
        console.log("error")
    }

    // Handle incoming messages from Twilio
    connection.on('message', async (message) => {
        try {

            const data = JSON.parse(message);
            switch (data.event) {
                case 'start':
                    console.log('Starting transcription...');
                    break;
                case 'media':
                    transcriptionService.send(data.media.payload);
                    break;
            }
        } catch (error) {
            console.error('Error parsing message:', error, 'Message:', message);
        }
    });

    function handleIntrupt() {
        config.stopStream = true;
        connection.send(
            JSON.stringify({
                event: 'clear',
            })
        );
    }








    transcriptionService.on('transcription', async (transcript_text) => {
        if (!transcript_text) return
        console.log('User', transcript_text);

        if (transcript_text) {
            config.stopStream = true;
            connection.send(
                JSON.stringify({
                    event: 'transcript',
                    transcript: {
                        value: transcript_text
                    }
                })
            );
        }

        if (transcript_text) {
            config.stopStream = true;
            connection.send(
                JSON.stringify({
                    event: 'clear',
                })
            );
        }

        connection.send(
            JSON.stringify({
                event: 'state',
                state: {
                    value: "Thinking..."
                }
            })
        );


        //send  response to assistant
        try {
            console.log('genrating response...');
            const text = await generateResponse(transcript_text);
            console.log("Bot: ", text);
            const videoBase64 = await generateVideo(text);
            console.log("response send successfully");
            connection.send(
                JSON.stringify({
                    event: 'media',
                    media: { payload:videoBase64, transcription: text }
                }))
        } catch (error) {
            console.log(error.message)
        }
    })




    // Handle connection close and log transcript
    connection.on('close', async () => {
        console.log(`Client disconnected`);
        transcriptionService.close();
    });

});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});