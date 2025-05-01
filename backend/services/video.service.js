import { exec } from 'child_process';
import * as fs from 'fs/promises';
import { generateAudio } from './audio.service.js';
import path from 'path';
import dotenv from 'dotenv';
dotenv.config();

function runFFmpeg(command,path='../Wav2Lip') {
    return new Promise((resolve, reject) => {
        exec(command, { cwd: path }, (error, stdout, stderr) => {
            if (error) {
                reject(`Error: ${error.message}`);
                return;
            }

            if (stderr) {
                console.error(`FFmpeg stderr: ${stderr}`);
            }

            resolve(`FFmpeg stdout: ${stdout}`);
        });
    });
}


export function generateVideo(text) {
    return new Promise(async (resolve, reject) => {
        const results = []
        const outputpath = path.join(process.cwd(), 'output.wav');
        const list = await generateAudio(text, outputpath);
        await Promise.all(list.map((l,i) => {
            const r = path.join(process.cwd(),`/results/${i}.mp4`)
            results.push(r);
            return runFFmpeg(`python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "./input/video.mp4" --audio "${l}" --nosmooth --outfile "${r}"`)
        }))

        const fileListPath = path.join(process.cwd(),"./fileList.txt");
        const fileContent = results.map(path => `file '${path}'`).join('\n');

        // Write the file list to a temporary file
        fs.writeFile(fileListPath, fileContent);

        // Build the ffmpeg command
        const outputPath =  path.join(process.cwd(),'./merged_video.mp4');
        const command = `ffmpeg -f concat -safe 0 -i "${fileListPath}" -c copy "${outputPath}"`;
        // Construct the command
        // const command = `python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "./input/video.mp4" --audio "${outputpath}" --nosmooth`;
        try {
            await runFFmpeg(command,'.');
            // Output path of the result video
            // const resultPath = '../Wav2Lip/results/result_voice.mp4';
            const data = await fs.readFile(outputPath);
            const base64 = data.toString('base64');
            fs.unlink(outputPath)
            
            resolve(base64);
        } catch (error) {
            console.error(`Error: ${error}`);
            reject(new Error('Failed to process the video.'));
        }
    });
}