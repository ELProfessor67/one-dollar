from text_to_speech import DeepgramTTS
import os
import dotenv
dotenv.load_dotenv()
import threading
import numpy as np
import librosa
import math
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import audio
# from face_detect import face_rect
from models import Wav2Lip
from batch_face import RetinaFace
from time import time
import ffmpeg
import os
from livekit import rtc
import asyncio
import audioop

WIDTH = 640
HEIGHT = 480
SAMPLES_PER_CHANNEL = 160
class Wav2lipService:
    warning_notice = True
    audio_chunks = []
    out_height = 480
    crop = [0, -1, 0, -1]
    mel_step_size = 16
    checkpoint_path = "checkpoints/wav2lip_gan.pth"
    box = [-1, -1, -1, -1]
    static = False
    pads = [0, 10, 0, 0]
    wav2lip_batch_size=128
    device="cpu"
    img_size=96
    video_file = "input/video.mp4"
    idle_video_file = "input/idle.mp4"
    model = detector = detector_model = None
    face_batch_size =  64 * 8
    isGeneratingVideo = False
    isSendingFrame = False
    generated_frames = []
    idle_frames = []
    dectected_faces = []
    buffer =  b''


    def __init__(self, ctx):
        self.ctx = ctx
        # self.do_load(self.checkpoint_path)

        self.video_stream = cv2.VideoCapture(self.video_file)
        self.fps =  self.video_stream.get(cv2.CAP_PROP_FPS)
        self.full_frames =  self.read_frames_from_video(self.video_stream)
        # self.frame_gen =  self.frame_generator( self.full_frames)


        self.idle_video_stream = cv2.VideoCapture(self.idle_video_file)
        self.idle_frames = self.read_frames_from_video( self.idle_video_stream)


        self.video_source = None
        self.audio_source = None

        print("Number of frames available for inference: "+str(len(self.full_frames))+"-"+str(len(self.idle_frames)))

        # self.TTS = DeepgramTTS(self.audio_handler)
        # self.TTS.send_text("Hello, can you hear me?")


    async def load_face_model(self):
        self.do_load(self.checkpoint_path)
        print("generating face frames...")
        self.dectected_faces = self.face_detect(self.full_frames)
        print("generated face frames")
        self.frame_gen =  self.frame_generator(self.full_frames,self.dectected_faces)

        await self.publish_video()
        await self.publish_audio()

        self.TTS = DeepgramTTS(self.audio_handler)
        self.TTS.send_text("Hello i am george washington.")

    def async_setup(self):
        asyncio.create_task(self.load_face_model())
        # await self.publish_video()
        # await self.publish_audio()


    async def publish_video(self):
        video_source = rtc.VideoSource(WIDTH, HEIGHT)
        video_track = rtc.LocalVideoTrack.create_video_track("video-track", video_source)
        video_options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=True,
            video_encoding=rtc.VideoEncoding(
                max_framerate=self.fps,
                max_bitrate=3_000_000
            ),
            # video_codec=rtc.VideoCodec.H264,
            # video_codec=rtc.VideoCodec.DESCRIPTOR,
            video_codec=rtc.VideoCodec.VP8,
        )
        await self.ctx.agent.publish_track(video_track, video_options)
        self.video_source = video_source
        asyncio.create_task(self.send_frame())
    
    async def publish_audio(self):
        audio_source = rtc.AudioSource(16000, 1)
        audio_track = rtc.LocalAudioTrack.create_audio_track("audio-track", audio_source)

        audio_options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE,
        )

        self.audio_source = audio_source
        await self.ctx.agent.publish_track(audio_track, audio_options)
        print("Audio Publish")
    


    def send(self, text):
        self.TTS.send_text(text=text)
        # self.TTS.flush()
    
    def audio_handler(self,pcm_data):
        self.audio_chunks.append(pcm_data)
        if self.isGeneratingVideo == False:
            self.isGeneratingVideo = True
            self.process_audio()
            # threading.Thread(target=self.process_audio).start()

        

    
    async def send_frame(self):
        idle_frames_count = 0
        while True:
            if idle_frames_count >= len(self.idle_frames):
                idle_frames_count = 0

            if len(self.generated_frames) != 0:
                pass
                self.isSendingFrame = True
                batch = self.generated_frames.pop(0)
                print(f"data: audio duration {len(batch[0]) / 16000} and frames {len(batch[1])} and fps {self.fps}")
                await asyncio.gather(
                    self.send_video_frames(batch[1]),
                    self.send_audio_chunk(batch[0])
                )

               
            else:
                frame = self.idle_frames[idle_frames_count]
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                argb_frame = bytearray(frame_rgba.tobytes())
                video_frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)
                self.video_source.capture_frame(video_frame)
                idle_frames_count += 1
                await asyncio.sleep(1/self.fps)



    async def send_audio_chunk(self, chunk_pcm):
        num_samples = len(chunk_pcm) // 2
        ptr = 0
        while True:
            if ptr + SAMPLES_PER_CHANNEL > num_samples:
                break

            start = ptr * 2
            end = (ptr + SAMPLES_PER_CHANNEL) * 2
            chunk_bytes = chunk_pcm[start:end]
            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
            ptr += SAMPLES_PER_CHANNEL

            audio_frame = rtc.AudioFrame.create(16000, 1, SAMPLES_PER_CHANNEL)
            audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)

            np.copyto(audio_data, chunk)


            await self.audio_source.capture_frame(audio_frame)
            await asyncio.sleep(SAMPLES_PER_CHANNEL / 16000)

        print("audio khatam")


    async def send_video_frames(self,frames):
        for frame in frames:
            self.video_source.capture_frame(frame)
            await asyncio.sleep(1 / self.fps)
        print("video khatam")


        

    def generate_video(self,chunk):
        # wav = self.pcm_to_wav_ffmpeg(chunk)
        wav = chunk
        audio_np = np.frombuffer(wav, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
        mel = audio.melspectrogram(audio_np)
        mel_chunks = self.get_mel_chunks(mel,self.fps)
        next(self.frame_gen)
        full_frames,face_coords = self.frame_gen.send(len(mel_chunks))

        batch_size = self.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(),face_coords, mel_chunks)

        frames_store = []
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            
        
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
        
         
            with torch.no_grad():
                pred = model(mel_batch, img_batch)
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))


                    f[y1:y2, x1:x2] = p

                    frame = cv2.resize(f, (WIDTH, HEIGHT))
                    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    argb_frame = bytearray(frame_rgba.tobytes())
                    video_frame =  rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)
                    frames_store.append(video_frame)
                   
                    
        self.generated_frames.append([chunk,frames_store])
        print("frame generated")
        if self.isSendingFrame == False:
            self.isSendingFrame = True
        

    def process_audio(self):
        while True:
            if(len(self.audio_chunks) == 0):
                self.isGeneratingVideo = False
                break
            print("audio generated")
            chunk = self.audio_chunks.pop(0)
            self.generate_video(chunk)
        self.isGeneratingVideo = False

            
                
                


    def generate_mel(self,audio_chunk):
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
        # Compute Mel Spectrogram
        sample_rate = 16000  # Ensure correct sample rate
        mel_spec = librosa.feature.melspectrogram(y=audio_np, sr=sample_rate, n_mels=128, fmax=8000)
        return mel_spec

   

    def _load(self,checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self,path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()



    def do_load(self,checkpoint_path):
        global model, detector, detector_model

        model = self.load_model(checkpoint_path)
        if torch.cuda.is_available():
            detector = RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")
        else:
            detector = RetinaFace( model_path="checkpoints/mobilenet.pth", network="mobilenet")
        detector_model = detector.model
        print("Models loaded")


    

    def face_rect(self,images):
        num_batches = math.ceil(len(images) / self.face_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * self.face_batch_size: (i + 1) * self.face_batch_size]
            all_faces = detector(batch)  # return faces list of all images
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret




    def get_smoothened_boxes(self,boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self,images):
        results = []
        pady1, pady2, padx1, padx2 = self.pads

        s = time()

        for image, rect in zip(images, self.face_rect(images)):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        print('face detect time:', time() - s)

        boxes = np.array(results)
        boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        return results


    def datagen(self,frames, face_coords, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = face_coords # BGR2RGB for CNN face detection
            else:
                face_det_results = [face_coords[0]]
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch



    def read_frames_from_video(self,video_stream):
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
                
            #resize
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(self.out_height * aspect_ratio), self.out_height))
            full_frames.append(frame)

        return full_frames


    def get_mel_chunks(self,mel,fps):
        mel_chunks = []
        mel_idx_multiplier = 80./fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.mel_step_size])
            i += 1
        
        return mel_chunks

    def pcm_to_wav_ffmpeg(self,pcm_chunk, sample_rate=16000, num_channels=1):
        process = (
            ffmpeg.input('pipe:0', format='s16le', ar=sample_rate, ac=num_channels)
            .output('pipe:1', format='wav')
            .run(capture_stdout=True, capture_stderr=True, input=pcm_chunk)
        )
        return process[0]  # Return WAV data


    def frame_generator(self, full_frames, detected_faces):
        index = 0
        total_frames = len(full_frames)
        
        while True:
            mel_length = yield  # Get mel length
            
            if mel_length is None:
                continue

            if index + mel_length <= total_frames:
                f = full_frames[index:index + mel_length]
                d = detected_faces[index:index + mel_length]
                yield f, d
                index += mel_length
            else:
                # Wrap-around logic
                remaining_f = full_frames[index:]
                remaining_d = detected_faces[index:]

                extra_needed = mel_length - len(remaining_f)
                wrapped_f = full_frames[:extra_needed]
                wrapped_d = detected_faces[:extra_needed]

                f = remaining_f + wrapped_f
                d = remaining_d + wrapped_d

                yield f, d
                index = extra_needed



    def fast_mel_spectrogram(self,audio_np):
        audio_tensor = torch.tensor(audio_np).cuda()  # Move to GPU
        mel = audio.melspectrogram(audio_tensor)  # Faster processing
        return mel.cpu().numpy()
