from flask import Flask, send_file, request, after_this_request, Response, jsonify, stream_with_context
from dotenv import load_dotenv
import time
import re
import os
import requests
from openai import OpenAI
import random
from moviepy.config import check
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx, CompositeAudioClip, TextClip, CompositeVideoClip, ImageClip
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.fx import Crop
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import assemblyai as aai
import time
import json
import srt
import shutil
from flask_cors import CORS
import numpy as np
from PIL import Image
import math


                                                    ### THIS IS FOR PROCESS EVERY API KEY AND ASSIGNING VARIABLES ARE HERE ###

client=OpenAI()
load_dotenv()
app = Flask(__name__)
CORS(app)
pexel_api_key = os.getenv("PEXEL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
client_Elevenlabs= ElevenLabs(api_key=elevenlabs_api_key)
aai.settings.api_key= assemblyai_api_key
transcriber = aai.Transcriber()
font_path = '../Font/bold_font.ttf'
keywordListVar = {}


                                        ### THIS USES OPENAI FOR CREATING THE SUBJECT FOR THE SCRIPT . THE TEXT IS THE VARIABLE THAT COMES FROM THE FRONTEND  ###

def aiCompletion(text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content":"you are a creative scriptwriter"
            },
            {
                "role": "user",
                "content": f"Create a concise script for a TikTok or Instagram Reel with a maximum of 110 words.The script should only include the narrator's spoken lines, without any scene descriptions or additional context. The topic is '{text}' Keep the tone engaging, conversational, and suitable for a short-form video. Do not exceed the word limit."
            }
            ]
    )
    print(completion.choices[0].message)
    return completion.choices[0].message.content


                                        ### THIS ALSO USES OPENAI FOR CREATING 3 ONE WORD KEYWORDS THAT WILL BE USED TO SEARCH FOR RELEVANT BACKGROUND VIDEO ###

def aiKeywords(text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content":"your task is to provide 3 one word keywords that is compatible with the script that will be used to find background videos your response should only be the 3 keywords example 'metal guitar chocolate'"
            },
            {
                "role": "user",
                "content": f"create one word keywords for this script {text} which must be the subject of the script and from this keyword create another 2 keywords that is almost synonyms with the keyword say example the script is about dog your response would be 'dog puppy husky'"
            }
            ]
    )
    print(completion.choices[0].message)
    return completion.choices[0].message.content


                                        ### THIS FUNCTION IS USED TO RETRIEVE VIDEOS FROM PEXELS API ITS FREE AND DOESNT COST ANYTHING  ###

def RetrieveVideos(searchTerm, filenumber, videoNumber):
    search = searchTerm
    url = f"https://api.pexels.com/videos/search?query={search}&per_page=5&min_duration=20"
    
    headers = {
        "Authorization": pexel_api_key
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        videos = data.get("videos",[])
        video_files = videos[filenumber].get("video_files",[])   
        first_video_url = video_files[0]["link"]
        print(first_video_url)

        downloadvideo(first_video_url, f"../Vid/Vid{videoNumber}.mp4")
        return videos
    else:
        print("failed")


                                        ### THIS WILL DOWNLOAD THE VIDEOS FROM THE JSON RETRIEVED BY THE RetrieveVideos FUNCTION ###

def downloadvideo(videoURL, outputPath):
    response = requests.get(videoURL, stream=True)
    if response.status_code == 200:
        with open(outputPath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("successfully downloaded")


                                        ### THIS WILL USE MOVIEPY TO CLIP THE VIDEO TO MAKE IT SUITABLE FOR INSTAGRAM OR TIKTOK REELS ###

def clipVideo(number):
    inputPath = f"../Vid/Vid{number}.mp4"
    outputPath = f"../Vid/Clipped/VidClip{number}.mp4"
    targetWidth = 540
    targetHeight = 960

    clip = VideoFileClip(inputPath)
    clip_resized = clip.resized(height=targetHeight)
    crop_width = clip_resized.size[0]
    if crop_width > targetWidth:
        excess_width = (crop_width - targetWidth) / 2
        clip_cropped = clip_resized.cropped(x1=excess_width, x2=crop_width - excess_width)
    else:
        clip_cropped = clip_resized

    clip_cropped.write_videofile(outputPath, codec="libx264", fps=30)


                                        ### THIS WILL CONVERT THE SCRIPT CREATED BY THE OPENAI TO SPEECH IN MP3 EXTENSION ###

def text_to_speech_file(text: str):
   
    urlForVoiceID="https://api.elevenlabs.io/v1/voices"
    headers = {
        "xi-api-key": elevenlabs_api_key
    }
    response = requests.get(urlForVoiceID, headers=headers)
    listofVoices = response.json()
    simplified_data = [{"name": voice["name"], "voice_id": voice["voice_id"]} for voice in listofVoices["voices"]]
    number = len(simplified_data)
    randomNumber = random.randint(0,number-1)
    print(randomNumber)
    randomVoice = simplified_data[randomNumber]["voice_id"]
    responseData = client_Elevenlabs.text_to_speech.convert(
        voice_id=randomVoice, 
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    save_file_path = f"../MP3/test2Subs.mp3"
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in responseData:
            if chunk:
                f.write(chunk)
    print(f"{save_file_path}: A new audio file was saved successfully!")

    return save_file_path


                                        ### THIS WILL TAKE THE MP3 CREATED BY ELEVENLABS AND CREATE A SUBTITLE WITH SRT EXTENSION ###

def srtCreate():
    transcript = transcriber.transcribe("../MP3/test2Subs.mp3")
    subtitle = transcript.export_subtitles_srt()
    print(subtitle)

    with open("../MP3/test2Subs.srt", "w") as srt_file:
        srt_file.write(subtitle)
    return "subtitle done"


                                        ### THIS RECONSTRUCT THE SRT SO THE SUB IS ONLY 3 WORDS PER LINE ###

def reconstructSub():
    with open("../MP3/test2Subs.srt", "r", encoding="utf-8") as file:
        subtitles = list(srt.parse(file.read()))

    # Process subtitles
    reformatted_subtitles = []
    for subtitle in subtitles:
        # Split the text into words and group them into shorter lines
        words = subtitle.content.split()
        wrapped_text = '\n'.join(
            [' '.join(words[i:i + 3]) for i in range(0, len(words), 3)]
        )
        # Update the subtitle content
        subtitle.content = wrapped_text
        reformatted_subtitles.append(subtitle)

    # Write the reformatted subtitles back to the file
    with open("../MP3/test2SubsShortened.srt", "w", encoding="utf-8") as file:
        file.write(srt.compose(reformatted_subtitles))

    print("Subtitle formatting complete. Saved to")


                                        ### THIS WILL MERGE ALL THE CLIPPED VIDEOS AND MAKE THE FINAL CLIP WITH THE MP3 SOUND ###

def mergeVideos():
    clip0 = VideoFileClip("../Img/TempImg/output.mp4")
    clip1 = VideoFileClip("../Vid/Clipped/VidClip0.mp4")
    clip2 = VideoFileClip("../Vid/Clipped/VidClip1.mp4")
    clip3 = VideoFileClip("../Vid/Clipped/VidClip2.mp4")
    clip4 = VideoFileClip("../Img/TempImg/output.mp4")
    
    audioDuration = getdurationAudio()
    
    
    clipsWithTransition = [
        clip0.with_end(10),
        clip1.with_end(7).with_effects([vfx.CrossFadeIn(1), vfx.CrossFadeOut(1)]),
        clip2.with_start(1).with_effects([vfx.CrossFadeIn(1), vfx.CrossFadeOut(1)]),
        clip3.with_effects([vfx.CrossFadeIn(1), vfx.CrossFadeOut(1)]),
        clip4.with_effects([vfx.CrossFadeIn(1)])

    ]
    audioClip = "../MP3/test2Subs.mp3"
    finalClip = concatenate_videoclips(clipsWithTransition, method="compose")
    ClippedClip = finalClip.subclipped(0, audioDuration +1)
    ClippedClip.write_videofile("../Vid/Clipped/FinalClip.mp4", audio=audioClip)
    return "done"


                                        ### THIS WILL ADD THE SUBTITLE PERMANENTLY IN THE MIDDLE OF THE SCREEN ###

def burnSubtitle():
    generator = lambda txt: TextClip("../Font/bold_font.ttf",txt, font_size=32, color='white', text_align="center", margin=(3,3))

    subtitle = SubtitlesClip("../MP3/test2SubsShortened.srt", make_textclip=generator)

    result = CompositeVideoClip([VideoFileClip("../Vid/Clipped/FinalClip.mp4"),subtitle.with_position("center","center")])
    result.write_videofile("../Vid/Clipped/FinalClipwithSub.mp4")


                                        ### USED FOR GETTING THE AUDIO DURATION ###

def getdurationAudio():
    audio = AudioFileClip("../MP3/test2Subs.mp3")
    duration = audio.duration
    return duration


                                                                        ### FUNCTIONS FOR GENERATING IMAGES ###


                                        ### AI FOR CREATING A GOOD PROMPT FOR DALLE 3 ###

def aiPrompt(text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content":"you are a instagram photographer"
            },
            {
                "role": "user",
                "content": f"Create a prompt with the subject {text} which have these parts specified the parts are the framing, film type, shoot context, lighting prompt, and the year and usage context optionally have vibe prompt, and shoot context. an example of your response is if the subject is a dog , the response would be : a close up, black and white studio photographic portrait of a dog with dramatic lighting, for a photo in a 1970s life magazine"
            }
            ]
    )
    print(completion.choices[0].message)
    return completion.choices[0].message.content


                                        ### GENERATE THE IMAGE USING OPENAI DALL E ###

def generateImg(text,title):
    path = "../Img/TempImg"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    response = client.images.generate(
        model='dall-e-3',
        prompt= text,
        size='1024x1792',
        quality="standard",
        n=1
    )
    imageGenerated = response.data[0].url
    downloadResponse = requests.get(imageGenerated)
    print (response.data[0].url)
    if downloadResponse.status_code == 200:
        with open(f"{path}/generatedPicture.png", "wb") as file:
            file.write(downloadResponse.content)
    print("Image downloaded successfully!")


                                        ### USED TO VIEW IMAGE ON THE FRONTEND ###

def streamImage(text):
    with open(f"../Img/TempImg/generatedPicture.png", "rb") as img_file:
        while chunk := img_file.read(4096):
            yield chunk


                                        ### API ENDPOINT FOR GENERATING VIDEO ###

@app.route("/generateVideo", methods=["POST"])
def postTest():
    path = "../Vid/Clipped/tempFolder"
    if os.path.exists(path):
        shutil.rmtree(path)
    data = request.form
    script = aiCompletion(data["subject"])
    keywords = aiKeywords(script)
    keywordsList = keywords.split(" ")
    subjectforImage = aiPromptforPhoto(keywordsList[0], script)
    responsePrompt = aiPrompt(subjectforImage)
    generateImg(responsePrompt, subjectforImage)
    imgResize()
    zoomingImage()
    RetrieveVideos(keywordsList[0], 1, 0)
    clipVideo(0)
    RetrieveVideos(keywordsList[1], 1, 1)
    clipVideo(1)
    RetrieveVideos(keywordsList[2], 1, 2)
    clipVideo(2)
    text_to_speech_file(script)
    srtCreate()
    mergeVideos()
    reconstructSub()
    burnSubtitle()
    file_path = "../Vid/Clipped/FinalClipwithSub.mp4"
    new_path = f"{path}/tempFile.mp4"
    os.mkdir(path)
    shutil.copy(file_path, new_path)
    return jsonify({"message" : "Video generated successfully"}, 200)


                                        ### API ENDPOINT FOR DOWNLOADING VIDEO ###

@app.route("/downloadVideo", methods=["POST"])
def download():
    return send_file("../Vid/Clipped/tempFolder/tempFile.mp4", as_attachment=True)


                                        ### API ENDPOINT FOR STREAMING VIDEO ###

@app.route("/video", methods=["GET"])
def helloWorld():
    def stream():
        with open(f"../Vid/Clipped/tempFolder/tempFile.mp4", "rb") as video:
            chunk = video.read(1024)
            while chunk:
                yield chunk
                chunk = video.read(1024)

    return Response(stream(), mimetype="video/mp4")


                                        ### API ENDPOINT FOR VIEWING THE IMAGE LIVE ###

@app.route("/streamImage")
def stream_image():
    return Response(stream_with_context(streamImage("test")), mimetype="image/png")


                                        ### API ENDPOINT FOR GENERATING IMAGE ###

@app.route("/generateImage", methods=["POST"])
def generateImage():
    data = request.form
    responsePrompt = aiPrompt(data["subject"])
    generateImg(responsePrompt, data["subject"])
    return jsonify({"message" : "Image generated successfully"}, 200)


                                        ### API ENDPOINT FOR DOWNLOADING THE IMAGE ###

@app.route("/downloadImage" , methods=["POST"])
def download_image():
    return send_file("../Img/TempImg/generatedPicture.png", as_attachment=True)





                                        ### THIS AREA IS FOR TESTING PURPOSE ONLY ###

imageTest = "../Img/TempImg/generatedPicture.png"




def zoom_in_effect(clip=imageTest, zoom_ratio=0.04):
    def effect(get_frame, t):
        img = Image.fromarray(get_frame(t))
        base_size = img.size

        new_size = [
            math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
            math.ceil(img.size[1] * (1 + (zoom_ratio * t)))
        ]

        # The new dimensions must be even.
        new_size[0] = new_size[0] + (new_size[0] % 2)
        new_size[1] = new_size[1] + (new_size[1] % 2)

        img = img.resize(new_size, Image.LANCZOS)

        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)

        img = img.crop([
            x, y, new_size[0] - x, new_size[1] - y
        ]).resize(base_size, Image.LANCZOS)
        
        result = np.array(img)
        img.close()

        return result

    return clip.transform(effect)


def aiPromptforPhoto(subject, script):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content":"you are a genius prompt engineer for dall e "
            },
            {
                "role": "user",
                "content": f"create a prompt with the subject {subject} and will be suit to the script of an instagram reel the script is {script} the prompt will be used as the subject for creating the cover photo you should only use less than 15 words. your response should be a little generic and no names or proper noun should be used since this will be used in generating image"
            }
            ]
    )
    return completion.choices[0].message.content

def zoomingImage():
    clip_img = ImageClip("../Img/TempImg/generatedPicture.png", duration=10)
    videoTest = zoom_in_effect(clip_img, 0.1)
    videoTest.write_videofile('../Img/TempImg/output.mp4', fps=24)


def imgResize():
    img = Image.open("../Img/TempImg/generatedPicture.png")
    img_resized = img.resize((540, 960), Image.LANCZOS)
    img_resized.save("../Img/TempImg/generatedPicture.png")

@app.route("/test", methods=["POST"])  
def test():
    
    print("message : Test generated successfully", "subjectforImage")
    return jsonify({"message" : "Test generated successfully"}, 200)

