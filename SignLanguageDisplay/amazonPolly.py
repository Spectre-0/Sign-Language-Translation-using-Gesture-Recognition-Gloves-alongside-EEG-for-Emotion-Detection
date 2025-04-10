
import boto3

#-------
#Adnan or whoever looks at the code You can ignore this part. I was using
#boto to save my credentials i didnt include it in my actual code. Sent you the access and secret key.
session = boto3.Session()
credentials = session.get_credentials()

if credentials:
    print("Access Key:", credentials.access_key)
    print("Secret Key:", credentials.secret_key)  # 
    print("Session Token:", credentials.token)  # 
else:
    print("No AWS credentials found.")
    
#From this point on you will need the code
#---------- 
import pygame
from io import BytesIO



# Initialize the Polly client
polly = boto3.client('polly', region_name='eu-west-2')

# 
# Adnan change this part of the functionality
#right now the voices will only speak what is printed before
# this was done just for testing the amazon voices
texts = {
    "happy": "This is my happy emotion. I feel so good today!",
    "sad": "This is my sad emotion. I don't know what to do.",
    "angry": "This is my angry emotion. I'm so upset right now!",
    "neutral": "This is my neutral emotion. Everything is just normal today."
}

# Male and female UK voices
voices = {
    'female': 'Amy',  # Amy is a British female voice
    'male': 'Brian'  # Brian is a British male voice
}

output_format = 'mp3'

# note for adnan - this part is to do with the way the wirds are said
# you can check out the documentation ill link it in the next comment
#https://docs.aws.amazon.com/polly/latest/dg/voice-speed-change-vip.html
ssml_templates = {
    "happy": '''<speak>
                    <prosody rate="fast" pitch="x-high" volume="loud">
                        This is my happy voice. I feel so good today! I'm so excited, I can't wait to share this joy with you!
                    </prosody>
                 </speak>''',
    "sad": '''<speak>
                  <prosody rate="slow" pitch="x-low" volume="medium">
                      This is my sad voice. I don't know what to do... I'm just feeling lost and a little hopeless right now.
                  </prosody>
               </speak>''',
    "angry": '''<speak>
                   <prosody rate="fast" volume="x-loud" pitch="x-low">
                       This is my angry emotion! I'm SO upset right now! I can't believe this is happening!!
                   </prosody>
                </speak>''',
    "neutral": '''<speak>
                     <prosody rate="medium" pitch="medium" volume="medium">
                         This is my neutral emotion. Everything is just normal today, nothing too exciting or bad happening.
                     </prosody>
                  </speak>'''
}

# Initialize pygame for audio playback
pygame.mixer.init()

# Generate and play speech for each emotion for both male and female voices
for gender, voice in voices.items():
    for emotion, text in texts.items():
        print(f"Generating and playing speech for {emotion} emotion with {gender} voice...")

        # Use SSML to generate the speech with emotional variation
        try:
            response = polly.synthesize_speech(
                Text=ssml_templates[emotion],
                VoiceId=voice,
                OutputFormat=output_format,
                Engine='standard',  # Standard engine, should work for both voices
                LanguageCode='en-GB',  # UK English
                TextType='ssml'  # Ensure SML used
            )

            # Get the audio stream and play it using pygame
            audio_stream = response['AudioStream'].read()
            audio_file = BytesIO(audio_stream)  # Convert the stream to a BytesIO object
            pygame.mixer.music.load(audio_file)  # Load the audio into pygame
            pygame.mixer.music.play()  # Play the audio

            # Wait for the audio to finish before moving to the next one
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # Wait until the audio finishes

            print(f"Speech with {emotion} emotion played successfully using {gender} voice.")
        except Exception as e:
            print(f"Error with {emotion} emotion using {gender} voice: {e}")

print("All speech synthesis complete!")
