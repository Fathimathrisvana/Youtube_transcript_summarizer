from flask import Flask, jsonify, render_template,send_file,request
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from googletrans import Translator
from gtts import gTTS
import os
import requests
app = Flask(__name__)


# Helper function to extract YouTube video ID from the URL
def extract_youtube_ids(video_url):
    pattern = r'(?:youtu.be/|youtube.com/(?:embed/|v/|watch\?v=|watch\?feature=player_embedded&v=))([\w\-]+)'
    match = re.search(pattern, video_url)
    if match:
        return match.group(1)
    else:
        return None

# Helper function to translate text
def g_translate(text, lang):
    if lang and lang.lower() != 'en':
        translator = Translator()
        text_parts = text.split('. ')
        translated_text = []
        for parts in text_parts:
            translated_text.append(translator.translate(parts, src='en', dest=lang).text)
        return ' '.join(translated_text) + '.'
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None  # Initialize summary to None
    audio_url = None
    translated_summary = None

    if request.method == 'POST':
        try:
            # Get the YouTube URL and language from the form
            youtube_url = request.form['youtube_url']
            language = request.form['language']

            # Extract the video ID from the URL
            video_id = extract_youtube_ids(youtube_url)

            if video_id:
                # Get the transcript for the video (assuming English)
                details = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US","pt",])

                transc = ''
                for item in details:
                    transc += item['text'] + ' '

                transcript = transc.strip()

                # Summarize the transcript
                model = T5ForConditionalGeneration.from_pretrained("t5-small")
                tokenizer = T5Tokenizer.from_pretrained("t5-small")
                inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=8192 ,
                                          truncation=True)
                outputs = model.generate(
                    inputs,
                    max_length=4096,
                    min_length=100,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Translate the summary to the requested language
                translated_summary = g_translate(summary, language)

                # Convert the translated summary to audio
                myobj = gTTS(text=translated_summary, lang=language, slow=False)
                audio_filename = "static/summary.mp3"
                myobj.save(audio_filename)

                # Store the audio URL
                audio_url = '/' + audio_filename

        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return render_template('index.html', summary=translated_summary, audio_url=audio_url)

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the summary and audio_url from the form or other sources
    summary = request.form.get('summary')
    audio_url = request.form.get('audio_url')
    
    # Render the index.html template with the provided variables
    return render_template('index.html', summary=summary, audio_url=audio_url)

@app.route('/download_summary/<path:translated_summary>', methods=['GET'])
def download_summary(translated_summary):
    if translated_summary:
        # Save the translated summary to a temporary text file
        with open("static/translated_summary.txt", "w", encoding="utf-8") as file:
            file.write(translated_summary)

        # Return the file for download
        return send_file("static/translated_summary.txt", as_attachment=True, download_name="translated_summary.txt")
    else:
        return "Translated summary not available."



if __name__ == '__main__':
    app.run(debug=True)