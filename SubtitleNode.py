import os
import subprocess
import whisper
import torch
from moviepy.editor import VideoFileClip

class SubtitleNode:
    def __init__(self):
        super().__init__()
        
        # Add inputs for video file, subtitle params, and font settings
        self.add_input('video_file', label="Video File", type="file")
        self.add_input('font_name', label="Font Name", type="text", default_value="Arial")
        self.add_input('font_size', label="Font Size", type="number", default_value=24)
        self.add_input('font_color', label="Font Color", type="color", default_value="#FFFFFF")
        self.add_input('subtitle_position', label="Subtitle Position", type="text", default_value="bottom")
        self.add_input('subtitle_style', label="Subtitle Style", type="text", default_value="normal")
        self.add_input('translate_to_english', label="Translate to English", type="boolean", default_value=False)

        # Output: processed video with embedded subtitles
        self.add_output('output_video', label="Output Video")

    def process(self):
        # Fetch input parameters
        video_file_path = self.get_input('video_file')
        font_name = self.get_input('font_name')
        font_size = self.get_input('font_size')
        font_color = self.get_input('font_color')
        subtitle_position = self.get_input('subtitle_position')
        subtitle_style = self.get_input('subtitle_style')
        translate_to_english = self.get_input('translate_to_english')

        # Extract audio from video and generate subtitles
        extracted_audio_name = extract_audio(video_file_path)
        params_dict = {
            "translate_to_english": translate_to_english,
            "is_upper": False,  # Allow user to control this via another input if needed
            "video_quality_key": "high",
            "eng_font": font_name,
            "subtitle_position": subtitle_position,
            "subtitle_style": subtitle_style
        }

        transcript_text = generate_transcript_matrix(extracted_audio_name, params_dict)
        vtt_path, srt_path = convert_transcript_to_subtitles(transcript_text, extracted_audio_name, params_dict)

        # Embed subtitles into video
        output_video = embed_subtitles(video_file_path, extracted_audio_name, params_dict)

        # Set the output to the resulting video
        self.set_output('output_video', output_video)

    def extract_audio(video_file_path):
        file_name_with_ext = os.path.basename(video_file_path)
        file_name = generate_unique_file_name(file_name_with_ext.split('.')[0])
    
        curr_audio_dir = f'{AUDIO_DIR}/{file_name}'
        os.makedirs(curr_audio_dir, exist_ok=True)
        audio_file_name = f'{file_name}.wav'
        audio_file_path = f'{curr_audio_dir}/{audio_file_name}'
    
        video_clip = VideoFileClip(video_file_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_file_path)
        video_clip.close()
    
        return file_name

    def generate_transcript_matrix(file_name, params_dict):
        curr_audio_dir = f'{AUDIO_DIR}/{file_name}'
        audio_file_name = f'{file_name}.wav'
        audio_file_path = f'{curr_audio_dir}/{audio_file_name}'
    
        model_name = "large-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_name, device)
        
        task = 'transcribe'
        if params_dict['translate_to_english']:
            task = 'translate'
    
        result = model.transcribe(
            audio_file_path,
            task=task,
            word_timestamps=True
        )
    
        segments = result['segments']
        transcript_matrix = []
        for i in range(len(segments)):
            words = segments[i]["words"]
            current_row = []
            for j in range(len(words)):
                word_instance = {
                    "start_time": int(words[j]["start"]*1000),
                    "end_time": int(words[j]["end"]*1000),
                    "word": words[j]["word"][1:]
                }
                current_row.append(word_instance)
            transcript_matrix.append(current_row)
    
        return transcript_matrix

    def convert_transcript_to_subtitles(transcript_matrix, file_name, params_dict):
        # Tạo phụ đề từ transcript_matrix dưới dạng `.vtt` và `.srt`
        lines = ["WEBVTT\n"]
        for i in range(len(transcript_matrix)):
            for j in range(len(transcript_matrix[i])):
                word = transcript_matrix[i][j]["word"]
                start_time = transcript_matrix[i][j]["start_time"]
                end_time = transcript_matrix[i][j]["end_time"]
                lines.append(f"{convert_time_for_vtt_and_srt(start_time, '.vtt')} --> {convert_time_for_vtt_and_srt(end_time, '.vtt')}\n{word}\n")
    
        vtt_text = "\n".join(lines)
        curr_subtitles_dir = f'{SUBTITLES_DIR}/{file_name}'
        os.makedirs(curr_subtitles_dir, exist_ok=True)
        vtt_subtitle_path = f'{curr_subtitles_dir}/{file_name}.vtt'
        
        with open(vtt_subtitle_path, 'w') as f:
            f.write(vtt_text)
        
        return vtt_subtitle_path

    def embed_subtitles(video_file_path, file_name, params_dict):
        curr_subtitles_dir = f"{SUBTITLES_DIR}/{file_name}"
        subtitles_path = f"{curr_subtitles_dir}/{file_name}.vtt"
        
        output_video_path = f"{TMP_OUTPUT_DIR}/{file_name}_output.mp4"
        font_name = params_dict["eng_font"]
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_file_path,
            "-vf", f"subtitles={subtitles_path}:force_style='Fontname={font_name},Fontsize={params_dict['font_size']},PrimaryColour=&H{params_dict['font_color']}'",
            '-c:a', 'copy',
            '-c:v', 'libx264',
            '-y',  # Overwrite output
            output_video_path
        ]
    
        subprocess.run(ffmpeg_cmd)
        return output_video_path

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "SubtitleNode": SubtitleNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtitleNode": "SubtitleNode"
}
