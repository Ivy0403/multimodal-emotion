import re
from faster_whisper import WhisperModel


class AudioTranscriber:
    def __init__(self, model_name="tiny", language="auto", beam_size=5, best_of=5, compute_type="default",
                 device="cuda"):
        """
        Initialize the AudioTranscriber with WhisperModel.

        :param model_name: Name of the Whisper model to use.
        :param language: Language of the audio (default: auto-detect).
        :param beam_size: Beam size for decoding.
        :param best_of: Number of best predictions to consider.
        :param compute_type: Compute type (e.g., 'default', 'float16').
        :param device: Device to use ('cpu' or 'cuda').
        """
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of

        try:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root="./models",
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WhisperModel: {e}")

    def transcribe(self, audio_file):
        """
        Transcribe the given audio file.

        :param audio_file: Path to the audio file to transcribe.
        :return: Transcription as a string.
        """
        try:
            segments, info = self.model.transcribe(
                audio_file,
                beam_size=self.beam_size,
                best_of=self.best_of,
                language=None if self.language == 'auto' else self.language
            )

            transcription = []
            for segment in segments:
                text = segment.text.strip().replace('&#39;', "'")
                text = re.sub(r'&#\d+;', '', text)

            # Append valid text to the transcription
                if text and not re.match(r'^[，。、？‘’“”；：（｛｝【】）:;"\'\s \d`!@#$%^&*()_+=.,?/\\-]*$', text):
                    transcription.append(text)

        # Combine all transcriptions into a single string
            return "\n".join(transcription)
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None


if __name__ == "__main__":
    try:
        transcriber = AudioTranscriber(
            model_name="tiny",
            language="en",
            beam_size=5,
            best_of=5,
            compute_type="default",
            device="cuda"
        )

        wav_file_path = "test.m4a"  # Replace with your audio file path
        transcription_result = transcriber.transcribe(wav_file_path)

        if transcription_result:
            print("Transcription Result:")
            print(transcription_result)
        else:
            print("No transcription available.")

    except Exception as error:
        print(f"An error occurred: {error}")
