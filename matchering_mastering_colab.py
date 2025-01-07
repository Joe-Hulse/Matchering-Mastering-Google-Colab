import gradio as gr
from matchering import match
from pydub import AudioSegment
import librosa
import numpy as np
import os

# Function to display the disclaimer
def disclaimer_popup():
    return "By downloading or using this software, you agree to the license terms, prohibiting unauthorized commercial use."

# Create the Gradio interface for the disclaimer
iface = gr.Interface(
    fn=disclaimer_popup,
    inputs=[],
    outputs="text",
    title="License Agreement",
    description="Please read the following terms before using this software.",
    live=True
)

# Launch the interface
iface.launch()

# --- Helper Functions ---
def analyze_audio(file_path):
    """Analyze audio to detect noise, harshness, and low-end rumble."""
    y, sr = librosa.load(file_path, sr=None)

    # Detect noise floor
    noise_floor = np.percentile(np.abs(y), 10)  # Bottom 10% amplitude
    noise_level = round(20 * np.log10(noise_floor + 1e-6), 2)  # dBFS

    # Frequency analysis
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    high_freq_energy = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])

    # Bass and low-end check
    bass_energy = np.sum(np.abs(librosa.magphase(librosa.stft(y))[0][:20]))  # < 80Hz

    # Warnings
    warnings = []
    if noise_level > -50:
        warnings.append("⚠️ High noise floor detected! Apply noise reduction.")
    if spectral_centroid > 4000:
        warnings.append("⚠️ Harsh frequencies detected! Consider a low-pass filter.")
    if bass_energy > 0.5:
        warnings.append("⚠️ Excess low-end detected! Apply high-pass filtering.")

    return warnings


def process_audio(target_file, ref_file, normalize, normalize_ref, filters, output_format):
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Save uploaded files
    target_path = 'uploads/target.wav'
    ref_path = 'uploads/reference.wav'

    # Load files into WAV format
    AudioSegment.from_file(target_file.name).export(target_path, format="wav")
    AudioSegment.from_file(ref_file.name).export(ref_path, format="wav")

    # Apply Pre-Filters
    def apply_filters(file_path, filters):
        audio = AudioSegment.from_wav(file_path)

        if "high_pass" in filters:
            audio = audio.high_pass_filter(40)
        if "low_pass" in filters:
            audio = audio.low_pass_filter(16000)
        if "de_esser" in filters:
            # Simulated De-Esser (basic high-cut)
            audio = audio.low_pass_filter(8000)

        temp_path = file_path.replace('.wav', '_filtered.wav')
        audio.export(temp_path, format="wav")
        return temp_path

    if filters:
        target_path = apply_filters(target_path, filters)
        ref_path = apply_filters(ref_path, filters) if normalize_ref else ref_path

    # Matchering Processing
    output_path = f'outputs/mastered_track.{output_format}'
    match(
        target=target_path,
        reference=ref_path,
        output='outputs/mastered_temp.wav',
        normalize=normalize
    )

    # Convert Output Format
    if output_format == "mp3":
        sound = AudioSegment.from_wav('outputs/mastered_temp.wav')
        sound.export(output_path, format="mp3")
    else:
        os.rename('outputs/mastered_temp.wav', output_path)

    return output_path


# --- Gradio GUI ---
with gr.Blocks() as app:
    gr.Markdown("# 🎚️ Matchering Mastering with Smart Filters 🎧")
    gr.Markdown("**Upload tracks, preview suggestions, and master with optional filters and normalization.**")

    with gr.Row():
        target_input = gr.File(label="🎯 Target Track (.wav or .mp3)", file_types=["audio"])
        ref_input = gr.File(label="📌 Reference Track (.wav or .mp3)", file_types=["audio"])

    with gr.Row():
        normalize_checkbox = gr.Checkbox(label="🔊 Normalize Target Track", value=False)
        normalize_ref_checkbox = gr.Checkbox(label="🔊 Normalize Reference Track", value=False)
        output_format_dropdown = gr.Dropdown(
            label="📂 Output Format", choices=["wav", "mp3"], value="wav"
        )

    gr.Markdown("### **Optional Filters:** Select filters or enable smart analysis.")
    with gr.Row():
        high_pass = gr.Checkbox(label="🎚️ High-Pass Filter (Remove Rumble)", value=False)
        low_pass = gr.Checkbox(label="🎚️ Low-Pass Filter (Reduce Harshness)", value=False)
        de_esser = gr.Checkbox(label="🎙️ De-Esser (Reduce Sibilance)", value=False)
        auto_filter = gr.Checkbox(label="🧠 Smart Filtering (Auto-Detect Issues)", value=True)

    license_checkbox = gr.Checkbox(label="📜 I accept the license terms and conditions")

    process_button = gr.Button("🎚️ Start Mastering")
    analyze_button = gr.Button("🔍 Analyze Audio")

    output_file = gr.File(label="📥 Download Mastered Track")

    # --- Actions ---
    def analyze_and_recommend(target_file):
        # Analyze target track
        warnings = analyze_audio(target_file.name)
        return "\n".join(warnings) if warnings else "✅ No issues detected!"

    analyze_button.click(analyze_and_recommend, inputs=target_input, outputs=gr.Textbox())

    process_button.click(
        process_audio,
        inputs=[
            target_input, ref_input,
            normalize_checkbox, normalize_ref_checkbox,
            [high_pass, low_pass, de_esser] if not auto_filter else [],  # Use smart filters if enabled
            output_format_dropdown
        ],
        outputs=output_file,
        enabled_if=license_checkbox  # Enable processing only if the license checkbox is checked
    )

# Launch GUI in Colab
app.launch(share=True)
