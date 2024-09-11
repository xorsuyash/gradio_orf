import gradio as gr
import numpy as np
import pandas as pd
from inference import get_text_embeddings,segment_audio_and_save,get_segment_loader
import onnxruntime as ort
import torch 
import torch.nn.functional as F
import numpy as np
from scipy.signal import resample

# Initialize the Whisper transcriber
providers = ['CUDAExecutionProvider']
view2_session = ort.InferenceSession("/home/suyash/gradio_orf/model_new_view1.onnx",providers=providers)
print("Execution Providers:", view2_session.get_providers())


custom_css = """
body { 
    background-color: white !important; /* Change background color to white */
    color: black !important; /* Change text color to black for better contrast */
}
"""

def sentences_to_list(filename):
  with open(filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    sentences = [line.strip() for line in lines]
    return sentences

text_list=sentences_to_list('texts.txt')

def resample_audio(audio,original_sample_rate,target_sample_rate):
    number_of_samples = round(len(audio) * float(target_sample_rate) / original_sample_rate)
    resampled_audio = resample(audio, number_of_samples)
    return resampled_audio

def transcribe(stream, new_chunk, text_to_match):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    
    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    
    resampled_stream=resample_audio(stream,sr,16000)
    # Match words from transcription with input text
    text_embeddings,acoustic_map=get_text_embeddings(text_to_match)
    for frame_length in [0.2,0.3,0.4,0.5,0.6]:
        segments=segment_audio_and_save(resampled_stream,sampling_rate=16000,frame_length_sec=frame_length)
        segment_loader=get_segment_loader(segments=segments,batch_size=128)
        for batch in segment_loader:
            view2_inputs = {view2_session.get_inputs()[0].name: batch}
            view2_output = view2_session.run(None, view2_inputs)
            onnx_emb = view2_output[0]
            torch_emb=torch.tensor(onnx_emb)
            torch_emb=torch_emb.to("cuda")
            text_embeddings=text_embeddings.to("cuda")
            audio_emb_normalized=F.normalize(torch_emb,p=2,dim=1)
            all_emb_normalized=F.normalize(text_embeddings,p=2,dim=1)
            cosim=torch.mm(audio_emb_normalized,all_emb_normalized.T)
            bool_emb=cosim>0.8
            true_indices = torch.nonzero(bool_emb, as_tuple=True)

            for idx in true_indices[1]:
                for k,v in acoustic_map[idx].items():
                    if acoustic_map[idx][k]!=1:
                        acoustic_map[idx][k]=1
        print(acoustic_map[0])
    #words_to_match = set(text_to_match.lower().split())
    #matched_words = [word for word in transcription.lower().split() if word in words_to_match]
    # HTML output with div styled like a box

    html_output = '''
    <div style="
        border: 2px solid #333; 
        padding: 10px; 
        border-radius: 5px; 
        background-color: white;  /* Set the background to solid white */
        color: black;  /* Set text to solid black */
        opacity: 1;  /* Ensure full opacity */
        font-size: 16px;
        max-height: 300px; 
        overflow-y: scroll;
    ">
    '''

    for word_dict in acoustic_map:
        for word, value in word_dict.items():
            color = "green" if value == 1 else "red"
            html_output += f'<span style="color: {color}; font-size: 16px;">{word} </span>'
    html_output += '</div>'
    
    return stream, html_output
    
with gr.Blocks(theme=gr.themes.Base(),css=custom_css) as demo:
    demo.load(None,
        None,
        js="""
  () => {
  const params = new URLSearchParams(window.location.search);
  if (!params.has('__theme')) {
    params.set('__theme', 'dark');
    window.location.search = params.toString();
  }
  }""",)
    gr.Markdown("# Acoustic Embedding Model")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], streaming=True)
        text_input = gr.Dropdown(label="Input Text to Match",choices=text_list)
    
    with gr.Row():
        output_matched = gr.HTML(label="Matched Words")
    
    stream_state = gr.State()
    
    audio_input.stream(
        fn=transcribe, 
        inputs=[stream_state, audio_input, text_input],
        outputs=[stream_state, output_matched],
        show_progress=False
    )

demo.queue().launch()