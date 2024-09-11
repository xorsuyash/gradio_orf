
import numpy as np
import os
from scipy.io.wavfile import write
from torch.utils.data import Dataset, DataLoader
import torch 
import librosa
import json 

MAX_MFCC_LEN=158
MAX_SEQ_LEN=28

def _load_vocab():
    with open("vocab.json",'r') as f:
        json_file=json.load(f)
    
    return dict(json_file)

VOCAB_DICT=_load_vocab()

def compute_mfcc(audio,sr=16000):
    
    if isinstance(audio,str): 
        y, sr = librosa.load(audio)
    else: 
        y=audio

    n_fft = min(2048, len(y))
    hop_length = n_fft // 4

    mfccs = librosa.feature.mfcc(y=y, sr=sr, 
                                     n_mfcc=13, 
                                     n_fft=n_fft, 
                                     hop_length=hop_length)

    width = min(9, mfccs.shape[1])
    if width < 3:
        width = 3
        
    width = min(width, mfccs.shape[1])

    if width % 2 == 0:
        width -= 1

    delta1 = librosa.feature.delta(mfccs, order=1, width=width)
    delta2 = librosa.feature.delta(mfccs, order=2, width=width)

    mfccs_combined = np.concatenate((mfccs, delta1, delta2), axis=0)
        
    return torch.tensor(mfccs_combined)

def segment_audio_and_save(audio, sampling_rate, frame_length_sec=0.4, stride_sec=0.01):
    frame_length_samples = int(frame_length_sec * sampling_rate)
    stride_length_samples = int(stride_sec * sampling_rate)
    
    num_samples = len(audio)
    segments = []
    
    for i, start_idx in enumerate(range(0, num_samples - frame_length_samples + 1, stride_length_samples)):
        end_idx = start_idx + frame_length_samples
        segment = audio[start_idx:end_idx]
        segments.append(segment)
    
    return segments

def preprocess_transcript(transcript):
    words=transcript.split(" ")
    processed_transcript=[]
    for text in words:
        for char in text:
            if char not in VOCAB_DICT.keys():
                text=text.replace(char,"")
        if len(text)>0:
            processed_transcript.append(text)
    
    new_transcript=" ".join(processed_transcript)
    return new_transcript

def pad_mfcc(mfcc,max_len):
    pad_width = max_len - mfcc.shape[1]
    padded_mfcc = torch.nn.functional.pad(mfcc, (0, pad_width), 'constant', 0)
    return padded_mfcc

def get_text_embeddings(text):
    embedding_store=torch.load('/home/suyash/gradio_orf/vector_store_new.pt')
    new_text=preprocess_transcript(text)
    acosutic_map=[]
    embeddings=[]
    for text in new_text.split(" "):
        embeddings.append(embedding_store[text])
        acosutic_map.append({text:0})
    embeddings=torch.stack(embeddings)
    embeddings=embeddings.squeeze(1)
    return embeddings, acosutic_map


class AudioDataset(Dataset):
    
    def __init__(self,segments):
        self.segments=segments
    
    def __len__(self):

        return len(self.segments)
    
    def __getitem__(self,idx):
        
        segment=self.segments[idx]
        mfcc=compute_mfcc(segment)
        padded_mfcc=pad_mfcc(mfcc,MAX_MFCC_LEN)

        return padded_mfcc

def collate_fn(batch):
    
    batch=torch.stack(batch)
    batch=batch.permute(0,2,1)
    return batch.detach().cpu().numpy()

def get_segment_loader(segments,batch_size):
    
    dataset=AudioDataset(segments)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

    return loader

