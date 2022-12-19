import torch
import numpy as np
import os
import argparse
import librosa
import re
import json
from string import punctuation
from g2p_en import G2p
import soundfile
from models.StyleSpeech import StyleSpeech
from text import text_to_sequence
import audio as Audio
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon_path):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))

    return torch.from_numpy(sequence).to(device=device)


def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    if sample_rate != 16000:
        wav = librosa.resample(wav, sample_rate, 16000)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)


def get_StyleSpeech(config, checkpoint_path):
    model = StyleSpeech(config).to(device=device)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()
    return model


def synthesize(args, model, _stft):   
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    src = preprocess_english(args.text, args.lexicon_path).unsqueeze(0)
    src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)

    ref_audio_list = args.ref_audio.split(',')
    ref_mel = preprocess_audio(ref_audio_list[0], _stft).transpose(0,1).unsqueeze(0)
    style_vector = model.get_style_vector(ref_mel)

    for x in range(2, 53):
        ref_audio = "phrases/p{index}_.wav".format(index = x)
        ref_mel_item = preprocess_audio(ref_audio, _stft).transpose(0,1).unsqueeze(0)
        style_vector_item = model.get_style_vector(ref_mel_item)
        style_vector = torch.add(style_vector, style_vector_item)
    average_style_vector = torch.div(style_vector, 52)

    #for index, ref_audio in enumerate(ref_audio_list):
    #    if index > 0:
    #        ref_mel_item = preprocess_audio(ref_audio, _stft).transpose(0,1).unsqueeze(0)
    #        style_vector_item = model.get_style_vector(ref_mel_item)
    #        style_vector = torch.add(style_vector, style_vector_item)

    # average_style_vector = torch.div(style_vector, len(ref_audio_list))

    # preprocess audio and text
    # ref_mel = preprocess_audio(args.ref_audio, _stft).transpose(0,1).unsqueeze(0)

    # Extract style vector
    # style_vector = model.get_style_vector(average_ref_mel)

    # Forward
    mel_output = model.inference(average_style_vector, src, src_len)[0]
    
    # mel_ref_ = average_ref_mel.cpu().squeeze().transpose(0, 1).detach()
    mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()
    
    # plotting
    # utils.plot_data([mel_ref_.numpy(), mel_.numpy()], ['Ref Spectrogram', 'Synthesized Spectrogram'], filename=os.path.join(save_path, 'plot.png'))
    vocoder = torch.hub.load(repo_or_dir='melgan-neurips', model='load_melgan', source='local')
    wav = vocoder.inverse(mel_).squeeze().cpu().numpy()  # audio (torch.tensor) -> (batch_size, 80, timesteps)
    soundfile.write(os.path.join(save_path, 'test.wav'), wav, 16000, 'PCM_16')
    print('Generate done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
        help="Path to the pretrained model")
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument("--save_path", type=str, default='results/')
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")
    parser.add_argument("--text", type=str, required=True,
        help="raw text to synthesize")
    parser.add_argument("--lexicon_path", type=str, default='lexicon/librispeech-lexicon.txt')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model = get_StyleSpeech(config, args.checkpoint_path)
    print('model is prepared')

    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax)

    # Synthesize
    synthesize(args, model, _stft)
