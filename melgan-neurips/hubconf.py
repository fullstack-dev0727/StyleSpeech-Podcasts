dependencies = ["torch", "librosa", "yaml"]
from mel2wav import MelVocoder
import os

def load_melgan(model_name="multi_speaker"):
    """
    Exposes a MelVocoder Interface
    Args:
        model_name (str): Supports only 2 models, 'linda_johnson' or 'multi_speaker'
    Returns:
        object (MelVocoder):  MelVocoder class.
            Default function (___call__) converts raw audio to mel
            inverse function convert mel to raw audio using MelGAN
    """
    print('local loaded!')
    return MelVocoder(path='melgan-neurips', github=False, model_name=model_name)
