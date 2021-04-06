import torch
import librosa    
from lang_trans.arabic import buckwalter

def load_file_to_data(file, srate = 16_000):
    batch = {} 
    speech, sampling_rate = librosa.load(file, sr=srate)
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch


def predict(data, model, processor, mode = 'rec', bw = False):
    if mode == 'rec':
        max_length = 128000
        features = processor(data["speech"][:max_length],
                            sampling_rate=data["sampling_rate"],
                            padding=True,
                            max_length=max_length, 
                            pad_to_multiple_of=max_length,
                            return_tensors="pt")
        
    else:
        max_length = 320000
        features = processor(data["speech"][:max_length], 
                        sampling_rate=data["sampling_rate"],
                        max_length=max_length,
                        pad_to_multiple_of=max_length,
                        padding=True, return_tensors="pt")
    input_values = features.input_values.to("cuda")
    attention_mask = features.attention_mask.to("cuda")
    with torch.no_grad():
        outputs = model(input_values, attention_mask = attention_mask)
    
    if mode == 'rec':
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        text =  processor.batch_decode(pred_ids)[0]

        if bw:
            text = buckwalter.untrans(text)
        return text 
    else:
        pred_ids = torch.argmax(outputs['logits'], dim=-1)[0]
        dialects = ['EGY','NOR','GLF','LAV','MSA']
        return dialects[pred_ids]