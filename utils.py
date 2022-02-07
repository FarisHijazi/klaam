import torch
import librosa    
from FastSpeech2.buckwalter import ar2bw, bw2ar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_file_to_data(file, srate = 16_000):
    batch = {} 
    speech, sampling_rate = librosa.load(file, sr=srate)
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch


def predict(data, model, processor, mode = 'rec', 
            bw = False, return_prob = False):
    if mode == 'rec':
        features = processor(data["speech"],
                            sampling_rate=data["sampling_rate"],
                            padding=True,
                            return_tensors="pt")
    else:
        features = processor(data["speech"], 
                        sampling_rate=data["sampling_rate"],
                        padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    try:
        attention_mask = features.attention_mask.to(device)
    except:
        attention_mask = None 
    with torch.no_grad():
        outputs = model(input_values, attention_mask = attention_mask)
    
    if mode == 'rec':
        if return_prob:
            raise('This parameter works for classification')
        pred_ids = torch.argmax(outputs.logits, dim=-1)
        text =  processor.batch_decode(pred_ids)[0]

        if bw:
            text = "".join([bw2ar[l] if l in bw2ar else l for l in text])
        return text 
    else:
        dialects = ['EGY','NOR','GLF','LAV','MSA']
        
        if not return_prob:
            pred_ids = torch.argmax(outputs['logits'], dim=-1)
            return dialects[pred_ids[0]]
        else:
            softmax = torch.nn.Softmax(dim = -1)
            probs = softmax(outputs['logits'])
            top_prob, top_lbls = torch.topk(probs[0], 5) 
            return {dialects[top_lbls[lbl]]:format(float(top_prob[lbl]),'.2f') for lbl in range(5)}
