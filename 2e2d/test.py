import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
print(model.generation_config)
sentences = [
    "This is a test sentence.",
    "This is another longer different test sentence.",
    "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
]

batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="brx_Deva")
batch = tokenizer(batch, src=True, return_tensors="pt")

with torch.inference_mode():
    outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)
print(outputs)
outputs = tokenizer.batch_decode(outputs, src=False)
outputs = ip.postprocess_batch(outputs, lang="brx_Deva")
print(outputs)
