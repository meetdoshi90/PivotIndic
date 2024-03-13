import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import evaluate
from tqdm import tqdm

metric = evaluate.load('sacrebleu',trust_remote_code=True)

src1 = '/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.eng_Latn'
src2 = '/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.hin_Deva'
tgt = '/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.gom_Deva'

src1_sents = []
tgt_sents = []

with open(src1) as f:
    src1_sents = f.readlines()
    src1_sents = [x.strip() for x in src1_sents]

with open(tgt) as f:
    tgt_sents = f.readlines()
    tgt_sents = [x.strip() for x in tgt_sents]

tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True).to('cuda:7')
print(model.generation_config)

preds = []

for i in tqdm(range(0,len(src1_sents),4)):
    batch = src1_sents[i:i+4]
    batch = ip.preprocess_batch(batch, src_lang="eng_Latn", tgt_lang="gom_Deva")
    batch = tokenizer(batch, src=True, return_tensors="pt").to('cuda:7')
    print(batch)
    with torch.inference_mode():
        outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)
    print(outputs)
    print('Tokens')
    outputs = tokenizer.batch_decode(outputs, src=False)
    outputs = ip.postprocess_batch(outputs, lang="gom_Deva")
    print(outputs)
    preds.extend(outputs)

print(len(preds),len(tgt_sents))
assert len(preds) == len(tgt_sents)

print(metric.compute(predictions=preds,references=tgt_sents))
