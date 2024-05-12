import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import evaluate
from tqdm import tqdm
import pandas as pd
from preprocess_translate import *

metric = evaluate.load('sacrebleu',trust_remote_code=True)

src_lang = 'eng_Latn'
tgt_lang = 'brx_Deva'

src1 = f'/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.{src_lang}'
#src2 = '/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.hin_Deva'
tgt = f'/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.{tgt_lang}'

src1_sents = []
tgt_sents = []

with open(src1) as f:
    src1_sents = f.readlines()
    src1_sents = [x.strip() for x in src1_sents]

with open(tgt) as f:
    tgt_sents = f.readlines()
    tgt_sents = [x.strip() for x in tgt_sents]

processed_tgt = caller(tgt_sents,tgt_lang,'false','false')

print(metric.compute(predictions=processed_tgt,references=processed_tgt,tokenize='none'))


tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True).to('cuda:0')
print(model.generation_config)

preds = []
#tgt_sents = [[x] for x in tgt_sents]

for i in tqdm(range(0,len(src1_sents),16)):
    batch = src1_sents[i:i+16]
    batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
    batch = tokenizer(
        batch,
        src=True,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to('cuda:0')
    #print(batch)
    with torch.inference_mode():
        outputs = model.generate(
            **batch, 
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1
            )
    #print(outputs)
    print('Tokens')
    outputs = tokenizer.batch_decode(outputs, src=False)
    outputs = ip.postprocess_batch(outputs, lang=tgt_lang)
    print(outputs)
    preds.extend(outputs)

print(len(preds),len(tgt_sents))
assert len(preds) == len(tgt_sents)

processed_preds = caller(preds,tgt_lang,'false','false')

print('Preprocessing', metric.compute(predictions=processed_preds,references=processed_tgt,tokenize='none'))
print('No preprocessing', metric.compute(predictions=preds,references=tgt_sents,tokenize='none'))

df = pd.DataFrame({
    'generated': preds,
    'target': tgt_sents
})

df.to_csv('gens.csv',index=False)

with open('preds.txt','w') as f:
    f.writelines([x+'\n' for x in preds])
with open('tgt.txt','w') as f:
    f.writelines([x+'\n' for x in tgt_sents])