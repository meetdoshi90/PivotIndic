from preprocess_translate import *
import evaluate

with open('brx_Deva-preds.txt') as f:
    preds = f.readlines()
    preds = [x.strip() for x in preds]
with open('brx_Deva-tgt.txt') as f:
    tgt = f.readlines()
    tgt = [x.strip() for x in tgt]



processed_tgt = caller(tgt,'brx_Deva','false','false')
processed_preds = caller(preds,'brx_Deva','false','false')

print(processed_preds[797])

print(processed_tgt[797])

metric = evaluate.load('sacrebleu',trust_remote_code=True)

print(metric.compute(predictions=preds,references=tgt))
print(metric.compute(predictions=processed_preds,references=tgt))
print(metric.compute(predictions=preds,references=processed_tgt))
print(metric.compute(predictions=processed_preds,references=processed_tgt))

print(metric.compute(predictions=preds,references=tgt,tokenize='none'))
print(metric.compute(predictions=processed_preds,references=tgt,tokenize='none'))
print(metric.compute(predictions=preds,references=processed_tgt,tokenize='none'))
print(metric.compute(predictions=processed_preds,references=processed_tgt,tokenize='none'))

