import torch
from typing import Optional, Any, Dict, List, NewType, Tuple,Union
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoConfig
from transformers import (
    BatchEncoding,
    PreTrainedModel,
    PretrainedConfig,
    GenerationConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    SchedulerType,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    IntervalStrategy,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import evaluate
from tqdm import tqdm
import pandas as pd
from preprocess_translate import *

def initialize_tokenizer(direction):
    tokenizer = IndicTransTokenizer(direction=direction)
    return tokenizer

device = "cuda:1" 

src_lang_1 = 'eng_Latn'
src_lang_2 = 'hin_Deva'
tgt_lang = 'brx_Deva'

src1 = f'/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.{src_lang_1}'
src2 = f'/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.{src_lang_2}'
tgt = f'/raid/nlp/pranavg/Multi-source-pivoting/data/IN22/IN22-Gen/test.{tgt_lang}'

CHECKPOINT_NAME = '/raid/nlp/pranavg/Multi-source-pivoting/PivotIndic/2e2d/model/checkpoint-13000'

class Config(PretrainedConfig):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

#class DualEncoderDualDecoder(torch.nn.Module, PyTorchModelHubMixin):
class DualEncoderDualDecoder(PreTrainedModel):
    config_class = Config

    def __init__(self,config=None,model_names=["ai4bharat/indictrans2-en-indic-1B", "ai4bharat/indictrans2-indic-indic-1B"],alpha=0.5,beta=0.5,PAD_ID=1,genconfig=None,scaling_lambda=10.0):
        super(DualEncoderDualDecoder,self).__init__(config=config)
        self.en_indic_model = self.initialize_model(model_names[0], "en-indic", "")
        self.indic_indic_model = self.initialize_model(model_names[1], "indic-indic", "")
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.tensor([beta]), requires_grad=True)
        self.scaling_lambda = scaling_lambda
        self.PAD_ID = PAD_ID
        self.is_encoder_decoder = True
        self.config = config
        self.generation_config = genconfig
        
    def get_loss(self,logits,targets):
        B, C, V = logits.shape
        logits = logits.reshape(B*C, V)
        targets = targets.reshape(B*C)
        PAD_ID = self.PAD_ID
        loss = F.cross_entropy(logits, targets,reduction='none')
        assert loss.shape==targets.shape
        loss_mask_1 = targets!=PAD_ID
        loss_masked = loss.where(loss_mask_1, torch.tensor(0.0))
        return loss_masked.sum()/torch.count_nonzero(loss_masked)

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: torch.LongTensor = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                shape1: Optional[int] = None,
                shape2: Optional[int] = None,
                return_dict: Optional[bool] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                **kwargs
        ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        eval = False
        if decoder_input_ids.shape != labels.shape:
            eval = True
        inputs_1 = {
            'input_ids': input_ids[:,:shape1],
            'attention_mask': attention_mask[:,:shape1],
        }
        inputs_2 = {
            'input_ids': input_ids[:,shape1:],
            'attention_mask': attention_mask[:,shape1:],
        }
        inputs_1 = BatchEncoding(inputs_1,tensor_type='pt')
        inputs_2 = BatchEncoding(inputs_2,tensor_type='pt')
        out_1 = self.en_indic_model(**inputs_1, decoder_input_ids=decoder_input_ids, return_dict=True)
        out_2 = self.indic_indic_model(**inputs_2, decoder_input_ids=decoder_input_ids, return_dict=True)
        normalized_out_1 = F.softmax(out_1.logits,dim=-1)
        normalized_out_2 = F.softmax(out_2.logits,dim=-1)
        out = self.alpha*normalized_out_1 + self.bias*normalized_out_2
        if eval == False:
            constraint_loss = torch.sum((self.alpha + self.bias - 1.0) ** 2)
            loss = self.get_loss(out,labels) + self.scaling_lambda*constraint_loss
        else:
            loss = None
        #output['loss'] = loss 
        return Seq2SeqLMOutput(
            loss=loss,
            logits=out,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
        )

    def prepare_inputs_for_generation(  
            self,
            input_ids: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            attention_mask: torch.LongTensor = None, 
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            shape1: Optional[int] = None,
            shape2: Optional[int] = None,
            **kwargs,):
        # if past_key_values is not None:
        #     raise KeyError('past key values not implemented')
        #print('\n'*10)
        # print('PREPARE INPUTS CALLED')
        # if past_key_values is not None:
        #     print('PAST KV', past_key_values.shape)
        # if attention_mask is not None:
        #     print('Attn mask', attention_mask.shape)
        # if decoder_input_ids is not None:
        #     print('dec ids', decoder_input_ids.shape)
        # if head_mask is not None:
        #     print('head mask', head_mask.shape)
        # if decoder_head_mask is not None:
        #     print('dec head mask', decoder_head_mask.shape)
        # if cross_attn_head_mask is not None:
        #     print('cross attn head mask', cross_attn_head_mask.shape)
        # if use_cache is not None:
        #     print('use cache', use_cache)
        # if encoder_outputs is not None:
        #     print('enc outs', encoder_outputs.shape)
        for arg in kwargs:
            print(arg, kwargs[arg])
        
        if input_ids.shape[1]>(shape1+shape2):
            #print('inp ids longer than shape1 + shape2',input_ids.shape)
            decoder_input_ids = torch.ones((input_ids.shape[0],1)).to(torch.long).to(device=device) * 2
            decoder_input_ids = torch.cat((decoder_input_ids,input_ids[:,shape1+shape2:]),dim=1)
            input_ids = input_ids[:,:shape1+shape2]
            attention_mask = attention_mask[:,:shape1+shape2]

        if decoder_input_ids is None:
            #print('prep inps dec inp ids None')
            decoder_input_ids = torch.ones((input_ids.shape[0],1)).to(torch.long).to(device=device) * 2
        # else:
        #     print('prep inps dec inp ids', decoder_input_ids.shape)
        if decoder_attention_mask is None:
            #print('prep inps dec attn mask None')
            decoder_attention_mask = torch.ones_like(decoder_input_ids).to(dtype=attention_mask.dtype).to(device=device)
        # else:
        #     print('prep inps dec attn mask', decoder_attention_mask.shape)
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # if inputs_embeds is not None and past_key_values is None:
        #     model_inputs = {"inputs_embeds": inputs_embeds}
        # else:
        # print('ip ids')
        # print(input_ids)
        # print(input_ids.shape)
        # print(decoder_input_ids)
        # print(encoder_outputs)
        # print('attention mask')
        # print(attention_mask)
        # print(attention_mask.shape)
        # print(decoder_attention_mask)
        # print(past_key_values)
        # print(labels) 
        # print(shape1)
        # print(shape2)
        # print(decoder_input_ids.shape)
        # print(past_key_values)
        # print(attention_mask.shape)
        # print(head_mask)
        # print(decoder_head_mask)
        # print(cross_attn_head_mask)
        # print(use_cache)
        # print(encoder_outputs)
        # return {
        #     "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        #     "encoder_outputs": encoder_outputs,
        #     "past_key_values": past_key_values,
        #     "decoder_input_ids": decoder_input_ids,
        #     "attention_mask": attention_mask,
        #     "head_mask": head_mask,
        #     "decoder_head_mask": decoder_head_mask,
        #     "cross_attn_head_mask": cross_attn_head_mask,
        #     "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        # }
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            # "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "shape1": shape1,
            "shape2": shape2,
            # "head_mask": head_mask,
            # "decoder_head_mask": decoder_head_mask,
            # "cross_attn_head_mask": cross_attn_head_mask,
            # "use_cache": False,  # change this to avoid caching (presumably for debugging)
        }
        
    def initialize_model(self,ckpt_dir, direction, quantization, train=True):
        if quantization == "4-bit":
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8-bit":
            qconfig = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        else:
            qconfig = None

        #tokenizer = IndicTransTokenizer(direction=direction)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
            use_cache=False
        )

        if qconfig == None:
            model = model.to(device)
            #model.half()
        
        if train==True:
            model.train()
        else:
            model.eval()
        
        return model
    


class GenConfig(GenerationConfig):
    max_new_tokens = 256
    early_stopping = True
    num_beams = 1

#config = Config.from_pretrained('/raid/nlp/pranavg/Multi-source-pivoting/PivotIndic/2e2d/model-en-hi-bo-ps/checkpoint-2400')


model_names = ["ai4bharat/indictrans2-en-indic-1B", "ai4bharat/indictrans2-indic-indic-1B"]
    
model = DualEncoderDualDecoder.from_pretrained(CHECKPOINT_NAME).to(device)
print(model.alpha.item(),model.bias.item(),'alpha beta')
ip = IndicProcessor(inference=False)
en_indic_tokenizer = initialize_tokenizer("en-indic")
indic_indic_tokenizer = initialize_tokenizer("indic-indic")
metric = evaluate.load('sacrebleu',trust_remote_code=True)
 
print(model.alpha)
print(model.bias)


IN22_src_1 = []
IN22_src_2 = []
IN22_tgt = []
with open(src1) as f:
    IN22_src_1 = f.readlines()
with open(src2) as f:
    IN22_src_2 = f.readlines()
with open(tgt) as f:
    IN22_tgt = f.readlines()

test_predictions = []
test_references = [x.strip() for x in IN22_tgt]
with torch.inference_mode():
    for test_inputs_1,test_inputs_2,test_tgt in tqdm(zip(IN22_src_1,IN22_src_2,IN22_tgt)):
        test_inputs_1 = [test_inputs_1]
        test_inputs_1 = ip.preprocess_batch(test_inputs_1, src_lang=src_lang_1, tgt_lang=tgt_lang)
        #print(test_inputs_1)
        test_inputs_1 = en_indic_tokenizer(
            test_inputs_1,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        #print(test_inputs_1)
        #print(en_indic_tokenizer.batch_decode(test_inputs_1.input_ids,src=True))
        test_inputs_2 = [test_inputs_2]
        test_inputs_2 = ip.preprocess_batch(test_inputs_2, src_lang=src_lang_2, tgt_lang=tgt_lang)
        #print(test_inputs_2)
        test_inputs_2 = indic_indic_tokenizer(
            test_inputs_2,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )

        test_shape1 = test_inputs_1.input_ids.shape[-1]
        test_shape2 = test_inputs_2.input_ids.shape[-1]
        inputs = {
            'input_ids': torch.cat((test_inputs_1.input_ids,test_inputs_2.input_ids),dim=1),
            'attention_mask': torch.cat((test_inputs_1.attention_mask,test_inputs_2.attention_mask),dim=1),
        }
        test_outputs = model.generate(
            inputs = inputs['input_ids'].to(device),
            attention_mask = inputs['attention_mask'].to(device),
            num_beams = 5,
            max_new_tokens = 256, 
            #early_stopping = True,
            #eos_token_id = 2,
            shape1 = test_shape1,
            shape2 = test_shape2,
            use_cache=False,
            labels = torch.ones((0,0)).to(device)
        )
        print('Output')
        test_outputs = test_outputs[:,test_shape1+test_shape2:]
        #print(test_outputs)
        outputs = en_indic_tokenizer.batch_decode(test_outputs, src=False)
        #print(outputs)
        outputs = ip.postprocess_batch(outputs, lang=tgt_lang)
        print('Test outputs', outputs)
        test_predictions.append(outputs[0])
        print('Actual output', test_tgt)

# print(test_predictions)
# print(test_references)
assert len(test_predictions)==len(test_references)


with open(f'{tgt_lang}-tgt.txt','w') as f:
    f.writelines([x+'\n' for x in test_references])

with open(f'{tgt_lang}-preds.txt','w') as f:
    f.writelines([x+'\n' for x in test_predictions])

processed_tgt = caller(test_references,tgt_lang,'false','false')
processed_preds = caller(test_predictions,tgt_lang,'false','false')

print(metric.compute(predictions=processed_preds,references=processed_tgt,tokenize='none'))


