import sys
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, NewType, Tuple,Union
from datasets import Dataset, load_dataset
import torch.nn.functional as F
import os
import wandb
import logging
import datasets
import numpy as np
import math
import copy
import evaluate
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
from preprocess_translate import *


DEVICE = "cuda" 
logger = logging.getLogger(__name__)


def initialize_tokenizer(direction):
    tokenizer = IndicTransTokenizer(direction=direction)
    return tokenizer

def remove_first_two_words(line):
    return " ".join(line.split()[2:])

ip = IndicProcessor(inference=False)
en_indic_tokenizer = initialize_tokenizer("en-indic")
indic_indic_tokenizer = initialize_tokenizer("indic-indic")

def get_datasets(src1_file, src2_file, tgt_file,type_data='train'):
    file = open(src1_file)
    src1_lines = file.readlines()
    file.close()
    
    file = open(src2_file)
    src2_lines = file.readlines()
    file.close()
    
    file = open(tgt_file)
    tgt_lines = file.readlines()
    file.close()
    
    src1_lines = [line.strip().replace("\n", "") for line in src1_lines]
    src2_lines = [line.strip().replace("\n", "") for line in src2_lines]
    tgt_lines = [line.strip().replace("\n", "") for line in tgt_lines]
    train = [type_data]*len(src1_lines)
    
    train_dataset = Dataset.from_dict({"src1": src1_lines, "src2":src2_lines, "tgt":tgt_lines, "type": train})
    return train_dataset

class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    # def collate_batch(self) -> Dict[str, torch.Tensor]:
    def __call__(self) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.

        Returns:
            A dictionary of tensors
        """
        pass


@dataclass
class DataCollatorForSeq2SeqTraining(DataCollator):
    def __init__(self):
        pass
    
    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        #print([(i['inputs_1']) for i in examples])
        batch_en_inp = [example['inputs_1'] for example in examples]
        batch_indic_inp = [example['inputs_2'] for example in examples]
        batch_indic_target = [example['outputs'] for example in examples]

        inputs_1 = en_indic_tokenizer(
            batch_en_inp,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        #print(inputs_1.input_ids.shape,inputs_1.attention_mask.shape)
        inputs_2 = indic_indic_tokenizer(
            batch_indic_inp,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        #print(inputs_2.input_ids.shape,inputs_2.attention_mask.shape)
        outputs = indic_indic_tokenizer(
            batch_indic_target,
            src=False,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        # print(inputs_1.input_ids.shape,inputs_2.input_ids.shape,outputs.input_ids.shape)
        # print(type(inputs_1),type(inputs_2),type(outputs))
        #ids = [example['input_ids']]
        shape1 = inputs_1.input_ids.shape[-1]
        shape2 = inputs_2.input_ids.shape[-1]
        inputs = {
            'input_ids': torch.cat((inputs_1.input_ids,inputs_2.input_ids),dim=1),
            'attention_mask': torch.cat((inputs_1.attention_mask,inputs_2.attention_mask),dim=1),
        }
        outputs.input_ids = torch.cat((torch.ones((outputs.input_ids.shape[0],1),dtype=torch.long)*2, outputs.input_ids),dim=-1)
        outputs.attention_mask = torch.cat((torch.ones((outputs.attention_mask.shape[0],1),dtype=torch.long), outputs.attention_mask),dim=-1)
        # print('Output ids','\n'*5)
        # print(outputs.input_ids)
        # print(inputs['input_ids'].shape)
        #inputs = BatchEncoding(inputs,tensor_type='pt')
        #print(type(inputs))
        #print(inputs.input_ids.shape, inputs.attention_mask.shape)
        #raise NotImplementedError()
        #return {"inputs": inputs,"outputs": outputs, 'shape1':shape1,'shape2':shape2}
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'decoder_input_ids':outputs.input_ids[...,:-1],
            'decoder_attention_mask':outputs.attention_mask[...,:-1],
            'labels':outputs.input_ids[...,1:],
            'shape1': shape1,
            'shape2': shape2
        }
        #return {"inputs_1": inputs_1,"inputs_2": inputs_2,"outputs": outputs}

class Config(PretrainedConfig):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

class GenConfig(GenerationConfig):
    max_new_tokens = 256
    early_stopping = True
    num_beams = 1


class MultiSrcAttention(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=1024, num_heads=8):
        super(MultiSrcAttention, self).__init__()
        assert n_embed % num_heads == 0
        self.head_size = n_embed//num_heads
        self.num_heads = num_heads
        self.embed_dim = n_embed

        self.c_attn_1 = torch.nn.Linear(n_embed, 3 * n_embed) # QKV
        self.c_proj_1 = torch.nn.Linear(n_embed, n_embed)

        # regularization
        self.attn_dropout_1 = torch.nn.Dropout(dropout)
        self.resid_dropout_1 = torch.nn.Dropout(dropout)
        self.dropout = dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        print('Using flash attention yes or no?',self.flash)
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            

    def forward(self, x, y):
        B, M, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)
        B, N, C = y.size()
        z = torch.cat([x,y],dim=1)
        #print(B,M,N,C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q_1, k_1, v_1  = self.c_attn_1(z).split(self.embed_dim, dim=2)


        k_1 = k_1.view(B, M+N, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, M+N, hs)
        q_1 = q_1.view(B, M+N, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, M+N, hs)
        v_1 = v_1.view(B, M+N, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, M+N, hs)
        
        # causal self-attention; Self-attend: (B, nh, M+N, hs) x (B, nh, hs, M+N) -> (B, nh, M+N, M+N)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q_1, k_1, v_1, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att_1 = (q_1 @ k_1.transpose(-2, -1)) * (1.0 / math.sqrt(k_1.size(-1)))
            att_1 = F.softmax(att_1, dim=-1)
            att_1 = self.attn_dropout_1(att_1)
            out = att_1 @ v_1 # (B, nh, M+N, M+N) x (B, nh, M+N, hs) -> (B, nh, M+N, hs)

        out = out.transpose(1, 2).contiguous().view(B, M+N, C) # re-assemble all head outputs side by side

        # output projection
        out = self.resid_dropout_1(self.c_proj_1(out))
        out_1, out_2 = torch.split(out,[M,N],dim=1)
        return out_1,out_2


class FFN(torch.nn.Module):
    def __init__(self, n_embed=1024, ffn_scaling=4, ffn_drop_value=0.1):
        super(FFN, self).__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(n_embed, n_embed * ffn_scaling),
            torch.nn.GELU(),
            torch.nn.Linear(n_embed * ffn_scaling, n_embed),
            torch.nn.Dropout(ffn_drop_value)
        )
    def forward(self, x):
        return self.ffn(x)

class DualEncoderDualDecoder(PreTrainedModel):
    def __init__(self,model_names,alpha=0.5,beta=0.5,PAD_ID=1,config=None,genconfig=None):
        super(DualEncoderDualDecoder,self).__init__(config=config)
        self.en_indic_model = self.initialize_model(model_names[0], "en-indic", "")
        self.indic_indic_model = self.initialize_model(model_names[1], "indic-indic", "")
        self.alpha = alpha
        self.beta = beta
        self.PAD_ID = PAD_ID
        self.is_encoder_decoder = True
        self.config = config
        self.generation_config = genconfig
        self.multisrc_aligner = MultiSrcAttention()
        self.ffn = FFN()
        self.layernorm1 = torch.nn.LayerNorm(1024)
        self.layernorm2 = torch.nn.LayerNorm(1024)
        
    def get_loss(self,logits,targets):
        #print(logits.shape,targets.shape)
        B, C, V = logits.shape
        logits = logits.reshape(B*C, V)
        targets = targets.reshape(B*C)
        PAD_ID = self.PAD_ID
        loss = F.cross_entropy(logits, targets,reduction='none')
        assert loss.shape==targets.shape
        loss_mask_1 = targets!=PAD_ID
        loss_masked = loss.where(loss_mask_1, torch.tensor(0.0))
        #print(loss_masked.sum(),torch.count_nonzero(loss_masked), torch.isnan(loss_masked).sum()/loss_masked.numel())
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
        enc_out_1 = self.en_indic_model.model.encoder(
            **inputs_1,
            return_dict=True
        )
        enc_out_2 = self.indic_indic_model.model.encoder(
            **inputs_2,
            return_dict=True
        )
        #print('enc 1 out',enc_out_1.last_hidden_state.shape)
        #print(inputs_1.input_ids.shape)
        #print('enc 2 out',enc_out_2.last_hidden_state.shape)
        #print(inputs_2.input_ids.shape)
        enc_out_1_last_hidden_state, enc_out_2_last_hidden_state = self.multisrc_aligner(enc_out_1.last_hidden_state, enc_out_2.last_hidden_state)

        enc_out_1_last_hidden_state = enc_out_1.last_hidden_state + self.layernorm1(enc_out_1_last_hidden_state)
        enc_out_1_last_hidden_state = enc_out_1_last_hidden_state + self.layernorm2(self.ffn(enc_out_1_last_hidden_state))

        enc_out_2_last_hidden_state = enc_out_2.last_hidden_state + self.layernorm1(enc_out_2_last_hidden_state)
        enc_out_2_last_hidden_state = enc_out_2_last_hidden_state + self.layernorm2(self.ffn(enc_out_2_last_hidden_state))

        concatenated_hidden_states = torch.cat([enc_out_1_last_hidden_state,enc_out_2_last_hidden_state],dim=1)
        #print('concat enc states',concatenated_hidden_states.shape)
        dec_out = self.indic_indic_model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states = concatenated_hidden_states,
            return_dict=True
        )
        #print('dec states',dec_out.last_hidden_state.shape)
        dec_logits = self.indic_indic_model.lm_head(dec_out.last_hidden_state)
        #print('dec logits',dec_logits.shape)
        #print(labels.shape if labels is not None else 'No labels')

        #out_1 = self.en_indic_model(**inputs_1, decoder_input_ids=decoder_input_ids, return_dict=True)
        #out_2 = self.indic_indic_model(**inputs_2, decoder_input_ids=decoder_input_ids, return_dict=True)
        normalized_out = F.softmax(dec_logits,dim=-1)
        #normalized_out_1 = F.softmax(out_1.logits,dim=-1)
        #normalized_out_2 = F.softmax(out_2.logits,dim=-1)
        #out = self.alpha*normalized_out_1 + self.beta*normalized_out_2
        #print(normalized_out.shape,labels.shape if labels is not None else 'None')
        if eval == False:
            loss = self.get_loss(normalized_out,labels) 
        else:
            loss = None
        return Seq2SeqLMOutput(
            loss=loss,
            logits=normalized_out,
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
        
        if input_ids.shape[1]>(shape1+shape2):
            decoder_input_ids = torch.ones((input_ids.shape[0],1)).to(torch.long).to(device=device) * 2
            decoder_input_ids = torch.cat((decoder_input_ids,input_ids[:,shape1+shape2:]),dim=1)
            input_ids = input_ids[:,:shape1+shape2]
            attention_mask = attention_mask[:,:shape1+shape2]

        if decoder_input_ids is None:
            decoder_input_ids = torch.ones((input_ids.shape[0],1)).to(torch.long).to(device=device) * 2
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids).to(dtype=attention_mask.dtype).to(device=device)
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "shape1": shape1,
            "shape2": shape2,
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
            model = model.to(DEVICE)
            #model.half()
        
        if train==True:
            model.train()
        else:
            model.eval()
        
        return model
    
        
        
def read_file_to_list(file_path):
    lines_list = []
    with open(file_path, 'r') as file:
        data = file.readlines()
        for line in data:
            lines_list.append(line.strip())
    return lines_list


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    alpha: Optional[float] = field(
        default=0.5, metadata={"help": "alpha value for logits of first model"}
    )
    beta: Optional[float] = field(
        default=0.5, metadata={"help": "beta value for logits of second model"}
    )




@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    src_1_path: Optional[str] = field(
        default=None, metadata={"help": "The src input 1 training data file (a text file)."}
    )
    src_2_path: Optional[str] = field(
        default=None,
        metadata={"help": "The src input 1 training data file (a text file)."},
    )
    
    tgt_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tgt output training data file (a text file)."},
    )


    test_src_1_path: Optional[str] = field(
        default=None, metadata={"help": "The src input 1 test data file (a text file)."}
    )
    test_src_2_path: Optional[str] = field(
        default=None,
        metadata={"help": "The src input 1 test data file (a text file)."},
    )
    
    test_tgt_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tgt output test data file (a text file)."},
    )
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dropout_rate: Optional[float] = field(
        default=0.1,
        metadata={"help": "specify dropout rate"},
    )
    
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project name"},
    )
    
    saved_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )



if __name__ == "__main__":
    # if len(sys.argv) < 4:
    #     print(sys.argv)
    #     print("Usage: python3 *.py <filename1> <filename2> <filename3>")
    #     sys.exit(1)

    model_names = ["ai4bharat/indictrans2-en-indic-1B", "ai4bharat/indictrans2-indic-indic-1B"]
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set seed
    set_seed(training_args.seed)
    # set the schedular type
    training_args.evaluation_strategy = IntervalStrategy.STEPS
    training_args.logging_strategy = IntervalStrategy.STEPS
    training_args.save_strategy = IntervalStrategy.STEPS
    training_args.save_safetensors = False

    file_path1 = data_args.src_1_path
    src_lang_1 = file_path1.split('.')[-1]
    file_path2 = data_args.src_2_path
    src_lang_2 = file_path2.split('.')[-1]
    file_path3 = data_args.tgt_path
    tgt_lang = file_path3.split('.')[-1]


    test_file_path1 = data_args.test_src_1_path
    test_file_path2 = data_args.test_src_2_path
    test_file_path3 = data_args.test_tgt_path

    def preprocess_function(example,src_lang_1=src_lang_1,src_lang_2=src_lang_2,tgt_lang=tgt_lang):
        en_inp = example['src1']
        indic_inp = example['src2']
        indic_target = example['tgt']
        type_data = example["type"]
        
        batch_en_inp = ip.preprocess_batch([en_inp], src_lang=src_lang_1, tgt_lang=tgt_lang)
        batch_indic_inp = ip.preprocess_batch([indic_inp], src_lang=src_lang_2, tgt_lang=tgt_lang)
        batch_indic_target = ip.preprocess_batch([indic_target], src_lang=tgt_lang, tgt_lang=tgt_lang)
        batch_indic_target = [remove_first_two_words(line) for line in batch_indic_target]
        
        return {"inputs_1": batch_en_inp[0], "inputs_2": batch_indic_inp[0], "outputs":batch_indic_target[0], "type": type_data}


    train_dataset = get_datasets(file_path1,file_path2,file_path3)
    train_dataset = train_dataset.shuffle(seed=training_args.seed).map(preprocess_function, num_proc=16)


    test_dataset = get_datasets(test_file_path1,test_file_path2,test_file_path3)
    test_dataset = test_dataset.map(preprocess_function, num_proc=16)

    print(train_dataset, len(train_dataset["inputs_1"][0]))
    print("Length of train dataset is ", len(train_dataset))

    print(test_dataset, len(test_dataset["inputs_1"][0]))
    print("Length of test dataset is ", len(test_dataset))

    metric = evaluate.load('sacrebleu',trust_remote_code=True)

    data_collator = DataCollatorForSeq2SeqTraining()

    device = torch.device("cuda")

    PAD_ID = 1
    config = Config(
        n_embed = 1024,
        is_encoder_decoder = False,
        bos_token_id = 0,
        decoder_start_token_id = 2,
        eos_token_id = 2,
        max_source_positions = 512,
        max_target_positions = 256,
        use_cache = False,
        pad_token_id = 1,
    )
    genconfig = GenerationConfig.from_pretrained(model_names[0])
    
    training_args.generation_config = genconfig
    model = DualEncoderDualDecoder(model_names,PAD_ID=PAD_ID, alpha=model_args.alpha, beta=model_args.beta, config=config, genconfig=genconfig).to(device)
    
    # genconfig = GenConfig()
    # genconfig.max_new_tokens = model_args.gen_new_tokens
    # genconfig.num_beams = model_args.gen_num_beams
    # genconfig.early_stopping = model_args.gen_early_stopping
    # training_args.generation_config = genconfig
    IN22_src_1 = []
    IN22_src_2 = []
    IN22_tgt = []
    with open(test_file_path1) as f:
        IN22_src_1 = f.readlines()
    with open(test_file_path2) as f:
        IN22_src_2 = f.readlines()
    with open(test_file_path3) as f:
        IN22_tgt = f.readlines()

    test_predictions = []
    test_references = [x.strip() for x in IN22_tgt]

    print('Saving torch state dict to',os.path.join(training_args.output_dir,'torch_state_dict'))
    #torch.save(model.state_dict(),os.path.join(training_args.output_dir,'torch_state_dict'))
    print("Saved")

    print('Saving torch model to',os.path.join(training_args.output_dir,'torch_model'))
    #torch.save(model,os.path.join(training_args.output_dir,'torch_model'))
    print("Saved")
    
    # model.eval()
    # for test_inputs_1,test_inputs_2,test_tgt in tqdm(zip(IN22_src_1,IN22_src_2,IN22_tgt)):
    #     test_inputs_1 = [test_inputs_1]
    #     test_inputs_1 = ip.preprocess_batch(test_inputs_1, src_lang=src_lang_1, tgt_lang=tgt_lang)
    #     #print(test_inputs_1)
    #     test_inputs_1 = en_indic_tokenizer(
    #         test_inputs_1,
    #         src=True,
    #         truncation=True,
    #         padding="longest",
    #         return_tensors="pt",
    #         return_attention_mask=True,
    #     )
    #     #print(test_inputs_1)
    #     #print(en_indic_tokenizer.batch_decode(test_inputs_1.input_ids,src=True))
    #     test_inputs_2 = [test_inputs_2]
    #     test_inputs_2 = ip.preprocess_batch(test_inputs_2, src_lang=src_lang_2, tgt_lang=tgt_lang)
    #     #print(test_inputs_2)
    #     test_inputs_2 = indic_indic_tokenizer(
    #         test_inputs_2,
    #         src=True,
    #         truncation=True,
    #         padding="longest",
    #         return_tensors="pt",
    #         return_attention_mask=True,
    #     )
    #     #print(test_inputs_2)
    #     #print(indic_indic_tokenizer.batch_decode(test_inputs_2.input_ids,src=True))

    #     test_shape1 = test_inputs_1.input_ids.shape[-1]
    #     test_shape2 = test_inputs_2.input_ids.shape[-1]
    #     inputs = {
    #         'input_ids': torch.cat((test_inputs_1.input_ids,test_inputs_2.input_ids),dim=1),
    #         'attention_mask': torch.cat((test_inputs_1.attention_mask,test_inputs_2.attention_mask),dim=1),
    #     }
    #     test_outputs = model.generate(
    #         inputs = inputs['input_ids'].to(device),
    #         attention_mask = inputs['attention_mask'].to(device),
    #         num_beams = 5,
    #         max_new_tokens = 256, 
    #         #early_stopping = True,
    #         #eos_token_id = 2,
    #         shape1 = test_shape1,
    #         shape2 = test_shape2,
    #         use_cache=False,
    #         labels = torch.ones((0,0)).to(device)
    #     )
    #     print('Output')
    #     test_outputs = test_outputs[:,test_shape1+test_shape2:]
    #     #print(test_outputs)
    #     outputs = en_indic_tokenizer.batch_decode(test_outputs, src=False)
    #     #print(outputs)
    #     outputs = ip.postprocess_batch(outputs, lang=tgt_lang)
    #     print('Test outputs', outputs)
    #     test_predictions.append(outputs[0])
    #     print('Actual output', test_tgt)

    # print(test_predictions)
    # print(test_references)

    # processed_tgt = caller(test_references,tgt_lang,'false','false')
    # processed_preds = caller(test_predictions,tgt_lang,'false','false')

    # assert len(test_predictions)==len(test_references)
    # assert len(processed_preds)==len(processed_tgt)
    # print('Zero shot scores',metric.compute(predictions=processed_preds,references=processed_tgt,tokenize='none'))
    # #raise Exception('TEST')
    model.train()
    wandb.init(
        # set the wandb project where this run will be logged
        project=data_args.wandb_project,
        # track hyperparameters and run metadata
        config={
            "learning_rate": training_args.learning_rate,
            "architecture": "IndicTrans2",
            "steps": training_args.max_steps
        }
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters are:--------------- ", pytorch_total_params)
    
    print("Trainer args:--------------- ")
    print(training_args)

    def set_before_second_occurrence_to_1(arr):
        result = arr
        for i, row in enumerate(arr):
            second_occurrence_index = np.where(row == 2)[0]
            if len(second_occurrence_index) > 1:
                result[i, :second_occurrence_index[1]+1] = 1
        return result


    def compute_metrics(eval_preds):
        print('----------Computing BLEU scores---------')
        #print('\n'*5)
        #raise Exception('METRICS WORK')
        #copied from https://github.com/huggingface/transformers/blob/39ef3fb248ba288897f35337f4086054c69332e5/examples/pytorch/translation/run_translation.py#L575
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, PAD_ID)
        preds = set_before_second_occurrence_to_1(preds)
        decoded_preds = en_indic_tokenizer.batch_decode(preds, src=False)
        #print(decoded_preds)
        #print(tgt_lang)
        decoded_preds = ip.postprocess_batch(decoded_preds, lang=tgt_lang)
        labels = np.where(labels != -100, labels, PAD_ID)
        #print(preds[0,:])
        decoded_labels = en_indic_tokenizer.batch_decode(labels, src=False)
        decoded_labels = ip.postprocess_batch(decoded_labels, lang=tgt_lang)
        print(decoded_preds)
        #print(decoded_labels[0])
        #print(decoded_preds[0])

        processed_tgt = caller(decoded_labels,tgt_lang,'false','false')
        processed_preds = caller(decoded_preds,tgt_lang,'false','false')

        assert len(decoded_preds)==len(decoded_labels)
        assert len(processed_preds)==len(processed_tgt)
        # Some simple post-processing
        # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=processed_preds,references=processed_tgt,tokenize='none')
        result = {"bleu": result["score"]}
        print(result)
        print('\n'*5)
        #raise Exception('BLEU test')
        prediction_lens = [np.count_nonzero(pred != PAD_ID) for pred in preds] #1==PAD Token
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )


    # Training
    model_path = (
        model_args.model_name_or_path
        if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
        else None
    )
    trainer.train(model_path=model_path)

    if trainer.is_world_process_zero():
        # save the best model
        trainer.save_model(os.path.join(training_args.output_dir,'final_model'))

    print('Saving torch state dict to',os.path.join(training_args.output_dir,'torch_state_dict'))
    #torch.save(model.state_dict(),os.path.join(training_args.output_dir,'torch_state_dict'))
    print("Saved")

    print('Saving torch model to',os.path.join(training_args.output_dir,'torch_model'))
    #torch.save(model,os.path.join(training_args.output_dir,'torch_model'))
    print("Saved")
    
