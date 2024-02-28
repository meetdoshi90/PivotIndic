import sys
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from tqdm import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, NewType, Tuple
from datasets import Dataset
import torch.nn.functional as F
import os
import wandb
import logging
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    SchedulerType,
    IntervalStrategy
)

alpha = 0.5
beta = 0.5
EPOCHS = 1
BATCH_SIZE = 4
optim_name = 'AdamW'
lr = 2e-4
wd = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
        return {"inputs_1": inputs_1,"inputs_2": inputs_2,"outputs": outputs}



class DualEncoderDualDecoder(torch.nn.Module):
    def __init__(self,model_names,alpha=0.5,beta=0.5,PAD_ID=1):
        super().__init__()
        self.en_indic_model = self.initialize_model(model_names[0], "en-indic", "")
        self.indic_indic_model = self.initialize_model(model_names[1], "indic-indic", "")
        self.alpha = alpha 
        self.beta = beta
        self.PAD_ID = PAD_ID
        
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

    def forward(self, inputs_1,inputs_2,outputs):
        out_1 = self.en_indic_model(**inputs_1, decoder_input_ids=outputs.input_ids[...,:-1], decoder_attention_mask=outputs.attention_mask[...,:-1])
        out_2 = self.indic_indic_model(**inputs_2, decoder_input_ids=outputs.input_ids[...,:-1], decoder_attention_mask=outputs.attention_mask[...,:-1])
        normalized_out_1 = F.softmax(out_1.logits,dim=-1)
        normalized_out_2 = F.softmax(out_2.logits,dim=-1)
        out = self.alpha*normalized_out_1 + self.beta*normalized_out_2
        output = {}
        output['logits'] = out 
        #print(outputs.input_ids[0,:-1],outputs.input_ids[0,1:])
        loss = self.get_loss(out,outputs.input_ids[...,1:])
        output['loss'] = loss
        return output 
        
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
        )

        if qconfig == None:
            model = model.to(DEVICE)
            model.half()
        
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
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
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

    print(train_dataset, len(train_dataset["inputs_1"][0]))
    print("Length of train dataset is ", len(train_dataset))

    data_collator = DataCollatorForSeq2SeqTraining()

    device = torch.device("cuda")

    model = DualEncoderDualDecoder(model_names).to(device)

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

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        # compute_metrics=compute_metrics_func,
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
