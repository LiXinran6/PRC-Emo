import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import json
from sklearn.metrics import classification_report
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import setup_chat_format, set_seed as trl_seed
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer
from transformers import set_seed as transf_seed
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np 
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
import random, glob
import shutil
from lightning import seed_everything 
from transformers.trainer_utils import get_last_checkpoint
from reformat_data_ft_llm_combine import process
from transformers import TrainerCallback
from peft import PeftModel, PeftConfig

#4090用不了这两个，需要禁用
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# 添加 numpy.ndarray 到安全白名单
torch.serialization.add_safe_globals([np.ndarray])
# 覆盖默认加载行为（确保所有 torch.load 调用生效）
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # 强制关闭安全模式
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

#随机种子设置
def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    trl_seed(seed)
    transf_seed(seed)
    
    """将对话格式转换为模型输入"""
def formatting_prompts_func(samples):
    prompt_texts = [tokenizer.apply_chat_template(
             sample[:-1], tokenize=False, add_generation_prompt=True) for sample in samples["messages"]]
    
    print("=="*50)
    print(prompt_texts[-1])
    print("=="*50)
    return prompt_texts

def split_label(sample):
    """提取并编码标签"""
    tokenized_lb = tokenizer.encode(sample['messages'][-1]['content'], padding='max_length',max_length=10 )
    sample['labels'] = tokenized_lb 
    return sample

class CurriculumDataset:
    """课程学习数据集管理器"""
    def __init__(self, dataset, bucket_number=8, curriculum=True):
        self.full_dataset = dataset
        self.bucket_number = bucket_number
        self.curriculum = curriculum
        self.buckets = None     #初始化桶列表为空，后续根据需要创建
        
        if self.curriculum:
            self.buckets = self._create_buckets()
            print(f"Created {len(self.buckets)} buckets for curriculum learning")
            for i, bucket in enumerate(self.buckets):
                print(f"Bucket {i}: {len(bucket)} samples, difficulty range: {min([s['difficulty'] for s in bucket]):.4f} - {max([s['difficulty'] for s in bucket]):.4f}")
    
    def _create_buckets(self):
        """根据difficulty字段创建桶"""
        # 按difficulty难度值升序排序
        sorted_data = sorted(self.full_dataset, key=lambda x: x.get('difficulty', 0))
        
        # 分桶
        bucket_size = len(sorted_data) // self.bucket_number
        buckets = []
        
        for i in range(self.bucket_number):
            start_idx = i * bucket_size
            if i == self.bucket_number - 1:  # 最后一个桶包含剩余所有数据
                end_idx = len(sorted_data)
            else:
                end_idx = (i + 1) * bucket_size
            
            bucket = sorted_data[start_idx:end_idx]
            buckets.append(bucket)
        
        return buckets
    
    def get_curriculum_dataset(self, step_index):
        """获取当前课程步骤的数据集"""
        if not self.curriculum or step_index >= len(self.buckets):
            # 如果不使用课程学习或已达到最后一步，返回全部数据
            data = list(self.full_dataset)
        else:
            # 返回从第0桶到第step_index桶的所有数据
            data = []
            for i in range(step_index + 1):
                data.extend(self.buckets[i])
        
        print(f"Curriculum step {step_index}: using {len(data)} samples")
        
        # 随机打乱数据
        random.shuffle(data)
        
        # 转换为HuggingFace Dataset格式
        return Dataset.from_list(data)

# 移除了课程学习相关逻辑的简化版Trainer
class SimplifiedTrainer(SFTTrainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 数据预处理参数配置
        self.data_process_args = argparse.Namespace(
            packing=False,
            dataset_text_field=None,
            max_seq_length=kwargs.get('max_seq_length', None),
            formatting_func=formatting_prompts_func,
            num_of_sequences=kwargs.get('num_of_sequences', 1024),
            chars_per_token=kwargs.get('chars_per_token', 3.6),
            remove_unused_columns=kwargs.get('args').remove_unused_columns if kwargs.get('args') is not None else True,
            dataset_kwargs=kwargs.get('dataset_kwargs', {})
        )
        self.eval_dataset = self._process_raw_data(kwargs.get('eval_dataset', None))
        print("len(eval dataset) = ",  len(self.eval_dataset))
    
    def _process_raw_data(self, dataset):
        """数据预处理流水线"""
        dataset2 = dataset.map(split_label)
        dataset = self._prepare_dataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                packing=False,
                dataset_text_field=None,
                max_seq_length=self.data_process_args.max_seq_length,
                formatting_func=self.data_process_args.formatting_func,
                num_of_sequences=self.data_process_args.num_of_sequences,
                chars_per_token=self.data_process_args.chars_per_token,
                remove_unused_columns=self.data_process_args.remove_unused_columns,
                **self.data_process_args.dataset_kwargs, 
            )
        dataset = dataset.add_column('labels', dataset2['labels'])
        return dataset 
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """获取评估数据加载器"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
            
        if "input_ids" not in eval_dataset.column_names and "labels" not in eval_dataset.column_names:
            eval_dataset = self._process_raw_data(eval_dataset)
            
        return super().get_eval_dataloader(eval_dataset)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only= None,
        ignore_keys = None,
        metric_key_prefix="eval",
    ) -> EvalLoopOutput:
        model = self.model
        model = model.to(dtype=torch.bfloat16)
        
        model.eval()
            
        all_preds = []
        all_labels = []
        all_raw_decoded = []
        
        def post_process(str_out):
            try:
                gen_text = str_out.split("assistant\n")[-1].split("<|im_end|>")[0]
            except:
                gen_text = "error"
            return gen_text
        
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(dataloader)):
                inputs = self._prepare_inputs(inputs)
                gen_kwargs = {'max_new_tokens': 10, 
                              'do_sample': False, 
                              'eos_token_id': self.tokenizer.eos_token_id, 
                              'pad_token_id': self.tokenizer.pad_token_id,
                              "temperature": 0.1,
                              }
                generated_tokens = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )
                labels = inputs.pop("labels")
                str_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                raw_decoded = [e for e in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)]
                str_decoded = [post_process(e) for e in raw_decoded]
                all_preds += str_decoded
                all_labels += str_labels
                all_raw_decoded += raw_decoded
        num_samples = len(dataloader)
        f1_weighted = f1_score(
            all_labels,
            all_preds,
            average=f"weighted",
        )
            # 添加新的评估指标
        f1_macro = f1_score(
            all_labels,
            all_preds,
            average="macro",
        )
        print(set(all_preds))  # 看预测标签有哪些
        print(set(all_labels))  # 看真实标签有哪些
        accuracy = accuracy_score(all_labels, all_preds)
        print(classification_report(all_labels, all_preds, digits=4))
        metrics = { f"{metric_key_prefix}_weighted-f1": f1_weighted,
                   f"{metric_key_prefix}_macro-f1": f1_macro,
                    f"{metric_key_prefix}_accuracy": accuracy}
            # 保存详细结果，包含所有指标
        result_data = {
            "metrics": metrics, 
            "detail_pred": list(zip(all_preds, all_labels, all_raw_decoded))
        }
        
        json.dump(result_data, 
              open(f"{self.args.output_dir}/result_{metric_key_prefix}_step-{self.state.global_step}.json", "wt"), 
              indent=1)
        
        # 打印所有指标到控制台，方便观察
       # print(f"\n=== {metric_key_prefix.upper()} METRICS ===")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print("=" * 30)
        
        del model
        torch.cuda.empty_cache()
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process ...')
    parser.add_argument('--do_train', action="store_true", help='fine tuning a LLM model with LoRA', default=False)
    parser.add_argument('--do_eval_test', action="store_true", help='eval on test set', default=False)
    parser.add_argument('--do_eval_dev', action="store_true", help='eval on dev set', default=False)
    parser.add_argument('--ft_model_path', type=str, default=None, help='fintuned model path') 
    parser.add_argument('--ft_model_id', type=str, default=None, help='fintuned model id for saving after train it')
    parser.add_argument('--prompting_type', type=str, default='spdescV2', help='prompting style in {cot, fewshot, zeroshot}')
    parser.add_argument('--base_model_id', type=str, default='/usr/3Tusr/lixinran/workspace/LLM/LLM_bases/LLaMA2/', help='base llm model id')
    parser.add_argument('--extract_prompting_llm_id', type=str, default='LLaMA2', help='base llm model id')
    parser.add_argument('--epoch', type=int, default=None, help='training epoch')
    parser.add_argument('--max_steps', type=int, default=None, help='training steps')
    parser.add_argument('--lr_scheduler', type=str, default='constant', help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate value')
    parser.add_argument('--seed', type=int, default=42, help='random seed value')
    parser.add_argument('--kshot', type=int, default=0, help='k shot examples for llm')
    parser.add_argument('--lora_r', type=int, default=32, help='lora rank')
    parser.add_argument('--eval_delay', type=int, default=200, help='eval delay')
    parser.add_argument('--window', type=int, default=5, help='local context window size')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length for chunking/packing')
    parser.add_argument('--re_gen_data', action="store_true", help='re generate data', default=False)
    parser.add_argument('--data_name', type=str,  help='data name in {iemocap, meld, emorynlp}', default='iemocap')
    parser.add_argument('--data_folder', type=str,  help='path folder save all data', default='./data/')
    parser.add_argument('--output_folder', type=str,  help='path folder save all data', default='./finetuned_llm/')
    # 课程学习相关参数
    parser.add_argument('--curriculum', action="store_true", help='enable curriculum learning', default=False)
    parser.add_argument('--bucket_number', type=int, default=8, help='number of buckets for curriculum learning')
    parser.add_argument('--curriculum_update_epochs', type=int, default=None, help='epochs between curriculum updates')

    args, unknown = parser.parse_known_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.prompting_type == 'zeroshot':
        args.kshot = 0
    print(args)
    
    set_random_seed(args.seed)
    
    all_path_folder_preprocessed_data = [f"{args.data_folder}/{args.data_name}.{d_type}.{args.kshot}shot_w{args.window}_{args.prompting_type}.jsonl" \
        for d_type in [ 'train' , 'valid',  'test']]
    if args.re_gen_data:
        process(all_path_folder_preprocessed_data, args) 
                    
    # Load jsonl data from disk
    full_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[0], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    valid_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[1], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    test_dataset = load_dataset("json", data_files=all_path_folder_preprocessed_data[2], split="train", cache_dir=f'{args.output_folder}/{args.ft_model_id}')
    
    # 创建课程学习管理器
    curriculum_manager = None
    if args.curriculum and args.do_train:
        print("Initializing curriculum learning...")
        curriculum_manager = CurriculumDataset(
            dataset=list(full_dataset), 
            bucket_number=args.bucket_number,
            curriculum=True
        )
    
    # 定义量化配置（全局）
    tensor_data_type = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=tensor_data_type
    )
    
    # Load model and tokenizer
    model_id = args.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 非训练模式下单独处理
    if not args.do_train:
        tensor_data_type = torch.float32
        ft_model_path = f"{args.output_folder}/{args.ft_model_id}" if args.ft_model_path is None else args.ft_model_path
        tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
        model = AutoPeftModelForCausalLM.from_pretrained(
            ft_model_path,
            device_map="auto",
            torch_dtype=tensor_data_type
        )
        model, tokenizer = setup_chat_format(model, tokenizer)
        
    # 设置课程学习更新频率
    if args.curriculum and args.curriculum_update_epochs is None:
        # 确保有足够的epoch数
        if args.epoch:
            args.curriculum_update_epochs = max(1, args.epoch // args.bucket_number)
        else:
            # 如果没有指定epoch数，则使用默认值
            args.curriculum_update_epochs = 1
    
    # 分阶段训练实现
    if args.do_train and args.curriculum:
        print("="*50)
        print("Starting curriculum learning with phased training")
        print(f"Total buckets: {args.bucket_number}")
        print(f"Epochs per phase: {args.curriculum_update_epochs}")
        print("="*50)
        
        # 计算每个阶段需要的epoch数
        remaining_epochs = args.epoch or 0  # 确保不是None
        base_output_dir = f'{args.output_folder}/{args.ft_model_id}'
        
        # 创建卸载文件夹
        offload_folder = os.path.join(base_output_dir, "offload")
        os.makedirs(offload_folder, exist_ok=True)
        
        # 第一阶段模型初始化
        model = None
        trainer = None
        
        for phase in range(args.bucket_number):
            print("\n" + "="*50)
            print(f"Starting training phase {phase+1}/{args.bucket_number}")
            print("="*50)
            
            # 获取当前阶段的训练集
            current_dataset = curriculum_manager.get_curriculum_dataset(phase)
            
            # 设置当前阶段的输出目录
            phase_output_dir = f"{base_output_dir}_phase_{phase}"
            os.makedirs(phase_output_dir, exist_ok=True)
            
            # 计算当前阶段训练的epoch数
            if remaining_epochs > 0:
                current_epochs = min(args.curriculum_update_epochs, remaining_epochs)
                remaining_epochs -= current_epochs
            else:
                current_epochs = 0
            
            # 如果当前阶段没有epoch需要训练，跳过
            if current_epochs <= 0:
                print(f"Skipping phase {phase} as no epochs remaining")
                continue
            
            # 创建训练参数
            training_args = TrainingArguments(
                output_dir=phase_output_dir,
                num_train_epochs=current_epochs,
                max_steps=args.max_steps,
                per_device_train_batch_size=1,  # 减少批大小以节省显存
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                save_total_limit=1,
                optim="adamw_torch_fused", 
                eval_delay=args.eval_delay,
                logging_steps=50,
                eval_steps=50,
                save_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model='weighted-f1',
                greater_is_better=True,
                eval_strategy='steps',
                logging_strategy='steps',
                save_strategy="steps",
                learning_rate=args.lr,
                bf16=True,
                tf32=False,
                max_grad_norm=0.3,
                warmup_ratio=0.03,
                lr_scheduler_type=args.lr_scheduler,
                push_to_hub=False,
                group_by_length=True,
                report_to="tensorboard",
            )

            # 如果是第一阶段，使用基础模型；否则加载上一阶段的模型
            if phase == 0:
                # 第一阶段：加载基础模型
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                    torch_dtype=tensor_data_type,
                    quantization_config=bnb_config
                )
                
                # 应用聊天格式
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.padding_side = 'left'
                model, tokenizer = setup_chat_format(model, tokenizer)
                
                # 保存原始模型的分词器
                tokenizer.save_pretrained(phase_output_dir)
                print(f"Tokenizer vocab size after setup_chat_format: {len(tokenizer)}")
                print(f"Model embed_tokens size: {model.get_input_embeddings().weight.shape}")
                print(f"Special tokens: {tokenizer.special_tokens_map}")
                
                # 保存这些信息到文件
                debug_info = {
                    "vocab_size": len(tokenizer),
                    "embed_shape": list(model.get_input_embeddings().weight.shape),
                    "special_tokens": tokenizer.special_tokens_map
                }
                with open(f"{phase_output_dir}/debug_info.json", "w") as f:
                    json.dump(debug_info, f, indent=2)
                
                # 第一阶段：创建带有LoRA配置的训练器
                trainer = SimplifiedTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=current_dataset,
                    eval_dataset=valid_dataset,
                    neftune_noise_alpha=5,
                    peft_config=LoraConfig(
                        lora_alpha=128,
                        lora_dropout=0.05,
                        r=args.lora_r,
                        bias="none",
                        target_modules="all-linear",
                        task_type="CAUSAL_LM", 
                    ),
                    max_seq_length=args.max_seq_len,
                    tokenizer=tokenizer,
                    packing=True,
                    dataset_kwargs={
                        "add_special_tokens": False,
                        "append_concat_token": False,
                    }
                )
                
            else:
                # 后续阶段：加载上一阶段保存的模型
                prev_phase_dir = f"{base_output_dir}_phase_{phase-1}"
                print(f"Loading model from previous phase: {prev_phase_dir}")
                
                # 释放内存
                if model is not None:
                    del model
                if trainer is not None:
                    del trainer
                torch.cuda.empty_cache()
                
                # **关键修改：先加载tokenizer，确保词汇表一致**
                tokenizer = AutoTokenizer.from_pretrained(prev_phase_dir)
                tokenizer.padding_side = 'left'
                
                # 首先加载配置
                peft_config = PeftConfig.from_pretrained(prev_phase_dir)
                
                # 加载基础模型
                base_model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    device_map="auto",
                    torch_dtype=tensor_data_type,
                    quantization_config=bnb_config,
                    offload_folder=offload_folder,
                )
                # 设置 gradient checkpointing，确保输入也可参与反向传播
                base_model.gradient_checkpointing_enable()
                base_model.enable_input_require_grads()
                # 应用聊天格式
                base_model, tokenizer = setup_chat_format(base_model, tokenizer)
                for name, param in base_model.named_parameters():
                    param.requires_grad = False
                # 加载上阶段 LoRA adapter
                model = PeftModel.from_pretrained(
                    base_model,
                    prev_phase_dir,
                    offload_folder=offload_folder,
                )
                for name, param in model.named_parameters():
                    if not any(key in name for key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                print("Embedding requires_grad:", model.get_input_embeddings().weight.requires_grad)
                print("Before parameter adjustment:")
                model.print_trainable_parameters()
                
                # **关键修改：确保LoRA参数可训练**
                # 方法1：使用内置方法
                model.train()  # 设置为训练模式
                
                # 方法2：手动设置LoRA参数为可训练
                for name, param in model.named_parameters():
                    if any(lora_key in name for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                        param.requires_grad = True
                        #print(f"Set {name} requires_grad = True")
                
                print("After parameter adjustment:")
                model.print_trainable_parameters()
                
                # **重要：后续阶段不要重新初始化LoRA配置**
                # 创建训练器时不传入peft_config，因为模型已经是PeftModel
                trainer = SimplifiedTrainer(
                    model=model,  # 直接使用已经加载LoRA的模型
                    args=training_args,
                    train_dataset=current_dataset,
                    eval_dataset=valid_dataset,
                    neftune_noise_alpha=5,
                    # **关键：不传入peft_config，因为模型已经是PeftModel**
                    peft_config=None,  
                    max_seq_length=args.max_seq_len,
                    tokenizer=tokenizer,
                    packing=True,
                    dataset_kwargs={
                        "add_special_tokens": False,
                        "append_concat_token": False,
                    }
                )
            
            # 开始训练当前阶段
            print(f"Training phase {phase} with {len(current_dataset)} samples for {current_epochs} epochs")
            
            # 训练前再次确认参数状态
            print("\n=== Final parameter check before training ===")
            trainable_params = 0
            all_params = 0
            for name, param in model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    #if 'lora' in name.lower():
                        #print(f"Trainable LoRA param: {name}, shape: {param.shape}")
            
            print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")
            
            if trainable_params == 0:
                raise RuntimeError("No trainable parameters found! Training will fail.")
            
            trainer.train()
            
            # 保存当前阶段模型
            trainer.save_model(phase_output_dir)
            print(f"Saved model for phase {phase} to {phase_output_dir}")
            
            # 保存分词器
            tokenizer.save_pretrained(phase_output_dir)
            print(f"Evaluating on test dataset after phase {phase}...")
            # 如果所有epoch已完成，提前退出
            if remaining_epochs <= 0:
                print("All epochs completed. Exiting curriculum training.")
                break

        if remaining_epochs > 0:
            print("\n" + "="*50)
            print(f"Starting final training on full dataset for {remaining_epochs} additional epoch(s)")
            print("="*50)
            print(f"len(full_dataset):{len(full_dataset)}")
            # 设置输出路径
            full_phase_output_dir = f"{base_output_dir}_final_full_finetune"
            os.makedirs(full_phase_output_dir, exist_ok=True)

            # 清理内存，加载最后一阶段模型
            del model
            del trainer
            torch.cuda.empty_cache()

            last_phase_dir = f"{base_output_dir}_phase_{args.bucket_number - 1}"
            tokenizer = AutoTokenizer.from_pretrained(last_phase_dir)
            tokenizer.padding_side = 'left'
            peft_config = PeftConfig.from_pretrained(last_phase_dir)

            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map="auto",
                torch_dtype=tensor_data_type,
                quantization_config=bnb_config,
                offload_folder=offload_folder,
            )
            base_model.gradient_checkpointing_enable()
            base_model.enable_input_require_grads()
            base_model, tokenizer = setup_chat_format(base_model, tokenizer)
            for name, param in base_model.named_parameters():
                param.requires_grad = False
            model = PeftModel.from_pretrained(
                base_model,
                last_phase_dir,
                offload_folder=offload_folder,
            )
            for name, param in model.named_parameters():
                if any(key in name for key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            model.train()

            # 新的训练参数（只修改 epoch）
            training_args = TrainingArguments(
                output_dir=full_phase_output_dir,
                num_train_epochs=remaining_epochs,
                max_steps=args.max_steps,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                save_total_limit=1,
                optim="adamw_torch_fused",
                eval_delay=args.eval_delay,
                logging_steps=50,
                eval_steps=50,
                save_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model='weighted-f1',
                greater_is_better=True,
                eval_strategy='steps',
                logging_strategy='steps',
                save_strategy="steps",
                learning_rate=args.lr,
                bf16=True,
                tf32=False,
                fp16=False,
                dataloader_pin_memory=False,
                max_grad_norm=0.3,
                warmup_ratio=0.03,
                lr_scheduler_type=args.lr_scheduler,
                push_to_hub=False,
                group_by_length=True,
                report_to="tensorboard",
            )

            trainer = SimplifiedTrainer(
                model=model,
                args=training_args,
                train_dataset=full_dataset,
                eval_dataset=valid_dataset,
                peft_config=None,
                neftune_noise_alpha=5,
                max_seq_length=args.max_seq_len,
                tokenizer=tokenizer,
                packing=True,
                dataset_kwargs={
                    "add_special_tokens": False,
                    "append_concat_token": False,
                }
            )

            print(f"Training final phase with {len(full_dataset)} samples for {remaining_epochs} epochs")
            trainer.train()
            
            # 保存最终模型和测试评估结果
            trainer.save_model(full_phase_output_dir)
            tokenizer.save_pretrained(full_phase_output_dir)

            test_results = trainer.evaluate(test_dataset, metric_key_prefix=f'test_final_full')
            all_results_file = os.path.join(base_output_dir, "all_phase_test_results.json")
            if os.path.exists(all_results_file):
                with open(all_results_file, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
            else:
                all_results = {}
            all_results.update({f"final_full": test_results})
            with open(all_results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"Saved final full-finetune model to: {full_phase_output_dir}")
            print(f"Final test results saved to: {all_results_file}")
            ft_model_path = full_phase_output_dir

        else:
            # 训练完成后，将最终模型复制到主目录
            final_phase_dir = f"{base_output_dir}_phase_{min(phase, args.bucket_number-1)}"
            print(f"\nCurriculum training completed. Final model saved to: {final_phase_dir}")
            ft_model_path = final_phase_dir

    # 非课程学习训练
    elif args.do_train:
        print("Starting standard training without curriculum learning")
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=tensor_data_type,
            quantization_config=bnb_config
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = 'left'
        model, tokenizer = setup_chat_format(model, tokenizer)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=f'{args.output_folder}/{args.ft_model_id}',
            num_train_epochs=args.epoch,
            max_steps=args.max_steps,
            per_device_train_batch_size=2,  # 减少批大小以节省显存
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            save_total_limit=1,
            optim="adamw_torch", 
            eval_delay=args.eval_delay,
            logging_steps=50,
            eval_steps=50,
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model='weighted-f1',
            greater_is_better=True,
            eval_strategy='steps',
            logging_strategy='steps',
            save_strategy="steps",
            learning_rate=args.lr,
            bf16=True,
            tf32=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type=args.lr_scheduler,
            push_to_hub=False,
            group_by_length=True,
            report_to="tensorboard",
        )

        trainer = SimplifiedTrainer(
            model=model,
            args=training_args,
            train_dataset=full_dataset,
            eval_dataset=valid_dataset,
            neftune_noise_alpha=5,
            peft_config=LoraConfig(
                lora_alpha=128,
                lora_dropout=0.05,
                r=args.lora_r,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM", 
            ),
            max_seq_length=args.max_seq_len,
            tokenizer=tokenizer,
            packing=True,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            }
        )
        
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(f'{args.output_folder}/{args.ft_model_id}')