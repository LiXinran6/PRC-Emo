
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from torch.utils.data import DataLoader
import traceback
from tqdm import tqdm
import json
import re 
from transformers import LlamaTokenizer, AutoModel, AutoTokenizer, LlamaForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = 'iemocap'
data_folder = './data/'
prompt_type = 'spdescV6'
# spdescV3: About 100 words describing the dialogue background and speaker background information.
# Based on the full context, briefly describe the overall background of the conversation 
# (e.g., location/setting, cause of occurrence), and provide a concise profile of the speaker {speaker_name} 
# (e.g., personality traits, possible occupations).
# spdescV4: About 250 words describing the dialogue background and speaker background information.
# spdescV5: About 250 words describing the speaker's personality characteristics.
# spdescV6: About 100 words describing the speaker's personality, generated using Qwen-14B.


print("Loading model ...")

model_name = '/home/pubNAS3/xinran/llm/Qwen3-14B'  #  standard model, please switch to your local model before running.
tensor_data_type = torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=tensor_data_type
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=tensor_data_type,
    device_map="cuda:0"
)

# Qwen3 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    padding_side="left"  
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Model Qwen3 loaded successfully from local path.")

# Add a function to process Qwen3's reasoning output
def extract_qwen3_output(generated_text, original_prompt):
    """
    Extract the actual output from Qwen3-generated text by removing the reasoning process.
    """
    # First, remove the original prompt
    if original_prompt in generated_text:
        generated_text = generated_text.replace(original_prompt, "")
    
    # Search for the </think> tag
    think_end_pattern = r'</think>\s*'
    match = re.search(think_end_pattern, generated_text, re.IGNORECASE)
    
    if match:
        # Extract text after the </think> tag
        actual_output = generated_text[match.end():].strip()
    else:
        # If the </think> tag is not found, use the remaining text (after prompt removal)
        actual_output = generated_text.strip()
    
    # Further clean up possible tags and extra whitespace
    actual_output = re.sub(r'<[^>]*>', '', actual_output)  # Remove other possible tags
    actual_output = actual_output.strip()
    
    return actual_output


def get_emotion_labels(dataset_name):
    label_map = {
        "iemocap": ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        "emorynlp": ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'],
        "meld": ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
        "dailydialog": ['no_emotion', 'happiness', 'sadness', 'surprise', 'anger', 'fear', 'disgust']
    }
    dataset_key = dataset_name.lower() if dataset_name else ""
    if dataset_key not in label_map:
        raise ValueError(f"Unknown dataset name '{dataset_name}' — no emotion label list found.")
    return label_map[dataset_key]


#面向对话数据的​​多维度批处理预处理器​​，主要完成对话数据的上下文整合、说话者关系建模和模型输入适配三大核心任务。
#A multi-dimensional batch preprocessor for dialogue data, primarily designed to perform three core tasks: context integration, speaker relationship modeling, and model input adaptation.
class BatchPreprocessor(object): 
    def __init__(self, tokenizer, dataset_name=None, window_ct=2) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name  = dataset_name
        self.window_ct = window_ct


    @staticmethod   #定义一个静态方法 load_raw_data，属于类的工具函数，不需要实例化对象就可以调用
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v['s_id'] = k   #s_id是对话的ID，不是句子ID
                new_data_list.append(v)
            return new_data_list
        elif isinstance(raw_data, list):
            return raw_data
            
    #说话者身份映射​
    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            # iemocap: label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
            speaker = {
                        "Ses01": {"F": "Mary", "M": "James"},
                        "Ses02": {"F": "Patricia", "M": "John"},
                        "Ses03": {"F": "Jennifer", "M": "Robert"},
                        "Ses04": {"F": "Linda", "M": "Michael"},
                        "Ses05": {"F": "Elizabeth", "M": "William"},
                    }
            s_id_first_part = s_id[:5] # 提取会话ID前缀（如Ses01）
            return speaker[s_id_first_part][gender].upper() #根据 session 名和性别查出姓名，并将其转为大写返回
        elif data_name in ['meld', "emorynlp"]:
            # emorynlp: label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
            # meld: label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
            gender_idx = gender.index(1) #从 one-hot 向量中获取索引值。例如 [1, 0, 0, 0, 0, 0] → 0，[0, 1] → 1
            return f"SPEAKER_{gender_idx}"#返回的说话者名字格式是 "SPEAKER_0" 或 "SPEAKER_1"
        elif data_name=='dailydialog':
            # dailydialog:  {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,  'anger': 4, 'fear': 5, 'disgust':6}
            return f"SPEAKER_{gender}"

    # Concatenate each sentence with its surrounding sentences within the specified around_window range
    # to form context-enriched text. The central sentence is marked with </s>. 
    # Each sentence is also prefixed with the corresponding speaker name.
    def sentence_mixed_by_surrounding(self, sentences, around_window, s_id, genders, data_name):
        new_sentences = []
        for i, cur_sent in enumerate(sentences):  # Index i and content of the current sentence
            tmp_s = ""
            # Sliding window over the range of sentences within around_window before and after the current one
            for j in range(max(0, i - around_window), min(len(sentences), i + around_window + 1)):
                # If the sentence j in the current window is the central (current) sentence,
                # add a special marker </s> before it to indicate the sentence boundary.
                if i == j:
                    tmp_s += " </s>"
                # Add formatted sentence information, including the speaker name 
                # (obtained via get_speaker_name) and the sentence text.
                # Example: " MICHAEL: I am fine." or " SPEAKER_1: Hello!"
                tmp_s += f" {self.get_speaker_name(s_id, genders[j], data_name=data_name)}: {sentences[j]}"
                if i == j:
                    # If it is the central sentence, also append </s> after it,
                    # enclosing the current sentence with </s> to help the model 
                    # identify which sentence is the “target” one.
                    tmp_s += " </s>"
            new_sentences.append(tmp_s)
        return new_sentences

    
    #该方法使得类的实例像函数一样可调用。批处理核心逻辑​​
    def __call__(self, batch):
        
        raw_sentences = []  # 存储处理后的所有句子
        raw_sentences_flatten = []  # 扁平化的句子列表
        labels = [] # 存储标签
        # 1. 计算批次中各对话的句子长度
        # masked tensor  
        lengths = [len(sample['sentences']) for sample in batch]#lengths存储每个样本的原始对话轮次数量
        max_len_conversation = max(lengths) #max_len_conversation确定批处理时的统一长度
        padding_utterance_masked = torch.BoolTensor([[False]*l_i+ [True]*(max_len_conversation - l_i) for l_i in lengths])
        # 创建一个布尔型张量，用来表示每个样本的句子是否需要填充。长度不足的部分填充True，其他为False
        # collect all sentences
        # - intra speaker
        ## 创建一个布尔型张量，表示每个对话中的说话者之间的关系，尺寸为(batch_size, max_len_conversation, max_len_conversation)
        intra_speaker_masekd_all = torch.BoolTensor(len(batch), max_len_conversation,max_len_conversation)
        for i, sample in enumerate(batch):# 遍历批次中的每个样本
        # 通过调用自定义函数 `sentence_mixed_by_surrounding` 对句子进行处理，这个函数将上下文窗口内的句子进行混合
            sentences_mixed_arround = self.sentence_mixed_by_surrounding(sample['sentences'], 
                                                                        around_window=self.window_ct, 
                                                                        s_id=sample['s_id'], 
                                                                        genders=sample['genders'],
                                                                        data_name=self.dataset_name)
        
            # conversation padding
            # 将处理后的句子列表填充到最大长度 
            padded_conversation = sentences_mixed_arround + ["<pad_sentence>"]* (max_len_conversation - lengths[i])
            raw_sentences.append(padded_conversation)# 添加到 raw_sentences
            raw_sentences_flatten += padded_conversation# 展平后的句子加入 raw_sentences_flatten

            # label padding 将标签填充到最大长度，不足的部分填充为 -1
            labels += [int(label) for label in sample['labels']] + [-1]* (max_len_conversation - lengths[i])

            # speaker创建一个方阵，用于标记说话者之间的关系，初始时全部为 False
            intra_speaker_masekd= torch.BoolTensor(len(padded_conversation),len(padded_conversation)).fill_(False)
            for j in  range(len( sample['genders'])):# 遍历当前样本中的每个句子的性别信息
                for k in  range(len( sample['genders'])):
                    gender_j = sample['genders'][j] # 第 j 个句子的说话者性别
                    gender_k = sample['genders'][k]  # 第 k 个句子的说话者性别

                    if gender_j == gender_k:    # 如果两句话的说话者性别相同，则认为它们是同一个说话者
                        intra_speaker_masekd[j][k] = True
                    else:
                        intra_speaker_masekd[j][k] = False

            intra_speaker_masekd_all[i] = intra_speaker_masekd   # 保存当前样本的说话者矩阵

        if len(labels)!= len(raw_sentences_flatten):     # 检查标签和句子数量是否一致
            print('len(labels)!= len(raw_sentences_flatten)')

        # utterance vectorizer
        # v_single_sentences = self._encoding(sample['sentences'])
        # 将展平后的对话文本进行分词处理，生成标准的Transformer输入格式
        contextual_sentences_ids = self.tokenizer(raw_sentences_flatten,  padding='longest', max_length=512, truncation=True, return_tensors='pt')
        # 通过torch.where找到所有分隔符（如</s>）的位置
        sent_indices, word_indices = torch.where(contextual_sentences_ids['input_ids'] == self.separate_token_id)
        #将每个句子的分隔符位置存入gr_sent_indices列表
        gr_sent_indices = [[] for e in range(len(raw_sentences_flatten))]# 创建一个空列表用于存储分隔符位置
        for sent_idx, w_idx in zip (sent_indices, word_indices):# 遍历所有的句子和单词索引
            gr_sent_indices[sent_idx].append(w_idx.item())# 将每个句子的分隔符索引保存到对应的位置
         # 生成句子级注意力掩码  
        cur_sentence_indexes_masked = torch.BoolTensor(contextual_sentences_ids['input_ids'].shape).fill_(False)
        for i in range(contextual_sentences_ids['input_ids'].shape[0]):
            if raw_sentences_flatten[i] =='<pad_sentence>':
                cur_sentence_indexes_masked[i][gr_sent_indices[i][0]] = True
                continue
            for j in range(contextual_sentences_ids['input_ids'].shape[1]):
                if  gr_sent_indices[i][0] <= j <= gr_sent_indices[i][1]:
                    cur_sentence_indexes_masked[i][j] = True

        return (contextual_sentences_ids, torch.LongTensor(labels), padding_utterance_masked, intra_speaker_masekd_all, cur_sentence_indexes_masked, raw_sentences) 

#核心改进​​：继承BatchPreprocessor，在父类基础上扩展情感分析功能，通过 emotion_labels 支持预定义的情感类别
#Core improvement: Inherits from BatchPreprocessor and extends it with emotion analysis functionality, enabling support for predefined emotion categories through emotion_labels.
class BatchPreprocessorLLM(BatchPreprocessor):#继承自 BatchPreprocessor，为大语言模型进行数据预处理
    def __init__(self, tokenizer, dataset_name=None, window_ct=0, emotion_labels=[]) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")#把特殊 token </s> 转换为其对应的 ID，并存为 self.separate_token_id
        self.dataset_name = dataset_name
        self.window_ct = window_ct
        self.emotion_labels = emotion_labels#保存情绪标签
        self.printted = False

    #和父类无区别
    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v['s_id'] = k
                new_data_list.append(v)
            return new_data_list
        elif isinstance(raw_data, list):
            return raw_data

    #和父类无区别
    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            # iemocap: label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
            speaker = {
                "Ses01": {"F": "Mary", "M": "James"},
                "Ses02": {"F": "Patricia", "M": "John"},
                "Ses03": {"F": "Jennifer", "M": "Robert"},
                "Ses04": {"F": "Linda", "M": "Michael"},
                "Ses05": {"F": "Elizabeth", "M": "William"},
            }
            s_id_first_part = s_id[:5]
            return speaker[s_id_first_part][gender].upper()
        elif data_name in ['meld', "emorynlp"]:
            # emorynlp: label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
            # meld: label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
            gender_idx = gender.index(1)
            return f"SPEAKER_{gender_idx}"
        elif data_name == 'dailydialog':
            # dailydialog:  {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,  'anger': 4, 'fear': 5, 'disgust':6}
            return f"SPEAKER_{gender}"

    def sentence_mixed_by_surrounding(self, sentences, around_window, s_id, genders, data_name):
        new_conversations = []
        align_sents = []    #存放每个“当前句子”带有说话人标注的版本
        for i, cur_sent in enumerate(sentences):
            tmp_s = ""
            for j in range(max(0, i-around_window), min(len(sentences), i+around_window+1)):
                u_j = f"{self.get_speaker_name(s_id, genders[j], data_name=data_name)}: {sentences[j]}"
                if i == j:
                    align_sents.append(u_j)
                tmp_s += f"\n{u_j}"
            new_conversations.append(tmp_s)
        #new_conversations: 所有拼接后的上下文句子串（用于输入 LLM 等）
        #align_sents: 所有的中心句子（用于对齐、标注或显示等）
        return new_conversations, align_sents

    def __call__(self, batch):
        raw_sentences = []
        raw_sentences_flatten = []
        labels = []
        speaker_info = []
        listener_info = []

        # masked tensor
        lengths = [len(sample['sentences']) for sample in batch]#每个样本的句子数量（即每轮对话的长度）。
        max_len_conversation = max(lengths)
        #创建一个布尔 mask，用于标记哪些位置是 padding：  维度为 [batch_size, max_len_conversation]
        padding_utterance_masked = torch.BoolTensor(
            [[False]*l_i + [True]*(max_len_conversation - l_i) for l_i in lengths])

        # collect all sentences
        # - intra speaker
        #创建一个形状为 [batch_size, max_len, max_len] 的布尔 tensor
        flatten_data = []
        intra_speaker_masekd_all = torch.BoolTensor(
            len(batch), max_len_conversation, max_len_conversation)
        for i, sample in enumerate(batch):
            #new_conversations：每个句子的上下文拼接版本
            #align_sents：每个中心句子本身（带 speaker）
            new_conversations, align_sents = self.sentence_mixed_by_surrounding(sample['sentences'],
                                                                                around_window=self.window_ct,
                                                                                s_id=sample['s_id'],
                                                                                genders=sample['genders'],
                                                                                data_name=self.dataset_name)
            #few-shot prompt 模板，用于提示 LLM 应该完成什么任务
            #给了三条示例：
            #给定对话和目标句子，输出该句子的情绪类别。
            #会作为上下文拼接到后面每条数据的 prompt 前。
            few_shot_example = """\n=======
Context: Given predefined emotional label set [happy, sad, neutral, angry, excited, frustrated], and bellow conversation: 
"
PATRICIA: You know, it's lovely here, the air is sweet.
PATRICIA: No, not sorry.  But, um. But I'm not gonna stay.
JOHN: The trouble is, I planned on sort of sneaking up on you on a period of a week or so.  But they take it for granted that we're all set.
PATRICIA: I knew they would, your mother anyway.
PATRICIA: Well, from her point of view, why else would I come?
PATRICIA: I guess this is why I came.
JOHN: I'm embarrassing you and I didn't want to tell it to you here.  I wanted some place we'd never been before.  A place where we'd be brand new to each other.
PATRICIA: Well, you started to write me
JOHN: You felt something that far back?
PATRICIA: Every day since.
JOHN: Ann, why didn't you let me know?
JOHN: Let's drive someplace.  I want to be alone with you.
JOHN: No.  Nothing like that.
"

Question: What is the emotion of the speaker at the utterance "PATRICIA: Well, from her point of view, why else would I come?"?
Answer: neutral

Question: What is the emotion of the speaker at the utterance "PATRICIA: I guess this is why I came."?
Answer: happy

Question: What is the emotion of the speaker at the utterance "JOHN: I'm embarrassing you and I didn't want to tell it to you here.  I wanted some place we'd never been before.  A place where we'd be brand new to each other."?
Answer: excited
"""
            #遍历样本中每个句子及其上下文。
            #conv: 当前句子的上下文拼接
            #utterance: 当前被标注的目标句子（带 speaker，如 "A: xxx"）
            #i_u当前这个句子在对话中的索引位置（第几句话）。
            for i_u, (conv, utterance) in enumerate(zip(new_conversations, align_sents)):
                prompt_extract_context_vect = few_shot_example + \
                    f"\n=======\nContext: Given predefined emotional label set [{', '.join(self.emotion_labels)}], and bellow conversation:\n\"{conv}\n\"\n\nQuestion: What is the emotion of the speaker at the utterance \"{utterance}\"?\nAnswer:"
                #打印 prompt（只打印一次，用于调试）
                if not self.printted:
                    print(prompt_extract_context_vect)
                    self.printted = True
                #把 prompt 文本转化为 token IDs
                #inputs 是一个字典，主要包含
                #"input_ids"：token 编号序列，形如 [[101, 3245, ...]]
                #"attention_mask"：注意力 mask，1 表示有效位置，0 表示 padding
                inputs = self.tokenizer(
                    prompt_extract_context_vect, return_tensors="pt")
                input_ids = inputs["input_ids"]
                #构建样本字典并保存到列表中
                flatten_data.append({
                    "s_id": sample['s_id'],
                    "u_idx": i_u,
                    "prompt_content": prompt_extract_context_vect,
                    "input_ids": input_ids,
                }
                )

        return flatten_data

#继承自 BatchPreprocessor，核心功能是 ​​为对话中的每个说话者生成当前的隐性情绪，用于后续大语言模型（LLM）分析
class BatchPreprocessorLLMSpeakerImplicitEmotion(BatchPreprocessor):
    def __init__(self, tokenizer, dataset_name=None, window_ct=2, emotion_labels=[]) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name = dataset_name
        self.window_ct = window_ct
        self.emotion_labels = emotion_labels

    @staticmethod
    def load_raw_data(path_data):
        raw_data = json.load(open(path_data))
        if isinstance(raw_data, dict):
            new_data_list = []
            for k, v in raw_data.items():
                v['s_id'] = k
                new_data_list.append(v)
            return new_data_list
        elif isinstance(raw_data, list):
            return raw_data

    @staticmethod
    def get_speaker_name(s_id, gender, data_name):
        if data_name == "iemocap":
            # iemocap: label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
            speaker = {
                "Ses01": {"F": "Mary", "M": "James"},
                "Ses02": {"F": "Patricia", "M": "John"},
                "Ses03": {"F": "Jennifer", "M": "Robert"},
                "Ses04": {"F": "Linda", "M": "Michael"},
                "Ses05": {"F": "Elizabeth", "M": "William"},
            }
            s_id_first_part = s_id[:5]
            return speaker[s_id_first_part][gender].upper()
        elif data_name in ['meld', "emorynlp"]:
            # emorynlp: label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
            # meld: label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
            gender_idx = gender.index(1)
            return f"SPEAKER_{gender_idx}"
        elif data_name == 'dailydialog':
            # dailydialog:  {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,  'anger': 4, 'fear': 5, 'disgust':6}
            return f"SPEAKER_{gender}"
    def preprocess(self, all_conversations):

        new_data = {}
        gr_by_len = {}#这是一个字典，用于根据 token 的长度将数据分组。分组后，能够批量解码以提高效率。
        for i, sample in enumerate(all_conversations):
        #遍历 all_conversations 中的每个对话样本 sample，并通过 enumerate 获得索引 i 和对应的对话内容。

            all_utterances = []#用来存储每个发言的完整文本（包括说话人的名字和发言内容）
            all_speaker_names = []#用来存储每个发言对应的说话人名称
            #遍历每个对话中的每个句子，i_u个数，u发言的句子
            for i_u, u in enumerate(sample['sentences']):
                if self.dataset_name == 'meld':
                    speaker_name = sample['speakers'][i_u]
                else:
                    speaker_name = self.get_speaker_name(sample['s_id'], sample['genders'][i_u], self.dataset_name)
                u_full_name = f'{speaker_name}: {u}'
                all_utterances.append(u_full_name)
                all_speaker_names.append(speaker_name)

            #将 all_utterances 列表中的所有发言通过换行符拼接在一起，形成完整的对话文本。这里是一个对话的所有发言句子
            full_conversation = "\n".join(all_utterances)   
            #用来存储每个说话人生成的 token ID（转换为模型输入的数字表示）。
            prompts_speaker_description_word_ids = {}
            #存储输入的提示语，用于生成描述。
            prompting_input = {}

            #遍历每个对话中的说话人，使用set去重
            for speaker_name in set(all_speaker_names):
                #要求模型根据给定的对话文本，描述该说话人的特征。prompting 格式化字符串中包括了完整对话以及对说话人描述的提示。

                prompting = ( 
                             "Given the conversation between speakers: \n"
                             f"{full_conversation}" 
                             "\n### Task ###\n"
                             # 对话背景和说话人描述V3，100词    V4，250词
                             #f"\nBased on the full context, briefly describe the overall background of the conversation (e.g., location/setting, cause of occurrence), and provide a concise profile of the speaker {speaker_name} (e.g., personality traits, possible occupations). \n"
                             # V5说话人性格，250词
                             # v6说话人性格，100词
                             f"In overall of above conversation, what do you think about the characteristics speaker {speaker_name}?\n"
                             "### Required JSON Format ###\n"
                             '''{\n"Response": "<description>"\n}\n''' 
                              "### Critical Rules ###\n"
                                "1. Output **ONLY** the JSON object\n"
                                "2. No explanations/thinking processes\n"
                                "3. Keep response to **around 100 words** - focus on quality over exact word count\n"
                             "Example Valid Response:\n"
                             '''{\n"Response": "Linda exhibits a pragmatic, unenthusiastic demeanor, contrasting with Michael's romanticized excitement. ..."\n}\n'''
                            )
                #通过 self.tokenizer 将生成的提示语 prompting 转换为模型的输入 ID
                prompts_speaker_description_word_ids[speaker_name] = self.tokenizer(
                    prompting, return_tensors="pt")["input_ids"]
                #将原始的提示语存储
                prompting_input[speaker_name] = prompting

                # group by len for batch decode by llm
                #使用 prompts_speaker_description_word_ids[speaker_name].shape[-1] 获取 token 序列的长度，并根据长度将数据分组。
                #这样做的目的是为了后续批量解码时能够根据长度进行分批处理，提高效率。
                if prompts_speaker_description_word_ids[speaker_name].shape[-1] not in gr_by_len:
                    gr_by_len[prompts_speaker_description_word_ids[speaker_name].shape[-1]] = []
                gr_by_len[prompts_speaker_description_word_ids[speaker_name].shape[-1]].append({
                    'w_ids': prompts_speaker_description_word_ids[speaker_name],
                    'conv_id': sample['s_id'],
                    'type_data': sample['type_data'],
                    "prompting_input": prompting,
                    'speaker_name': speaker_name,
                    'all_speaker_names': all_speaker_names
                })
        #这是一个字典，按照 token 长度分组的样本。每个长度对应的值是一个列表，包含了每个分组的详细信息。
        return gr_by_len



raw_data = []
for type_data in ['valid', 'test', 'train']:
#for type_data in ['valid']:
    #生成数据集名称模板
    data_name_pattern = f'{dataset_name}.{type_data}'
    #生成处理后数据的存储路径
    path_processed_data = f'{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split("/")[-1]}.json'
    #调用自定义类的静态方法加载原始JSON数据
    org_raw_data = BatchPreprocessorLLMSpeakerImplicitEmotion.load_raw_data(
        f"{data_folder}/{data_name_pattern}.json")
    #检查已处理文件是否存在，若存在则加载
    if os.path.exists(path_processed_data):
        #读取已经处理好的数据
        processed_data = json.load(open(path_processed_data, 'rt'))
        #打印信息，说明该类型（如 train）的数据中，多少条样本已经被处理过
        print(
            f'- sucessful processed {len(processed_data)}/{len(org_raw_data)} conversations in data-type ={type_data}')
        #将已处理的数据备份一份（防止出错后原文件丢失），文件名后缀 _backup.json
        json.dump(processed_data, open(
            path_processed_data+"_backup.json", 'wt'), indent=2)
        #将未被处理过的样本从原始数据中过滤出来
        org_raw_data = [e for e in org_raw_data if e['s_id']
                        not in processed_data]
    #打印还有多少个样本需要继续处理
    print(
        f'- Continue process {len(org_raw_data)} conversations in data-type ={type_data}')
    for e in org_raw_data:#为未处理的样本增加字段 type_data，记录它是 train/valid/test 的哪一种
        e['type_data'] = type_data
    #将这些未处理样本加入到总的 raw_data 列表中
    raw_data = raw_data + org_raw_data

# Initialize a BatchPreprocessorLLMSpeakerDescription instance
# Purpose: to construct the input prompt (dialogue + speaker info),
# specify the tokenizer for text processing,
# and configure parameters such as emotion labels and context window size.
data_preprocessor = BatchPreprocessorLLMSpeakerImplicitEmotion(
    tokenizer,
    dataset_name=dataset_name,
    window_ct=5,
    emotion_labels=['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']
)

# Preprocess the raw data by calling the preprocess() method defined above.
'''
gr_by_len = {
    token_len_1: [dict1, dict2, ...],
    token_len_2: [dict3, dict4, ...],
    ...
}
'''
gr_by_len = data_preprocessor.preprocess(raw_data)
all_data = {}
print_one_time = True  # control variable, print inference output only once

# Iterate over each group of speakers grouped by different prompt token lengths (to speed up batch processing)
# len_promting: the token sequence length of the prompt (integer)
# speaker_promts: list of data samples whose prompt token length equals len_promting
for len_promting, speaker_promts in tqdm(gr_by_len.items(), desc="Prompt Length Groups"):
    for batch_size in [32, 16, 8, 4, 2, 1]:  # try different batch sizes (from large to small)
        # for batch_size in [4, 2, 1]:
        try:
            # Extract all text prompts in the current group
            all_promtings_texts = [e['prompting_input'] for e in speaker_promts]

            # Wrap these texts with PyTorch DataLoader for batch processing
            data_loader = DataLoader(all_promtings_texts,
                                     batch_size=batch_size,
                                     shuffle=False)

            # Initialize output list to collect personality descriptions for each speaker
            output_sp_desc = []
            with torch.no_grad():  # run without gradient computation
                # Iterate through each batch (e.g., 8 prompts) to send into the LLM
                for i, speaker_promts_in_batch in enumerate(data_loader):
                    # Tokenize the batch texts into tensor format for model input
                    inputs = tokenizer(
                        speaker_promts_in_batch,
                        return_tensors="pt",
                        padding=True,   # enable padding for batch processing
                        truncation=True,
                        max_length=2048  # Qwen3 supports longer sequences
                    )
                    input_ids = inputs["input_ids"].to("cuda")
                    attention_mask = inputs["attention_mask"].to("cuda")

                    '''
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=600
                    )
                    '''
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=800,
                        do_sample=True,
                        temperature=0.3,         # low randomness for stable generation
                        top_p=0.85,              # slightly increased for diversity
                        repetition_penalty=1.2,  # important! prevents repetition
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    # Decode model outputs (token sequences) back to readable text
                    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    # Key modification: use the new function to remove Qwen3’s reasoning process
                    for j, e in enumerate(output_text):
                        cleaned_output = extract_qwen3_output(e, all_promtings_texts[j])

                        # New: parse JSON string outputs into Python dictionaries
                        try:
                            # If cleaned_output is a JSON string, parse it into a dictionary
                            if isinstance(cleaned_output, str) and cleaned_output.strip().startswith('{'):
                                parsed_output = json.loads(cleaned_output)
                                output_sp_desc.append(parsed_output)
                            else:
                                # If not JSON format, keep the raw string
                                output_sp_desc.append(cleaned_output)
                        except json.JSONDecodeError as e:
                            print(f"JSON parsing error (index {j}): {e}")
                            print(f"Raw string: {cleaned_output}")
                            # Keep the original string if parsing fails
                            output_sp_desc.append(cleaned_output)

                    if print_one_time:
                        print("Raw generated output_text:")
                        print(output_text[0] if output_text else "No output")
                        print("Processed output_sp_desc:")
                        print(output_sp_desc[0] if output_sp_desc else "No output")
                        print_one_time = False

                # Assign the processed speaker description back to the corresponding data samples
                for i, out in enumerate(output_sp_desc):
                    # If the output is a dictionary containing the key 'Response', use its value
                    if isinstance(out, dict) and 'Response' in out:
                        speaker_promts[i]['sp_desc'] = out['Response']
                    else:
                        # Otherwise, keep the raw output
                        speaker_promts[i]['sp_desc'] = out
            
            break  # Stop trying smaller batch sizes once successful

        except Exception as e:
            traceback.print_exc()
            print(e)
            if batch_size == 1:
                print(["Errr "] * 10)
                print("CUDA out of Memory (batch = 1)!!!!")


# save results
for type_data in ['valid', 'test', 'train']:
    data_name_pattern = f'{dataset_name}.{type_data}'
    path_processed_data = f'{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split("/")[-1]}.json'
    processed_data = {}
    
    if os.path.exists(path_processed_data):
        processed_data = json.load(open(path_processed_data, 'rt'))
        print(
            f'- load processed [old] {len(processed_data)} conversations in data-type ={type_data}')
    
    all_data = {}
    for len_promting, speaker_promts in gr_by_len.items():
        for description in speaker_promts:
            if type_data != description['type_data']:
                continue

            if description['conv_id'] not in all_data:
                all_data[description['conv_id']] = {
                    'all_speaker_names': description['all_speaker_names'],
                    'vocab_sp2desc':  {}
                }
            all_data[description['conv_id']
                     ]['vocab_sp2desc'][description['speaker_name']] = description['sp_desc']
    
    print(
        f'- sucessful processed [new] {len(all_data)} conversations in data-type ={type_data}')
    
    all_data_new = {}
    for k, v in all_data.items():
        all_data_new[k] = []
        for sp_name in v['all_speaker_names']:
            all_data_new[k].append(v['vocab_sp2desc'][sp_name])
    
    print(
        f'- update processed [new] {len(all_data_new)} + [old] {len(processed_data)} conversations in data-type ={type_data}')
    
    all_data_new.update(processed_data)
    json.dump(all_data_new, open(f'{path_processed_data}', 'wt'), indent=2)