
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
prompt_type = 'ImplicitEmotion_V3'



print("Loading model ...")
# trained with chat and instruction

model_name = '/home/pubNAS3/xinran/llm/Qwen3-14B'  # standard model, please switch to your local model before running.
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
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded successfully from local path.")

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
class BatchPreprocessor(object): 
    def __init__(self, tokenizer, dataset_name=None, window_ct=2) -> None:
        self.tokenizer = tokenizer
        self.separate_token_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.dataset_name  = dataset_name
        self.window_ct = window_ct

    #从原始的 JSON 文件中读取原始数据并进行统一格式处理
    '''
    将字典
    {
    "id1": {"text": "example1"},
    "id2": {"text": "example2"}
    }
    转化为列表
    [
    {"s_id": "id1", "text": "example1"},
    {"s_id": "id2", "text": "example2"}
    ]
    '''
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

    #   将每个句子与其前后 around_window 个句子拼接起来，构成带上下文的信息，中心句用 </s> 标记。每句都加上对应说话人前缀。
    def sentence_mixed_by_surrounding(self, sentences, around_window, s_id, genders, data_name):
        new_sentences = []
        for i, cur_sent in enumerate(sentences): #当前句子的索引 i 和内容 cur_sent
            tmp_s = ""
            #当前句子前后 around_window 个句子的范围内滑动窗口
            for j in range(max(0, i-around_window), min(len(sentences), i+around_window+1)):
                #当前窗口中的句子 j 恰好是中心句（当前处理的句子），在它前面加一个特殊标记 </s> 作为句子边界
                if i == j:
                    tmp_s += " </s>"
                #添加格式化的句子信息，包括说话人姓名（通过 get_speaker_name 获得）和句子内容。
                #" MICHAEL: I am fine." 或 " SPEAKER_1: Hello!"
                tmp_s +=  f" {self.get_speaker_name(s_id, genders[j], data_name=data_name)}: {sentences[j]}"
                if i == j:
                    #如果是中心句，再在后面也加一个 </s>，将其完全用 </s> 包裹起来，便于模型知道哪一句是“当前”句子。
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

# Inherits from BatchPreprocessor.
# Core functionality: generates implicit emotions for each speaker in a dialogue,
# which are later used for large language model (LLM) analysis.
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
            # IEMOCAP: label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
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
            # EMORYNLP: {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
            # MELD: {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
            gender_idx = gender.index(1)
            return f"SPEAKER_{gender_idx}"
        elif data_name == 'dailydialog':
            # DailyDialog: {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3, 'anger': 4, 'fear': 5, 'disgust':6}
            return f"SPEAKER_{gender}"

    # Process all dialogue data and convert them into a format suitable for model input.
    def preprocess(self, all_conversations):

        gr_by_len = {}  # A dictionary for grouping data by token length.
                        # Grouping allows batch decoding for improved efficiency.
        for _, sample in enumerate(all_conversations):
            # Iterate through each dialogue sample in all_conversations using enumerate
            # to get both the index and its content.

            # Build a list of utterances with speaker tags, e.g., “SPEAKER: utterance”
            tagged = []
            speaker_sequence = []
            for idx, utt in enumerate(sample['sentences']):
                if self.dataset_name == 'meld':
                    name = sample['speakers'][idx]
                else:
                    name = self.get_speaker_name(sample['s_id'], sample['genders'][idx], self.dataset_name)
                tagged.append(f'{name}: "{utt}"')
                speaker_sequence.append(name)

            # For each utterance in the dialogue, create a prompt that includes
            # the current sentence and a window of previous sentences.
            for idx, utt in enumerate(tagged):
                window_size = self.window_ct       # context window size
                window_start = max(0, idx - window_size + 1)
                # window_start = 0
                context = tagged[window_start: idx + 1]
                # Format the context as a continuous string block
                convo_str = ' '.join(context)
                # print(convo_str)
                current_speaker = tagged[idx].split(':')[0]
                emotion_list = get_emotion_labels(self.dataset_name)
                # emotion_options = '/'.join(emotion_list)

                '''
                prompt = (
                    "You are an expert in understanding subtle human emotions.\n"
                    "Your job is to analyze both the **surface emotion** (what the person seems to express outwardly) and the **implicit emotion** (what the person might actually feel inside), based on both the conversation and common sense reasoning.\n "
                    "Do NOT use emotion labels like 'sadness' or 'anger'. Instead, describe the emotional expression and internal feeling using **a short, natural-language sentence** (no less than 30 words each).\n"
                    "The following conversation between speakers (within ### marks) involves multiple turns.\n"
                    f"### {convo_str} ###\n"
                    f"Focus on the current speaker in the last turn: {utt}\n"
                    f"Please use your common sense to infer the *surface emotion* and *implicit emotion* elicited in {current_speaker} by their utterance, "
                    "\n\n### Critical Format Rules ###\n"
                    "1. Output **ONLY** the two-line response in the specified format\n"
                    "2. Do **NOT** include any thinking process\n"
                    "3. Do **NOT** use markdown or code blocks\n"
                    "4. If unsure, make a reasonable guess but still follow the format\n"
                    "Respond exactly in the format below (without additional text):\n"
                    f"SurfaceEmotion: <how the speaker appears to feel, based on their words and tone.>\n"
                    "ImplicitEmotion: <what the speaker might truly feel inside, even if not directly expressed.>\n"
                )
                '''
                prompt = (
                    "You are an expert in analyzing human surface and implicit emotions through conversation context.\n"
                    "### Task ###\n"
                    "1. Analyze **explicit emotion** (outward expression based on their words and tone.) and **implicit emotion** (true inner feeling, even if not directly expressed.)\n"
                    "2. Use **natural language descriptions** (no emotion labels like 'sappiness')\n"
                    "3. Use **at least 20 words**, but no more than 50 words each\n"
                    "4. You MUST take into account the entire past conversation context, including what the current speaker and others have said earlier."
                    "### Conversation Context ###\n"
                    f"{convo_str}\n\n"
                    "### Focus Utterance ###\n"
                    f"{current_speaker}: {utt}\n\n"
                    "### Required JSON Format ###\n"
                    '''{\n"ExplicitEmotion": "<description>",\n"ImplicitEmotion": "<description>"\n}''' + "\n"
                    "### Critical Rules ###\n"
                    "1. Output **ONLY** the JSON object\n"
                    "2. No explanations/thinking processes\n"
                    "3. No markdown/code formatting\n"
                    "4. Keys must be exactly as shown\n\n"
                    "Example Valid Response:\n"
                    '''{\n"ExplicitEmotion": "The speaker's cheerful tone and frequent use of positive adjectives suggest outwardly optimistic engagement...",\n"ImplicitEmotion": "Underneath the enthusiastic delivery, a slight hesitation in phrasing hints at unspoken reservations..."\n}'''
                )

                inputs = self.tokenizer(prompt, return_tensors='pt')
                length = inputs['input_ids'].shape[-1]
                entry = {
                    'input_ids': inputs['input_ids'],
                    'conv_id': sample['s_id'],
                    'utter_idx': idx,
                    'prompting_input': prompt,
                    'type_data': sample['type_data'],
                    'speaker_name': current_speaker,
                    'all_speakers': speaker_sequence
                }
                gr_by_len.setdefault(length, []).append(entry)
        return gr_by_len


raw_data = []
for type_data in ['valid', 'test', 'train']:
    # for type_data in ['valid']:
    # Generate the dataset name pattern
    data_name_pattern = f'{dataset_name}.{type_data}'
    # Generate the save path for processed data
    path_processed_data = f'{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split("/")[-1]}.json'
    # Load the original JSON data using the static method from the custom class
    org_raw_data = BatchPreprocessorLLMSpeakerImplicitEmotion.load_raw_data(
        f"{data_folder}/{data_name_pattern}.json"
    )
    # Check if the processed file already exists; if it does, load it
    if os.path.exists(path_processed_data):
        # Read the already processed data
        processed_data = json.load(open(path_processed_data, 'rt'))
        # Print information about how many samples in this data type (e.g., train)
        # have already been successfully processed
        print(
            f'- Successfully processed {len(processed_data)}/{len(org_raw_data)} conversations in data-type = {type_data}'
        )
        # Backup the processed file (to prevent data loss in case of errors)
        # The backup file has the suffix "_backup.json"
        json.dump(
            processed_data,
            open(path_processed_data + "_backup.json", 'wt'),
            indent=2
        )
        # Filter out already processed samples from the original data
        org_raw_data = [
            e for e in org_raw_data if e['s_id'] not in processed_data
        ]
    # Print how many remaining samples still need to be processed
    print(
        f'- Continue processing {len(org_raw_data)} conversations in data-type = {type_data}'
    )
    # Add a new field "type_data" to each unprocessed sample,
    # indicating whether it belongs to train / valid / test
    for e in org_raw_data:
        e['type_data'] = type_data
    # Add these unprocessed samples to the total raw_data list
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
print_one_time = True  # Control variable — only print inference output once

# Iterate over each group of samples with different prompt token lengths (for faster batch processing)
# len_promting: integer representing the token sequence length of the prompt
# speaker_promts: list of samples whose prompt token length equals len_promting
for len_promting, speaker_promts in tqdm(gr_by_len.items(), desc="Prompt Length Groups"):
    for batch_size in [32, 16, 8, 4, 2, 1]:  # Try different batch sizes (from large to small)
        # for batch_size in [4, 2, 1]:
        try:
            # Extract all prompt texts in the current group
            all_promtings_texts = [e['prompting_input'] for e in speaker_promts]

            # Wrap these texts using PyTorch DataLoader for efficient batch processing
            data_loader = DataLoader(all_promtings_texts,
                                     batch_size=batch_size,
                                     shuffle=False)

            # Initialize an output list to store personality/emotion descriptions for each speaker
            output_sp_desc = []
            with torch.no_grad():  # Run in inference mode (no gradient computation)
                # Iterate through each batch (e.g., 8 prompts) to feed into the LLM
                for i, speaker_promts_in_batch in enumerate(data_loader):
                    # Tokenize each batch of text prompts into tensors for model input
                    inputs = tokenizer(
                        speaker_promts_in_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024
                    ).to(device)

                    '''
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=600
                    )
                    '''
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=800,          # Limit the output length to a reasonable range
                        temperature=0.3,             # Lower randomness (0 = deterministic, 1 = highly random)
                        top_p=0.9,                   # Nucleus sampling to control diversity
                        do_sample=True,              # Enable sampling but keep it controlled
                        repetition_penalty=1.2,      # Penalize repetition to avoid redundancy
                        eos_token_id=tokenizer.eos_token_id,  # Explicit end-of-sequence token
                        pad_token_id=tokenizer.eos_token_id,  # Padding alignment handling
                        num_return_sequences=1,      # Ensure a single output per sample
                    )

                    # Decode model outputs (token sequences) into human-readable text
                    generated_texts = tokenizer.batch_decode(
                        outputs, skip_special_tokens=True)

                    # Remove the original prompt from generated text to keep only the model’s response.
                    # (Some models reproduce the prompt before generating answers, so we strip it out.)
                    batch_outputs = []
                    for j, gen_text in enumerate(generated_texts):
                        prompt_text = speaker_promts_in_batch[j]
                        generated_part = gen_text.replace(prompt_text, "", 1).strip()

                        # Try parsing as JSON format
                        SurfaceEmotion, ImplicitEmotion = "", ""
                        try:
                            # Extract a JSON object from the text using regex
                            json_match = re.search(r"\{\s*\"ExplicitEmotion\".*?\}", generated_part, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                emotion_data = json.loads(json_str)
                                SurfaceEmotion = emotion_data.get("ExplicitEmotion", "")
                                ImplicitEmotion = emotion_data.get("ImplicitEmotion", "")
                            else:
                                raise ValueError("No JSON found")

                        except Exception as e:
                            # ------------------------------------
                            # Fallback: search for "ImplicitEmotion" from the end of the text
                            # ------------------------------------
                            implicit_pos = generated_part.rfind("ImplicitEmotion:")
                            if implicit_pos != -1:
                                # Extract content following "ImplicitEmotion:"
                                ImplicitEmotion = generated_part[implicit_pos + len("ImplicitEmotion:"):].strip()
                                # Remove any trailing "ExplicitEmotion" content if present
                                next_surface_pos = ImplicitEmotion.find("ExplicitEmotion:")
                                if next_surface_pos != -1:
                                    ImplicitEmotion = ImplicitEmotion[:next_surface_pos].strip()
                            else:
                                ImplicitEmotion = generated_part  # Fallback: keep the full text

                            # ------------------------------------
                            # Fallback: search for "ExplicitEmotion" from the end of the text
                            # ------------------------------------
                            surface_pos = generated_part.rfind("ExplicitEmotion:")
                            if surface_pos != -1:
                                SurfaceEmotion = generated_part[surface_pos + len("ExplicitEmotion:"):].strip()
                                # Remove any trailing "ImplicitEmotion" content if present
                                next_implicit_pos = SurfaceEmotion.find("ImplicitEmotion:")
                                if next_implicit_pos != -1:
                                    SurfaceEmotion = SurfaceEmotion[:next_implicit_pos].strip()

                        # Store structured results
                        batch_outputs.append({
                            "surface_emotion": SurfaceEmotion,
                            "implicit_emotion": ImplicitEmotion
                        })
                        output_sp_desc.append({
                            "surface_emotion": SurfaceEmotion,
                            "implicit_emotion": ImplicitEmotion
                        })

                    # Print one sample of inference output for debugging or verification
                    if print_one_time:
                        print("▶ Example Prompt:", speaker_promts_in_batch[3])
                        print("▶ Generated Text:", generated_texts[3])
                        print("▶ Structured Output:", batch_outputs)
                        for i in range(min(5, len(output_sp_desc))):
                            print(f"{i+1}.", output_sp_desc[-len(output_sp_desc) + i])
                        print_one_time = False
                
                # Write the generated results back into the original data structure
                # Add 'surface_emotion' and 'implicit_emotion' fields to each entry
                for i, out in enumerate(output_sp_desc):
                    speaker_promts[i]['surface_emotion'] = out['surface_emotion']
                    speaker_promts[i]['implicit_emotion'] = out['implicit_emotion']
            
            break  # Stop trying smaller batch sizes once successful

        except Exception as e:
            traceback.print_exc()
            print(e)
            if batch_size == 1:
                print(["Errr "] * 10)
                print("CUDA out of Memory (batch = 1)!!!!")

# Iterate over the three dataset types: validation set, test set, and training set
for type_data in ['valid', 'test', 'train']:
    # for type_data in ['valid']:
    data_name_pattern = f'{dataset_name}.{type_data}'
    path_processed_data = f'{data_folder}/{data_name_pattern}_{prompt_type}_{model_name.split("/")[-1]}.json' 

    # Initialize a dictionary to store previously processed dialogue data
    processed_data = {}

    # If previously processed data for this dataset type exists, load it to avoid redundant processing
    if os.path.exists(path_processed_data):
        processed_data = json.load(open(path_processed_data, 'rt'))
        print(f'- Loaded processed [old] {len(processed_data)} conversations for data-type = {type_data}')

    # Initialize a new dictionary to store newly processed data for the current dataset type
    new_data = {}

    # Iterate over all prompt-length groups (these come from the earlier preprocess() results)
    for len_promting, speaker_promts in gr_by_len.items():
        for entry in speaker_promts:
            # Skip entries that do not belong to the current dataset type
            if type_data != entry['type_data']:
                continue  # Only process samples of the current type_data; skip others

            conv_id = entry['conv_id']
            utter_idx = entry['utter_idx']

            # If this conversation ID has not yet been added to new_data, initialize it
            if conv_id not in new_data:
                new_data[conv_id] = {
                    'utterances': [],   # Store all utterances in their original order
                    'predictions': {}   # Store model predictions by utterance index
                }

            # If utterances are not yet populated, retrieve the original conversation data
            if not new_data[conv_id]['utterances']:
                raw_conv = next((c for c in raw_data if c['s_id'] == conv_id), None)
                if raw_conv:
                    speaker_sequence = entry['all_speakers']
                    print(entry['all_speakers'])
                    new_data[conv_id]['utterances'] = [
                        f"{speaker}: {text}" 
                        for speaker, text in zip(speaker_sequence, raw_conv['sentences'])
                    ]

            # Add the model-generated description (speaker emotion prediction) for this utterance
            new_data[conv_id]['predictions'][str(utter_idx)] = {
                'surface_emotion': entry.get('surface_emotion', 'unknown'),
                'implicit_emotion': entry.get('implicit_emotion', ''),
                'prompt': entry['prompting_input']
            }

    # Print the number of newly processed conversations
    print(f'- Successfully processed [new] {len(all_data)} conversations for data-type = {type_data}')

    # Merge newly processed data with any existing processed data
    final_data = {}
    for conv_id in set(processed_data.keys()) | set(new_data.keys()):
        # Prioritize the newly processed data
        if conv_id in new_data:
            final_data[conv_id] = new_data[conv_id]
        else:
            final_data[conv_id] = processed_data[conv_id]

        # Ensure structural consistency of data fields
        final_data[conv_id].setdefault('utterances', [])
        final_data[conv_id].setdefault('predictions', {})

    # Convert the dictionary format {"A": descA, "B": descB} into an ordered list
    # to preserve the original speaker/utterance order.
    formatted_output = {}
    for conv_id, data in final_data.items():
        # Reconstruct results according to the original utterance order
        ordered_predictions = []
        for idx in range(len(data['utterances'])):
            pred = data['predictions'].get(str(idx), {
                'surface_emotion': 'No prediction',
                'implicit_emotion': 'No prediction'
            })
            ordered_predictions.append(pred)

        formatted_output[conv_id] = {
            'utterances': data['utterances'],
            'emotion_predictions': ordered_predictions
        }

    # Save the final processed output as JSON
    with open(path_processed_data, 'w') as f:
        json.dump(formatted_output, f, indent=2)

    print(f'- Saved {len(formatted_output)} conversations for {type_data}')
