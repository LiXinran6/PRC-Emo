#将原始对话数据转换为适合大语言模型（LLM）训练的提示工程格式
import json
import re
import faiss  # 添加FAISS向量检索库
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
# === 新增：加载情感检索库并构建FAISS索引 ===
def load_retrieval_library(retrieval_path):
    """加载情感检索库并构建FAISS索引"""
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    # 提取向量和元数据
    vectors = []
    metadata = []
    # ===== 新增：带进度条的遍历 =====
    print(f"加载检索库 ({len(retrieval_data)} 个样本):")
    for sample in tqdm(retrieval_data, desc="Processing", unit="sample"):
        vectors.append(sample['vector'])
        metadata.append({
            'text': sample['text'],
            'label': sample['label'],
            'dataset': sample['dataset'],
            'conversation_id': sample['conversation_id'],
            'utterance_id': sample['utterance_id']
        })
    
    # 转换为numpy数组
    vectors = np.array(vectors).astype('float32')
    #faiss.normalize_L2(vectors)#归一化向量
    # 创建FAISS索引
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension) #这里是L2距离
    #index = faiss.IndexFlatIP(dimension)  # 使用内积代替L2距离
    index.add(vectors)
    
    return index, metadata, vectors

# 初始化检索库（全局只加载一次）
RETRIEVAL_PATH = "./data/Emotion_Retrieval_Library.json"
print(f"使用的是：{RETRIEVAL_PATH}\n")
retrieval_index, retrieval_metadata, retrieval_vectors = load_retrieval_library(RETRIEVAL_PATH)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda')  # 与检索库相同的向量模型
# === 新增：检索相似样本函数 ===
def retrieve_similar_samples(query_text, current_id, d_type, k=3):
    """
    从检索库中查找相似样本
    :param query_text: 查询文本
    :param current_id: 当前样本ID (用于排除自身)
    :param d_type: 数据类型 (train/valid/test)
    :param k: 返回的相似样本数量
    """
    # 生成查询向量
    query_vec = sentence_model.encode([query_text], convert_to_tensor=False)
    query_vec = np.array(query_vec).astype('float32')
    #faiss.normalize_L2(query_vec)
    # FAISS检索
    distances, indices = retrieval_index.search(query_vec, k+5)  # 多取几个备选
    
    results = []
    # 修复：同时遍历距离和索引
    for dist, idx in zip(distances[0], indices[0]):
        # 跳过无效索引
        if idx < 0 or idx >= len(retrieval_metadata):
            continue
            
        sample = retrieval_metadata[idx]
        
        # 训练集排除自身样本
        if d_type == 'train' and sample['conversation_id'] == current_id[0] and sample['utterance_id'] == current_id[1]:
            continue
            
        results.append({
            'text': sample['text'],
            'label': sample['label'],
            'dataset': sample['dataset'],
            'distance': float(dist)
            #'similarity': float(dist)  # 直接使用距离值或相似度值
        })
        
        if len(results) >= k:
            break
    #results.sort(key=lambda x: x['similarity'], reverse=True) # 按相似度降序排序
    return results

#对话情感分析任务的提示工程生成器
data_name_pattern = 'train'
#根据数据集类型生成可读的说话者标识
def get_speaker_name(s_id, speaker_info, data_name):
    """
    参数重命名为 speaker_info：
    - IEMOCAP：gender 参数（原逻辑）
    - MELD：直接传入 speakers 列表中的姓名
    - DailyDialog：保留原 gender 编码逻辑
    """
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
        return speaker[s_id_first_part][speaker_info].upper()
    if data_name == "meld":
        return speaker_info 
    if data_name in ['meld', "emorynlp"]:
        # emorynlp: label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        # meld: label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        gender_idx = speaker_info.index(1) 
        return f"SPEAKER_{gender_idx}"
    elif data_name=='dailydialog':
        # dailydialog:  {'no_emotion': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3,  'anger': 4, 'fear': 5, 'disgust':6}
        return f"SPEAKER_{speaker_info}"

#对话上下文窗口生成器
def flatten_conversation_mixed_by_surrounding(conv, around_window, s_id, genders, data_name):
    new_data = []
    ## 生成包含前后对话的上下文窗口（窗口大小由around_window控制）
    for i, cur_sent in enumerate(conv):
        tmp_window = []
        for j in range(max(0, i-around_window), min(len(conv), i+around_window+1)):
            tmp_window.append(f" {get_speaker_name(s_id, genders[j], data_name=data_name)}: {conv[j]}")

        new_data.append(tmp_window)
    return new_data

#标签映射器
def get_label_map(data_name):
    all_data_label_map = {
        "iemocap":   ['happy','sad','neutral','angry','excited','frustrated'],
        "emorynlp":  ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'],
        "meld":  ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
        "dailydialog":  ['no_emotion', 'happiness', 'sadness', 'surprise',  'anger', 'fear', 'disgust']
    }
    return all_data_label_map[data_name]

#对描述性文本（可能包含特殊标记 <s> 和 </s>）进行预处理，去除不需要的部分和多余空格。
def preprocess_desc_speaker(str_in):
    str_in = str_in.split("</s>")[0].replace("<s>", "").replace("\n", " ")
    str_out = re.sub(r" {2,}", " ",  str_in)
    return str_out
#生成多轮对话标注任务输入
def gen_default_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data=None):
    new_conv = []   #用于保存“说话人名 + 语句”的格式化对话。
    samples = []    #最后返回的结果列表，每个元素是一个包含 prompt 对话的样本
    for i,sent in enumerate(conv['sentences']):
        #如果是 MELD 数据集，用 speakers[i]；否则用 genders[i]（如 IEMOCAP 用性别标识）。
        param = conv['speakers'][i] if data_name == "meld" else conv['genders'][i]
        #用 get_speaker_name 函数获取标准化的说话人名（如 Speaker A、Male Speaker 等）。
        sent_name = get_speaker_name(s_id, param, data_name)
        #构造格式为：SpeakerA: sentence
        new_sent =  f'{sent_name}: {sent}'
        #将格式化句子加入到 new_conv 列表中。
        new_conv.append(new_sent)
    #把所有格式化后的句子拼接成一个字符串，用换行符分隔（用于后续 context）.即，对话的所有句子
    conv_str = "\n".join(new_conv)
    
    #生成“局部对话上下文”的列表 flatten_conv
    #flatten_conv[i] 表示针对第 i 句话，取其前后 around_window 个句子构成的上下文（带说话人名）。
    #将每个发言，构造为包含上下文的发言形式
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['speakers'] if data_name == "meld" else conv['genders'], data_name)
    
    for i, sent in enumerate(new_conv):
        #定义一个 system prompt：设定模型的角色是情感识别专家
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.\n'
        #将第 i 个样本的局部上下文拼接成字符串。构造 context 提示：引导模型基于上下文来理解当前语句的语义。
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = f"\n### Given the following conversation as a context \n{conv_str}"
        speaker_name = get_speaker_name(s_id, conv['speakers'][i] if data_name == "meld" else conv['genders'][i], data_name)
        #构造用户问题提示，即让模型判断这个说话人当前这句话的情绪。
        q_msg =  f'Based on above conversation, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
        #获取当前语句第i句的具体真实情绪标签。   
        label_msg =  get_label_map(data_name)[conv['labels'][i]]

        #构造一组完整的对话消息，包括三部分：
        '''
        system：设定角色和上下文；
        user：提出问题（哪句情绪是什么）；
        assistant：真实标签作为答案。
        '''
        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
    return samples
     
def gen_spdescV2_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data):
    new_conv = []
    for i,sent in enumerate(conv['sentences']):
        param = conv['speakers'][i] if data_name == "meld" else conv['genders'][i]
        sent_name = get_speaker_name(s_id, param, data_name)
        new_sent =  f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['speakers'] if data_name == "meld" else conv['genders'], data_name)
    
    samples = []
    for i, sent in enumerate(new_conv):
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.'
        speaker_name = get_speaker_name(s_id, conv['speakers'][i] if data_name == "meld" else conv['genders'][i], data_name)
        
        #得到角色的特征描述
        desc_str = desc_speaker_data[s_id][i].replace("\n", " ")
        desc_msg = f'\n### Given the characteristic of this speaker, {speaker_name}: \n{desc_str}'
        
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = f"\n### Given the following conversation as a context \n{conv_str}"
        
        q_msg =  f'Based on above conversation and characteristic of the speakers, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
        label_msg = get_label_map(data_name)[conv['labels'][i]]
        
        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + desc_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
        
    return samples 

def gen_spdescV3_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data, retrieval_library, d_type):
    new_conv = []
    raw_utterances = conv['sentences']  # 新增：保存原始发言
    for i,sent in enumerate(conv['sentences']):
        param = conv['speakers'][i] if data_name == "meld" else conv['genders'][i]
        sent_name = get_speaker_name(s_id, param, data_name)
        new_sent =  f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['speakers'] if data_name == "meld" else conv['genders'], data_name)
    
    samples = []
    for i, sent in enumerate(new_conv):
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.\nYour goal is to infer the most accurate **emotion label** for a given utterance.\n'
        speaker_name = get_speaker_name(s_id, conv['speakers'][i] if data_name == "meld" else conv['genders'][i], data_name)
        raw_utterance = raw_utterances[i]  # 获取原始发言文本
        response = desc_speaker_data[s_id][i]["response"].replace("\n", " ")
        # 从检索库中查找相似情感样本（训练集排除自身）
        similar_samples = retrieve_similar_samples(
            query_text=raw_utterance,
            current_id=(s_id, i),
            d_type=d_type,
            k=3  # 返回3个最相似样本
        )

        # 构建示范文本
        demonstration_str = "### Reference Similar Emotional Expressions:\n"
        for idx, sample in enumerate(similar_samples, 1):
            demonstration_str += (
                f"{idx}. \"{sample['text']}\" → {sample['label']} "
                f"(from {sample['dataset']}, distance: {sample['distance']:.2f})\n"
            )
        demonstration_str += "\n"
        note = "Note: *Explicit Emotion* refers to the emotional state that a person outwardly expresses through their words, tone, facial expressions, or behavior. It is typically visible and can be directly perceived by others. For instance, someone loudly complaining might show a surface emotion of frustration or annoyance.\n*Implicit Emotion*, on the other hand, is the underlying emotional tendency that may not be directly expressed but can be inferred through context, common sense reasoning, or deeper understanding of the speaker’s situation. These emotions often reflect the speaker’s true feelings. \n"
        
        implicit_info_msg = (
            f"\n### Speaker: {speaker_name}\n"
            f"- {response}\n"
        )
        desc_msg = f'\n### Given the background of the conversation and the profile of this speaker: {implicit_info_msg}\n'
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = (f"\n### Given the following conversation as a context \n{conv_str}\n")
        
        emotion_labels = get_label_map(data_name)
        labels_msg = f"### Available emotion labels: {', '.join(emotion_labels)}\n\n"
        

        q_msg =  f'Based on above conversation, background of the conversation and the profile of the speaker, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
        label_msg = get_label_map(data_name)[conv['labels'][i]]
        
        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + desc_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
        
    return samples 

def gen_spdescV5_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data, retrieval_library, d_type):
    new_conv = []
    raw_utterances = conv['sentences']  # 新增：保存原始发言
    for i,sent in enumerate(conv['sentences']):
        param = conv['speakers'][i] if data_name == "meld" else conv['genders'][i]
        sent_name = get_speaker_name(s_id, param, data_name)
        new_sent =  f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['speakers'] if data_name == "meld" else conv['genders'], data_name)
    
    samples = []
    for i, sent in enumerate(new_conv):
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.\nYour goal is to infer the most accurate **emotion label** for a given utterance.\n'
        speaker_name = get_speaker_name(s_id, conv['speakers'][i] if data_name == "meld" else conv['genders'][i], data_name)
        raw_utterance = raw_utterances[i]  # 获取原始发言文本
        response = desc_speaker_data[s_id][i]["response"].replace("\n", " ")
        # 从检索库中查找相似情感样本（训练集排除自身）
        similar_samples = retrieve_similar_samples(
            query_text=raw_utterance,
            current_id=(s_id, i),
            d_type=d_type,
            k=3  # 返回3个最相似样本
        )

        # 构建示范文本
        demonstration_str = "### Reference Similar Emotional Expressions:\n"
        for idx, sample in enumerate(similar_samples, 1):
            demonstration_str += (
                f"{idx}. \"{sample['text']}\" → {sample['label']} "
                f"(from {sample['dataset']}, similarity: {sample['similarity']:.2f})\n"
            )
        demonstration_str += "\n"
        note = "Note: *Explicit Emotion* refers to the emotional state that a person outwardly expresses through their words, tone, facial expressions, or behavior. It is typically visible and can be directly perceived by others. For instance, someone loudly complaining might show a surface emotion of frustration or annoyance.\n*Implicit Emotion*, on the other hand, is the underlying emotional tendency that may not be directly expressed but can be inferred through context, common sense reasoning, or deeper understanding of the speaker’s situation. These emotions often reflect the speaker’s true feelings. \n"
        
        implicit_info_msg = (
            f"\n### Speaker: {speaker_name}\n"
            f"- {response}\n"
        )
        desc_msg = f'\n### Given the characteristic of this speaker: {implicit_info_msg}\n'
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = (f"\n### Given the following conversation as a context \n{conv_str}\n")
        
        emotion_labels = get_label_map(data_name)
        labels_msg = f"### Available emotion labels: {', '.join(emotion_labels)}\n\n"
        

        q_msg =  f'Based on above conversation and characteristic of the speakers, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
        label_msg = get_label_map(data_name)[conv['labels'][i]]
        
        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + desc_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
        
    return samples 

def gen_ImplicitEmotion_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data):
    new_conv = []
    for i,sent in enumerate(conv['sentences']):
        param = conv['speakers'][i] if data_name == "meld" else conv['genders'][i]
        sent_name = get_speaker_name(s_id, param, data_name)
        new_sent =  f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['speakers'] if data_name == "meld" else conv['genders'], data_name)
    
    samples = []
    for i, sent in enumerate(new_conv):
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.\n'
        speaker_name = get_speaker_name(s_id, conv['speakers'][i] if data_name == "meld" else conv['genders'][i], data_name)
        
        #得到隐性标签和情绪描述的特征描述
        emotion_label = desc_speaker_data[s_id][i]['emotion_label'].replace("\n", " ")
        emotion_desc = desc_speaker_data[s_id][i]['emotion_desc'].replace("\n", " ")
        note = "Note: **Implicit emotion** refers to the internal emotional tendency that is not directly expressed but can help in understanding the speaker's true feelings.\n"
        desc_msg = f'\n### Given the speaker’s implicit emotion label and its interpretation:\n'
        implicit_info_msg = (
            f"\n### Speaker: {speaker_name}\n"
            f"- Implicit Emotion Label: {emotion_label}\n"
            f"- Implicit Emotion Interpretation: {emotion_desc}\n"
        )
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = f"\n### Given the following conversation as a context \n{conv_str}"
        
        q_msg =  f'Based on above conversation and implicit emotion label and its interpretation of the speakers, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
        label_msg = get_label_map(data_name)[conv['labels'][i]]
        
        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + note + desc_msg + implicit_info_msg + local_context_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
        
    return samples 


def gen_ImplicitEmotion_V2_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data):
    new_conv = []
    for i,sent in enumerate(conv['sentences']):
        param = conv['speakers'][i] if data_name == "meld" else conv['genders'][i]
        sent_name = get_speaker_name(s_id, param, data_name)
        new_sent =  f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['speakers'] if data_name == "meld" else conv['genders'], data_name)
    
    samples = []
    for i, sent in enumerate(new_conv):
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.\nYour goal is to infer the most accurate **emotion label** for a given utterance.\n'
        speaker_name = get_speaker_name(s_id, conv['speakers'][i] if data_name == "meld" else conv['genders'][i], data_name)
        
        #得到隐性标签和情绪描述的特征描述
        emotion_surface = desc_speaker_data[s_id][i]['surface_emotion'].replace("\n", " ")
        emotion_implicit = desc_speaker_data[s_id][i]['implicit_emotion'].replace("\n", " ")
        note = "Note: *Surface emotion* refers to the emotional state that a person outwardly expresses through their words, tone, facial expressions, or behavior. It is typically visible and can be directly perceived by others. For instance, someone loudly complaining might show a surface emotion of frustration or annoyance.\n*Implicit emotion*, on the other hand, is the underlying emotional tendency that may not be directly expressed but can be inferred through context, common sense reasoning, or deeper understanding of the speaker’s situation. These emotions often reflect the speaker’s true feelings. \n"
        desc_msg = f'\n### Given the speaker’s Surface Emotion Interpretation and Implicit Emotion Interpretation in the utterance \"{conv["sentences"][i]}\":'
        implicit_info_msg = (
            f"\n### Speaker: {speaker_name}\n"
            f"- Surface Emotion Interpretation: {emotion_surface}\n"
            f"- Implicit Emotion Interpretation: {emotion_implicit}\n"
        )
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = f"\n### Given the following conversation as a context \n{conv_str}\n"
        
        q_msg =  f'Based on above conversation and Surface Emotion Interpretation and Implicit Emotion Interpretation, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
        label_msg = get_label_map(data_name)[conv['labels'][i]]
        
        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + note + local_context_msg + desc_msg + implicit_info_msg },
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
        
    return samples 

def gen_ImplicitEmotion_V3_prompting_messages(data_name, conv, around_window, s_id, desc_speaker_data, retrieval_library, d_type):
    new_conv = []
    raw_utterances = conv['sentences']  # 新增：保存原始发言
    for i,sent in enumerate(conv['sentences']):
        param = conv['speakers'][i] if data_name == "meld" else conv['genders'][i]
        sent_name = get_speaker_name(s_id, param, data_name)
        new_sent =  f'{sent_name}: {sent}'
        new_conv.append(new_sent)
    conv_str = "\n".join(new_conv)
    
    flatten_conv = flatten_conversation_mixed_by_surrounding(conv['sentences'], around_window, s_id, conv['speakers'] if data_name == "meld" else conv['genders'], data_name)
    
    samples = []
    for i, sent in enumerate(new_conv):
        system_msg = f'### You are an expert at analyzing the emotion of utterances among speakers in a conversation.\nYour goal is to infer the most accurate **emotion label** for a given utterance.\n'
        speaker_name = get_speaker_name(s_id, conv['speakers'][i] if data_name == "meld" else conv['genders'][i], data_name)
        raw_utterance = raw_utterances[i]  # 获取原始发言文本
        #得到隐性标签和情绪描述的特征描述
        emotion_surface = desc_speaker_data[s_id][i]['surface_emotion'].replace("\n", " ")
        emotion_implicit = desc_speaker_data[s_id][i]['implicit_emotion'].replace("\n", " ")
        # 从检索库中查找相似情感样本（训练集排除自身）
        similar_samples = retrieve_similar_samples(
            query_text=raw_utterance,
            current_id=(s_id, i),
            d_type=d_type,
            k=3  # 返回3个最相似样本
        )

        # 构建示范文本
        demonstration_str = "### Reference Similar Emotional Expressions:\n"
        for idx, sample in enumerate(similar_samples, 1):
            demonstration_str += (
                f"{idx}. \"{sample['text']}\" → {sample['label']} "
                f"(from {sample['dataset']}, distance: {sample['distance']:.2f})\n"
                #f"(from {sample['dataset']}, similarity: {sample['similarity']:.2f})\n"
            )
        demonstration_str += "\n"
        note = "Note: *Explicit Emotion* refers to the emotional state that a person outwardly expresses through their words, tone, facial expressions, or behavior. It is typically visible and can be directly perceived by others. For instance, someone loudly complaining might show a surface emotion of frustration or annoyance.\n*Implicit Emotion*, on the other hand, is the underlying emotional tendency that may not be directly expressed but can be inferred through context, common sense reasoning, or deeper understanding of the speaker’s situation. These emotions often reflect the speaker’s true feelings. \n"
        desc_msg = f'\n### Given the speaker’s Explicit Emotion Interpretation and Implicit Emotion Interpretation in the utterance \"{conv["sentences"][i]}\":'
        implicit_info_msg = (
            f"\n### Speaker: {speaker_name}\n"
            f"- Explicit Emotion Interpretation: {emotion_surface}\n"
            f"- Implicit Emotion Interpretation: {emotion_implicit}\n"
        )
        conv_str = "\n".join(flatten_conv[i])
        local_context_msg = (f"\n### Given the following conversation as a context \n{conv_str}\n"
                             )
        
        emotion_labels = get_label_map(data_name)
        labels_msg = f"### Available emotion labels: {', '.join(emotion_labels)}\n\n"
        

        q_msg =  f'Based on above conversation, similar emotional expressions, Explicit Emotion Interpretation and Implicit Emotion Interpretation, which emotional label of {speaker_name} in the utterance \"{conv["sentences"][i]}\".'
        label_msg = get_label_map(data_name)[conv['labels'][i]]
        
        samples.append({
            "messages":  [
                {'role': "system", 'content': system_msg + note  + local_context_msg + desc_msg + implicit_info_msg + demonstration_str + labels_msg},
                {'role': "user", 'content': q_msg},
                {'role': "assistant", 'content': label_msg},
            ]
        })
        
    return samples 

def process(paths_folder_preprocessed_data, args):
    
    process_kwargs = {}
    #paths_folder_preprocessed_data: 一个列表，包含预处理数据的路径
    for path_folder_preprocessed_data in paths_folder_preprocessed_data:
        
        d_type = 'train' if '.train.' in path_folder_preprocessed_data else \
                'valid' if '.valid.' in path_folder_preprocessed_data else \
                'test' if '.test.' in path_folder_preprocessed_data else None  
        '''
        folder_data: 数据所在文件夹
        around_window: 设定用于构建上下文窗口大小
        data_name: 数据集名称，如 "meld"
        path_data_out: 最终输出文件路径
        prompting_type: prompt 类型，如 "default" 或 "spdescV2"
        extract_prompting_llm_id: LLM 的 ID，用于加载说话人描述信息
        '''
        folder_data = args.data_folder
        around_window = args.window
        data_name = args.data_name
        path_data_out = path_folder_preprocessed_data
        prompting_type = args.prompting_type
        extract_prompting_llm_id = args.extract_prompting_llm_id 
        
        raw_data = f'{folder_data}/{data_name}.{d_type}.json'
        org_data = json.load(open(raw_data)) # ; org_data = dict([(k,v) for k,v in org_data.items()][:10])
        
        new_format = []
        
        # if use speaker description -> load raw data and preprocess
        #prompt 类型不是 "default"，则从指定文件中加载“说话人描述”
        if prompting_type not in ["default" ]:
            desc_speaker_data = json.load(open(f'{folder_data}/{data_name}.{d_type}_{prompting_type}_{extract_prompting_llm_id}.json'))
            processed_desc_speaker_data = {}
            #如果 prompt 类型中包含 "spdesc"，就用 preprocess_desc_speaker 函数对每条描述做预处理：
            if desc_speaker_data is not None and "spdescV2" == prompting_type:
                for s_id, desc_all_conv in desc_speaker_data.items():
                    processed_desc_speaker_data[s_id] = [preprocess_desc_speaker(spdesc) for spdesc in desc_all_conv]
                #最终存入 desc_speaker_data 字典中
                desc_speaker_data = processed_desc_speaker_data
            combined_dict = {}
            if  desc_speaker_data is not None and prompting_type in ["spdescV3", "spdescV4", "spdescV5"]:
                for s_id, conv_data in desc_speaker_data.items():
                    if not isinstance(conv_data, list):
                        continue
                    response_list = []
                    for item in conv_data:
                        if isinstance(item, str):
                            try:
                                # 替换非法反斜杠（如：\c 变成 \\c）
                                safe_item = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', item)
                                parsed = json.loads(safe_item)
                                if "Response" in parsed:
                                    response_list.append({"response": parsed["Response"]})
                                else:
                                    response_list.append({"response": "null"})
                            except json.JSONDecodeError as e:
                                print(f"❌ JSON解析失败，s_id: {s_id}，错误信息: {e}")
                                response_list.append({"response": "null"})
                    combined_dict[s_id] = response_list

                desc_speaker_data = combined_dict
                #print(desc_speaker_data)
            if  desc_speaker_data is not None and "ImplicitEmotion" == prompting_type:
                combined_dict = {
                    s_id: [
                        {
                            "emotion_label": emotion['implicit_emotion'],
                            "emotion_desc": emotion['emotion_desc']
                        } for emotion in conv_data['emotion_predictions']
                    ]
                    for s_id, conv_data in desc_speaker_data.items()
                }
                desc_speaker_data = combined_dict
            if  desc_speaker_data is not None and prompting_type in ["ImplicitEmotion_V2", "ImplicitEmotion_V2_byQwen3_14b", "ImplicitEmotion_V3","ImplicitEmotion_V4" ]:
                combined_dict = {
                    s_id: [
                        {
                            "surface_emotion": emotion['surface_emotion'],
                            "implicit_emotion": emotion['implicit_emotion']
                        } for emotion in conv_data['emotion_predictions']
                    ]
                    for s_id, conv_data in desc_speaker_data.items()
                }
                desc_speaker_data = combined_dict
        else:#是 default 类型，则不使用说话人描述
            desc_speaker_data = None
            
        # path data out 设置输出文件路径,默认将原始路径 .json 替换成 .jsonl 并加上 prompting 类型和 window
        path_processed_data = raw_data.replace(".json", f".0shot_w{around_window}_{prompting_type}.jsonl") if path_data_out is None else path_data_out
        
        # prompting process function 
        process_function_map = {
            "spdescV2": gen_spdescV2_prompting_messages,
            "spdescV3": gen_spdescV3_prompting_messages,
            "spdescV4": gen_spdescV3_prompting_messages,
            "spdescV5": gen_spdescV5_prompting_messages,
            "default": gen_default_prompting_messages,
            "ImplicitEmotion" : gen_ImplicitEmotion_prompting_messages,
            'ImplicitEmotion_V2' : gen_ImplicitEmotion_V2_prompting_messages,
            'ImplicitEmotion_V2_byQwen3_14b' : gen_ImplicitEmotion_V2_prompting_messages,
            'ImplicitEmotion_V3' : gen_ImplicitEmotion_V3_prompting_messages,
            #'ImplicitEmotion_V4': gen_ImplicitEmotion_V4_prompting_messages
        }
        #预定义的函数映射表中选出用于生成 prompt 的函数。如果找不到就默认用 "default"。
        process_func = process_function_map.get(prompting_type, process_function_map['default'])
        print(f"- process prompting by {process_func.__name__}")

        #遍历原始数据中的每条对话
        for s_id, conv in org_data.items(): 
           # 根据不同的prompting_type构造不同的参数
            if prompting_type in ['ImplicitEmotion_V3', 'spdescV3','spdescV4', 'spdescV5']:
                # ImplicitEmotion_V3需要额外的retrieval_library和d_type参数
                samples = process_func(data_name, conv, around_window, s_id, desc_speaker_data, None, d_type)
            else:
                # 其他函数使用标准参数
                samples = process_func(data_name, conv, around_window, s_id, desc_speaker_data)
            #将所有样本拼接到 new_format 中
            new_format = new_format + samples
        #每条 sample 转成 JSON 字符串格式。    
        with open(f'{path_processed_data}', 'wt') as f:
            new_format = [json.dumps(e) for e in new_format]
            f.write("\n".join(new_format))


# if __name__=="__main__":
#     process('train', around_window=5, use_spdesc=True)
#     process('test', around_window=5, use_spdesc=True)
#     process('valid', around_window=5, use_spdesc=True)

