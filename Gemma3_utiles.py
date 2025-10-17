from huggingface_hub import login
import pdb
from datasets import load_dataset
from PIL import Image
import random
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer
import datetime
import functools
import base64
from io import BytesIO
# Login into Hugging Face Hub
 
def set_seed(seed: int = 42):
    random.seed(seed) # Python 内置的 random 模块
    np.random.seed(seed) # NumPy 库
    if torch.cuda.is_available():
        torch.manual_seed(seed) # PyTorch CPU
        torch.cuda.manual_seed_all(seed) # PyTorch GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.manual_seed(seed)

def lora_trainable_report(model, lora_config):
    #use lora, parameter could be set through lora_config
    #report which layer are trainable
    peft_model = get_peft_model(model, lora_config)
    print("\n--- LoRA layers ---")
    lora_layers_found = []
    for name, module in peft_model.named_modules():
        #print(name, module)
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B')or hasattr(module, 'lora_'):
            lora_layers_found.append(name)
            #print(f"LoRA Layer (by attribute check): {name}")
    if not lora_layers_found:
        print("\n--- LoRA layers not found---")
    print(lora_layers_found)

def fully_finetune_trainable_report(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_final = 0
    trainable_name = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_final += param.numel()
            trainable_name.append(name)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters (final): {trainable_params_final:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params_final / total_params:.2f}%")
    print("Trainable layer name: ", trainable_name)
    
def fully_trainable_set(model, finetune_range):
    #finetune range:all, vision_tower, multi_modal_projector, language_model, lm_head
    if 'all' in finetune_range:
        fully_finetune_trainable_report(model)
        return True
    else:
        for param in model.parameters():
            param.requires_grad = False
        print("All model parameters are firsty frozen.")
        if "vision_tower" in finetune_range:
            vision_tower = model.model.vision_tower
            for name, param in vision_tower.named_parameters():
                param.requires_grad = True
            fully_finetune_trainable_report(model)
        if "multi_modal_projector" in finetune_range:
            multi_modal_projector = model.model.multi_modal_projector
            for name, param in multi_modal_projector.named_parameters():
                param.requires_grad = True
            fully_finetune_trainable_report(model)
        if "language_model" in finetune_range:
            language_model = model.model.language_model
            for name, param in language_model.named_parameters():
                param.requires_grad = True
            fully_finetune_trainable_report(model)
        if "lm_head" in finetune_range:
            lm_head = model.model.lm_head
            for name, param in lm_head.named_parameters():
                param.requires_grad = True
            fully_finetune_trainable_report(model)
    
def process_vision_info(messages):
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        role = msg.get("role", )
        if role == 'user':
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]
            for element in content:
                if isinstance(element, dict) and (
                    "image_url" in element or element.get("type") == "image_url"
                ) and element.get("image_url") != None:
                    image_url = element["image_url"].get('url')
                    header, base64_data = image_url.split(',', 1)
                    image_data = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    #image.show()
                    image_inputs.append(image)
        
    return image_inputs

def process_vision_info_inference(messages):
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        role = msg.get("role", )
        if role == 'user':
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]
            for element in content:
                if isinstance(element, dict) and (
                    "image" in element or element.get("type") == "image"
                ) and element.get("image") != None:
                    image_url = element["image"]
                    header, base64_data = image_url.split(',', 1)
                    image_data = base64.b64decode(base64_data)
                    #pdb.set_trace()
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    #image.show()
                    image_inputs.append(image)
        
    return image_inputs

def process_text_info(messages):
    #tutorial use apply_chat_template to form below strings
    #'<bos><start_of_turn>user\nYou are an expert product description writer for Amazon.\n\nCreate a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.\nOnly return description. The description should be SEO optimized and for a better mobile search experience.\n\n<PRODUCT>\nLEISURE ARTS A Dishcloth A Month\n</PRODUCT>\n\n<CATEGORY>\nHome & Kitchen | Home Décor | Kids\' Room Décor\n</CATEGORY><start_of_image><end_of_turn>\n<start_of_turn>model\nAdorable & functional dishcloths!  LEISURE ARTS\' "A Dishcloth A Month" collection adds charming style to your kitchen or kids\' room.  Perfect for everyday use or as unique home décor.  Shop now!<end_of_turn>\n'
    #We do same thing directly and be more flexible
    # Iterate through each conversation
    text_inputs = 'bos'
    user_count = 0   
    for msg in messages:
        # Get content (system + user + assistant)
        role = msg.get("role", )
        if role == 'system':
            content = msg.get("content", [])
            sysmtem_text = '<start_of_turn>'+'usr\n'+content[0]['text']+'\n<start_of_image><start_of_image>\n'
        if role == 'user':
            user_count += 1
            content = msg.get("content", [])
            for element in content:
                if isinstance(element, dict) and (
                "text" in element or element.get("type") == "text"
                    ) and element.get("text") != None:
                    if user_count == 1:
                        text_inputs += sysmtem_text
                    text_inputs += element['text']
                    text_inputs += "<end_of_turn>"
                    text_inputs += "\n"
        if role == 'assistant':
            content = msg.get("content", [])
            text_inputs += "<start_of_turn>model\n"
            text_inputs += content[0]['text']
            text_inputs += "<end_of_turn>"
            #text_inputs += "\n"
    #text_inputs = '<bos><start_of_turn><start_of_image><start_of_image>usr\n'+system_text+'\n\n'+user_text+"<end_of_turn>"+"\n"+"<start_of_turn>"+assistant_text+'<end_of_turn>\n'     
    return text_inputs

def process_text_info_inference(messages):
    #tutorial use apply_chat_template to form below strings
    #'<bos><start_of_turn>user\nYou are an expert product description writer for Amazon.\n\nCreate a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.\nOnly return description. The description should be SEO optimized and for a better mobile search experience.\n\n<PRODUCT>\nLEISURE ARTS A Dishcloth A Month\n</PRODUCT>\n\n<CATEGORY>\nHome & Kitchen | Home Décor | Kids\' Room Décor\n</CATEGORY><start_of_image><end_of_turn>\n<start_of_turn>model\nAdorable & functional dishcloths!  LEISURE ARTS\' "A Dishcloth A Month" collection adds charming style to your kitchen or kids\' room.  Perfect for everyday use or as unique home décor.  Shop now!<end_of_turn>\n'
    #We do same thing directly and be more flexible
    # Iterate through each conversation
    text_inputs = '<bos>'
        
    for msg in messages:
        # Get content (system + user + assistant)
        role = msg.get("role", )
        if role == 'system':
            content = msg.get("content", [])
            sysmtem_text = '<start_of_turn>'+'usr\n'+content[0]['text']+'\n<start_of_image><start_of_image>\n'
        if role == 'user':
            content = msg.get("content", [])
            for element in content:
                if isinstance(element, dict) and (
                "text" in element or element.get("type") == "text"
                    ) and element.get("text") != None:
                    text_inputs += sysmtem_text
                    text_inputs += element['text']
                    text_inputs += "<end_of_turn>"
                    text_inputs += "\n"
        if role == 'assistant':
            content = msg.get("content", [])
            text_inputs += "<start_of_turn>model\n"
            text_inputs += content[0]['text']
            text_inputs += "<end_of_turn>"
            #text_inputs += "\n"
    #text_inputs = '<bos><start_of_turn><start_of_image><start_of_image>usr\n'+system_text+'\n\n'+user_text+"<end_of_turn>"+"\n"+"<start_of_turn>"+assistant_text+'<end_of_turn>\n'     
    return text_inputs

def find_subsequence_occurrences(main_list, sub_list):
    """
    在一个主列表中查找给定子序列的所有起始索引。

    参数:
        main_list (list): 要在其中搜索的列表。
        sub_list (list): 要搜索的子序列。

    返回:
        list: 找到子序列的所有起始索引的列表。
              如果子序列未找到或为空，则返回空列表。
    """
    # 边界情况处理
    if not sub_list:
        # 如果子序列为空，根据实际需求决定：
        # - 返回空列表（这里采用的策略，因为没有具体序列可以“找到”）
        # - 或者返回所有可能的插入点，这通常不是用户期望的“查找”
        return []
    if not main_list:
        return [] # 在空列表中无法找到任何东西

    occurrences = []
    len_main = len(main_list)
    len_sub = len(sub_list)

    # 如果子序列比主列表还长，则不可能找到
    if len_sub > len_main:
        return []

    # 遍历主列表，直到剩余长度不足以包含整个子序列
    for i in range(len_main - len_sub + 1):
        # 检查当前切片是否与子序列匹配
        if main_list[i : i + len_sub] == sub_list:
            occurrences.append([i, i+len_sub]) # 如果匹配，则记录当前起始索引

    return occurrences

def find_mask_idx(batch, processor):
    #return the none -100 idx
    single_time_target_phrase = ["<start_of_turn>model\nGrasp result: none-stable", "<start_of_turn>model\nGrasp result: stable"]
    multiple_times_target_phrase = ["selected_cause_most_likely", "reason_most_likely",
                        "selected_cause_second_likely", "reason_second_likely",
                        "selected_most_reliable_action", "action_most_likely",
                        "selected_second_reliable_action", "action_second_likely"
                        ]
    special_token = ["<start_of_turn>model\n", "<end_of_turn>"]
    #trasnlate phrase into ids
    #['<start_of_turn>', 'G', 'rasp', '▁result', ':', '▁none', '-', 'stable'], ['<start_of_turn>', 'G', 'rasp', '▁result', ':', '▁stable']
    single_time_phrase_ids = [processor.tokenizer(i, return_tensors="pt", add_special_tokens=False)['input_ids'] for i in single_time_target_phrase]
    multiple_times_phrase_ids = [processor.tokenizer(i, return_tensors="pt", add_special_tokens=False)['input_ids'] for i in multiple_times_target_phrase]
    special_token_ids = [processor.tokenizer(i, return_tensors="pt", add_special_tokens=False)['input_ids'] for i in special_token]
    
    start_end_idx_phrase = [[] for i in range(len(batch['input_ids']))]
    for i in range(len(batch['input_ids'])):
        for j in range(len(single_time_phrase_ids)):
            start_end = find_subsequence_occurrences(batch['input_ids'][i].tolist(), single_time_phrase_ids[j][0].tolist())
            if start_end:
                start_end_idx_phrase[i].append([start_end[0][0]+5, start_end[0][1]])
    #pdb.set_trace()
    for i in range(len(batch['input_ids'])):
        for j in range(len(multiple_times_phrase_ids)):
            start_end = find_subsequence_occurrences(batch['input_ids'][i].tolist(), multiple_times_phrase_ids[j][0].tolist())
            if start_end:
                start_end_idx_phrase[i].append(start_end[-1]) #keywords show multiple times
    #locate the un-masked idx range
    none_masked_phrase_idx = [[] for i in range(len(batch['input_ids']))]
    for i in range(len(none_masked_phrase_idx)):
        if len(start_end_idx_phrase[i]) == 1:
            none_masked_phrase_idx[i].append(start_end_idx_phrase[i][0])
        else:    
            for j in range(len(start_end_idx_phrase[i])):
                if j == 0:
                    none_masked_phrase_idx[i].append(start_end_idx_phrase[i][0])
                elif j == len(start_end_idx_phrase[i])-1:
                    none_masked_phrase_idx[i].append([start_end_idx_phrase[i][j][-1]+1, len(batch['input_ids'][i].tolist())-1])
                else:
                    none_masked_phrase_idx[i].append([start_end_idx_phrase[i][j][-1]+1, start_end_idx_phrase[i][j+1][0]])
    
    start_end_idx_sotm = [[] for i in range(len(batch['input_ids']))]
    start_end_idx_eot = [[] for i in range(len(batch['input_ids']))]
    for i in range(len(batch['input_ids'])):
        for j in range(len(special_token_ids)):
            start_end = find_subsequence_occurrences(batch['input_ids'][i].tolist(), special_token_ids[j][0].tolist())
            if start_end and j==0:
                start_end_idx_sotm[i].extend(start_end)
            if start_end and j==1:
                start_end_idx_eot[i].extend(start_end)
    none_masked_special_idx = [[] for i in range(len(batch['input_ids']))]
    for i in range(len(batch['input_ids'])):
        # for j in range(len(start_end_idx_sotm[i])):
        #     #none_masked_special_idx[i].append([start_end_idx_sotm[i][j][-1], start_end_idx_eot[i][j*2+1][0]+1])
        #pdb.set_trace()
        none_masked_special_idx[i].append([start_end_idx_sotm[i][-1][-1], start_end_idx_eot[i][-1][0]+1])
            
    # decode the none-masked contents for check 
    # for i in range(len(none_masked_special_idx)):
    #     for j in range(len(none_masked_special_idx[i])):
    #         print(processor.tokenizer.convert_ids_to_tokens(batch["input_ids"][i][none_masked_special_idx[i][j][0]:none_masked_special_idx[i][j][1]]))
    # #pdb.set_trace()
    return  none_masked_phrase_idx, none_masked_special_idx

def decode_for_verification(processor, labels):
    for i in range(labels.size(0)):
        for j in range(labels.size(1)):
            label_ids = labels[i][j]
            valid_ids = label_ids[label_ids != -100]
            text = processor.tokenizer.convert_ids_to_tokens(valid_ids)
            if text != []:
                print(f"Sample {i}: {text}")
        pdb.set_trace()
        
        
        