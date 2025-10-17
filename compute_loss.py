# memo loss idea
# 1. add classification head (not good for multiple task)
# 2. sentence level classification using mask (first try this)
# 3. evaluate in sentence level, compute_metrics,  最常用的是计算两个嵌入向量的余弦相似度 (Cosine Similarity)。 of embedding, to do in future
import math
import numpy as np
import torch
from transformers import EvalPrediction, AutoTokenizer, Trainer, TrainingArguments # 导入必要的类
# from Gemma3_utiles import find_subsequence_occurrences
from torch.nn import CrossEntropyLoss
import pdb
from accelerate import Accelerator
from transformers import AutoProcessor
from sentence_transformers import SentenceTransformer, util, SimilarityFunction

accelerator = Accelerator()
def connect_tokens_to_sentence(token_list):
    sentence_parts = []
    punctuation = {'.', ',', ':', ';', '!', '?', '/', '"'} # 可以根据需要添加更多标点

    for i, token in enumerate(token_list):
        token = token.strip() # 移除 token 自身可能带的空格

        if token == '\n':
            sentence_parts.append('\n')
        elif token in punctuation:
            # 如果上一个不是空白或换行，则将标点紧跟其后
            if sentence_parts and sentence_parts[-1] not in [' ', '\n']:
                sentence_parts[-1] += token
            else: # 否则直接添加，前面有空格
                sentence_parts.append(token)
        else:
            # 如果当前不是第一个部分，且上一个不是换行，则添加空格
            if sentence_parts and sentence_parts[-1] != '\n':
                sentence_parts.append(' ')
            sentence_parts.append(token)
            
    # 最终连接所有部分
    return "".join(sentence_parts).strip()

class EvalStats:
    def __init__(self):
        self.token_correct = 0
        self.token_total = 0
        self.sent_correct = 0
        self.sent_total = 0

    def reset(self):
        self.token_correct = 0
        self.token_total = 0
        self.sent_correct = 0
        self.sent_total = 0
stats = EvalStats()      
def task1_compute_metric(eval_preds: EvalPrediction, compute_result=False):
    # <class 'transformers.trainer_utils.EvalPrediction'> (6, 2072, 262208)
    #
    #eval_preds[0][0] numpy prediction, [0][1] hidden state
    #eval_preds[1] label_ids
    # loss is calculated position based, so we need to locate the position first
    #predictions = np.argmax(eval_preds[0][0], axis=-1)
    #pdb.set_trace()
    #print("!!!!!!!!!!!!!!")
    if not compute_result:
        logits = eval_preds[0][0]
        label_ids = eval_preds[1]#.long()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = label_ids[..., 1:].contiguous()
        processor = AutoProcessor.from_pretrained("google/gemma-3-4b-pt")
        # Get predictions
        predictions = shift_logits.argmax(dim=-1)

        # Create mask for non-padding tokens (assuming ignore_index is -100)
        mask = shift_labels != -100

        # Calculate accuracy only on non-padding tokens
        correct_predictions = (predictions == shift_labels) & mask
        total_tokens = mask.sum()
        correct_tokens = correct_predictions.sum()
        stats.token_total += total_tokens.sum()
        stats.token_correct += correct_tokens.sum()
        # Compute the mean token accuracy and log it
        #total_sum = total_tokens.sum()
        #accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        #print("!!!!!!!! customize ", correct_tokens.sum(), total_sum)        
        decoded_preds = []
        decoded_labels = []

        for pred_, label_ in zip(predictions, shift_labels):
            # 1. 去掉 label 里为 -100 的部分
            mask = label_ != -100
            valid_label_ids = label_[mask]
            valid_pred_ids = pred_[mask]
            pred_str = processor.batch_decode(valid_pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            label_str = processor.batch_decode(valid_label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            decoded_preds.append(pred_str)
            decoded_labels.append(label_str)
        
        matches = [p == l for p, l in zip(decoded_preds, decoded_labels)]
        #sentence_accuracy = sum(matches) / len(matches)
        stats.sent_correct += sum(matches)
        stats.sent_total += len(matches)
        #print(stats.token_correct, stats.token_total, stats.sent_correct, stats.sent_total)
        return {}
    else:
        #pdb.set_trace()
        # Final summary after all batches
        token_acc = stats.token_correct / stats.token_total if stats.token_total > 0 else 0.0
        sent_acc = stats.sent_correct / stats.sent_total if stats.sent_total > 0 else 0.0
        #pdb.set_trace()
        stats.reset()  # reset after final calculation
        return {
            "self_calculate_token_accuracy": token_acc,
            "eval_accuracy": sent_acc
        }
    
#return {"eval_accuracy":sentence_accuracy, "self_calculate_token_accuracy": accuracy}
stats = EvalStats()  
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', similarity_fn_name=SimilarityFunction.COSINE)
def task2_compute_metric(eval_preds: EvalPrediction, compute_result=False):
    # <class 'transformers.trainer_utils.EvalPrediction'> (6, 2072, 262208)
    #
    #eval_preds[0][0] numpy prediction, [0][1] hidden state
    #eval_preds[1] label_ids
    # loss is calculated position based, so we need to locate the position first
    #predictions = np.argmax(eval_preds[0][0], axis=-1)
    #pdb.set_trace()
    
    if not compute_result:
        logits = eval_preds[0][0]
        label_ids = eval_preds[1]#.long()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = label_ids[..., 1:].contiguous()
        processor = AutoProcessor.from_pretrained("google/gemma-3-4b-pt")
        # Get predictions
        predictions = shift_logits.argmax(dim=-1)

        # Create mask for non-padding tokens (assuming ignore_index is -100)
        mask = shift_labels != -100

        # Calculate accuracy only on non-padding tokens
        # correct_predictions = (predictions == shift_labels) & mask
        # total_tokens = mask.sum()
        # correct_tokens = correct_predictions.sum()
        # stats.token_total += total_tokens.sum()
        # stats.token_correct += correct_tokens.sum()
        # Compute the mean token accuracy and log it
        #total_sum = total_tokens.sum()
        #accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        #print("!!!!!!!! customize ", correct_tokens.sum(), total_sum)        
        decoded_preds = []
        decoded_labels = []

        for pred_, label_ in zip(predictions, shift_labels):
            # 1. 去掉 label 里为 -100 的部分
            mask = label_ != -100
            valid_label_ids = label_[mask]
            valid_pred_ids = pred_[mask]
            correct_predictions = valid_pred_ids == valid_label_ids
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()
            stats.token_total += 1
            stats.token_correct += correct_tokens.sum()/total_tokens.sum()
           
            #generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            #label_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(valid_label_ids, ground_truth_ids)]
            pred_str = processor.batch_decode([valid_pred_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            label_str = processor.batch_decode([valid_label_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            pred_embedding = embedding_model.encode(pred_str, convert_to_tensor=True)
            label_embedding = embedding_model.encode(label_str, convert_to_tensor=True)
            similarity = embedding_model.similarity(pred_embedding, label_embedding)
            stats.sent_correct += similarity
            stats.sent_total += 1
            # decoded_preds.append(pred_str)
            # decoded_labels.append(label_str)
        
        #print(len(decoded_preds), len(similarity))
        print(stats.token_correct, stats.token_total, stats.sent_correct, stats.sent_total)
        return {}
    else:
        #pdb.set_trace()
        # Final summary after all batches
        token_acc = stats.token_correct / stats.token_total if stats.token_total > 0 else 0.0
        sent_acc = stats.sent_correct / stats.sent_total if stats.sent_total > 0 else 0.0
        #pdb.set_trace()
        stats.reset()  # reset after final calculation
        return {
            "self_calculate_token_accuracy": token_acc,
            "eval_accuracy": sent_acc
        }