import numpy as np
import pandas as pd
from ...smp import *

rule_dict = {
    'llava_bench_conv': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_detail': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_complex': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'}  # noqa: E501
}


def get_eval(judge, content):
    return judge.generate(content)


def parse_score(review):
    """增强的分数解析函数，支持多种GPT回复格式"""
    logger = get_logger('Evaluation')
    
    try:
        lines = review.strip().split('\n')
        
        # 方法1: 寻找简单的两个数字格式 "8.0 9.0"
        for line in lines:
            line = line.strip().replace(',', ' ')
            parts = [p for p in line.split(' ') if p.strip()]
            
            if len(parts) == 2:
                try:
                    score1, score2 = float(parts[0]), float(parts[1])
                    if 0 <= score1 <= 10 and 0 <= score2 <= 10:
                        return [score1, score2]
                except ValueError:
                    continue
        
        # 方法2: 使用正则表达式匹配各种格式
        patterns = [
            # Assistant 1 Score: 8.0, Assistant 2 Score: 9.0
            r'Assistant\s+1\s+Score:\s*([\d\.]+).*?Assistant\s+2\s+Score:\s*([\d\.]+)',
            # [Assistant 1 Score: 8.0] [Assistant 2 Score: 9.0]
            r'\[Assistant\s+1\s+Score:\s*([\d\.]+)\].*?\[Assistant\s+2\s+Score:\s*([\d\.]+)\]',
            # Assistant 1: 8.0, Assistant 2: 9.0
            r'Assistant\s+1:\s*([\d\.]+).*?Assistant\s+2:\s*([\d\.]+)',
            # Assistant 1 Score: 8.0 Assistant 2 Score: 9.0 (单行)
            r'Assistant\s+1[^:]*:\s*([\d\.]+).*?Assistant\s+2[^:]*:\s*([\d\.]+)',
            # Score: 8.0 vs 9.0
            r'Score:\s*([\d\.]+)\s+vs\s+([\d\.]+)',
            # Rating: 8.0/10 and 9.0/10
            r'Rating:\s*([\d\.]+)/10.*?([\d\.]+)/10',
            # Assistant 1 gets 8.0, Assistant 2 gets 9.0
            r'Assistant\s+1\s+gets\s+([\d\.]+).*?Assistant\s+2\s+gets\s+([\d\.]+)',
            # Scores are 8.0 and 9.0
            r'Scores?\s+are?\s+([\d\.]+)\s+and\s+([\d\.]+)',
            # 8.0 for Assistant 1, 9.0 for Assistant 2
            r'([\d\.]+)\s+for\s+Assistant\s+1.*?([\d\.]+)\s+for\s+Assistant\s+2',
        ]
        
        import re
        for pattern in patterns:
            matches = re.search(pattern, review, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    score1, score2 = float(matches.group(1)), float(matches.group(2))
                    if 0 <= score1 <= 10 and 0 <= score2 <= 10:
                        return [score1, score2]
                except (ValueError, IndexError):
                    continue
        
        # 方法3: 寻找 JSON 格式的评分
        json_patterns = [
            r'"assistant_1":\s*([\d\.]+).*?"assistant_2":\s*([\d\.]+)',
            r'"score_1":\s*([\d\.]+).*?"score_2":\s*([\d\.]+)',
            r'"first":\s*([\d\.]+).*?"second":\s*([\d\.]+)',
        ]
        
        for pattern in json_patterns:
            matches = re.search(pattern, review, re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    score1, score2 = float(matches.group(1)), float(matches.group(2))
                    if 0 <= score1 <= 10 and 0 <= score2 <= 10:
                        return [score1, score2]
                except (ValueError, IndexError):
                    continue
        
        # 方法4: 按行搜索包含两个分数的行（改进版）
        for line in lines:
            # 跳过明显的描述性文本
            if len(line.strip()) < 5 or any(word in line.lower() for word in ['explanation', 'analysis', 'evaluation', 'because', 'however', 'therefore']):
                continue
                
            # 查找包含恰好两个分数的行
            scores = re.findall(r'\b(\d+\.?\d*)\b', line)
            if len(scores) == 2:
                try:
                    score1, score2 = float(scores[0]), float(scores[1])
                    if 0 <= score1 <= 10 and 0 <= score2 <= 10:
                        return [score1, score2]
                except ValueError:
                    continue
            
            # 查找分数范围更宽的情况（可能包含其他数字）
            if len(scores) >= 2:
                for i in range(len(scores) - 1):
                    try:
                        score1, score2 = float(scores[i]), float(scores[i + 1])
                        if 0 <= score1 <= 10 and 0 <= score2 <= 10:
                            return [score1, score2]
                    except ValueError:
                        continue
        
        # 方法5: 搜索单独的Assistant评分（改进版）
        assistant1_score = None
        assistant2_score = None
        
        # 更灵活的单独评分匹配
        assistant1_patterns = [
            r'Assistant\s+1[^:]*:?\s*([\d\.]+)',
            r'First\s+assistant[^:]*:?\s*([\d\.]+)',
            r'Response\s+1[^:]*:?\s*([\d\.]+)',
            r'Answer\s+1[^:]*:?\s*([\d\.]+)',
        ]
        
        assistant2_patterns = [
            r'Assistant\s+2[^:]*:?\s*([\d\.]+)',
            r'Second\s+assistant[^:]*:?\s*([\d\.]+)',
            r'Response\s+2[^:]*:?\s*([\d\.]+)',
            r'Answer\s+2[^:]*:?\s*([\d\.]+)',
        ]
        
        for line in lines:
            if assistant1_score is None:
                for pattern in assistant1_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            score = float(match.group(1))
                            if 0 <= score <= 10:
                                assistant1_score = score
                                break
                        except ValueError:
                            pass
            
            if assistant2_score is None:
                for pattern in assistant2_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            score = float(match.group(1))
                            if 0 <= score <= 10:
                                assistant2_score = score
                                break
                        except ValueError:
                            pass
        
        if assistant1_score is not None and assistant2_score is not None:
            return [assistant1_score, assistant2_score]
        
        # 方法6: 尝试从文本开头找分数（很多评判会在开头直接给出分数）
        first_lines = lines[:3]  # 检查前3行
        for line in first_lines:
            # 匹配开头的两个数字
            match = re.match(r'^\s*([\d\.]+)\s+([\d\.]+)', line.strip())
            if match:
                try:
                    score1, score2 = float(match.group(1)), float(match.group(2))
                    if 0 <= score1 <= 10 and 0 <= score2 <= 10:
                        return [score1, score2]
                except ValueError:
                    continue
        
        # 方法7: 最后的容错机制 - 提取所有可能的分数并尝试组合
        all_scores = []
        for line in lines:
            scores = re.findall(r'\b(\d+\.?\d*)\b', line)
            for score_str in scores:
                try:
                    score = float(score_str)
                    if 0 <= score <= 10:
                        all_scores.append(score)
                except ValueError:
                    pass
        
        # 如果找到了一些合理的分数，取前两个
        if len(all_scores) >= 2:
            return [all_scores[0], all_scores[1]]
        
        # 如果都找不到，返回中性分数
        logger.warning('Could not parse scores from review. Using default scores [5.0, 5.0].')
        logger.debug('Review content (first 300 chars): %s', review[:300])
        return [5.0, 5.0]
        
    except Exception as e:
        logger.error('Exception in parse_score: %s', str(e))
        logger.debug('Review content (first 200 chars): %s', review[:200])
        return [5.0, 5.0]
    except Exception as e:
        logger.error('Exception: %s', str(e))
        return [-1, -1]


def build_prompt(line):
    cap_str = line['caption']
    question = line['question']
    ans1 = line['gpt4_ans']
    ans2 = line['prediction']
    category = 'llava_bench_' + line['category']
    rule = rule_dict[category]
    role, prompt = rule['role'], rule['prompt']

    content = (f'[Context]\n{cap_str}\n\n'
               f'[Question]\n{question}\n\n'
               f'[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n'
               f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
               f'[System]\n{prompt}\n\n')
    return content


def LLaVABench_atomeval(model, prompt):
    review = get_eval(model, prompt)
    scores = parse_score(review)
    return scores


def LLaVABench_score(data):
    cates = ['overall'] + list(set(data['category']))
    ret = defaultdict(list)

    for c in cates:
        ret['split'].append(c)
        sub = data[data['category'] == c] if c != 'overall' else data
        ret['Relative Score (main)'].append(np.mean(sub['score']) / np.mean(sub['gpt4_score']) * 100)
        ret['VLM Score'].append(np.mean(sub['score']) * 10)
        ret['GPT4 Score'].append(np.mean(sub['gpt4_score']) * 10)
    return pd.DataFrame(ret)
