import torch
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen, listinstr
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from ..dataset import DATASET_TYPE
import pandas as pd
import string
import os


class IDEFICS(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceM4/idefics-9b-instruct', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2
        from transformers import IdeficsForVisionText2Text, AutoProcessor

        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map='auto'
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=False,
            trust_remote_code=True
        )
        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.file_root = osp.dirname(__file__)
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

    def generate_inner(self, message, dataset=None):
        prompts = (
            ['Users:']
            + [msg['value'] if msg['type'] == 'text' else Image.open(msg['value']) for msg in message]
            + ['<end_of_utterance>', '\nAssistant: ']
        )
        inputs = self.processor(
            prompts, add_end_of_utterance_token=False, return_tensors='pt'
        ).to('cuda')
        exit_condition = self.processor.tokenizer(
            '<end_of_utterance>', add_special_tokens=False
        ).input_ids
        bad_words_ids = self.processor.tokenizer(
            ['<image>', '<fake_token_around_image>'], add_special_tokens=False
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            **self.kwargs,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        text = generated_text[0].split('\nAssistant: ')[-1]
        return text


class IDEFICS2(BaseModel):  # 🔧 确保类名是 IDEFICS2（全大写）
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceM4/idefics2-8b', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        if 'Idefics3' in self.model_path.lower():
            warnings.warn('Install transfomers from source: PR https://github.com/open-compass/VLMEvalKit/pull/379')
            warnings.warn('Reference: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3')
        
        # 🔧 添加本地路径支持
        local_model_paths = {
            'HuggingFaceM4/idefics2-8b': '/data_all/intern05/models/idefics2-8b',
        }
        
        # 检查是否是已知的模型名称，如果是则使用本地路径
        if model_path in local_model_paths:
            actual_model_path = local_model_paths[model_path]
            print(f"🔄 使用本地模型路径: {model_path} -> {actual_model_path}")
        elif os.path.exists(model_path):
            actual_model_path = model_path
            print(f"✅ 使用传入的本地路径: {actual_model_path}")
        else:
            actual_model_path = model_path
            print(f"🌐 使用原始模型路径: {actual_model_path}")
        
        print(f"🔧 正在加载模型: {actual_model_path}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                actual_model_path,
                local_files_only=False,  # 允许访问本地缓存
                trust_remote_code=True
            )
            
            model = AutoModelForVision2Seq.from_pretrained(
                actual_model_path,
                torch_dtype=torch.bfloat16,
                _attn_implementation='flash_attention_2',
                device_map='cpu',
                local_files_only=False,  # 允许访问本地缓存
                trust_remote_code=True)
            self.model = model.to('cuda')
            print(f"✅ 模型加载成功: {actual_model_path}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 尝试强制离线模式
            print("🔧 尝试强制离线模式...")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    actual_model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                model = AutoModelForVision2Seq.from_pretrained(
                    actual_model_path,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation='flash_attention_2',
                    device_map='cpu',
                    local_files_only=True,
                    trust_remote_code=True)
                self.model = model.to('cuda')
                print(f"✅ 离线模式加载成功: {actual_model_path}")
            except Exception as e2:
                print(f"❌ 离线模式也失败: {e2}")
                raise e

        kwargs_default = {'max_new_tokens': 1024}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )
        torch.cuda.empty_cache()

    def _process(self, formatted_messages, formatted_images):
        inputs = self.processor(
            text=formatted_messages, images=formatted_images, return_tensors='pt'
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False, change_the_img_place=False):
        msgs = []
        if change_the_img_place:
            msgs.append({'type': 'text', 'value': message[0]['value']})
            msgs.extend(message[1:])
        else:
            msgs = message
        
        formatted_messages = ""
        formatted_images = []
        
        for msg in msgs:
            if msg['type'] == 'text':
                formatted_messages += msg['value']
                if add_brief:
                    formatted_messages += "\nPlease answer briefly."
                if add_yes_or_no:
                    formatted_messages += "\nPlease answer yes or no."
            elif msg['type'] == 'image':
                formatted_images.append(load_image(msg['value']))
                formatted_messages += "<image>"
        
        return formatted_messages, formatted_images

    def build_prompt_puremcq(self, message):
        return self.build_prompt_default(message)

    def build_prompt_mt(self, message):
        return self.build_prompt_default(message)

    def build_prompt_mmbench(self, message):
        return self.build_prompt_default(message)

    def build_prompt_mmmu(self, message):
        return self.build_prompt_default(message)

    def build_prompt_mathvista(self, message):
        return self.build_prompt_default(message)

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip()
        return response

    def generate_inner(self, message, dataset=None):
        if dataset in [
            'MMBench_DEV_EN', 'MMBench_DEV_EN_V11',
            'MMBench_TEST_EN', 'MMBench_TEST_EN_V11',
            'MMBench_DEV_CN', 'MMBench_DEV_CN_V11',
            'MMBench_TEST_CN', 'MMBench_TEST_CN_V11',
            'MMBench', 'MMBench_V11', 'MMBench_CN', 'MMBench_CN_V11'
        ]:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ['MathVista_MINI']:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in [
            'MME',
            'MMVet',
            'OCRVQA_TEST',
            'OCRVQA_TESTCORE',
            'TextVQA_VAL',
            'ChartQA_TEST',
            'DocVQA_VAL',
            'DocVQA_TEST',
            'InfoVQA_VAL',
            'InfoVQA_TEST',
        ]:
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == 'HallusionBench':
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            'MMStar',
            'SEEDBench_IMG',
            'AI2D_TEST',
            'ScienceQA_VAL',
            'ScienceQA_TEST',
        ]:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif listinstr(['MLVU','TempCompass','MVBench'], dataset):
            formatted_messages, formatted_images = self.build_prompt_default(message, change_the_img_place=True)
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip()
        return response
