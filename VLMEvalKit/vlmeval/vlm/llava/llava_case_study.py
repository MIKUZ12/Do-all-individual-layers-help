import torch
from PIL import Image
import os
import numpy as np
import pandas as pd
import logging
from vlmeval.vlm.base import BaseModel
from vlmeval.dataset import load
import pdb
import pickle
class LLaVA_Next(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='lmms-lab/llama3-llava-next-8b', **kwargs):
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
        )

        self.model_path = model_path
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).cuda()

        model = model.eval()
        self.model = model.cuda()

        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=1
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        logging.warning(
            f"Following kwargs received: {self.kwargs}, will use as generation config."
        )

    def output_process(self, answer):
        # Process the answer to clean up any unwanted tokens
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        # Additional processing to remove specific tokens can be added here
        return answer

    def generate_inner(self, message, dataset=None):
        content, images = [], []
        for msg in message:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": msg["value"]})
            else:
                content.append({"type": "image"})
                images.append(Image.open(msg["value"]).convert("RGB"))
        
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(prompt, images, return_tensors="pt").to(
            "cuda", torch.float16
        )
        
        output = self.model.generate(**inputs, **self.kwargs, return_dict_in_generate=True, output_hidden_states=True)
       
        # numpy_data = [tensor.cpu().numpy() for tensor in output['hidden_states'][0][0]]

        # pdb.set_trace()
        # len(output['hidden_states'][0][0][1]):4  len(output['hidden_states'][0][0][2]):1  len(output['hidden_states'][0][0][63]):4 len(output['hidden_states'][0][0][64]):1
        additional_tensor = output['hidden_states'][0][1].cpu().numpy()

        print(additional_tensor)
        pdb.set_trace()
        # Define the directory path using dataset_name and the index from dataset
        if dataset is not None:
            # output_dir = f'/ssddata/shiqi/reason/case_study/{dataset["dataset_name"]}/{dataset["index"]}/'
            output_dir = f'/ssddata/shiqi/reason/case_study/{dataset["dataset_name"]}_ans/{dataset["index"]}/'
            os.makedirs(output_dir, exist_ok=True)  
        else:
            output_dir = '/ssddata/shiqi/reason/case_study/default_dir/'

        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the tuple of tensors to one .npy file
        # np.save(os.path.join(output_dir, 'output_hidden_states_tuple.npy'), numpy_data, allow_pickle=True)
        # print("Saved tuple to output_hidden_states_tuple.npy")
        with open(os.path.join(output_dir, 'output_hidden_states_tuple.npy'), 'wb') as f:
            # pickle.dump(output['hidden_states'][0][0], f)
            ans=tuple(output['hidden_states'][i][0] for i in range(1, len(output['hidden_states'])))
            # pdb.set_trace()
            pickle.dump(ans,f)
        # pdb.set_trace()
        # Save the additional tensor to another .npy file

        # np.save(os.path.join(output_dir, 'output_hidden_states_additional.npy'), additional_tensor)
        print("Saved additional tensor to output_hidden_states_additional.npy")

        answer = self.processor.decode(output[0][-1], skip_special_token=True)
        answer = self.output_process(answer)
        print(answer)
        print(len(output['hidden_states']))
        # pdb.set_trace()
        return answer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset_name = 'MathVista_MINI'
    datasetfile_name = f'/ssddata/model_hub/{dataset_name}.tsv'
    id=29
    llava_model = LLaVA_Next('llava-hf/llama3-llava-next-8b-hf')
    length=len(load(datasetfile_name))
    for id in range(length):
        dataset = load(datasetfile_name).iloc[id]
    
        # Add dataset_name and index to dataset for easy access
        dataset["dataset_name"] = dataset_name
        text = dataset.get('question_for_eval', dataset['question'])
        image = f"/ssddata/model_hub/images/{dataset_name}/{dataset['index']}.jpg"
        
        messages = [
            {"type": "text", "value": text},
            {"type": "image", "value": image},
        ]

        output = llava_model.generate_inner(messages, dataset=dataset)