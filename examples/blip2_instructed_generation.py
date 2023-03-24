#!/usr/bin/env python
# coding: utf-8

# #### Large RAM is required to load the larger models. Running on GPU can optimize inference speed.

# In[ ]:


import sys
if 'google.colab' in sys.modules:
    print('Running in Colab.')
    get_ipython().system('pip3 install salesforce-lavis')


# In[ ]:


import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess


# #### Load an example image

# In[5]:


img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
display(raw_image.resize((596, 437)))


# In[3]:


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# #### Load pretrained/finetuned BLIP2 captioning model

# In[13]:


# we associate a model with its preprocessors to make it easier for inference.
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
)

# Other available models:
# 
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
# )

vis_processors.keys()


# #### prepare the image as model input using the associated processors

# In[8]:


image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


# #### generate caption using beam search

# In[ ]:


model.generate({"image": image})


# #### generate multiple captions using nucleus sampling

# In[ ]:


# due to the non-determinstic nature of necleus sampling, you may get different captions.
model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)


# #### instructed zero-shot vision-to-language generation

# In[14]:


model.generate({"image": image, "prompt": "Question: which city is this? Answer:"})


# In[15]:


model.generate({
    "image": image,
    "prompt": "Question: which city is this? Answer: singapore. Question: why?"})


# In[21]:


context = [
    ("which city is this?", "singapore"),
    ("why?", "it has a statue of a merlion"),
]
question = "where is the name merlion coming from?"
template = "Question: {} Answer: {}."

prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"

print(prompt)


# In[27]:


model.generate(
    {
    "image": image,
    "prompt": prompt
    },
    use_nucleus_sampling=False,
)

