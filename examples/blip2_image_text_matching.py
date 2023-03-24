#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor


# #### Load an example image and text

# In[ ]:


raw_image = Image.open("../docs/_static/merlion.png").convert("RGB")
display(raw_image.resize((596, 437)))


# In[ ]:


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# In[ ]:


caption = "merlion in Singapore"


# #### Load model and preprocessors

# In[ ]:


model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
# model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)


# #### Preprocess image and text inputs

# In[ ]:


img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
txt = text_processors["eval"](caption)


# #### Compute image-text matching (ITM) score

# In[ ]:


itm_output = model({"image": img, "text_input": txt}, match_head="itm")
itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')


# In[ ]:


itc_score = model({"image": img, "text_input": txt}, match_head='itc')
print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

