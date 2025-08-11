from typing import Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference.recipe_suggestion_model import RecipeSuggestionModel
from inference.encoder import Encoder

import torch
from fastapi import FastAPI

app   = FastAPI()
recipeSuggestionModel = RecipeSuggestionModel()
[model, tokenizer, device] = recipeSuggestionModel.getModel()

@app.get("/prompt/{prompt}")
def read_prompt(prompt):

    # Tokenize prompt
    encoder = (Encoder(device,tokenizer)
               .setPrompt(prompt)
               .encode())
    
    # Generate (greedy)
    output_ids = model.generate(
        input_ids=encoder["input_ids"],
        attention_mask=encoder["attention_mask"],
        max_new_tokens=recipeSuggestionModel.getMaxNewToken(),
        do_sample=False
    )

    # Decode full sequence (prompt + completion)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {
        "response": text
    }
