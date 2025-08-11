from transformers import AutoTokenizer, AutoModelForCausalLM

class RecipeSuggestionModel:
    def __init__(self):
        self.modelId = "microsoft/Phi-3-mini-4k-instruct"
        self.maxNew = 120
        self.device = "cuda"
        pass
        
    def getModel(self):
        tokenizer = AutoTokenizer.from_pretrained( self.modelId )
        model = AutoModelForCausalLM.from_pretrained( self.modelId )
        model = model.to(self.device)
        
        return model.eval(), tokenizer, self.device
    
    def getMaxNewToken(self):
        return self.maxNew