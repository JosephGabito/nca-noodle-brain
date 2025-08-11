class Encoder:
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
        pass
    
    def setPrompt(self,prompt):
        self.prompt = prompt
        return self
    
    def encode(self):
        enc = self.tokenizer(self.prompt, return_tensors="pt")
        enc["input_ids"] = enc["input_ids"].to(self.device)
        enc["attention_mask"] = enc["attention_mask"].to(self.device)
        return enc