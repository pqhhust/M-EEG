# clinical_t5_encoder.py

from typing import Union, List, Tuple

import torch
from transformers import AutoTokenizer, T5EncoderModel

MODEL_NAME = "hossboll/clinical-t5"


class ClinicalT5Encoder:
    """
    Module:
      input : raw text (str hoặc list[str])
      output: đầu ra encoder của T5 (last_hidden_state) + attention_mask
              - last_hidden_state: (batch_size, seq_len, hidden_size)
              - attention_mask:   (batch_size, seq_len)
    """
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = T5EncoderModel.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        texts : str hoặc list[str]
        max_length : int
            độ dài tối đa khi tokenize

        Returns
        -------
        last_hidden_state : torch.Tensor
            Tensor shape (batch_size, seq_len, hidden_size)
        attention_mask : torch.Tensor
            Tensor shape (batch_size, seq_len)
        """
        if isinstance(texts, str):
            texts = [texts]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # outputs.last_hidden_state: (B, L, H)
        return outputs.last_hidden_state