"""Text encoder"""

from typing import List, Union

import torch
from jaxtyping import Float
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
import torchvision.transforms as T

from nerfstudio.utils.rich_utils import CONSOLE


class ClipEncoder(nn.Module):
    """Clip encoder
    Args:
        pretrained_clip_path: path to the pretrained clip model
        device: device to use
    """

    def __init__(
        self,
        pretrained_clip_path: str = "openai/clip-vit-large-patch14",
        device: str = "cuda:0",
        cache_empty_text_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_clip_path)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_clip_path)
        self.vision_encoder = CLIPVisionModel.from_pretrained(pretrained_clip_path)
        self.vision_preprocess = T.Compose(
            [
                T.Resize(size=224, antialias=True),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.device = device
        self.cache_empty_text_embeddings = cache_empty_text_embeddings
        self.empty_text_embeddings = None

        CONSOLE.print("Text encoder loaded!")

    def get_text_embeds_inner(self, prompt: Union[str, List[str]]) -> Float[Tensor, "2 max_length embed_dim"]:
        """Get text embeddings for prompt
        Args:
            prompt: Prompt text
        Returns:
            Text embeddings
        """

        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings

    def get_text_embeds(self, prompt: Union[str, List[str]]) -> Float[Tensor, "2 max_length embed_dim"]:
        """Get text embeddings for prompt.
        This fuction has logic to cache empty text embeddings to speed up repeated calls with empty strings.
        Args:
            prompt: Prompt text
        Returns:
            Text embeddings
        """

        if self.cache_empty_text_embeddings:
            if isinstance(prompt, str) and prompt == "":
                if self.empty_text_embeddings == None:
                    self.empty_text_embeddings = self.get_text_embeds_inner("")
                return self.empty_text_embeddings
            elif isinstance(prompt, list) and all(p == "" for p in prompt):
                if self.empty_text_embeddings == None:
                    self.empty_text_embeddings = self.get_text_embeds_inner("")
                return self.empty_text_embeddings.repeat(len(prompt), 1, 1)

        text_embeddings = self.get_text_embeds_inner(prompt)
        return text_embeddings

    def get_image_embeds(self, images):
        """
        b x 3 x h x w
        """
        x = self.vision_preprocess(images)
        vision_outputs = self.vision_encoder(x)
        return vision_outputs.pooler_output
