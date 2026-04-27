import torch
from sentence_transformers import SentenceTransformer

from ..utils import set_hf_token


class TextEmbedder:

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        hf_token: str=None,
        cache_dir: str="./cache_predql"
    ) -> None:
        set_hf_token(hf_token)

        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_dir,
            device=device
        )
        self.none_values = {None, "None", "none", "NULL", "null", "NaN", "nan", ""}

    @torch.no_grad()
    def __call__(self, sentences: str | list[str]) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]

        filt_sentences = self._clean_sentences(sentences)

        return self.model.encode(filt_sentences, convert_to_tensor=True)

    def _clean_sentences(self, sentences: list[str]) -> list[str]:
        return [str(s) if str(s) not in self.none_values else "" for s in sentences]
