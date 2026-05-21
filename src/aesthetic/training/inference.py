import torch
import torch.nn.functional as F
from aesthetic.training.models.pair_cls import PairClassifier
from aesthetic.training.models.four_cls import AestheticClassifier


class InferenceEngine:
    """
    Inference engine to evaluate features and compute win probabilities
    for Elo rating generation or 4-class aesthetic scoring.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "pair",
        feature_dim: int = 1024,
        device: str = "cuda",
    ):
        self.device = device
        self.model_type = model_type

        if model_type == "pair":
            self.model = PairClassifier(feature_dim=feature_dim)
        elif model_type == "four_class":
            self.model = AestheticClassifier(feature_dim=feature_dim, num_classes=4)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_features(self, *args) -> torch.Tensor:
        """
        For 'pair' model:
            Expects (emb1, emb2).
            Returns the probability of emb1 winning (Class 0).
        For 'four_class' model:
            Expects (features,).
            Returns the class probabilities.
        """
        if self.model_type == "pair":
            if len(args) != 2:
                raise ValueError("Pair model expects (emb1, emb2)")

            emb1 = args[0].to(self.device)
            emb2 = args[1].to(self.device)

            logits = self.model(emb1, emb2)
            probs = F.softmax(logits, dim=1)

            # Class 0 means left (emb1) wins. Return its probability.
            return probs[:, 0]
        else:
            if len(args) != 1:
                raise ValueError("Four-class model expects (features,)")

            features = args[0].to(self.device)
            logits = self.model(features)
            return F.softmax(logits, dim=1)
