import clip
import torch
import torch.nn as nn


class CLIPEmbed(nn.Module):
    def __init__(self, model_version='ViT-L/14', device="cuda", **kwargs):
        super().__init__()
        self.clip_model, self.transform = clip.load(model_version, device, **kwargs)
        self.device = device

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            x = self.transform(x).unsqueeze(0).to(self.device)
            x = self.clip_model.encode_image(x)
        return x


class CLIPReward(nn.Module):
    def __init__(
        self,
        *,
        model: CLIPEmbed,
        alpha: float,
        target_prompts: torch.Tensor,
        baseline_prompts: torch.Tensor,
    ) -> None:
        """CLIP Reward function that modifies the CLIP vector space by
        projecting all vectors onto the line spanned by the prompt and
        a baseline prompt. The alpha parameter controls the degree of
        projection. A value of 0.0 means that the reward function is
        equivalent to the CLIP reward function. A value of 1.0 means
        that the vector space is completely projected onto the line
        and becomes a 1D space. Any value in between is a linear
        interpolation between the two.

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.embed_module = model
        target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
        baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)

        self.alpha = alpha
        projection = self.compute_projection(alpha)
        self.register_buffer("projection", projection)

    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_images(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        breakpoint()
        y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        return y

    @staticmethod
    def tokenize_prompts(x, device='cuda') -> torch.Tensor:
        """Tokenize a list of prompts."""
        return clip.tokenize(x).to(device)

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.embed_module.clip_model.encode_text(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x):
        return self.embed_module.forward(x)


def load_reward_model(target_prompts, baseline_prompts, alpha, *, model_version='ViT-L/14', device="cuda", **kwargs):
    return CLIPReward(
        model=CLIPEmbed(model_version, device, **kwargs),
        alpha=alpha,
        target_prompts=CLIPReward.tokenize_prompts(target_prompts, device),
        baseline_prompts=CLIPReward.tokenize_prompts(baseline_prompts, device),
    ).eval()


if __name__ == "__main__":
    model = load_reward_model(['a', 'b', 'c', 'd', 'e'], 'baseline', 0.5, size=(224, 224), color=False, format='tensor')
    model(torch.randn(224, 224))
