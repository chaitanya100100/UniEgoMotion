import torch
from loguru import logger
from utils.torch_utils import to_tensor


class ImageFeats(object):
    def __init__(self, data_dir, split, img_feat_type):
        self.data_dir = data_dir
        self.split = split
        self.img_feat_type = img_feat_type
        self.load_img_feats()

    def load_img_feats(self):

        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            processed_path = f"{self.data_dir}/uniegomotion/egoview_dinov2_{self.split}.pt"
            logger.info(f"Loading dinov2 feats from {processed_path}.")
            self.ego_img_feats = torch.load(processed_path, weights_only=False)
            return

        if self.img_feat_type == "egovideo":
            processed_path = f"{self.data_dir}/uniegomotion/egoview_egovideo_{self.split}.pt"
            logger.info(f"Loading egovideo feats from {processed_path}.")
            self.ego_img_feats = torch.load(processed_path, weights_only=False)
            self.ego_img_feats = to_tensor(self.ego_img_feats)
            return

        if self.img_feat_type in ["clip_all"]:
            processed_path = f"{self.data_dir}/uniegomotion/egoview_clip_{self.split}.pt"
            logger.info(f"Loading clip feats from {processed_path}.")
            self.ego_img_feats = torch.load(processed_path, weights_only=False)
            return

        raise AttributeError

    def get_img_feats(self, seq_name, start_idx, end_idx):
        take_name, st, en = seq_name.split("___")
        start_idx = start_idx + int(st)
        end_idx = end_idx + int(st)
        assert end_idx <= int(en)
        del seq_name

        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            start_idx = start_idx // 3
            end_idx = end_idx // 3
            embs = (
                self.ego_img_feats["feats"][take_name][start_idx // 2 : end_idx // 2 + 1].clone().float()
            )  # T/2 x 5 x 1024 because dino feats are at 5 fps

            # duplicate the embs to make it 10 fps
            # assert start_idx % 2 == 0
            embs = embs[:, None].repeat(1, 2, 1, 1).flatten(0, 1)  # T x 5 x 1024
            if start_idx % 2 != 0:
                embs = embs[1:]

            if embs.shape[0] == (end_idx - start_idx + 2):  # handle partial segment
                embs = embs[:-1]

            assert embs.shape[0] == (end_idx - start_idx + 1)

            valid_img_embs = torch.ones(embs.shape[0], dtype=torch.long)
            if self.img_feat_type == "dinov2":  # remove register tokens
                embs = embs[:, 0]  # T x 1024
            return embs, valid_img_embs

        if self.img_feat_type == "egovideo":

            take_feats = self.ego_img_feats["feats"][take_name]
            feat_st = self.ego_img_feats["start_frame"][take_name]
            feat_en = self.ego_img_feats["end_frame"][take_name]
            feat_idx = ((feat_st + feat_en) // 2).long()

            q_idx = torch.arange(start_idx, end_idx + 1, 3, dtype=torch.long)

            # egovideo features are at short video segment level, not per frame
            # duplicate the features to make it per frame
            embs = batched_linear_interpolate(take_feats[None], feat_idx[None], q_idx[None])[0]
            assert embs.shape[0] == (end_idx // 3 - start_idx // 3 + 1)
            embs = torch.nn.functional.normalize(embs.float(), dim=-1)
            valid_img_embs = torch.ones(embs.shape[0], dtype=torch.long)

            return embs, valid_img_embs

        if self.img_feat_type in ["clip_all"]:
            start_idx = start_idx // 3
            end_idx = end_idx // 3
            embs = self.ego_img_feats["feats"][take_name][start_idx : end_idx + 1].clone().float()
            assert embs.shape[0] == (end_idx - start_idx + 1)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            valid_img_embs = torch.ones(embs.shape[0], dtype=torch.long)
            return embs, valid_img_embs
        raise AttributeError


def batched_linear_interpolate(A, t_i, t):
    """
    Efficient batched linear interpolation in PyTorch.

    Args:
        A: Tensor of shape (B, T, D) - batch of time series data.
        t_i: Tensor of shape (B, T) - time points corresponding to A.
        t: Tensor of shape (B, N) - query timestamps to interpolate.

    Returns:
        A_interp: Tensor of shape (B, N, D) - interpolated values.
    """
    B, T, D = A.shape  # Batch size, time steps, feature dimension
    _, N = t.shape  # Number of query points per batch

    # Ensure time steps are sorted
    assert torch.all(t_i[..., :-1] <= t_i[..., 1:]), "t_i must be sorted in increasing order"

    # Find indices where `t` would be inserted in `t_i`
    indices = torch.searchsorted(t_i, t, right=True)  # (B, N)

    # Get lower and upper bounds, clamping to valid indices
    lower_indices = torch.clamp(indices - 1, 0, T - 1)
    upper_indices = torch.clamp(indices, 0, T - 1)

    # Gather corresponding time values and A values
    t_lower = torch.gather(t_i, 1, lower_indices)  # (B, N)
    t_upper = torch.gather(t_i, 1, upper_indices)  # (B, N)
    A_lower = torch.gather(A, 1, lower_indices.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)
    A_upper = torch.gather(A, 1, upper_indices.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)

    # Compute interpolation weights
    t_range = (t_upper - t_lower).clamp(min=1e-6)  # Avoid division by zero
    alpha = (t - t_lower) / t_range  # (B, N)

    # Compute interpolated values
    A_interp = (1 - alpha.unsqueeze(-1)) * A_lower + alpha.unsqueeze(-1) * A_upper  # (B, N, D)

    return A_interp
