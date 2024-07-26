import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, trange
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from config import DiffusionConfig, scheduler_types

class CallableMeta(type):
    def __call__(cls, *args, **kwargs):
        if 'x' in kwargs or len(args) > 0:
            instance = cls()
            return instance.forward(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)

class Classifier(nn.Module, metaclass=CallableMeta):
    scheduler: scheduler_types = None
    unet: UNet2DConditionModel = None
    latent_size = 64
    n_tries = 1
    n_samples = [50, 500]
    keep = [2, 1]
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    seed = 42
    dtype = torch.float32
    n_timesteps = DiffusionConfig.NUM_TRAINING_STEPS.value
    batch_size = 4
    
    def __init__(self):
        super(Classifier, self).__init__()
        self.samples = self.__class__.n_samples
        self.max_samples = max(self.samples) 
        self.n_timesteps = self.__class__.n_timesteps
        self.keep = self.__class__.keep
        self.n_tries = self.__class__.n_tries
        self.batch_size = self.__class__.batch_size
        self.device = self.__class__.device
        self.dtype = self.__class__.dtype

        self.noise = torch.randn(self.max_samples * self.__class__.n_tries, 4, self.__class__.latent_size, self.__class__.latent_size)
        self.scheduler = self.__class__.scheduler
        self.unet = self.__class__.unet

        self.to(self.__class__.device)
        self.to(self.__class__.dtype)

    def forward_aux(self, x, ts, noise_ids, embedding, embeds_ids):

        errors = torch.zeros(len(ts))
        ids = 0

        with torch.no_grad():
            for _ in trange(len(ts) // self.batch_size + int(len(ts) % self.batch_size != 0), leave=False):
                batch_ts = torch.tensor(ts[ids: ids + self.batch_size])
                noise = self.noise[noise_ids[ids: ids + self.batch_size]]
                noised_x = x * (self.scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(self.device) + \
                    x * ((1 - self.scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(self.device)
                text_input = embedding[embeds_ids[ids: ids + self.batch_size]]
                noise_pred = self.unet(noised_x, batch_ts.to(self.device), encoder_hidden_states=text_input).sample
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
                errors[ids: ids + len(batch_ts)] = error.detach().cpu()
                ids += len(batch_ts)

        return errors

    def forward(self, x, embedding):
        data = dict()
        evaluated = set()
        prompt_ids = list(range(len(embedding)))

        start = self.n_timesteps // self.max_samples // 2
        to_eval = list(range(start, self.n_timesteps, self.n_timesteps // self.max_samples))[:self.max_samples]

        for keep, samples in zip(self.keep, self.samples):
            ts = list()
            noise_ids = list()
            embeds_ids = list()
            curr_to_eval = to_eval[len(to_eval) // samples // 2::len(to_eval) // samples][:samples]
            curr_to_eval = [t for t in curr_to_eval if t not in evaluated]
            for prompt_id in prompt_ids:
                for t_id, t in enumerate(curr_to_eval, start=len(evaluated)):
                    ts.extend([t] * self.n_tries)
                    noise_ids.extend(list(range(self.n_tries * t_id, self.n_tries * (t_id + 1))))
                    embeds_ids.extend([prompt_id] * self.n_tries)

            evaluated.update(curr_to_eval)
            print(ts)

            errors = self.forward_aux(x, ts, noise_ids, embedding, embeds_ids)

            for prompt_id in prompt_ids:
                mask = torch.tensor(embeds_ids) == prompt_id
                prompt_ts = torch.tensor(ts)[mask]
                prompt_pred_errors = errors[mask]
                if prompt_id not in data:
                    data[prompt_id] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
                else:
                    data[prompt_id]['t'] = torch.cat([data[prompt_id]['t'], prompt_ts])
                    data[prompt_id]['pred_errors'] = torch.cat([data[prompt_id]['pred_errors'], prompt_pred_errors])

            errors = [-data[prompt_id]['pred_errors'].mean() for prompt_id in prompt_ids]
            best_ids = torch.topk(torch.tensor(errors), k=keep, dim=0).indices.tolist()
            prompt_ids = [prompt_ids[i] for i in best_ids]

        pred_ids = prompt_ids[0]
        return pred_ids, data



if __name__ == "__main__":
    x = torch.randn(1, 10) 
    print(Classifier(x))