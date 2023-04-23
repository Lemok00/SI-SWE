from cleanfid import fid
import torchvision
import torch
from utils.training import log
from utils.mess_proj import message_to_tensor, tensor_to_message


def evaluate_fid(self, iter_idx):
    from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
    thread_pool = ThreadPoolExecutor(4)

    def save_img(batch, batch_idx, batch_size, nums):
        for i in range(batch.shape[0]):
            idx = batch_idx * batch_size + i
            if idx > nums:
                break
            torchvision.utils.save_image(batch[i],
                                         f'{self.args.eval_save_path}/{idx:05d}.png',
                                         normalize=True,
                                         range=(-1, 1))

    nums = 10000
    all_task = []
    with torch.no_grad():
        for i in range(nums // self.batch_size):
            fake_img = self.generate_image(ema=True)
            fake_img = fake_img.cpu().data
            all_task.append(thread_pool.submit(save_img, fake_img, i, self.batch_size, nums))
    wait(all_task, return_when=ALL_COMPLETED)

    if 'Bedroom' in self.args.dataset_path:
        if not fid.test_stats_exists('Bedroom', 'clean'):
            assert self.args.dataset_type == 'normal'
            fid.make_custom_stats('Bedroom', '../dataset/Bedroom/Samples_256', mode='clean')
        fid_score = fid.compute_fid(self.args.eval_save_path, dataset_name='Bedroom',
                                    mode="clean", dataset_split="custom")
    elif 'Church' in self.args.dataset_path and self.args.resolution == 256:
        fid_score = fid.compute_fid(self.args.eval_save_path, dataset_name='lsun_church',
                                    dataset_res=256, dataset_split="trainfull")
    elif 'FFHQ' in self.args.dataset_path and self.args.resolution == 256:
        fid_score = fid.compute_fid(self.args.eval_save_path, dataset_name='ffhq',
                                    dataset_res=256, dataset_split="trainval70k")
    else:
        fid_score = fid.compute_fid(self.args.eval_save_path, self.args.fid_ref_path)

    log(f'[{iter_idx:08d}/{self.args.iters:08d}] FID: {fid_score:.2f}', self.args.eval_log_path)

def evaluate_fid_and_kid(self, iter_idx):
    from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
    thread_pool = ThreadPoolExecutor(4)

    def save_img(batch, batch_idx, batch_size, nums):
        for i in range(batch.shape[0]):
            idx = batch_idx * batch_size + i
            if idx > nums:
                break
            torchvision.utils.save_image(batch[i],
                                         f'{self.args.eval_save_path}/{idx:05d}.png',
                                         normalize=True,
                                         range=(-1, 1))

    nums = 10000
    all_task = []
    with torch.no_grad():
        for i in range(nums // self.batch_size):
            fake_img = self.generate_image(ema=True)
            fake_img = fake_img.cpu().data
            all_task.append(thread_pool.submit(save_img, fake_img, i, self.batch_size, nums))
    wait(all_task, return_when=ALL_COMPLETED)

    if 'Bedroom' in self.args.dataset_path:
        fid_score = fid.compute_fid(self.args.eval_save_path, dataset_name='bedroom',
                                    mode="clean", dataset_split="custom", verbose=False)
        kid_score = fid.compute_kid(self.args.eval_save_path, dataset_name='bedroom',
                                    mode="clean", dataset_split="custom")
    elif 'Church' in self.args.dataset_path:
        fid_score = fid.compute_fid(self.args.eval_save_path, dataset_name='church',
                                    mode="clean", dataset_split="custom", verbose=False)
        kid_score = fid.compute_kid(self.args.eval_save_path, dataset_name='church',
                                    mode="clean", dataset_split="custom")
    elif 'FFHQ' in self.args.dataset_path:
        fid_score = fid.compute_fid(self.args.eval_save_path, dataset_name='ffhq',
                                    mode="clean", dataset_split="custom", verbose=False)
        kid_score = fid.compute_kid(self.args.eval_save_path, dataset_name='ffhq',
                                    mode="clean", dataset_split="custom")
    else:
        raise NotImplementedError

    log(f'[{iter_idx:08d}/{self.args.iters:08d}] FID: {fid_score:.2f}; KID: {kid_score*1000:.2f}', self.args.eval_log_path)

def evaluate_acc(self, iter_idx):
    if iter_idx > self.args.start_train_E:
        nums = 1000
        ACCs = []
        with torch.no_grad():
            for i in range(nums // self.batch_size):
                message = torch.randint(low=0, high=1, size=(self.batch_size, self.stru_dim))
                z_s = message_to_tensor(message, sigma=1, delta=0.5).to(self.device)
                _, z_t = self.sample_z()
                fake_img = self.generate_image((z_s, z_t), ema=True)
                rec_z = self.recover(fake_img, z_t, self.args.recover_iter_max)
                rec_message = tensor_to_message(rec_z, sigma=1)

                BER = torch.mean(torch.abs(message - rec_message))
                ACC = 1 - BER
                ACCs.append(ACC.item())

        ACC = sum(ACCs) / len(ACCs)

        log(f'[{iter_idx:08d}/{self.args.iters:08d}] ACC: {ACC:.4f}', self.args.eval_log_path)
