import argparse
import os
#
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
#
torch.cuda.empty_cache()
#
from models import vision_transformer as vits
from prompters import PadPrompter, PatchPrompter

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import str2bool, get_params_groups, finetune_params, freeze, unfreeze, cosine_lr
from util.cluster_and_log_utils import log_accs_from_preds
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator
#, get_param_groups

from config import clip_pretrain_path, dino_pretrain_path
from models import extend_clip_vit

parser = argparse.ArgumentParser(description='SPTNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

parser.add_argument('--warmup_model_dir', type=str, default=None)
parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aircraft, herbarium_19, food101')
parser.add_argument('--prop_train_labels', type=float, default=0.5)
parser.add_argument('--use_ssb_splits', action='store_true', default=True)

parser.add_argument('--grad_from_block', type=int, default=11)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr2', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--transform', type=str, default='imagenet')
parser.add_argument('--sup_weight', type=float, default=0.35)
parser.add_argument('--n_views', default=2, type=int)
parser.add_argument('--lamb', type=float, default=0.1, help='The balance factor.')
parser.add_argument('--model', type=str, default='dino')
parser.add_argument('--model_path', type=str)

parser.add_argument('--freq_rep_learn', type=int)
parser.add_argument('--prompt_size', type=int)
parser.add_argument('--prompt_type', type=str, default='patch')
parser.add_argument('--pretrained_model_path', type=str)

parser.add_argument('--memax_weight', type=float, default=2)
parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

parser.add_argument('--fp16', action='store_true', default=True)
parser.add_argument('--eval_freq', default=1, type=int)

parser.add_argument('--num_words', default=7, type=int)
parser.add_argument('--words_drop_ratio', default=0.2, type=float)
parser.add_argument('--word_dim', default=512, type=int)   
parser.add_argument('--t_model_path', type=str, default=None, help='Path to pretrained text model')

# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
device = torch.device('cuda')
args = get_class_splits(args)

class SPTNetWithText(nn.Module):
    def __init__(self,
                 num_words = 7,
                 word_dim = 512,
                 words_drop_ratio = 0.2
                 ):
        super(SPTNetWithText, self).__init__()
        self.num_words = num_words
        self.word_dim = word_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concat_projector = nn.Linear(512 + 768, 768)
        self.projector = nn.Linear(in_features=512, out_features=num_words*word_dim).to(device)
                     load_extend_CLIP(
    "/path/to/ViT-B-16.pt",  # Use local path instead of "ViT-B/16"
    device="cuda"
)
        self.model1, _ = extend_clip_vit.load_extend_CLIP("./backbone/ViT-B-16.pt", torch.device(device), n_ctx=self.num_words+2)
        self.model2, _ = extend_clip_vit.load_extend_CLIP("./backbone/ViT-B-16.pt", torch.device(device), n_ctx=77)

        for params in self.model2.parameters():
            params.requires_grad = False

        self.words_drop_ratio = words_drop_ratio

        for params in self.model1.parameters():
            params.requires_grad = False
        
    def text_encoder(self, text_descriptions):
        text_tokens = extend_clip_vit.tokenize(text_descriptions).to(self.device)
        text_features = self.model2.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def genPf(self, pseudo_words, drop_or_not):
        if pseudo_words.shape[0] == 0:
            raise ValueError("Shape of pseudo words is 0")
        valid_mask = self._drop_word(pseudo_words)
        pseudo_text, end_token_ids = self.model1.prepare_pseudo_text_tensor(
            pseudo_words, valid_mask=valid_mask,w_dropout=drop_or_not
        )
        pseudo_feats = self.model1.encode_pseudo_text(pseudo_text, end_token_ids, text_pe=True)

        return pseudo_feats
    
    def _drop_word(self, pseudo_words):
        p = self.words_drop_ratio
        num_preds, num_words, _ = pseudo_words.shape
        mask = F.dropout(pseudo_words.new_ones(num_preds, num_words),
                         p=p,
                         training=self.training)
        start_end_mask = torch.ones_like(mask[:, :1])
        is_empty = mask.sum(dim=-1) == 0.0
        mask[is_empty, 0] = 1.0
        mask[mask>0.0] = 1.0
        valid_mask = torch.cat([start_end_mask, mask, start_end_mask], dim=-1)

        return valid_mask
    
    def forward(self, x, drop_or_not=False):
        img_feats = self.model2.encode_image(x).float()
        pseudo_words = self.projector(img_feats).view(-1, self.num_words, self.word_dim)
        pseudo_text_features = self.genPf(pseudo_words, drop_or_not).float()

        concatenated_features = torch.cat([
                                            F.normalize(img_feats, dim=-1), 
                                            F.normalize(pseudo_text_features, dim=-1)], dim=1)
        fused_features = self.concat_projector(concatenated_features)

        return fused_features


def construct_gcd_loss(prompter, backbone, projector, model_t, ln_t, images, class_labels, mask_lab, cluster_criterion, epoch, args):
    if prompter is None:
        image_features = backbone(images)
    else:
        image_features = backbone(prompter(images))


    concatenated_features = model_t(images)

    student_proj, student_out = projector(concatenated_features)
    teacher_out = student_out.detach()

    # clustering, sup
    sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
    cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

    # clustering, unsup
    cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
    avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
    me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
    cluster_loss += args.memax_weight * me_max_loss

    # represent learning, unsup
    contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
    contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

    # representation learning, sup
    student_proj1 = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
    student_proj1 = F.normalize(student_proj1, dim=-1)
    sup_con_labels = class_labels[mask_lab]
    sup_con_loss = SupConLoss()(student_proj1, labels=sup_con_labels)


    # Combined losses
    loss = (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
    loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss

    return loss, concatenated_features, student_out



def train(prompter, backbone, projector, model_t, ln_t, train_loader, optimizer, optimizer_cls, 
          exp_lr_scheduler, exp_lr_scheduler_cls, cluster_criterion, epoch, args):

    prompter.train()
    backbone.train()
    projector.train()
    model_t.eval()
    ln_t.train()

    num_batches_per_epoch = len(train_loader)
    switch_to_cls = False

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if (batch_idx + 1) % args.freq_rep_learn == 0:
            switch_to_cls = not switch_to_cls

        step = num_batches_per_epoch * epoch + batch_idx
        exp_lr_scheduler(step)

        images, class_labels, uq_idxs, mask_lab = batch
        mask_lab = mask_lab[:, 0]
        class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()

        if switch_to_cls: # train classifier
            freeze(prompter)
            args.grad_from_block = 11
            finetune_params(backbone, args)

            with torch.cuda.amp.autocast(args.fp16_scaler is not None):
                images = torch.cat([images[0].cuda(non_blocking=True), 
                                  prompter(images[0].cuda(non_blocking=True)).detach()], dim=0)
                loss, feats, outs = construct_gcd_loss(None, backbone, projector, model_t, ln_t, 
                                                     images, class_labels, mask_lab, cluster_criterion, epoch, args)

            optimizer_cls.zero_grad()
            if args.fp16_scaler is None:
                loss.backward()
                optimizer_cls.step()
            else:
                args.fp16_scaler.scale(loss).backward()
                args.fp16_scaler.step(optimizer_cls)
                args.fp16_scaler.update()

        else: # train prompter
            unfreeze(prompter)
            args.grad_from_block = 20 # large enough
            finetune_params(backbone, args)

            with torch.cuda.amp.autocast(args.fp16_scaler is not None):
                images = torch.cat(images, dim=0).cuda(non_blocking=True)
                loss, feats, outs = construct_gcd_loss(prompter, backbone, projector, model_t, ln_t,
                                                     images, class_labels, mask_lab, cluster_criterion, epoch, args)

            optimizer.zero_grad()
            if args.fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                args.fp16_scaler.scale(loss).backward()
                args.fp16_scaler.step(optimizer)
                args.fp16_scaler.update()

    exp_lr_scheduler_cls.step()


def test(model, model_t, ln_t, test_loader, epoch, save_name, args):
    model.eval()
    model_t.eval()
    ln_t.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            # Get concatenated features
            feats = model_t(images)
            _, logits = projector(feats)
            
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, 
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # Hyper-paramters
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.proj_dim = 256
    args.num_mlp_layers = 3
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_ctgs = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # BASE MODEL
    # ----------------------
    backbone = vits.__dict__['vit_base']().to(device)
    args.patch_size = 16
        
    if args.prompt_type == 'patch':
        args.prompt_size = 1
        prompter = PatchPrompter(args)

    elif args.prompt_type == 'all':
        args.prompt_size = 30
        prompter1 = PadPrompter(args)
        args.prompt_size = 1
        prompter2 = PatchPrompter(args)
        prompter = nn.Sequential(prompter1, prompter2)

    print(args)

    finetune_params(backbone, args) # HOW MUCH OF BASE MODEL TO FINETUNE

    # ----------------------
    # CLS HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    
    classifier = nn.Sequential(backbone, projector).cuda()
    state_dict = torch.load(args.pretrained_model_path, map_location='cpu')
    classifier.load_state_dict(state_dict, strict=False)###
    model = nn.Sequential(prompter, classifier).cuda()
    model_t = SPTNetWithText(num_words=args.num_words,
                              word_dim=args.word_dim,
                              words_drop_ratio=args.words_drop_ratio).to(device)
    ln_t = nn.Linear(in_features=512, out_features=768).to(device)
    for params in model_t.parameters():
        params.requires_grad = False

    # ----------------------
    # OPTIMIZATION
    # ----------------------
    params = [
         {'params': [p for param_group in get_params_groups(prompter) 
                   for p in param_group['params']], 'lr': args.lr},
         {'params': [p for p in ln_t.parameters() if p.requires_grad], 'lr': args.lr}
     ]
        
    # Initialize optimizers
    optimizer = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=0)
    optimizer_cls = SGD([p for param_group in get_params_groups(classifier) for p in param_group['params']], lr=args.lr2, momentum=args.momentum, weight_decay=args.weight_decay)
        
    args.fp16_scaler = None
    if args.fp16:
        args.fp16_scaler = torch.cuda.amp.GradScaler()

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # CONTRASTIVE TRANSFORM
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    # DATASETS
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # ------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    total_steps = len(train_loader) * args.epochs
    exp_lr_scheduler = cosine_lr(optimizer, args.lr, 1000, total_steps)
    exp_lr_scheduler_cls = lr_scheduler.CosineAnnealingLR(
            optimizer_cls,
            T_max=args.epochs,
            eta_min=args.lr2 * 0.1,
        )
    
    # ----------------------
    # TRAIN
    # ----------------------

    for epoch in range(args.epochs):
        train(prompter, backbone, projector, model_t, ln_t, train_loader, 
              optimizer, optimizer_cls, exp_lr_scheduler, exp_lr_scheduler_cls, 
              cluster_criterion, epoch, args)
        
        if epoch % 10 == 0:
            with torch.no_grad():
                # Get a sample batch for monitoring
                sample_images, _, _, _ = next(iter(train_loader))
                sample_images = sample_images[0].to(device)
                
                # Get features
                img_feats = model_t.model2.encode_image(sample_images).float()
                pseudo_words = model_t.projector(img_feats).view(-1, args.num_words, args.word_dim)
                print(f"Pseudo word norms: {pseudo_words.norm(dim=-1).mean(0)}")
                
                # Get similarity
                visual_features = backbone(sample_images)
                text_features = ln_t(model_t(sample_images))
                sim = F.cosine_similarity(visual_features, text_features).mean()
                print(f"Visual-Text similarity: {sim.item():.4f}")

    
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                all_acc, old_acc, new_acc = test(model, model_t, ln_t, test_loader_labelled, 
                                                           epoch, 'Train ACC Unlabelled', args)
                
                torch.save({
                    'model': model.state_dict(),
                    'model_t': model_t.state_dict(),
                    'ln_t': ln_t.state_dict(),
                    'projector': projector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, os.path.join(args.model_path, 'dinoB16_best_trainul.pt'))
