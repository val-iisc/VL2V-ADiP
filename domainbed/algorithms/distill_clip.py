import clip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from domainbed import networks
from domainbed.algorithms import Algorithm
from domainbed.algorithms.miro import (
    MeanEncoder,
    VarianceEncoder,
    get_shapes,
    ForwardModel,
    get_optimizer,
)
from domainbed.networks.ur_networks import URFeaturizer
from domainbed.lib import misc
from tqdm import tqdm
from itertools import chain

dims = {
    "RN50": 1024,
    "RN101": 512,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class CLIP:
    """CLIP interface to save memory during algoritm save"""

    def __init__(self, hparams):

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. comment out the hparams line
        2. In line self.clip_model=clip.load, give the backbone "ViT-B/16"
        """
        ##########################

        self.device = "cuda"
        self.hparams = hparams
        self.clip_model = clip.load(self.hparams["clip_backbone"], device=self.device)[
            0
        ].float()
        self.clip_model.eval()
        self.clip_model.to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def get_img_feat(self, x):
        """Get normalized image embeddings"""

        clip_img_feat = self.clip_model.encode_image(x)
        clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        return clip_img_feat

    def get_txt_feat(self, labels):
        """Get normalized text embeddings"""

        clip_txt_feat = []
        for i in range(labels.size(0)):
            feat = self.zeroshot_weights[labels[i].item()]
            clip_txt_feat.append(feat)
        clip_txt_feat = torch.stack(clip_txt_feat, dim=0)
        return clip_txt_feat

    def get_zeroshot_classifier(self, classnames, templates):
        logit_scale = self.clip_model.logit_scale

        print("Getting zeroshot weights.")
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [
                    template.format(class_name=classname) for template in templates
                ]

                # Embeddings for each class
                texts = clip.tokenize(texts).to(self.device)  # tokenize
                embeddings = self.clip_model.encode_text(
                    texts
                )  # embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()
                zeroshot_weights.append(embeddings)

            # Computing zero-shot weights
            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(self.device)
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
            zeroshot_weights *= logit_scale.exp()
            zeroshot_weights = zeroshot_weights.squeeze().float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        classification_head = ClassificationHead(
            normalize=True, weights=zeroshot_weights
        )
        self.zeroshot_weights = zeroshot_weights
        return classification_head, zeroshot_weights


class DistillCLIP(Algorithm):
    """Distillation from CLIP"""

    def __init__(
        self, input_shape, num_classes, num_domains, hparams, clip_model, tgt_dom=0
    ):
        super(DistillCLIP, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. SET FEATURIZER = timm.models.vit_base_patch_16_224
        2. IGNORE num_domains, hparams, tgt_dom
        3. set lmd=0.1
        4. prompt_style = 2
        5. get classnames from your ID code
        """
        ##########################

        self.hparams = hparams
        self.lmd = hparams["lmd"]
        self.cls_num = num_classes
        self.dom_num = num_domains + 1
        self.prompt_style = hparams["prompt_style"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        # [#] zero-shot classifier from CLIP

        # Loading CLIP model
        classnames = [name.replace("_", " ") for name in hparams["class_names"]]

        # type 1 - {class}
        if self.prompt_style == 1:
            print("Prompt Template: \{class\}")
            templates = ["{class_name}"]

        # type 2 - a photo of a {class}
        elif self.prompt_style == 2:
            print("Prompt Template: a photo of a \{class\}")
            templates = ["a photo of a {class_name}"]

        # type 3 - a {dom} photo of a {class}
        elif self.prompt_style == 3:
            print("Prompt Template: a \{dom\} photo of a \{class\}")
            templates = [
                "an art of a {class_name}",
                "a clipart of a {class_name}",
                "a photo of a {class_name} product",
                "a photo of a {class_name}",
            ]

        # type 4 - a {dom} photo of a {small / big} {class}
        elif self.prompt_style == 4:
            print("Prompt Template: a \{dom\} photo of a \{small/big\} \{class\}")
            templates = [
                "an art photo of a {size} {{class_name}}",
                "a clipart photo of a {size} {{class_name}}",
                "a product photo of a {size} {{class_name}}",
                "a real photo of a {size} {{class_name}}",
            ]

        # type 5 - a {dom} photo of a {class} w/o target
        elif self.prompt_style == 5:
            print("Prompt Template: a \{dom\} photo of a \{class\} w/o target")
            templates = [
                "an art of a {class_name}",
                "a clipart of a {class_name}",
                "a photo of a {class_name} product",
                "a photo of a {class_name}",
            ]
            del templates[tgt_dom]

        # type 6 - a photo of a {class_label}
        elif self.prompt_style == 6:
            classnames = [str(i) for i in range(len(classnames))]
            print("Prompt Template: a photo of a \{class_label\}")
            templates = ["a photo of a {class_name}"]

        print(classnames)
        self.clip_cls, self.zeroshot_weights = clip_model.get_zeroshot_classifier(
            classnames, templates
        )


class DFC_STAGE1(DistillCLIP):
    """
    - ImNet frozen w/ CLIP loss
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE1, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 1 - ImNet frozen w/ CLIP loss\n")

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. set embed_dim=512
        """
        ##########################

        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.proj_lyr.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE11(DistillCLIP):
    """
    - ImNet frozen w/ CLIP loss, act
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE11, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 1 - ImNet frozen w/ CLIP loss, ACT\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.proj_lyr.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = F.gelu(self.proj_lyr(img_feat))
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(F.gelu(x))

    def forward(self, x):
        return self.predict(x)


class DFC_CLIP_INIT(DistillCLIP):
    """
    - ImNet frozen w/ CLIP loss, no projection (CLIP Init)
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_CLIP_INIT, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nImNet frozen w/ CLIP loss, no projection (CLIP Init)\n")

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. set embed_dim=512
        """
        ##########################

        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        train_params = chain(
            self.featurizer.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            img_feat_norm = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, img_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE2(DistillCLIP):
    """
    Stage 2 - Proj lyr frozen w/ CLIP loss
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE2, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. set embed_dim=512
        """
        ##########################

        print("\n\nStage 2 - Proj lyr frozen w/ CLIP loss\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        # # CLIP features
        # clip_img_feat = self.clip_model.encode_image(all_x)
        # clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        # clip_txt_feat = []
        # for i in range(all_y.size(0)):
        #     feat = self.zeroshot_weights[all_y[i].item()]
        #     clip_txt_feat.append(feat)
        # clip_txt_feat = torch.stack(clip_txt_feat, dim=0)

        # img_feat = self.featurizer(all_x)
        # proj_feat = self.proj_lyr(img_feat)
        # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # # Losses
        # kd_loss_1 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))
        # kd_loss_2 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))
        # loss = self.lmd * kd_loss_1 + (1 - self.lmd) * kd_loss_2

        # # [#] Backward pass

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return {"loss": loss.item()}

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE21(DistillCLIP):
    """
    Stage 2 - Proj lyr frozen w/ CLIP loss, ACT
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE21, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 2 - Proj lyr frozen w/ CLIP loss, ACT\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # [#] cutmix forward pass
        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = F.gelu(self.proj_lyr(img_feat))
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = F.gelu(self.proj_lyr(x))
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE22(DistillCLIP):
    """
    Stage 2 - Proj lyr frozen w/ CLIP loss, FULL
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE22, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 2 - Proj lyr frozen w/ CLIP loss, FULL\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.classifier.parameters(),
            self.proj_lyr.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE31(DistillCLIP):
    """
    Stage 3 - classifier only
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE31, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - classifier only \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.classifier.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        # # CLIP features
        # clip_img_feat = self.clip_model.encode_image(all_x)
        # clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        # clip_txt_feat = []
        # for i in range(all_y.size(0)):
        #     feat = self.zeroshot_weights[all_y[i].item()]
        #     clip_txt_feat.append(feat)
        # clip_txt_feat = torch.stack(clip_txt_feat, dim=0)

        # img_feat = self.featurizer(all_x)
        # proj_feat = self.proj_lyr(img_feat)
        # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # # Losses
        # kd_loss_1 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))
        # kd_loss_2 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))
        # loss = self.lmd * kd_loss_1 + (1 - self.lmd) * kd_loss_2

        # # [#] Backward pass

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return {"loss": loss.item()}

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            # clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            # clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # # forward pass
            # img_feat = self.featurizer(all_x)
            # proj_feat = self.proj_lyr(img_feat)
            # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # # loss comp.
            # loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE32(DistillCLIP):
    """
    Stage 3 - Full network
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE32, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - classifier only \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.proj_lyr.parameters(),
            self.classifier.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        # # CLIP features
        # clip_img_feat = self.clip_model.encode_image(all_x)
        # clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        # clip_txt_feat = []
        # for i in range(all_y.size(0)):
        #     feat = self.zeroshot_weights[all_y[i].item()]
        #     clip_txt_feat.append(feat)
        # clip_txt_feat = torch.stack(clip_txt_feat, dim=0)

        # img_feat = self.featurizer(all_x)
        # proj_feat = self.proj_lyr(img_feat)
        # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # # Losses
        # kd_loss_1 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))
        # kd_loss_2 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))
        # loss = self.lmd * kd_loss_1 + (1 - self.lmd) * kd_loss_2

        # # [#] Backward pass

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return {"loss": loss.item()}

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            # clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            # clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # # forward pass
            # img_feat = self.featurizer(all_x)
            # proj_feat = self.proj_lyr(img_feat)
            # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # # loss comp.
            # loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE33(DistillCLIP):
    """
    Stage 3 - Classifier only, new cls
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE33, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - Classifier only, new cls \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.classifier.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE34(DistillCLIP):
    """
    Stage 3 - full network, new cls
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE34, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - full network, new cls \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.proj_lyr.parameters(),
            self.classifier.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE35(DistillCLIP):
    """
    Stage 3 - classifier only, new cls, no proj
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE35, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - classifier only, new cls, no proj \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.new_cls = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.new_cls.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        return self.new_cls(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE36(DistillCLIP):
    """
    Stage 3 - Stage 2 init, KLD distillation
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE36, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - Stage 2 init, KLD distillation \n")
        self.tmp = hparams["tmp"]
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.new_cls = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.new_cls.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # [#] forward pass and loss
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
        clip_logits = self.clip_cls(clip_img_feat) / self.tmp

        # forward pass of student
        img_feat = self.featurizer(all_x)
        logits = self.new_cls(img_feat) / self.tmp

        # cos sim loss
        ce_loss = F.cross_entropy(logits, all_y)
        sim_loss = F.kl_div(logits.softmax(dim=-1), clip_logits.softmax(dim=-1))
        loss = ce_loss + (self.lmd * sim_loss)

        # [#] backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        return self.new_cls(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE2_MIRO(DistillCLIP):
    """
    - Stage 2 training on backbone + MIRO
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DFC_STAGE2_MIRO, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        print("\n\n Stage 2 training on backbone \n")
        self.embed_dim = 512
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.ld = hparams.ld

        # train_params = chain(
        #     self.featurizer.parameters(),
        # )
        # for param in self.pre_featurizer.parameters():
        #     param.requires_grad = False

        # build mean/var encoders
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
        self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])
        # optimizer
        parameters = [
            {"params": self.featurizer.parameters()},
            {
                "params": self.mean_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
            {
                "params": self.var_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        # self.optimizer = torch.optim.Adam(
        #     train_params,
        #     lr=self.hparams["lr"],
        #     weight_decay=self.hparams["weight_decay"],
        # )

    def update(self, x, y, **kwargs):
        # all_x = torch.cat([x for x, y in minibatches])
        # all_y = torch.cat([y for x, y in minibatches])
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # [#] Forward pass

        # CLIP features
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
        clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

        # clip_img_feat = self.clip_model.encode_image(all_x)
        # clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        # clip_txt_feat = []
        # for i in range(all_y.size(0)):
        #     feat = self.zeroshot_weights[all_y[i].item()]
        #     clip_txt_feat.append(feat)
        # clip_txt_feat = torch.stack(clip_txt_feat, dim=0)

        img_feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        proj_feat = self.proj_lyr(img_feat)
        proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # CLIP cos sim loss
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))
        loss = self.lmd * kd_loss_1 + (1 - self.lmd) * kd_loss_2

        # MIRO loss
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.0
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.0

        loss += reg_loss * self.ld

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE3_old(Algorithm):
    """Mutual-Information Regularization with Oracle"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ld = hparams.ld

        # build mean/var encoders
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
        self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])

        # optimizer
        parameters = [
            {"params": self.network.parameters()},
            {
                "params": self.mean_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
            {
                "params": self.var_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        logit = self.classifier(feat)
        loss = F.cross_entropy(logit, all_y)

        # MIRO
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.0
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.0

        loss += reg_loss * self.ld

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "reg_loss": reg_loss.item()}

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model


class DFC_STAGE3_MIRO(DistillCLIP):
    """
    - Stage 3 training on backbone + MIRO
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE3_MIRO, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model=clip_model
        )
        print("\n\n Stage 3 w/ MIRO ...\n")
        self.embed_dim = 512

        # [#] stage 3 - clip cls
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)

        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        # self.network = nn.Sequential(self.featurizer, self.proj_lyr, self.classifier)
        self.ld = hparams.ld

        # train_params = chain(
        #     self.featurizer.parameters(),
        # )
        # for param in self.pre_featurizer.parameters():
        #     param.requires_grad = False

        # build mean/var encoders
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
        self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])
        # optimizer
        # {"params": self.classifier.parameters()},
        # parameters = [
        #     {"params": self.classifier.parameters()},
        #     {
        #         "params": self.mean_encoders.parameters(),
        #         "lr": hparams.lr * hparams.lr_mult,
        #     },
        #     {
        #         "params": self.var_encoders.parameters(),
        #         "lr": hparams.lr * hparams.lr_mult,
        #     },
        # ]
        parameters = [
            {"params": self.featurizer.parameters()},
            {"params": self.proj_lyr.parameters()},
            {"params": self.classifier.parameters()},
            {
                "params": self.mean_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
            {
                "params": self.var_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        # all_x = torch.cat([x for x, y in minibatches])
        # all_y = torch.cat([y for x, y in minibatches])
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # [#] MIRO forward pass

        # forward pass of RN50
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        proj_feat = self.proj_lyr(feat)
        logit = self.classifier(proj_feat)
        loss = F.cross_entropy(logit, all_y)

        # MIRO loss
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.0
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.0

        loss += reg_loss * self.ld

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)
