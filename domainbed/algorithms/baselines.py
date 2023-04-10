from .distill_clip import *

class DFC_PLAIN_1(DistillCLIP):
    """
    - Naive distillation - 
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

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        
        super(DFC_PLAIN_1, self).__init__(input_shape, num_classes, num_domains, hparams)
        print("\n\nStage 1 - ImNet frozen w/ CLIP loss\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.classifier = self.clip_cls
        # self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.classifier.parameters(),
            # self.proj_lyr.parameters(),
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
        clip_img_feat = self.clip_model.encode_image(all_x).float()
        
        # softmax distillation
        img_feat = self.featurizer(all_x)
        clip_logits = self.clip_cls(clip_img_feat)
        rn50_logits = self.classifier(img_feat)
        
        # losses
        loss1 = F.kl_div(rn50_logits.softmax(dim=-1), clip_logits.softmax(dim=-1))
        loss2 = F.cross_entropy(rn50_logits, all_y)
        loss = (loss1 + loss2) / 2

        # [#] backward pass

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
    
    
class DFC_PLAIN_2(DistillCLIP):
    """
    - Image emb distillation
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
        
        super(DFC_PLAIN_2, self).__init__(input_shape, num_classes, num_domains, hparams, clip_model=clip_model)
        print("\n\nImage emb distillation\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
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
          
        # [#] forward pass and loss
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)  
        
        # IE distillation
        img_feat = self.featurizer(all_x)
        proj_feat = self.proj_lyr(img_feat)
        proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)
        
        # cos sim loss
        loss= -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))

        # [#] backward pass

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
    
    
class DFC_PLAIN_3(DistillCLIP):
    """
    - Text emb distillation
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
        
        super(DFC_PLAIN_3, self).__init__(input_shape, num_classes, num_domains, hparams, clip_model=clip_model)
        print("\n\nText emb distillation\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
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
          
        # [#] forward pass and loss
        clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)  
        
        # IE distillation
        img_feat = self.featurizer(all_x)
        proj_feat = self.proj_lyr(img_feat)
        proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)
        
        # cos sim loss
        loss= -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))

        # [#] backward pass

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
    
    
class DFC_PLAIN_4(DistillCLIP):
    """
    - Distillation using teacher's softmax prediction (cos sim)
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
        
        super(DFC_PLAIN_4, self).__init__(input_shape, num_classes, num_domains, hparams, clip_model=clip_model)
        print("\n\nDistillation using teacher's softmax prediction (cos sim)\n")
        self.tmp = hparams["tmp"]
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        train_params = chain(
            self.featurizer.parameters(),
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
          
        # [#] forward pass and loss
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)  
        clip_logits = self.clip_cls(clip_img_feat) / self.tmp
        
        # forward pass of student
        img_feat = self.featurizer(all_x)
        logits = self.classifier(img_feat) / self.tmp
        
        # cos sim loss
        ce_loss = F.cross_entropy(logits, all_y)
        sim_loss = -torch.mean(F.cosine_similarity(logits.softmax(dim=-1), clip_logits.softmax(dim=-1)))
        loss = ce_loss + (self.lmd * sim_loss)

        # [#] backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)
    
    
class DFC_PLAIN_5(DistillCLIP):
    """
    - Distillation using teacher's softmax prediction (KLD)
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
        
        super(DFC_PLAIN_5, self).__init__(input_shape, num_classes, num_domains, hparams, clip_model=clip_model)
        print("\n\nDistillation using teacher's softmax prediction (KLD)\n")
        self.tmp = hparams["tmp"]
        self.embed_dim = dims[hparams["clip_backbone"]]
        # self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        train_params = chain(
            self.featurizer.parameters(),
            # self.proj_lyr.parameters(),
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
          
        # [#] forward pass and loss
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)  
        clip_logits = self.clip_cls(clip_img_feat) / self.tmp
        
        # forward pass of student
        img_feat = self.featurizer(all_x)
        # prj_feat = self.proj_lyr(img_feat)
        logits = self.classifier(img_feat) / self.tmp
        
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
        # x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)
    
    
class DFC_PLAIN_52(DistillCLIP):
    """
    - Distillation using teacher's softmax prediction (KLD), activation
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
        
        super(DFC_PLAIN_52, self).__init__(input_shape, num_classes, num_domains, hparams, clip_model=clip_model)
        print("\n\nDistillation using teacher's softmax prediction (KLD), Act\n")
        self.tmp = hparams["tmp"]
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        # self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
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
          
        # [#] forward pass and loss
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)  
        clip_logits = self.clip_cls(clip_img_feat) / self.tmp
        
        # forward pass of student
        img_feat = self.featurizer(all_x)
        prj_feat = self.proj_lyr(img_feat)
        act_prj_feat = F.gelu(prj_feat)
        logits = self.classifier(act_prj_feat) / self.tmp
        
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
        x = self.proj_lyr(x)
        return self.classifier(F.gelu(x))

    def forward(self, x):
        return self.predict(x)
    
    
class DFC_PLAIN_53(DistillCLIP):
    """
    - Distillation using teacher's softmax prediction (KLD), CLIP cls
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
        
        super(DFC_PLAIN_53, self).__init__(input_shape, num_classes, num_domains, hparams, clip_model=clip_model)
        print("\n\nDistillation using teacher's softmax prediction (KLD), CLIP cls\n")
        self.tmp = hparams["tmp"]
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        self.classifier = self.clip_cls
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
          
        # [#] forward pass and loss
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)  
        clip_logits = self.clip_cls(clip_img_feat) / self.tmp
        
        # forward pass of student
        img_feat = self.featurizer(all_x)
        prj_feat = self.proj_lyr(img_feat)
        logits = self.classifier(prj_feat) / self.tmp
        
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
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)
    
class DFC_PLAIN_6(DistillCLIP):
    """
    Our distillation method in 1 stage
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
        super(DFC_PLAIN_6, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nOur distillation method in 1 stage\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        # self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
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
    
    
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]
    
class DFC_CLIP_FT(DistillCLIP):
    """
    - Fine-tuning CLIP's IE using softmax over similarity
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
        
        super(DFC_CLIP_FT, self).__init__(input_shape, num_classes, num_domains, hparams, clip_model=clip_model)
        print("\n\nFine-tuning CLIP's IE using softmax over similarity\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        train_params = chain(
            self.featurizer.parameters(),
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
          
        # [#] forward pass and loss
        
        # forward pass of student
        img_feat = self.featurizer(all_x)
        logits = self.classifier(img_feat)
        loss = F.cross_entropy(logits, all_y)
        
        # label_mask = one_hot_embedding(all_y, logits.size(1))
        # label_mask = label_mask.cuda()
        
        # correct_logit = torch.sum(label_mask * logits, dim=1)
        # wrong_logit, _ = torch.max((1 - label_mask) * logits - 1e4 * label_mask, axis=1)
        # loss = - torch.sum(F.relu(correct_logit - all_logits + 50))
        # loss = - torch.sum(F.relu(correct_logit - wrong_logit + 50))

        # [#] backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)