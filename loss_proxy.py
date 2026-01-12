from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import torch

def loss_calu(predict, target, config):
    # Initialize loss weights
    w1 = 1
    w2 = 1
    loss_fn = CrossEntropyLoss()
    
    # Unpack target data: image, attribute labels, object labels, composition (target) labels and image_path
    batch_img, batch_attr, batch_obj, batch_target, image = target
    
    # Move ground truth labels to GPU
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    
    # Unpack predictions: A and O might represent intermediate attribute/object features
    total_logits, A, O = predict
    
    # Unpack logits for both Visual Proxies and Textual Embeddings
    # Proxy: Usually refers to learnable visual prototypes or visual branch outputs
    # Text: Usually refers to linguistic/semantic branch outputs
    logits_com_proxy, logits_attr_proxy, logits_obj_proxy, \
    logits_com_text, logits_attr_text, logits_obj_text = total_logits
    
    # 1. Classification Losses for the Visual Proxy Branch
    loss_logit_com_proxy = loss_fn(logits_com_proxy, batch_target) # Composition loss
    loss_logit_attr_proxy = loss_fn(logits_attr_proxy, batch_attr) # Attribute loss
    loss_logit_obj_proxy = loss_fn(logits_obj_proxy, batch_obj)    # Object loss
    
    # 2. Classification Losses for the Textual/Semantic Branch
    loss_logit_com_text = loss_fn(logits_com_text, batch_target)
    loss_logit_attr_text = loss_fn(logits_attr_text, batch_attr)
    loss_logit_obj_text = loss_fn(logits_obj_text, batch_obj)
    
    #Alignment Loss via KL Divergence
    # Aims to align the distribution of the visual proxy with the textual semantic space
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    
    # KLDivLoss expects log-probabilities for input (p) and probabilities for target (q)
    p_com = F.log_softmax(logits_com_proxy, dim=1)  
    q_com = torch.softmax(logits_com_text, dim=1) 
    loss_kl = kl_loss(p_com, q_com)
    
    # 4. Loss Grouping
    # Total losses for Text and Vision branches respectively
    loss_text = loss_logit_com_text + loss_logit_attr_text + loss_logit_obj_text
    loss_vision = loss_logit_com_proxy + loss_logit_attr_proxy + loss_logit_obj_proxy
    
    # Aggregate losses by component (Composition, Attribute, Object)
    loss_ct = loss_logit_com_text + loss_logit_com_proxy # Combined Composition Loss
    loss_at = loss_logit_attr_text + loss_logit_attr_proxy # Combined Attribute Loss
    loss_ot = loss_logit_obj_text + loss_logit_obj_proxy   # Combined Object Loss
    
    # 5. Final Weighted Total Loss
    # Combines composition loss, balanced attribute/object losses, and the alignment loss (KL)
    loss = w1 * loss_ct + w2 * (loss_at + loss_ot) + loss_kl
    
    return loss, loss_logit_com_proxy, loss_logit_attr_proxy, loss_logit_obj_proxy, \
           loss_logit_com_text, loss_logit_attr_text, loss_logit_obj_text, loss_kl