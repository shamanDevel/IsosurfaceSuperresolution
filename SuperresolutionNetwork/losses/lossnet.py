import torch
import torch.nn as nn
import torch.nn.functional as F

from .lossbuilder import LossBuilder

class LossNet(nn.Module):
    """
    Main Loss Module.

    device: cpu or cuda
    opt: command line arguments (Namespace object) with:
     - losses: list of loss terms with weighting as string
     - further parameters depending on the losses

    """
    def __init__(self, device, input_channels, output_channels, high_res, padding, opt):
        super(LossNet, self).__init__()
        self.padding = padding
        self.upsample = opt.upsample
        self.input_channels = input_channels
        self.output_channels = output_channels
        #List of tuples (name, weight or None)
        self.loss_list = [(s.split(':')[0],s.split(':')[1]) if ':' in s else (s,None) for s in opt.losses.split(',')]
        
        # Build losses and weights
        builder = LossBuilder(device)
        self.loss_dict = {}
        self.weight_dict = {}

        self.loss_dict['mse'] = builder.mse() #always use mse for psnr
        self.weight_dict['mse'] = 0.0

        content_layers = []
        style_layers = []
        self.discriminator = None
        for name,weight in self.loss_list:
            if 'mse'==name or 'l2'==name or 'l2_loss'==name:
                self.weight_dict['mse'] = float(weight) if weight is not None else 1.0
            elif 'inverse_mse'==name:
                self.loss_dict['inverse_mse'] = builder.inverse_mse()
                self.weight_dict['inverse_mse'] = float(weight) if weight is not None else 1.0
            elif 'fft_mse'==name:
                self.loss_dict['fft_mse'] = builder.fft_mse()
                self.weight_dict['fft_mse'] = float(weight) if weight is not None else 1.0
            elif 'l1'==name or 'l1_loss'==name:
                self.loss_dict['l1'] = builder.l1_loss()
                self.weight_dict['l1'] = float(weight) if weight is not None else 1.0
            elif 'tl2'==name or 'temp-l2'==name:
                self.loss_dict['temp-l2'] = builder.temporal_l2()
                self.weight_dict['temp-l2'] = float(weight) if weight is not None else 1.0
            elif 'perceptual'==name:
                content_layers = [(s.split(':')[0],float(s.split(':')[1])) if ':' in s else (s,1) for s in opt.perceptualLossLayers.split(',')]
                #content_layers = [('conv_4',1), ('conv_12',1)]
                self.weight_dict['perceptual'] = float(weight) if weight is not None else 1.0
            elif 'texture'==name:
                style_layers = [(s.split(':')[0],float(s.split(':')[1])) if ':' in s else (s,1) for s in opt.textureLossLayers.split(',')]
                #style_layers = [('conv_1',1), ('conv_3',1), ('conv_5',1)]
                self.weight_dict['texture'] = float(weight) if weight is not None else 1.0
            elif 'adv'==name or 'gan'==name:
                self.discriminator, self.adv_loss = builder.gan_loss(
                    opt.discriminator, high_res,
                    input_channels + (output_channels+1),
                    opt)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = False
                self.discriminator_clip_weights = False
            elif 'wgan'==name:
                self.discriminator, self.adv_loss = builder.wgan_loss(
                    opt.discriminator, high_res,
                    input_channels + (output_channels+1),
                    opt)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = False
                self.discriminator_clip_weights = True
            elif 'wgan-gp'==name: # Wasserstein-GAN with gradient penalty
                self.discriminator, self.adv_loss = builder.wgan_loss(
                    opt.discriminator, high_res,
                    input_channels + (output_channels+1),
                    opt,
                    gradient_penalty = True)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = False
                self.discriminator_clip_weights = False
            elif 'tadv'==name or 'tgan'==name: #temporal adversary
                self.discriminator, self.adv_loss = builder.gan_loss(
                    opt.discriminator, high_res,
                    input_channels + 2*(output_channels+1),
                    opt)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = True
                self.discriminator_clip_weights = False
            elif 'twgan'==name: #temporal Wassertein GAN
                self.discriminator, self.adv_loss = builder.wgan_loss(
                    opt.discriminator, high_res,
                    input_channels + 2*(output_channels+1),
                    opt)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = True
                self.discriminator_clip_weights = True
            elif 'twgan-gp'==name: #temporal Wassertein GAN with gradient penalty
                self.discriminator, self.adv_loss = builder.wgan_loss(
                    opt.discriminator, high_res,
                    input_channels + 2*(output_channels+1),
                    opt,
                    gradient_penalty = True)
                self.weight_dict['adv'] = float(weight) if weight is not None else 1.0
                self.discriminator_use_previous_image = True
                self.discriminator_clip_weights = False
            else:
                raise ValueError('unknown loss %s'%name)

        if len(content_layers)>0 or len(style_layers)>0:
            self.pt_loss, self.style_losses, self.content_losses = \
                    builder.get_style_and_content_loss(dict(content_layers), dict(style_layers))

        self.loss_dict = nn.ModuleDict(self.loss_dict)
        print('Loss weights:', self.weight_dict)

    def print_summary(self, gt_shape, pred_shape, input_shape, prev_pred_warped_shape, num_batches, device):
        #Print networks for VGG + Discriminator
        import torchsummary
        if 'perceptual' in self.weight_dict.keys() or 'texture' in self.weight_dict.keys():
            print('VGG (Perceptual + Style loss)')
            torchsummary.summary(self.pt_loss, gt_shape, 2*num_batches, device=device.type)
        if self.discriminator is not None:
            print('Discriminator:')
            res = gt_shape[1]
            if self.discriminator_use_previous_image:
                input_images_shape = (gt_shape[0]+1+input_shape[0]+prev_pred_warped_shape[0], res, res)
            else:
                input_images_shape = (gt_shape[0]+1+input_shape[0], res, res)
            torchsummary.summary(
                self.discriminator,
                input_images_shape, 
                2*num_batches,
                device=device.type)


    @staticmethod
    def pad(img, border):
        """
        overwrites the border of 'img' with zeros.
        The size of the border is specified by 'border'.
        The output size is not changed.
        """
        if border==0: 
            return img
        b,c,h,w = img.shape
        img_crop = img[:,:,border:h-border,border:h-border]
        img_pad = F.pad(img_crop, (border, border, border, border), 'constant', 0)
        _,_,h2,w2 = img_pad.shape
        assert(h==h2)
        assert(w==w2)
        return img_pad

    def forward(self, gt, pred, input, prev_pred_warped):
        """
        gt: ground truth high resolution image (B x C=output_channels x 4W x 4H)
        pred: predicted high resolution image (B x C=output_channels x 4W x 4H)
        input: low resolution input image (B x C=input_channels x W x H)
               Only used for the discriminator, can be None if only the other losses are used
        prev_pred_warped: predicted image from the previous frame warped by the flow
               Shape: B x C x 4W x 4H
               with C = output_channels + 1 (warped mask)
               Only used for temporal losses, can be None if only the other losses are used
        """

        B, Cout, Hhigh, Whigh = gt.shape
        assert Cout == self.output_channels
        assert gt.shape == pred.shape
        B2, Cin, H, W = input.shape
        assert B == B2
        assert Cin == self.input_channels
        _, Cout2, _, _ = prev_pred_warped.shape
        #assert Cout2 == Cout + 1

        generator_loss = 0.0
        loss_values = {}

        # zero border padding
        gt = LossNet.pad(gt, self.padding)
        pred = LossNet.pad(pred, self.padding)
        if prev_pred_warped is not None:
            prev_pred_warped = LossNet.pad(prev_pred_warped, self.padding)

        # normal, simple losses, uses gt+pred
        for name in ['mse','inverse_mse','fft_mse','l1']:
            if name in self.weight_dict.keys():
                loss = self.loss_dict[name](gt, pred)
                loss_values[name] = loss.item()
                generator_loss += self.weight_dict[name] * loss

        # special losses: perceptual+texture, uses gt+pred
        if 'perceptual' in self.weight_dict.keys() or 'texture' in self.weight_dict.keys():
            style_weight=self.weight_dict.get('texture', 0)
            content_weight=self.weight_dict.get('perceptual', 0)
            style_score = 0
            content_score = 0

            input_images = torch.cat([gt, pred], dim=0)
            self.pt_loss(input_images)

            for sl in self.style_losses:
                style_score += sl.loss
            for cl in self.content_losses:
                content_score += cl.loss

            generator_loss += style_weight * style_score + content_weight * content_score
            if 'perceptual' in self.weight_dict.keys():
                loss_values['perceptual'] = content_score.item()
            if 'texture' in self.weight_dict.keys():
                loss_values['texture'] = style_score.item()

        # special: discriminator, uses input+pred+prev_pred_warped
        if 'adv' in self.weight_dict.keys():
            input_high = F.interpolate(input, 
                                       size=(gt.shape[-2],gt.shape[-1]),
                                      mode=self.upsample)
            pred_with_mask = torch.cat([pred, input_high[:,3:4,:,:]], dim=1)
            if self.discriminator_use_previous_image:
                input_images = torch.cat([input_high, pred_with_mask, prev_pred_warped], dim=1)
            else:
                input_images = torch.cat([input_high, pred_with_mask], dim=1)
            input_images = LossNet.pad(input_images, self.padding)
            gen_loss = self.adv_loss(self.discriminator(input_images))
            loss_values['discr_pred'] = gen_loss.item()
            generator_loss += self.weight_dict['adv'] * gen_loss

        # special: temporal l2 loss, uses input (for the mask) + pred + prev_warped
        if 'temp-l2' in self.weight_dict.keys():
            pred_with_mask = torch.cat([
                pred,
                F.interpolate(input[:,3:4,:,:], size=(gt.shape[-2],gt.shape[-1]), mode=self.upsample)
                ], dim=1)
            prev_warped_with_mask = prev_pred_warped
            loss = self.loss_dict['temp-l2'](pred_with_mask, prev_warped_with_mask)
            loss_values['temp-l2'] = loss
            generator_loss += self.weight_dict['temp-l2'] * loss

        return generator_loss, loss_values

    def train_discriminator(self, input,
                            gt_high, gt_prev_warped,
                            pred_high, pred_prev_warped):
        """
        Let Cin = input_channels (RGB + mask + optional normal + depth)
        Let Cout = output_channels + 1 (RGB + mask)
        Expected shapes:
         - input: low-resolution input, B x Cin x H x W
         - gt_high: ground truth high res image, B x Cout x 4H x 4W
         - gt_prev_warped: ground truth previous image warped, B x Cout x 4H x 4W
         - pred_high: predicted high res image, B x Cout x 4H x 4W
         - pred_prev_warped: predicted previous high res image, warped, B x Cout x 4H x 4W
        Note that the mask of the high-resolution image is not part of the generator,
         but interpolated later.
        """

        B, Cin, H, W = input.shape
        assert Cin == self.input_channels
        B2, Cout, Hhigh, Whigh = gt_high.shape
        assert B2 == B
        assert Cout == self.output_channels + 1
        assert gt_prev_warped.shape == gt_high.shape
        assert pred_high.shape == gt_high.shape
        assert pred_prev_warped.shape == gt_high.shape

        assert 'adv' in self.weight_dict.keys()
        B, Cout, Hhigh, Whigh = gt_high.shape

        # assemble input
        input_high = F.interpolate(input, size=(Hhigh, Whigh), mode=self.upsample)
        if self.discriminator_use_previous_image:
            gt_input = torch.cat([input_high, gt_high, gt_prev_warped], dim=1)
            pred_input = torch.cat([input_high, pred_high, pred_prev_warped], dim=1)
        else:
            gt_input = torch.cat([input_high, gt_high], dim=1)
            pred_input = torch.cat([input_high, pred_high], dim=1)
        gt_input = LossNet.pad(gt_input, self.padding)
        pred_input = LossNet.pad(pred_input, self.padding)

        discr_loss, gt_score, pred_score = self.adv_loss.train_discr(
            gt_input, pred_input, self.discriminator)
        return discr_loss, gt_score, pred_score
