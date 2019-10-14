import torch
import torch.nn as nn
import torch.nn.functional as F

# EnhanceNet: Single Image Super-Resolution through Automated Texture Synthesis - Sajjadi et.al.
# https://github.com/geonm/EnhanceNet-Tensorflow
class EnhanceNet(nn.Module):
    def __init__(self, upscale_factor, input_channels, channel_mask, output_channels, opt):
        '''
        Additional options:
        upsample: nearest, bilinear, or pixelShuffler
        recon_type: residual or direct
        use_bn: for batch normalization of the residual blocks
        '''
        super(EnhanceNet, self).__init__()
        assert(upscale_factor==4)
        self.upscale_factor = upscale_factor
        self.upsample = opt.upsample
        self.recon_type = opt.reconType
        self.use_bn = opt.useBN
        self.channel_mask = channel_mask
        self.input_channels = input_channels
        self.output_channels = output_channels

        self._enhancenet(input_channels, output_channels)
        self._initialize_weights()

    def _preprocess(self, images):
        #pp_images = images / 255.0
        ## simple mean shift
        images = images * 2.0 - 1.0

        return images
    
    def _postprocess(self, images):
        pp_images = ((images + 1.0) / 2.0)# * 255.0
        
        return pp_images

    def _upsample(self, factor):
        if self.upsample == 'nearest':
            return nn.Upsample(scale_factor=factor, mode='nearest')
        elif self.upsample == 'bilinear':
            return nn.Upsample(scale_factor=factor, mode='bilinear')
        elif self.upsample == 'bicubic':
            return nn.Upsample(scale_factor=factor, mode='bicubic')
        else: #pixelShuffle
            return nn.PixelShuffle(self.factor)

    #@profile
    def _recon_image(self, inputs, outputs):
        '''
        LR to HR -> inputs: LR, outputs: HR
        HR to LR -> inputs: HR, outputs: LR
        '''

        # check if we have recovered that model from a checkpoint prior to unshaded networks
        if not hasattr(self, 'channel_mask'):
            self.channel_mask = [0, 1, 2] # fallback to default RGB mode
        if not hasattr(self, 'output_channels'):
            self.output_channels = 3 # fallback to default RGB mode

        #resized_inputs = F.interpolate(inputs[:,self.channel_mask,:,:], 
        #inputs_masked = inputs[:,self.channel_mask,:,:]  #time per hit: 58463.9
        channel_mask_length = len(self.channel_mask)
        inputs_masked = inputs[:,0:channel_mask_length,:,:] #time per hit: 289.8
        resized_inputs = F.interpolate(inputs_masked, 
                                       size=[outputs.shape[2], 
                                             outputs.shape[3]], 
                                       mode=self.upsample)
        if self.recon_type == 'residual':
            if channel_mask_length==self.output_channels:
                recon_outputs = resized_inputs + outputs
            elif channel_mask_length<self.output_channels:
                recon_outputs = torch.cat([
                    resized_inputs + outputs[:,0:len(self.channel_mask),:,:],
                    outputs[:,len(self.channel_mask):,:,:]],
                    dim=1)
            else:
                raise ValueError("number of output channels must be at least the number of masked input channels")
        else:
            recon_outputs = outputs
        
        #resized_inputs = self._postprocess(resized_inputs)
        #resized_inputs = tf.cast(tf.clip_by_value(resized_inputs, 0, 255), tf.uint8)
        #tf.summary.image('4_bicubic image', resized_inputs)

        #recon_outputs = self._postprocess(recon_outputs)
        
        return recon_outputs, outputs
        
    def _enhancenet(self, input_channels, output_channels):
        self.preblock = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU())
            
        self.blocks = []
        for idx in range(10):
            if self.use_bn:
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64)
                    ))
            else:
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    ))
        self.blocks = nn.ModuleList(self.blocks)
            
        self.postblock = nn.Sequential(
                self._upsample(2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                self._upsample(2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, output_channels, 3, padding=1),
            )

    def _initialize_weights(self):
        def init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.orthogonal_(m.weight, torch.nn.init.calculate_gain('relu'))
        for block in self.blocks:
            block.apply(init)      

    #@profile
    def forward(self, inputs):
        #inputs = self._preprocess(inputs)

        features = self.preblock(inputs)
        for block in self.blocks:
            features = features + block(features)
        outputs = self.postblock(features)

        outputs, residual = self._recon_image(inputs, outputs)
        return outputs, residual