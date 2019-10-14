"""
Generator networks.
Possible networks: EnhanceNet, SubpixelNet, TecoGAN.

Each net takes two arguments:
 - upscale_factor, currently always expected to be 4
 - opt: Namespace object of options: (if supported by the network)
    - upsample: upsample mode, can be 'nearest', 'bilinear' or 'pixelShuffler'
    - reconType: if 'residual', the whole network is a residual network
    - useBn: True if batch normalization should be used
    - numResidualLayers: integer specifying the number of residual layers

"""

from .enhancenet import EnhanceNet
from .subpixelnet import SubpixelNet
from .tecogan import TecoGAN
from .rcan import RCAN
from .videotools import VideoTools

def createNetwork(name, upscale_factor, input_channels, channel_mask, output_channels, additional_opt):
    print('upscale_factor:', upscale_factor)
    print('input_channels:', input_channels)
    print('channel_mask:', channel_mask)
    print('output_channels:', output_channels)
    """
    Creates the network for single image superresolution.
    Parameters:
     - upscale_factor: the upscale factor of the network, assumed to be 4
     - input_channels: the number of input channels of the low resolution image.
        Can vary from setting to setting to include rgb, depth, normal, warped previous frames
     - channel_mask: selection of input channels that match the output channels.
        Used for the residual architecture
     - output_chanels: the number of ouput channels of the high resolution image.
        Can vary from setting to setting
     - additional_opt: additional command line parameters to the networks
    """
    model = None
    if name.lower()=='SubpixelNet'.lower():
        model = SubpixelNet(upscale_factor, input_channels, channel_mask, output_channels, additional_opt)
    elif name.lower()=='EnhanceNet'.lower():
        model = EnhanceNet(upscale_factor, input_channels, channel_mask, output_channels, additional_opt)
    elif name.lower()=='TecoGAN'.lower():
        model = TecoGAN(upscale_factor, input_channels, channel_mask, output_channels, additional_opt)
    elif name.lower()=='RCAN'.lower():
        model = RCAN(upscale_factor, input_channels, channel_mask, output_channels, additional_opt)
    else:
        raise ValueError('Unknown model %s'%name)
    return model
