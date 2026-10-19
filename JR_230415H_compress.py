import cv2
import numpy as np
from PIL import Image

def rgb_to_YCrCb(img):
    """

    This function is convert image RGB to YCrCb

    Args : Image 
    Returns : Converted image

    Commonly used :

        Y = 0.299R + 0.587G + 0.114B
        
        Cr = 128 + 0.5R - 0.418688G - 0.081312B

        Cb = 128 - 0.168736R - 0.331264G + 0.5B

    """
    B = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    R = img[:, :, 2].astype(float)

    Y  = 0.299000 * R + 0.587000 * G + 0.114000 * B
    Cr = 128 + 0.500000 * R - 0.418688 * G - 0.081312 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.500000 * B

    Y  = np.clip(np.round(Y),  0, 255).astype(np.uint8)
    Cr = np.clip(np.round(Cr), 0, 255).astype(np.uint8)
    Cb = np.clip(np.round(Cb), 0, 255).astype(np.uint8)

    ycrcb_img = np.stack((Y, Cr, Cb), axis=2)

    return ycrcb_img

def channel_norm_chroma_sub_sampeling(img):
    """
    Channel Normalization and the Chroma Sub Sampleing
    """

    Y = img[:,:, 0].astype(float)
    Cr = img[:,:,1].astype(float)
    Cb = img[:,:,2].astype(float)

    # 4:2:0 Chroma Subsampling
    Cb_sub = Cb[::2, ::2]
    Cr_sub = Cr[::2, ::2]

    # Channel normalization 
    Y  = Y  - 128
    Cb_sub = Cb_sub - 128
    Cr_sub = Cr_sub - 128

    return Y, Cr_sub, Cb_sub

def dct_8x8(block):
    """
    Compute the 2D DCT for one 8x8 level-shifted block.
    """
    block = np.asarray(block, dtype=np.float64)

    result = np.zeros((8, 8), dtype=np.float64)

    for u in range(8):
        for v in range(8):
            total = 0.0
            for x in range(8):
                for y in range(8):
                    total += (
                        block[x, y] * np.cos((2 * x + 1) * u * np.pi / 16)* np.cos((2 * y + 1) * v * np.pi / 16))
            cu = 1 / np.sqrt(2) if u == 0 else 1.0
            cv = 1 / np.sqrt(2) if v == 0 else 1.0

            result[u, v] = 0.25 * cu * cv * total
    return result

def dct_channel(channel):
    """
    Apply 8x8 DCT to every block in a 2D channel.
    The channel dimensions must be multiples of 8.
    """
    channel = np.asarray(channel, dtype=np.float64)
    height, width = channel.shape

    transformed = np.zeros_like(channel)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = channel[i:i+8, j:j+8]
            transformed[i:i+8, j:j+8] = dct_8x8(block)

    return transformed


if __name__ == "__main__":
    # Read image
    img = cv2.imread('kodim23.png')
    print(img.shape)
    # Check for bbp value
    num_channels = len(img.shape)
    bits_per_channel = img.dtype.itemsize*8 # convert to bits
    bpp = num_channels*bits_per_channel
    print(f"Image has {bpp}\n")

    #########################################
    #        Color Space Conversion         #
    #########################################
    conversion_img = rgb_to_YCrCb(img)
    print("Image Conversion RGB to YCrCb Done!\n")

    #######################################################
    # Channel value normalization and chroma sub-sampling #
    #######################################################
    Y, Cr, Cb = channel_norm_chroma_sub_sampeling(conversion_img)
    print("Image Conversion channel_norm_chroma_sub_sampeling Done!\n")

    ####################
    #     Apply DCT    #
    ####################
    Y_DCT = dct_channel(Y)
    Cr_DCT = dct_channel(Cr)
    Cb_DCT = dct_channel(Cb)
    print(f"DCT applied Successfully")



