import cv2
import numpy as np
import heapq
from collections import Counter

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

def Quntization(channel, channel_type):
    # Standard JPEG luminance quantization table
    QY = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77],
        [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101],
        [72, 92, 95, 98,112,100,103, 99]
    ], dtype=np.float64)

    # Standard JPEG chrominance quantization table
    QC = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float64)

    height, width = channel.shape

    if channel_type == "Y":
        Q = QY
    elif channel_type == "C":
        Q = QC
    else:
        raise ValueError("channel_type must be 'Y' or 'C'")

    Q_full = np.tile(Q, (height // 8, width // 8))
    return np.rint(channel / Q_full)

def zigzag(block):
    # Zig-zag scan order for an 8x8 block (list of (row, col))
    ZIGZAG = sorted(((r, c) for r in range(8) for c in range(8)), key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else p[1]))

    """
    Read an 8x8 block in zig-zag order -> 64 values (DC first).
    """
    return np.array([block[r, c] for r, c in ZIGZAG], dtype=np.int32)

def run_length_encoding(channel):
    """
    JPEG style RLE for a quantized channel.

    DC : stored as the difference from the previous block's DC  (Top Left conner element)
    AC : stored as (run_of_zeros, value) pairs (15, 0) = 16 zeros (ZRL), (0, 0) = End Of Block (EOB)
    """
    height, width = channel.shape
    dc_values = []
    ac_values = []
    prev_dc = 0

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            zz = zigzag(channel[i:i+8, j:j+8])

            # DC differential
            dc_values.append(int(zz[0]) - prev_dc)
            prev_dc = int(zz[0])

            # AC run-length
            pairs = []
            run = 0
            for value in zz[1:]:
                if value == 0:
                    run += 1
                else:
                    while run > 15:
                        pairs.append((15, 0))
                        run -= 16
                    pairs.append((run, int(value)))
                    run = 0
            if run > 0:
                pairs.append((0, 0))
            ac_values.append(pairs)

    return dc_values, ac_values

def size_category(value):
    """
    Number of bits needed for |value| (JPEG 'size' / category).
    0 -> 0, ±1 -> 1, ±2..3 -> 2, ±4..7 -> 3, ...
    """
    return int(abs(value)).bit_length()


def amplitude_bits(value, size):
    """
    Extra bits that store the exact value inside its category.
    Negative values use one's complement (JPEG rule).
    """
    if size == 0:
        return ""
    if value < 0:
        value = value + (1 << size) - 1
    return format(value, f"0{size}b")


def build_huffman_table(symbols):
    """
    Build a Huffman code {symbol: bit string} from a list of symbols.
    """
    freq = Counter(symbols)

    # Only one kind of symbol , give it code "0"
    if len(freq) == 1:
        return {next(iter(freq)): "0"}

    # heap items: [frequency, tie_breaker, [symbol, code], [symbol, code], ...]
    heap = [[f, i, [sym, ""]] for i, (sym, f) in enumerate(freq.items())]
    heapq.heapify(heap)
    counter = len(heap)

    while len(heap) > 1:
        low = heapq.heappop(heap)
        high = heapq.heappop(heap)
        for pair in low[2:]:
            pair[1] = "0" + pair[1]
        for pair in high[2:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [low[0] + high[0], counter] + low[2:] + high[2:])
        counter += 1

    return {sym: code for sym, code in heap[0][2:]}


def dc_symbols(dc_values):
    return [size_category(v) for v in dc_values]


def ac_symbols(ac_values):
    # (run, size) ; EOB stays (0, 0) and ZRL stays (15, 0)
    return [(run, size_category(v)) for block in ac_values for run, v in block]


def huffman_encode(dc_values, ac_values, dc_table, ac_table):
    """
    Encode one channel into a bit string.
    Per block: DC code + DC bits, then for each AC pair: AC code + AC bits.
    """
    bits = []
    for dc, block in zip(dc_values, ac_values):
        size = size_category(dc)
        bits.append(dc_table[size])
        bits.append(amplitude_bits(dc, size))

        for run, value in block:
            size = size_category(value)
            bits.append(ac_table[(run, size)])
            bits.append(amplitude_bits(value, size))

    return "".join(bits)
if __name__ == "__main__":
    # Read image
    img = cv2.imread('kodim23.png')
    print(img.shape)
    # Check for bbp value
    num_channels = len(img.shape)
    bits_per_channel = img.dtype.itemsize*8 # convert to bits
    bpp = num_channels*bits_per_channel
    print(f"Image has {bpp}bbp\n")

    #########################################
    #        Color Space Conversion         #
    #########################################
    conversion_img = rgb_to_YCrCb(img)
    print(f"Image Conversion RGB to YCrCb Done!\n")

    #######################################################
    # Channel value normalization and chroma sub-sampling #
    #######################################################
    Y, Cr, Cb = channel_norm_chroma_sub_sampeling(conversion_img)
    print(f"Image Conversion channel_norm_chroma_sub_sampeling Done!\n")

    ####################
    #     Apply DCT    #
    ####################
    Y_DCT = dct_channel(Y)
    print(Y_DCT.shape)
    Cr_DCT = dct_channel(Cr)
    print(Cr_DCT.shape)
    Cb_DCT = dct_channel(Cb)
    print(f"DCT applied Successfully")

    ################
    # Quantization #
    ################
    Y_Qua = Quntization(Y_DCT, "Y")
    Cr_Qua = Quntization(Cr_DCT, "C")
    Cb_Qua = Quntization(Cb_DCT, "C")
    print(f"Quantization is Successful")

    ################
    #     RLE      #
    ################
    Y_dc,  Y_ac  = run_length_encoding(Y_Qua)
    Cr_dc, Cr_ac = run_length_encoding(Cr_Qua)
    Cb_dc, Cb_ac = run_length_encoding(Cb_Qua)
    print(f"Run Length Encoding is Successful")

    ######################
    #  Huffman Encoding  #
    ######################
    Y_dc_table = build_huffman_table(dc_symbols(Y_dc))
    Y_ac_table = build_huffman_table(ac_symbols(Y_ac))
    C_dc_table = build_huffman_table(dc_symbols(Cr_dc + Cb_dc))
    C_ac_table = build_huffman_table(ac_symbols(Cr_ac + Cb_ac))

    Y_bits  = huffman_encode(Y_dc,  Y_ac,  Y_dc_table, Y_ac_table)
    Cr_bits = huffman_encode(Cr_dc, Cr_ac, C_dc_table, C_ac_table)
    Cb_bits = huffman_encode(Cb_dc, Cb_ac, C_dc_table, C_ac_table)
    print(f"Huffman Encoding is Successful")

    total_bits = len(Y_bits) + len(Cr_bits) + len(Cb_bits)
    h, w = img.shape[:2]
    print(f"Compressed size : {total_bits // 8} bytes")
    print(f"Bits per pixel  : {total_bits / (h * w):.3f} bpp")
    print(f"Compression ratio : {24 * h * w / total_bits:.2f} : 1")



