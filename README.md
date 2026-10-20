# JPEG_Compression_Algo

JPEG Compression Algorithm by scartch without using opencv.

Setup : 
```
git clone <repo_url>

python -m venv .venv

source .venv/bin/activate

pip install -r req.txt

```

Run Code : 

```
python JR_230415H_compress.py
```

Detailed methods included in the report in Docs.

Steps :


## JPEG Compression Pipeline

The JPEG compression process converts the RGB image into
YCbCr, reduces chrominance resolution, transforms image
blocks into frequency coefficients, and encodes the
quantized coefficients to produce a compressed bitstream.

```mermaid
flowchart TD
    A["Input Image: 24-bit RGB"] --> B["1. Color Space Conversion: RGB to YCbCr"]
    B --> C["2. Chroma Subsampling: 4:2:0"]
    C --> D["Level Shifting: Subtract 128"]
    D --> E["3. 8x8 Block DCT"]
    E --> F["4. Quantization"]
    F --> G["Zigzag Ordering"]
    G --> H["5. Run-Length Encoding"]
    H --> I["6. Huffman Entropy Encoding"]
    I --> J["Compressed JPEG Bitstream"]
```
