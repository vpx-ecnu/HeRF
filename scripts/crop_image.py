
import sys
from PIL import Image

margin = 10
image_path = sys.argv[1]
output_path = sys.argv[2]
img = Image.open(image_path)
w_down, h_down = img.size
img = img.crop((margin, margin, w_down - margin, h_down - margin))
img.save(image_path)