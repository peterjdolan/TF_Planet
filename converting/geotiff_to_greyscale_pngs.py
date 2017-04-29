# Requires TiffFile to be installed.
#
# Run from the directory in which the .tif files are located
# Outputs greyscale PNG files, one for each channel of the
# original.

import png
import os
import tifffile

for filename in os.listdir("."):
  if filename.endswith(".tif"):
    print("Converting " + filename)
    with tifffile.TiffFile(filename) as image:
      for channel in range(4):
        channel_data = image.asarray()[:,:,channel]
        out_filename = "{}.{}.png".format(filename, channel)
        print("Writing " + out_filename)
        with open(out_filename, 'wb') as f:
          writer = png.Writer(width=256, height=256, bitdepth=16, greyscale=True)
          writer.write(f, channel_data.tolist())
