import io
import numpy as np
import pytest
from PIL import Image
from rssr.uploads import decode_image

def encoded(size=(16,16),mode='RGB',format='PNG'):
    b=io.BytesIO();Image.new(mode,size).save(b,format=format);return b.getvalue()

def test_maximum_and_grayscale():
    assert decode_image(encoded((512,512))).shape==(512,512,3)
    y=decode_image(encoded(mode='L'));assert y.dtype==np.uint8 and y.shape==(16,16,3)

def test_rejected_inputs_are_clear():
    for data,message in [(encoded((1025,1)),'1024'),(encoded(mode='RGBA'),'opaque'),
                         (b'not an image','readable'),(b'0'*(8*1024*1024+1),'8 MB'),
                         (encoded(format='BMP'),'PNG or JPEG')]:
        with pytest.raises(ValueError,match=message):decode_image(data)
