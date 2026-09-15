from astropy.io import fits
import numpy as np
filepath = "/home/bshra/Transient_Object_Classifier/TAO_transients/data/AGN/CSS071204:100029+071116.fits"


def load_object_images(filepath):
    with fits.open(filepath) as hdul:
        image_stack = []
        for hdu in hdul:
            if hdu.data is None:
                continue
            image_stack.append(hdu.data)
        images = np.array(image_stack)
