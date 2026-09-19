from __future__ import print_function

from datetime import datetime
import os
from pathlib import Path

import cv2 as cv
import argparse

import logging



#should be the same value as in AI
low_threshold:int =25


logger = logging.getLogger(__name__)

def CannyThreshold(low_threshold : float|int,
                   image_name : str,
                   show_image : bool = False,
                   save_image : bool = False,
                   saved_image_name : Path = None) -> bool :

    src = cv.imread(cv.samples.findFile(image_name))
    if src is None:
        logger.error(f"Could not open or find the image [{image_name}]")
        return False

    #TODO Use cv2.GaussianBlur() to reduce noise. This can help improve edge detection. Check out our guide on Python OpenCV cv2.GaussianBlur() for more details.
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(src_gray, (3,3))
    ratio = 3
    kernel_size = 3

    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))

    if show_image:
        window_name = 'Edge Map'
        cv.namedWindow(window_name)
        cv.imshow(window_name, dst)
        cv.waitKey()
    # Save the result
    if save_image:
        return cv.imwrite(saved_image_name, dst)

    return True


def CannyImage(image_name : str)->bool:
    image_name_path=Path(image_name)
    if not Path.exists(image_name_path):
        logger.error(f"file does not exist [{image_name}]")
        return False

    return CannyThreshold(low_threshold=low_threshold,
                   image_name=image_name,
                   show_image=True)

def CannyDir(dir_name : str, output_dir : str)->bool:
    if output_dir is None:
        logger.error("ouput directory shall be provided")
        return False

    dir_name_path=Path(dir_name)
    if not Path.exists(dir_name_path):
        logger.error(f"dir does not exist [{dir_name_path}]")
        return False

    new_location=Path(output_dir)
    if new_location.exists():
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = new_location.with_name(new_location.name + '_' + backup_timestamp)
        os.rename(new_location, backup_path)
        logger.info(f"Result folder already exists, renaming it: {new_location} -> {backup_path}")

    size_root_path=len(dir_name_path.parts)

    #we assume here we have only jpg images
    result=True
    for path_img in dir_name_path.rglob('*.jpg'):
        new_path = new_location.joinpath(*path_img.parts[size_root_path:])
        logger.info(f"{path_img.name} -> {new_path}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        res=CannyThreshold(low_threshold=low_threshold,
                   image_name=str(path_img),
                   save_image=True,
                   saved_image_name=str(new_path))
        if not res:
            result=False

    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Canny Edge Detector')
    parser.add_argument("-image", nargs='?')
    parser.add_argument("-dir", nargs='?')
    parser.add_argument("-output", nargs='?')
    try:
        args = parser.parse_args()
        if args.dir is None:
            if args.image is None:
                print("missing image argument")
                parser.print_help()
            else:
                print('processing '+ args.image)
                CannyImage(args.image)
        else:
            if args.output is None:
                print("missing output dir argument")
                parser.print_help()
            else:
                CannyDir(args.dir, args.output)

    except (argparse.ArgumentError or argparse.ArgumentTypeError):
        print("failed to parse arguments")
        parser.print_help()
        raise

