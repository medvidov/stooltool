import uuid
import os
import sys
import pdb
from PIL import Image
import re


def attempt_transform_pipeline(image_path, crop, to_greyscale):
    """
    Attempts to transform image by cropping it in the center to 128x128 and make
    image black and white

    Input:
        image_path: path to image file
        crop: if we will crop image
        to_greyscale: if we will turn the image into greyscale

    Output:
        Transformed image of type Image, None if transform unsucessful
    """

    # Applies transformations to image if possible, otherwise prints an exception
    try:
        im = Image.open(image_path)
        if im.size[0] > im.size[1]:
            im = im.rotate(90)
        if crop and not to_greyscale:
            cropped_image = crop_image_from_centerpoint(im, im.size[1], im.size[1])
            return cropped_image
        if to_greyscale and not crop:
            return im.convert("L")
        if to_greyscale and crop:
            cropped_image = crop_image_from_centerpoint(im, im.size[1], im.size[1])
            return cropped_image.convert("L")
        else:
            return im
    except Exception as e:
        print("ERR: %s: %s" % (image_path, e))


def crop_image_from_centerpoint(im, new_height: int, new_width: int):
    """
    https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    Crops the image in the center by new_height and new_width

    Input:
        im: PIL image variable
        new_height: int new height dimension
        new_width: int new width dimension

    Output:
        Cropped image of type Image

    TODO:
        Allow XY as a point instead of requiring it to be perfectly centered
    """

    # If the image would be bigger after being cropped, tells us
    if max(im.size) < new_width:
        raise Exception(
            "ERR: %s: too small to crop. Size is %s" % (image_path, im.size)
        )
    width, height = im.size  # Get dimensions
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop from the center of the image
    new_image = im.crop((left, top, right, bottom))

    return new_image


def save_images_with_uuid(images: list, new_ids: list, output_dir: str = "output"):
    """
    Save out list of images as JPEGs to an output directory

    Input:
        images: list of images of type Image
        new_ids: list of new uuids for images
        output_dir: Optional param of type string. Output directory name

    Output:
        None
    """
    # Make ouputs directory
    os.makedirs(output_dir, exist_ok=True)

    # Save each image
    for i in range(len(images)):
        try:
            image_name = str(new_ids[i])
            image_out_path = os.path.join(output_dir, image_name + ".jpg")
            images[i].save(image_out_path)
        except OSError:
            images[i] = images[i].convert('RGB')
            image_name = str(new_ids[i])
            image_out_path = os.path.join(output_dir, image_name + ".jpg")
            images[i].save(image_out_path)
        except Exception as e:
            print("Save Error: %s" % e)


def save_images_without_uuid(images: list, ids: list, output_dir: str = "output"):
    """
    Save out list of images as JPEGs to an output directory without uuids

    Input:
        images: list of images of type Image
        output_dir: Optional param of type string. Output directory name

    Output:
        None
    """
    # Make ouputs directory
    os.makedirs(output_dir, exist_ok=True)

    # Save images with original names
    for i in range(len(images)):
        id = re.split("\.|\\\\", ids[i])

        try:
            image_name = str(id[1])
            image_out_path = os.path.join(output_dir, image_name + ".jpg")
            images[i].save(image_out_path)
        except Exception as e:
            print("Save Error: %s" % e)


def transform_images_from_dir(image_filenames: list, crop, to_greyscale):
    """
    Gets all image file paths and performs crop and black/white transformations on those images

    Input:
        image_filenames: a list of paths to target images

    Output:
        List of transformed images of type Image
    """
    # Get transformed images and return
    res = [
        x
        for x in [
            attempt_transform_pipeline(i, crop, to_greyscale) for i in image_filenames
        ]
        if x != None
    ]
    return res


def main(
    target_dir: str, output_dir: str = "output", crop=True, to_greyscale=True, UUID=True
):
    """
    Calls the transform function on the directory containing the images and saves all transformed images.

    Input:
        target_dir: string of path to directory containing all images
        bw: apply black and white transformation
        crop: apply cropping transformation
        uuid: apply new UUIDs to images
    Output:
        None
    """
    # Get transformed images
    # Get image paths
    # Make output dir to remove errors

    os.makedirs(output_dir, exist_ok=True)

    image_filenames = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.lower().endswith(".jpeg") or f.lower().endswith(".jpg") or f.lower().endswith(".png")
    ]

    # generate UUID mapping
    new_uuids = [
        uuid.uuid4() for x in image_filenames
    ]  # not all will necessarily be used
    for i in range(len(image_filenames)):
        # print(image_filenames[i])
        os.system(
            "echo '%s,%s'>> output/mappings.csv" % (image_filenames[i], new_uuids[i])
        )

    # Process and transform images
    print("%s total input images" % len(image_filenames))
    results = transform_images_from_dir(image_filenames, crop, to_greyscale)
    print("%s processed images" % len(results))

    # Save images
    if UUID:
        save_images_with_uuid(results, new_uuids, output_dir)
    if not UUID:
        save_images_without_uuid(results, image_filenames, output_dir)


if __name__ == "__main__":
    # If less than 2 arguments, clarifies usage
    if len(sys.argv) < 2:
        print(
            "Usage: python transform_images.py [Images Directory] [-o <Output Directory>] [-c] [-g] [-u]"
        )
        sys.exit(0)

    # Default arguments, takes in source directory only
    if len(sys.argv) == 2:
        main(sys.argv[1])

    # Handles options for cropping, greyscale, UUID, output
    if len(sys.argv) > 2:
        crop = False
        to_greyscale = False
        UUID = False
        output = ""
        curr_arg = 2
        while len(sys.argv) - 1 >= curr_arg:
            if sys.argv[curr_arg] == "-o":
                try:
                    output = sys.argv[curr_arg + 1]
                except Exception as e:
                    print(
                        "Usage: python transform_images.py [Images Directory] [-o <Output Directory>] [-c] [-g] [-u]"
                    )
                    sys.exit(0)
            if sys.argv[curr_arg] == "-c":
                crop = True
            if sys.argv[curr_arg] == "-g":
                to_greyscale = True
            if sys.argv[curr_arg] == "-u":
                UUID = True
            curr_arg = curr_arg + 1
        if output:
            main(sys.argv[1], output, crop, to_greyscale, UUID)
        main(sys.argv[1], crop=crop, to_greyscale=to_greyscale, UUID=UUID)
