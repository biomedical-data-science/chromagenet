import os
import numpy as np
import skimage as ski
from aicsimageio import AICSImage  # re-exported: notebooks call czi_utils.AICSImage


def get_channel_image_and_metadata(czi_img, czi_filepath, channel_idx):
    """
    Extract an image and metadata for a specific channel from a CZI file.

    Args:
        czi_img: The CZI image object.
        czi_filepath (str): The file path of the CZI image.
        channel_idx (int): Index of the channel to extract.

    Returns:
        img (numpy.ndarray): The channel image data.
        metadata_dic (dict): Metadata dictionary containing information about the image.
    """
    # Extract the channel image data
    img = czi_img.get_image_data("ZXY", C=channel_idx, T=0)

    # Extract batch ID and nucleus ID from file paths
    batch_id = os.path.basename(os.path.dirname(czi_filepath))
    nuc_id = os.path.basename(czi_filepath)[:-4]
    nuc_id = nuc_id.replace("Image ", "nuc_")

    # Fix the Dimension object to make it a serialisable dictionary for JSON
    pix_sizes = czi_img.physical_pixel_sizes
    original_res = {"Z": pix_sizes.Z, "X": pix_sizes.X, "Y": pix_sizes.Y}
    original_dims = {k: v for k, v in czi_img.dims.items()}

    metadata_dic = {
        "original_res": original_res,
        "original_dims": original_dims,
        "original_num_channels": czi_img.dims.C,
        "original_channel_names": czi_img.channel_names,
        "original_sigma_noise": ski.restoration.estimate_sigma(img),
        "original_intensity_sum": img.sum(),
        "original_intensity_mean": img.mean(),
        "original_intensity_max": img.max(),
        "original_intensity_min": img.min(),
        "nuc_id": nuc_id,
        "batch_id": batch_id,
    }

    return img, metadata_dic


def load_czi_image(czi_filepath, czi_img, channel_names):
    """
    Load a CZI image file and extract a specific channel along with metadata.

    Args:
        czi_filepath (str): The file path of the CZI image.
        channel_names (list of str): The names of the channel to extract.

    Returns:
        img (numpy.ndarray): The channel image data.
        metadata_dic (dict): Metadata dictionary containing information about the image.
    """

    # Check if the CZI image has channel names information
    if hasattr(czi_img, "get_channel_names"):
        available_channel_names = czi_img.get_channel_names()
    elif hasattr(czi_img, "channel_names"):
        available_channel_names = czi_img.channel_names
    else:
        print(
            "WARNING: Skiping image. No AICSimage compatible channel detected"
        )
        return None, None

    # Find the index of the specified channel
    for target_ch in channel_names:

        matching_ch = [ch for ch in available_channel_names if target_ch in ch]

        if matching_ch:
            break

    if not matching_ch:
        print(f"WARNING, Skiping image. No {channel_names} channels.")
        return None, None

    channel_idx = available_channel_names.index(matching_ch[0])

    # Extract the channel image and metadata using the index
    return get_channel_image_and_metadata(czi_img, czi_filepath, channel_idx)


def from_zxy_to_xyz(array_list):
    """Transpose the image dimensions from ZXY to XYZ coordinates"""
    return [np.transpose(a, axes=[1, 2, 0]) for a in array_list]
