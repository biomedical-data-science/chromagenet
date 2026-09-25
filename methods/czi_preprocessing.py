import os
import gc
import zarr
import random
from tqdm import tqdm
from PIL import Image
from glob import glob
import numpy as np
import pandas as pd
import skimage as ski
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
import multiprocessing as mp
import functools

from scipy.ndimage import binary_fill_holes
from aicsimageio import AICSImage


random.seed(2023)


class CZIPreprocessing:
    """
    A class for preprocessing CZI microscopy image data with customizable parameters.

    Input Parameters:
        input_dir (str): The directory containing input CZI files.
        output_dir (str): The directory where preprocessed data will be saved.
        conditions (list): A list of conditions or samples to process.
        resolution (float): A float representing the desired isotropic resolution for resizing.
        normalization (str): The normalization method to apply (e.g. 'minmax', 'zscore').
        channel (str): The specific channel to process.
        resize (bool): Whether to perform resizing along all dimensions (default: True).
        resize_z (bool): Whether to perform z-axis resizing (default: True).
        outliers (list): A list containing previously detected outlier filepaths (optional).
        override (bool): Whether to save a new version of the file if it already exists.
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        conditions,
        resolution,
        normalization,
        channels,
        resize=True,
        resize_z=True,
        outliers=None,
        override=False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.conditions = conditions
        self.outliers = outliers
        self.res = resolution
        self.norm = normalization
        self.channels = channels
        self.resize = resize
        self.resize_z = resize_z
        self.override = override

    def save_as_3D_array(self, extension="npz"):
        """
        Save CZI images as 3D arrays with optional zip or npz compression.

        Parameters:
            extension (str, optional): Output file extension ('zip' or 'npz'). Default is 'zip'.

        Returns:
            list: List of output directories where the processed data is saved.
        """
        # Create output directory if it does not exist already
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_dir_list = []  # List to store output directories

        # For each condition
        for cond in self.conditions:
            # Create condition folder if it does not exist already
            class_dir = f"{self.output_dir}/{cond}"
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Get a list of image files (CZI format) within the condition directory
            img_list = glob(
                f"{self.input_dir}/{cond}/**/*.czi", recursive=True
            )

            # For each image within this condition
            for i, img_path in enumerate(img_list, 1):
                out_path = self._get_nuc_filename(img_path, class_dir)

                # If this cell was already processed and override mode is not enabled - skip
                if not self.override:
                    if os.path.exists(
                        f"{out_path}_zarr.zip"
                    ) or os.path.exists(f"{out_path}.npz"):
                        continue

                # Skip outliers
                if self.outliers is not None:
                    if img_path in self.outliers:
                        print("Outlier identified - skipping")
                        continue

                # Print information to the user
                print(img_path)
                print("Image Number : " + str(i))

                # Preprocess the CZI image and obtain image data, nucleus mask, and metadata
                img, nuc_mask, metadata = self.czi_image_preprocessing(
                    img_path, plot_path=out_path
                )

                # If the image could not be extracted, skip to the next image
                if img is None:
                    print(
                        "Image could not be extracted from the CZI file - skipping"
                    )
                    continue

                # Update metadata with relevant information
                metadata["czi_path"] = img_path
                metadata["condition"] = cond
                metadata["normalization"] = self.norm
                metadata["resolution"] = self.res
                metadata["resized"] = self.resize
                output_dir_list.extend(out_path)

                # If the chosen extension is 'zip', save data as zarr format
                if extension == "zip":
                    zarr.save(
                        f"{out_path}.{extension}",
                        img=img.astype(np.float32),
                        nuc_mask=nuc_mask.astype(float),
                    )
                    # Save metadata as a CSV file
                    pd.Series(metadata).to_csv(f"{out_path}_metadata.csv")

                # If the chosen extension is 'npz', save data as compressed numpy format
                elif extension == "npz":
                    np.savez_compressed(
                        out_path,
                        img=img.astype(np.float32),
                        nuc_mask=nuc_mask.astype(float),
                        metadata=metadata,
                    )

                gc.collect()  # Perform garbage collection to free up memory

        return output_dir_list


    def _save_as_3D_array_wrapper(self, img_path, class_dir, cond):
        
        out_path = self._get_nuc_filename(img_path, class_dir)
    
        # If this cell was already processed and override mode is not enabled - skip
        if not self.override:
            if os.path.exists(
                f"{out_path}_zarr.zip"
            ) or os.path.exists(f"{out_path}.npz"):
                return
    
        # Skip if the image is an outlier
        if self.outliers is not None and img_path in self.outliers:
            print(f"Outlier identified - skipping {img_path}")
            return
    
        # Preprocess the CZI image and obtain image data, nucleus mask, and metadata
        img, nuc_mask, metadata = self.czi_image_preprocessing(
            img_path, plot_path=out_path
        )
    
        # If the image could not be extracted, skip to the next image
        if img is None:
            print(
                "Image could not be extracted from the CZI file - skipping"
            )
            return
    
        # Update metadata
        metadata.update({
            "czi_path": img_path,
            "condition": cond,
            "normalization": self.norm,
            "resolution": self.res,
            "resized": self.resize
        })
    
        np.savez_compressed(
            out_path,
            img=img.astype(np.float32),
            nuc_mask=nuc_mask.astype(float),
            metadata=metadata,
        )

    
    def save_as_3D_array_multiprocessing(self, n_cores):
        """
        Save CZI images as 3D arrays with optional zip or npz compression.

        Parameters:
            extension (str, optional): Output file extension ('zip' or 'npz'). Default is 'zip'.

        Returns:
            list: List of output directories where the processed data is saved.
        """
        # Create output directory if it does not exist already
        os.makedirs(self.output_dir, exist_ok=True)

        # For each condition
        for cond in self.conditions:
            print(cond)
            # Create condition folder if it does not exist already
            class_dir = f"{self.output_dir}/{cond}"
            os.makedirs(class_dir, exist_ok=True)
            
            # Get a list of image files (CZI format) within the condition directory
            img_list = glob(
                f"{self.input_dir}/{cond}/**/*.czi", recursive=True
            )

            with mp.Pool(processes=n_cores) as pool:
                pool.map(functools.partial(self._save_as_3D_array_wrapper, 
                                           class_dir=class_dir, cond=cond), 
                         img_list)

    
    def save_as_2D_image(self):
        """
        Save CZI images as 2D images in PNG format.

        This function processes CZI images, extracts individual slides, and saves them as 2D PNG images.

        Returns:
            list: List of output directories where the processed data is saved.
        """
        # Create output directory if it does not exist already
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_dir_list = []  # List to store output directories

        # For each condition
        for cond in self.conditions:
            # Create condition folder if it does not exist already
            class_dir = f"{self.output_dir}/{cond}"
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Get a list of image files (CZI format) within the condition directory
            print(f"preprocessing condition {cond}")
            czi_ims = glob(f"{self.input_dir}/{cond}/**/*.czi", recursive=True)

            # For each image within this condition
            for i, czi_path in enumerate(tqdm(czi_ims, unit="ims")):
                # Preprocess the CZI image and obtain image data, nucleus mask, and metadata
                img, nuc_mask, metadata = self.czi_image_preprocessing(
                    czi_path,
                )

                # If the image could not be extracted, skip to the next image
                if img is None:
                    print(
                        "Image could not be extracted from the CZI file - skipping"
                    )
                    continue

                # For each slide of the intensity and nuclear mask
                for s, _ in enumerate(nuc_mask):
                    slide, slide_mask = img[s, :, :], nuc_mask[s, :, :]

                    nuc_mask_ratio = (
                        np.count_nonzero(slide_mask) / slide_mask.size
                    )
                    # If at least 40% of the original slide is covered by the nuclear mask
                    if nuc_mask_ratio >= 0.4:
                        output_path = f"{class_dir}/nuc{i}_slide{s}.png"

                        # Save the slide as a grayscale image in the 0-255 intensity range
                        slide = Image.fromarray(np.uint8(slide * 255), "L")
                        slide.save(output_path)

        return output_dir_list

    def _get_rgb_slide_updown(self, img, mask, s):
        """
        Get RGB image slices from the upper, current, and lower slides.

        This function takes the upper slide, the current slide, and the lower slide from the input
        image and mask, and combines them into a single RGB image with 3 channels.

        Args:
            img (numpy.ndarray): Input image data.
            mask (numpy.ndarray): Input mask data.
            s (int): Index of the current slide.

        Returns:
            numpy.ndarray: RGB image slice with 3 channels.
            numpy.ndarray: RGB mask slice with 3 channels.
        """
        # Take the upper slide, the current slide and the down slide
        rgb_mask = mask[s - 1 : s + 2, :, :]
        rgb_slide = img[s - 1 : s + 2, :, :]

        # Convert the 3 slides to an image of 3 channels
        rgb_slide = rgb_slide.transpose(1, 2, 0)
        rgb_mask = rgb_mask.transpose(1, 2, 0)

        return rgb_slide, rgb_mask

    def _get_rgb_slide_experimental(self, slide):
        """
        Generate a RGB image from a single slide using a combination of different filters.

        This function applies image filters to the input slide, normalizes the results
        within the 0-1 range, and creates an RGB image by stacking the original slide
        and filtered versions.

        Args:
            slide (numpy.ndarray): Input slide data.

        Returns:
            numpy.ndarray: RGB image created from the original slide and filtered versions.
        """
        # Apply two different filters to enhance image features
        filt = self._apply_filter(slide, "sobel")
        filt2 = self._apply_filter(slide, "meijering")

        # Normalize the filtered image to the 0 - 1 interval
        filt *= 1.0 / np.max(filt)
        filt2 *= 1.0 / np.max(filt2)

        # Stack the original slide and the two filtered images into an RGB image
        rgb_slide = np.stack((slide, filt, filt2), axis=-1)

        return rgb_slide

    def _create_img_directories(
        self, output_dir, datasets, conditions, save_mask, class_nums
    ):
        """
        Create directories for storing image data and optional masks.

        Args:
            output_dir (str): The root output directory.
            datasets (list): List of dataset names.
            conditions (list): List of condition names.
            save_mask (bool): Whether to create directories for masks.
            class_nums (list): List of class numbers corresponding to conditions.
        """
        # Create the root output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Else, delete the contents of the folder to avoid duplicates
        #else:
        #    shutil.rmtree(output_dir)

        # Create subdirectories for each dataset
        for ds in datasets:
            if not os.path.exists(f"{output_dir}/{ds}"):
                os.makedirs(f"{output_dir}/{ds}")

            # Create subdirectories for each class within the condition
            for num, cond in zip(class_nums, conditions):
                if not os.path.exists(f"{output_dir}/{ds}/{num}_{cond}"):
                    os.makedirs(f"{output_dir}/{ds}/{num}_{cond}")

        # If saving masks, create corresponding mask directories
        if save_mask:
            output_mask_dir = f"{output_dir}_mask"

            # Create mask directory if it doesn't exist
            if not os.path.exists(output_mask_dir):
                os.makedirs(output_mask_dir)

            # Create subdirectories for each dataset
            for ds in datasets:
                if not os.path.exists(f"{output_mask_dir}/{ds}"):
                    os.makedirs(f"{output_mask_dir}/{ds}")

                # Create subdirectories for each condition
                for num, cond in zip(class_nums, conditions):
                    if not os.path.exists(
                        f"{output_mask_dir}/{ds}/{num}_{cond}"
                    ):
                        os.makedirs(f"{output_mask_dir}/{ds}/{num}_{cond}")

    def npz_to_keras_png(
        self,
        output_img_dir,
        conditions,
        channel_mode,
        slide_plane="XY",
        special_mode=None,
        class_nums=(0, 1),
        datasets=["train", "test"],
        data_splits=[0.9, 0.1],
        min_nuc_ratio=0.4,
        standardize=False,
        fix_size=None,
        save_mask=True,
        seed=2023,
    ):
        """
        Convert .npz image files to Keras-compatible .png format while organizing them
        into specified datasets.

        Args:
            output_img_dir (str): Output directory for saving the images.
            conditions (list): List of condition names.
            channel_mode (str): Color channel mode (e.g., 'L' for grayscale).
            special_mode (str): Special processing mode for images (optional).
            class_nums (tuple): Tuple of class numbers (e.g., (0, 1)).
            datasets (list): List of dataset names (e.g., ["train", "validation", "test"]).
            data_splits (list): List of data split ratios for each dataset.
            min_nuc_ratio (float): Minimum nucleus mask ratio for including an image.
            fix_size (tuple): Tuple specifying the fixed image size (optional).
            save_mask (bool): Whether to save masks alongside images.
            seed (int): Random seed for shuffling images.

        Returns:
            List of output directories where images were saved.
        """
        # Create output directories for images and masks
        self._create_img_directories(
            output_img_dir,
            datasets,
            conditions,
            save_mask,
            class_nums=class_nums,
        )

        # Set a random seed for reproducibility
        random.seed(seed)

        PLANE2INT = {"XY": 0, "ZY": 1, "ZX": 2}

        # For each biological condition
        for num, cond in zip(class_nums, conditions):
            print(f"preprocessing condition {cond}")

            # Find all npz files for the current condition and shuffle them
            npz_ims = glob(f"{self.output_dir}/{cond}/*.npz")
            random.shuffle(npz_ims)

            target_ds = []

            # Assign images to train, validation, or test datasets based on splits
            for ds, split in zip(datasets, data_splits):
                target_ds.extend([ds] * int(len(npz_ims) * split))

            # For each image and target dataset
            for zipped in tqdm(zip(npz_ims, target_ds), unit="ims"):
                # Load image data from the npz file
                npz, ds = zipped
                image = np.load(npz, allow_pickle=True)
                img = image["img"]
                mask = image["nuc_mask"].astype(int)

                # Recover metadata from as a dictionary
                # Somehow I need the [()] index to recover the dictionary
                metadata = image["metadata"][()]
                nuc_id = metadata["nuc_id"]
                batch_id = metadata["batch_id"]

                n_slices = range(mask.shape[PLANE2INT[slide_plane]])
                # For each slide in the image and mask
                for s in n_slices:
                    # Take slide in the XY plane
                    if slide_plane == "XY":
                        slide, slide_mask = img[s, :, :], mask[s, :, :]
                    elif slide_plane == "ZY":
                        slide, slide_mask = img[:, s, :], mask[:, s, :]
                    elif slide_plane == "ZX":
                        slide, slide_mask = img[:, :, s], mask[:, :, s]

                    nuc_mask_ratio = (
                        np.count_nonzero(slide_mask) / slide_mask.size
                    )

                    # If at least min_nuc_ratio% of the original slide is covered by the nuclear mask
                    if nuc_mask_ratio >= min_nuc_ratio:
                        output_path = (
                            f"{output_img_dir}/{ds}/{num}_{cond}/"
                            f"{batch_id}_{nuc_id}_{s}.png"
                        )

                        # Preprocess for the "up and down" RGB scenario
                        if special_mode == "up and down":
                            slide, slide_mask = self._get_rgb_slide_updown(
                                img, mask, s
                            )

                        if standardize:
                            slide = (slide - slide.min()) / (slide.max() - slide.min())
                        
                        slide = slide * slide_mask

                        # If a certain fix_size was given, pad the nucleus and center it
                        if fix_size is not None:
                            dim_diff = np.array(fix_size) - np.array(
                                slide.shape
                            )
                            if any(n < 0 for n in dim_diff):
                                print(
                                    "Dimension is bigger than fixed size - skipping"
                                )
                                continue
                            slide = center_and_pad_nuc(slide, fix_size)
                            slide_mask = center_and_pad_nuc(
                                slide_mask, fix_size
                            )

                        # Preprocess for the "experimental" RGB scenario
                        if special_mode == "experimental":
                            slide = self._get_rgb_slide_experimental(slide)

                        # Save as a .PNG image
                        slide = Image.fromarray(
                            np.uint8(slide * 255), channel_mode
                        )
                        slide.save(output_path)

                        # Save the mask if this option was chosen
                        if save_mask:
                            output_mask_dir = f"{output_img_dir}_mask"
                            mask_path = (
                                f"{output_mask_dir}/{ds}/{num}_{cond}/"
                                f"{batch_id}_{nuc_id}_{s}.png"
                            )
                            slide_mask = Image.fromarray(
                                np.uint8(slide_mask), channel_mode
                            )
                            slide_mask.save(mask_path)

    def _get_channel_image_and_metadata(
        self, czi_img, czi_filepath, channel_idx
    ):
        """
        Extract an image and metadata for a specific channel from a CZI file.

        Args:
            self: The instance of the class.
            czi_img: The CZI image object.
            czi_filepath (str): The file path of the CZI image.
            channel_idx (int): Index of the channel to extract.

        Returns:
            img (numpy.ndarray): The channel image data.
            metadata_dic (dict): Metadata dictionary containing information about the image.
        """
        # Extract the channel image data
        img = czi_img.get_image_data("ZXY", C=channel_idx)

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

    def _load_czi_image(self, czi_filepath):
        """
        Load a CZI image file and extract a specific channel along with metadata.

        Args:
            czi_filepath (str): The file path of the CZI image.

        Returns:
            img (numpy.ndarray): The channel image data.
            metadata_dic (dict): Metadata dictionary containing information about the image.
        """
        # Load the CZI image using AICSImage
        czi_img = AICSImage(czi_filepath)

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
        for target_ch in self.channels:

            matching_ch = [
                ch for ch in available_channel_names if target_ch in ch
            ]

            if matching_ch:
                break

        if not matching_ch:
            print(f"WARNING, Skiping image. No {self.channels} channels.")
            return None, None

        channel_idx = available_channel_names.index(matching_ch[0])

        # Extract the channel image and metadata using the index
        return self._get_channel_image_and_metadata(
            czi_img, czi_filepath, channel_idx
        )

    def _resize_image(self, img, pixel_sizes):
        """
        Resize an image while preserving its aspect ratio, achieving the same resolution
        for each of the X, Y and Z dimensions.

        Args:
            self: The instance of the class.
            img (numpy.ndarray): The input image to be resized.
            pixel_sizes (dict): Dictionary containing pixel sizes for Z, X, and Y axes.

        Returns:
            img (numpy.ndarray): The resized image.
        """
        # Calculate the new shape based on desired pixel sizes and current resolution
        new_shape = (
            np.round(img.shape[0] * pixel_sizes["Z"] / self.res),
            np.round(img.shape[1] * pixel_sizes["X"] / self.res),
            np.round(img.shape[2] * pixel_sizes["Y"] / self.res),
        )

        # Check that the number of pixels is the same in X and Y
        for i, _ in enumerate(new_shape[1:], 1):
            if new_shape[i] > img.shape[i]:
                print("Skipping due to different X and Y resolution")
                return None

        # Resize the image using scikit-image, preserving the image range and applying anti-aliasing
        # About anti-aliasing:
        # Whether to apply a Gaussian filter to smooth the image prior to
        # downsampling. It is crucial to filter when downsampling the image to
        # avoid aliasing artifacts. If not specified, it is set to True when
        # downsampling an image whose data type is not bool.
        if not self.resize_z:
            new_shape[0] = img.shape[0]

        img = ski.transform.resize(
            img,
            new_shape,
            preserve_range=True,
            anti_aliasing=True,
        )

        print("Rescaling image. New Image Size: ", img.shape)
        return img

    def _zscore_normalization(self, img, mask):
        """Perform Z-score normalization (Standardization) in the
        image region inside the supplied nuclear mask"""
        roi = img[mask.astype(bool)]
        img[mask.astype(bool)] = (img[mask.astype(bool)] - roi.mean()) / roi.std()

        return img

    def _mean_normalization(self, img, mask):
        """Perform Mean normalization in the image region inside
        the supplied nuclear mask"""
        roi = img[mask.astype(bool)]
        img -= roi.mean()
        return img

    def _pad_center(self, array_list, padding=(3, 3)):
        """
        Pad each array in array_list with zeros to center it within a specified padding.

        Args:
            array_list (list of numpy.ndarray): List of arrays to be padded.
            padding (tuple): Number of pixels to pad for each dimension.

        Returns:
            list of numpy.ndarray: List of padded arrays.
        """
        # Determine the number of dimensions in the arrays
        n_dims = len(array_list[0].shape)

        # Create a pad width list for each dimension using the specified padding
        pad_width = [padding for n in range(n_dims)]

        # Pad each array in array_list with zeros using the specified padding
        return [np.pad(a, pad_width=pad_width) for a in array_list]

    def _from_zxy_to_xyz(self, array_list):
        """Transpose the image dimensions from ZXY to XYZ coordinates"""
        return [np.transpose(a, axes=[1, 2, 0]) for a in array_list]

    def _trim_zeros(self, img, mask):
        """Returns a trimmed view of an n-D array excluding regions containing
        only zeros"""

        # Find non-zero indices in the mask for each dimension
        non_zero = np.nonzero(mask)

        # Create slices that define the trimmed region
        slices = tuple(slice(idx.min(), idx.max() + 1) for idx in non_zero)

        # Return the trimmed image and mask using the defined slices
        return img[slices], mask[slices]

    def _get_nucleus_mask(self, img, tolerance=0.2):
        """Batch calling different segmentation steps
        INPUT:
            img: image numpy 3D array
            res: image resolution
        """
        sigma = 0.1 / self.res

        smooth_img = ski.filters.gaussian(img, sigma=sigma)
        otsu_thres = ski.filters.threshold_otsu(smooth_img)
        lower_thres = otsu_thres * (1 - tolerance)
        upper_thres = otsu_thres * (1 + tolerance)

        # applies histerisis thresholding of TOLERANCE
        mask = ski.filters.apply_hysteresis_threshold(
            smooth_img, lower_thres, upper_thres
        )

        # fills holes
        mask = ski.morphology.binary_closing(mask, ski.morphology.ball(5))
        mask = binary_fill_holes(mask, ski.morphology.ball(5))

        return mask

    def _plot_seg(self, img, mask, plane, out_path, contour=True, plot=False):
        """
        Plots an input image with contoured segmentation in a matrix of subplots
        with different slices in the indicated plane.

        Args:
            img (numpy.ndarray): Input image as a 3D numpy array.
            mask (numpy.ndarray): Segmented image as a binary 3D mask.
            plane (str): Plane to plot/slice (e.g., 'XY', 'ZX', 'ZY').
            out_path (str): Filename for the resulting image.
            contour (bool): Whether to plot the contour of the nucleus mask.
            plot (bool): Whether to plot the image on the terminal.

        Returns:
            None
        """
        # Constants for subplot layout
        n_bs = 3  # Number of rows per subplot
        n_total = 15  # Total number of slices
        n_cols = int(np.ceil(n_total / n_bs))  # Number of columns
        fig, axes = plt.subplots(3, n_cols, figsize=(15, 10))

        # For each of the created axes and 2D slides, plot depending on the chosen plane
        for i, ax in enumerate(axes.flat):
            if plane == "XY":
                slice_img = img[
                    i * int(np.floor(img.shape[0] / n_total)), :, :
                ]
                slice_mask = mask[
                    i * int(np.floor(img.shape[0] / n_total)), :, :
                ]
            elif plane == "ZX":
                slice_img = img[
                    :, :, i * int(np.floor(img.shape[2] / n_total))
                ]
                slice_mask = mask[
                    :, :, i * int(np.floor(img.shape[2] / n_total))
                ]
            elif plane == "ZY":
                slice_img = img[
                    :, i * int(np.floor(img.shape[1] / n_total)), :
                ]
                slice_mask = mask[
                    :, i * int(np.floor(mask.shape[1] / n_total)), :
                ]

            # Display the image and optional contour
            ax.imshow(slice_img, vmax=np.max(img.flatten()) * 0.7)
            if contour:
                ax.contour(slice_mask)
            ax.set_title(f"PLANE : {plane}")

        # Save the resulting image and close
        plt.savefig(out_path)
        plt.close()

        # Optionally, display the plot on the terminal
        if plot:
            plt.show()

    def _get_nuc_filename(self, inp, output, suffix="XY", ext=".png"):
        """
        Generates an output filename for .png inspection images based on input.

        Args:
            input (str): Input filename of .czi image.
            output (str): Working output folder.
            suffix (str): Suffix to be added before the file extension.
            ext (str): File extension for the output filename.

        Returns:
            str: The generated output filename.
        """

        # Extract batch ID and nucleus ID from input path
        batch_id = os.path.basename(os.path.dirname(inp))
        nuc_id = os.path.basename(inp)[:-4]
        nuc_id = nuc_id.replace("Image ", "nuc_")

        # Create the filename based on batch and nucleus ID
        filename = f"{output}/{batch_id}_{nuc_id}"

        return filename

    def czi_image_preprocessing(self, czi_path, plot_path=None):
        """
        Preprocesses a CZI image, including denoising, resizing, segmentation,
        normalization, and optional plotting.

        Args:
            self: The instance of the class.
            czi_path (str): Path to the .czi image file.
            plot_path (str): Path for saving optional plots (if provided).

        Returns:
            img (ndarray): Preprocessed image data.
            nuc_mask (ndarray): Segmented nucleus mask.
            metadata_dic (dict): Dictionary containing image metadata.
        """
        # Load CZI image and metadata
        czi_img, metadata_dic = self._load_czi_image(czi_path)

        # Check if image loading was successful
        if not isinstance(czi_img, np.ndarray):
            return None, None, None

        # Denoise using the total variation in the 3D image
        img = ski.restoration.denoise_tv_chambolle(czi_img, weight=0.01)

        # Resize image with isotropic interpolation
        if self.resize:
            img = self._resize_image(img, metadata_dic["original_res"])
            if img is None:
                return None, None, None

        # Get segmentation mask of the nucleus
        nuc_mask = self._get_nucleus_mask(img)
        # Make sure it's a boolean matrix
        nuc_mask = nuc_mask.astype(bool)

        # Check if a valid mask was obtained
        if len(np.unique(nuc_mask)) == 1:
            print("No mask found - skipping")
            return None, None, None

        # Remove zero dimensions from the image and mask
        img, nuc_mask = self._trim_zeros(img, nuc_mask)

        # Pad the image and mask to a desired shape (centering the object)
        img, nuc_mask = self._pad_center([img, nuc_mask])

        # TV-chambolle and resizing make the intensity values very small.
        # Rescale intensity values to the 0-1 interval
        # min_int = np.min(img[nuc_mask])
        # max_int = np.max(img[nuc_mask])
        # img[nuc_mask] = (img[nuc_mask] - min_int) / (max_int - min_int)

        # Eliminate residual intensities outside the nucleus mask
        img = img * nuc_mask

        # Normalize the image using different methods (z-score or mean)
        if self.norm == "z_score":
            img = self._zscore_normalization(img, mask=nuc_mask)
        elif self.norm == "mean":
            img = self._mean_normalization(img, mask=nuc_mask)

        print(img.shape)

        # Optional plotting of segmented images
        if plot_path:
            for suffix in ["XY", "ZY"]:
                out_path = f"{plot_path}_{suffix}.png"
                self._plot_seg(img, nuc_mask, suffix, out_path, contour=True)

        return img, nuc_mask, metadata_dic

    def _apply_filter(self, img, filter="None"):
        """Apply image filtering techniques to enhance features or edges
        in the image"""
        sigmas = range(1, 5, 1)

        if filter == "roberts":
            img = ski.filters.roberts(img)
        elif filter == "sobel":
            img = ski.filters.sobel(img)
        elif filter == "scharr":
            img = ski.filters.scharr(img)
        elif filter == "gabor":
            gabor_kernel = ski.filters.gabor_kernel(frequency=0.8)
            img = ndi.convolve(
                img, np.imag(gabor_kernel), mode="reflect", cval=0
            )
        elif filter == "sato":
            img = ski.filters.sato(img, sigmas=sigmas)
        elif filter == "meijering":
            img = ski.filters.meijering(img, sigmas=sigmas)
        elif filter == "frangi":
            img = ski.filters.frangi(img, sigmas=sigmas)
        elif filter == "farid":
            img = ski.filters.farid(img)

        return img


def center_and_pad_nuc(img, goal_size):
    """
    Center and pad a Numpy array to achieve a target size.

    This function takes a Numpy array 'img' and adjusts its size to match the
    specified 'goal_size'. It centers the input array and pads it with zeros
    as needed to reach the desired dimensions.

    Args:
        img (numpy.ndarray): The input Numpy array to be centered and padded.
        goal_size (tuple of int): A tuple specifying the target size (shape)
            that the 'img' should be adjusted to.

    Returns:
        numpy.ndarray: The centered and padded Numpy array.
    """
    # Calculate the differences in dimensions
    dim_diffs = np.array(goal_size) - np.array(img.shape)

    # Create a list of tuples specifying the number of zeros to add before
    # and after the nucleus in each dimension, ensuring even distribution
    pad_width = [(d // 2, (d // 2) + (d % 2)) for d in dim_diffs]

    # Pad the 'img' array with zeros based on 'pad_width'
    return np.pad(img, pad_width, mode="constant", constant_values=0)