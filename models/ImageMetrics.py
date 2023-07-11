import lpips
import numpy
import torch
import json
from skimage.metrics import structural_similarity as ssim

class ImageMetrics:
    @staticmethod
    def __l2__metric__tensor(first_image: torch.Tensor, second_image: torch.Tensor, range=255.):
        return ImageMetrics.__l2_metric__numpy(first_image.numpy(), second_image.numpy(), range)

    @staticmethod
    def __l2_metric__numpy(first_image: numpy.ndarray, second_image: numpy.ndarray, range=255.):
        return lpips.l2(first_image, second_image, range)

    @staticmethod
    def l2_metric(first_image, second_image, range=255.):
        if type(first_image) != type(second_image):
            raise Exception('Images are not of the same type')

        if type(first_image) == numpy.ndarray:
            return ImageMetrics.__l2_metric__numpy(first_image, second_image, range)
        if type(first_image) == torch.Tensor:
            return ImageMetrics.__l2__metric__tensor(first_image, second_image, range)
        raise Exception('Unsupported image type')

    @staticmethod
    def __psnr__metric__tensor(first_image: torch.Tensor, second_image: torch.Tensor, peak=255.):
        return ImageMetrics.__psnr_metric__numpy(first_image.numpy(), second_image.numpy(), peak)

    @staticmethod
    def __psnr_metric__numpy(first_image: numpy.ndarray, second_image: numpy.ndarray, peak=255.):
        return lpips.psnr(first_image, second_image, peak)

    @staticmethod
    def psnr_metric(first_image, second_image, peak=1.):
        if type(first_image) != type(second_image):
            raise Exception('Images are not of the same type')

        if type(first_image) == numpy.ndarray:
            return ImageMetrics.__psnr_metric__numpy(first_image, second_image, peak)
        if type(first_image) == torch.Tensor:
            return ImageMetrics.__psnr__metric__tensor(first_image, second_image, peak)
        raise Exception('Unsupported image type')

    @staticmethod
    def __ssim__metric__tensor(first_image: torch.Tensor, second_image: torch.Tensor, range=255.):
        return ImageMetrics.__ssim_metric__numpy(first_image.numpy(), second_image.numpy(), range)

    @staticmethod
    def __ssim_metric__numpy(first_image: numpy.ndarray, second_image: numpy.ndarray, range=255.):
        return ssim(first_image, second_image, channel_axis=2)

    @staticmethod
    def ssim_metric(first_image, second_image, range=255.):
        if type(first_image) != type(second_image):
            raise Exception('Images are not of the same type')

        if type(first_image) == numpy.ndarray:
            return ImageMetrics.__ssim_metric__numpy(first_image, second_image, range)
        if type(first_image) == torch.Tensor:
            return ImageMetrics.__ssim__metric__tensor(first_image, second_image, range)
        raise Exception('Unsupported image type')

    @staticmethod
    def __lpips__metric__tensor(model: lpips.LPIPS, first_image: torch.Tensor, second_image: torch.Tensor):
        return model(first_image, second_image).detach().numpy().flatten()[0]

    @staticmethod
    def lpips_metric(first_image, second_image, model):
        if type(first_image) != type(second_image):
            raise Exception('Images are not of the same type')

        if type(first_image) == torch.Tensor:
            return ImageMetrics.__lpips__metric__tensor(model, first_image, second_image)
        raise Exception('Unsupported image type')

    @staticmethod
    def metric_export(first_image, second_image, interpolation, lpips_model):
        dictionary = {
            "name": interpolation,
            "psnr": str(ImageMetrics.psnr_metric(first_image, second_image, peak=255.)),
            "l2": str(ImageMetrics.l2_metric(first_image, second_image)),
            "ssim": str(ImageMetrics.ssim_metric(first_image, second_image)),
            "lpips": str(ImageMetrics.lpips_metric(torch.tensor(first_image).permute(2, 0, 1),
                                                   torch.tensor(second_image).permute(2, 0, 1), lpips_model))
        }
        return dictionary

    @staticmethod
    def metric_export_all(metrics, file_name):
        json_object = json.dumps(metrics, indent=4)
        with open(f'{file_name}_metrics.json', "w") as outfile:
            outfile.write(json_object)