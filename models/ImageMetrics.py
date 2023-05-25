import lpips
import numpy
import torch


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
    def __dssim__metric__tensor(first_image: torch.Tensor, second_image: torch.Tensor, range=255.):
        return ImageMetrics.__dssim_metric__numpy(first_image.numpy(), second_image.numpy(), range)

    @staticmethod
    def __dssim_metric__numpy(first_image: numpy.ndarray, second_image: numpy.ndarray, range=255.):
        return lpips.dssim(first_image, second_image, range)

    @staticmethod
    def dssim_metric(first_image, second_image, range=255.):
        if type(first_image) != type(second_image):
            raise Exception('Images are not of the same type')

        if type(first_image) == numpy.ndarray:
            return ImageMetrics.__dssim_metric__numpy(first_image, second_image, range)
        if type(first_image) == torch.Tensor:
            return ImageMetrics.__dssim__metric__tensor(first_image, second_image, range)
        raise Exception('Unsupported image type')

    @staticmethod
    def __lpips__metric__tensor(model: lpips.LPIPS, first_image: torch.Tensor, second_image: torch.Tensor):
        return model(first_image, second_image).detach().numpy().flatten()[0]

    @staticmethod
    def lpips_metric(first_image, second_image, net='vgg'):
        model = lpips.LPIPS(net=net, verbose=False)
        if type(first_image) != type(second_image):
            raise Exception('Images are not of the same type')

        if type(first_image) == torch.Tensor:
            return ImageMetrics.__lpips__metric__tensor(model, first_image, second_image)
        raise Exception('Unsupported image type')
