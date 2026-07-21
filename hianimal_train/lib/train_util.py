def reshape_multiview_tensors(image_tensor, calibration_tensor):
    image_tensor = image_tensor.view(
        image_tensor.shape[0] * image_tensor.shape[1], *image_tensor.shape[2:]
    )
    calibration_tensor = calibration_tensor.view(
        calibration_tensor.shape[0] * calibration_tensor.shape[1],
        *calibration_tensor.shape[2:],
    )
    return image_tensor, calibration_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    sample_tensor = sample_tensor.unsqueeze(1).repeat(1, num_views, 1, 1)
    return sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1], *sample_tensor.shape[2:]
    )


def adjust_learning_rate(optimizer, epoch, learning_rate, schedule, gamma):
    if epoch in schedule:
        learning_rate *= gamma
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
    return learning_rate
