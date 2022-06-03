from mrcnn.config import Config


class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "puzzle piece_piece_detector_inference_config"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + puzzle piece

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
