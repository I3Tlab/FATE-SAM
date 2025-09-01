import argparse
from predictor_utils import *
from notebooks.dataset_loader import ImageLabelLoader

def run_fate_sam_prediction(
    test_image_path,
    test_label_path,
    support_images_path,
    support_labels_path,
    num_classes
):
    loader = ImageLabelLoader(support_images_path, support_labels_path)
    support_images, support_labels = load_support_data_from_loader(loader)

    dice_df, seg_predictions = run_single_image_inference(
        query_image_path=test_image_path,
        query_label_path=test_label_path,
        support_images=support_images,
        support_labels=support_labels,
        num_classes=num_classes
    )
    
    return dice_df, seg_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FATE-SAM prediction for a test image.")
    parser.add_argument('--test_image_path', type=str, required=True, help='Path to the test image')
    parser.add_argument('--test_label_path', type=str, required=True, help='Path to the test label')
    parser.add_argument('--support_images_path', type=str, required=True, help='Path to the support images')
    parser.add_argument('--support_labels_path', type=str, required=True, help='Path to the support labels')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes (default: 8)')

    args = parser.parse_args()

    run_fate_sam_prediction(
        test_image_path=args.test_image_path,
        test_label_path=args.test_label_path,
        support_images_path=args.support_images_path,
        support_labels_path=args.support_labels_path,
        num_classes=args.num_classes
    )