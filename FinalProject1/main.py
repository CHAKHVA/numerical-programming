import cv2
from tests.test_edge_detector import test_canny_detector

if __name__ == "__main__":
    # Define parameter sets
    parameter_sets = [
        {
            'name': 'Default',
            'low_threshold': 0.1,
            'high_threshold': 0.3,
            'kernel_size': 5,
            'sigma': 1.4
        },
        {
            'name': 'Fine Detail',
            'low_threshold': 0.05,
            'high_threshold': 0.2,
            'kernel_size': 3,
            'sigma': 1.0
        },
        {
            'name': 'Smooth',
            'low_threshold': 0.2,
            'high_threshold': 0.4,
            'kernel_size': 7,
            'sigma': 2.0
        }
    ]

    # Test each parameter set
    for image_num in range(1, 7):  # for test1.png through test6.png
        image_path = f'images/test{image_num}.png'
        print(f"\nProcessing {image_path}")

        for params in parameter_sets:
            print(f"\nTesting with {params['name']} parameters:")
            print(f"Low threshold: {params['low_threshold']}")
            print(f"High threshold: {params['high_threshold']}")
            print(f"Kernel size: {params['kernel_size']}")
            print(f"Sigma: {params['sigma']}")

            try:
                # Get results
                results = test_canny_detector(image_path, params)

                # Display results
                for name, img in results.items():
                    cv2.imshow(f"{name} - {params['name']}", img)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"Error with {params['name']} parameters: {e}")