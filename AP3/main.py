import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_and_resize(image, target_size, method):
    interpolation = cv2.INTER_NEAREST if method == 'nearest' else cv2.INTER_CUBIC
    resized = cv2.resize(image, target_size, interpolation=interpolation)
    return resized


def calculate_error(original, resized):
    original_resized = cv2.resize(original, (resized.shape[1], resized.shape[0]), cv2.INTER_CUBIC)
    orig_f = original_resized.astype(float)
    resized_f = resized.astype(float)
    error_matrix = np.abs(orig_f - resized_f)
    error_2d = error_matrix.reshape(-1, error_matrix.shape[-1])

    return {
        'frobenius': np.linalg.norm(error_2d, 'fro'),
        'infinity': np.linalg.norm(error_2d, np.inf),
        'mean_absolute_error': np.mean(error_matrix)
    }


def zoom_analysis(image, zoom_factors, methods):
    results = {}
    original_size = image.shape[:2]

    for method in methods:
        results[method] = []
        for factor in zoom_factors:
            new_size = (int(original_size[1] * factor), int(original_size[0] * factor))
            resized = load_and_resize(image, new_size, method)
            error = calculate_error(image, resized)
            results[method].append({
                'factor': factor,
                'error': error,
                'resized_image': resized
            })
    return results


bad_image = cv2.imread('images/bad.jpg')
good_image = cv2.imread('images/good.jpg')

if bad_image is None or good_image is None:
    raise ValueError("Could not load one or both images")

initial_size = (512, 512)
bad_image = cv2.resize(bad_image, initial_size)
good_image = cv2.resize(good_image, initial_size)

zoom_factors = [0.5, 1.5, 2.0, 3.0]
methods = ['nearest', 'bicubic']

good_results = zoom_analysis(good_image, zoom_factors, methods)
bad_results = zoom_analysis(bad_image, zoom_factors, methods)

plt.figure(figsize=(20, 20))

plt.subplot(4, 2, 1)
for method in methods:
    good_errors = [r['error']['mean_absolute_error'] for r in good_results[method]]
    plt.plot(zoom_factors, good_errors, '-o', label=f'{method.capitalize()} (Good Image)')
    bad_errors = [r['error']['mean_absolute_error'] for r in bad_results[method]]
    plt.plot(zoom_factors, bad_errors, '--o', label=f'{method.capitalize()} (Bad Image)')

plt.title('Mean Absolute Error vs Zoom Factor')
plt.xlabel('Zoom Factor')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)

plt.subplot(4, 2, 3)
plt.imshow(cv2.cvtColor(good_image, cv2.COLOR_BGR2RGB))
plt.title('Good Image (Original)')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.imshow(cv2.cvtColor(bad_image, cv2.COLOR_BGR2RGB))
plt.title('Bad Image (Original)')
plt.axis('off')

zoom_factor_idx = zoom_factors.index(2.0)
plt.subplot(4, 2, 5)
plt.imshow(cv2.cvtColor(good_results['nearest'][zoom_factor_idx]['resized_image'], cv2.COLOR_BGR2RGB))
plt.title('Good Image - Nearest Neighbor (2x zoom)')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.imshow(cv2.cvtColor(bad_results['nearest'][zoom_factor_idx]['resized_image'], cv2.COLOR_BGR2RGB))
plt.title('Bad Image - Nearest Neighbor (2x zoom)')
plt.axis('off')

plt.subplot(4, 2, 7)
plt.imshow(cv2.cvtColor(good_results['bicubic'][zoom_factor_idx]['resized_image'], cv2.COLOR_BGR2RGB))
plt.title('Good Image - Bicubic (2x zoom)')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.imshow(cv2.cvtColor(bad_results['bicubic'][zoom_factor_idx]['resized_image'], cv2.COLOR_BGR2RGB))
plt.title('Bad Image - Bicubic (2x zoom)')
plt.axis('off')

plt.tight_layout()
plt.show()