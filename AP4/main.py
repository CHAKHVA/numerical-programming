import cv2
import numpy as np
import matplotlib.pyplot as plt


def bezier_curve(p0, p1, p2, p3, num_points=20):
    t = np.linspace(0, 1, num_points)
    points = []

    for t_i in t:
        x = (1 - t_i) ** 3 * p0[0] + \
            3 * (1 - t_i) ** 2 * t_i * p1[0] + \
            3 * (1 - t_i) * t_i ** 2 * p2[0] + \
            t_i ** 3 * p3[0]

        y = (1 - t_i) ** 3 * p0[1] + \
            3 * (1 - t_i) ** 2 * t_i * p1[1] + \
            3 * (1 - t_i) * t_i ** 2 * p2[1] + \
            t_i ** 3 * p3[1]

        points.append((int(x), int(y)))

    return np.array(points)


def create_puzzle_edge(start, end, direction=1, tab_height=0.15):
    start = np.array(start)
    end = np.array(end)

    edge = end - start
    length = np.linalg.norm(edge)
    normal = np.array([-edge[1], edge[0]]) / length

    tab_size = length * tab_height * direction

    if direction != 0:
        # First curve (start to tab peak)
        curve1_p0 = start
        curve1_p1 = start + edge * 0.2
        curve1_p2 = start + edge * 0.3 + normal * tab_size * 0.8
        curve1_p3 = start + edge * 0.5 + normal * tab_size

        points1 = bezier_curve(curve1_p0, curve1_p1, curve1_p2, curve1_p3)

        # Second curve (tab peak to end)
        curve2_p0 = curve1_p3
        curve2_p1 = start + edge * 0.7 + normal * tab_size * 0.8
        curve2_p2 = start + edge * 0.8
        curve2_p3 = end

        points2 = bezier_curve(curve2_p0, curve2_p1, curve2_p2, curve2_p3)

        return np.vstack([points1, points2])
    else:
        return np.array([start, end])


def generate_puzzle_pieces(image, rows=4, cols=6, gap=3):
    height, width = image.shape[:2]
    piece_height = height // rows
    piece_width = width // cols

    # Create output array with white background
    output = np.ones_like(image) * 255

    # Generate edge patterns
    horizontal_edges = np.random.choice([-1, 1], size=(rows - 1, cols))
    vertical_edges = np.random.choice([-1, 1], size=(rows, cols - 1))

    for i in range(rows):
        for j in range(cols):
            # Create mask for the piece
            mask = np.zeros((height, width), dtype=np.uint8)
            points = []

            # Calculate base coordinates with gap
            x1 = j * piece_width
            y1 = i * piece_height
            x2 = (j + 1) * piece_width
            y2 = (i + 1) * piece_height

            # Generate edges with Bezier curves
            # Top edge
            if i == 0:
                points.extend([(x1, y1), (x2, y1)])
            else:
                points.extend(create_puzzle_edge((x1, y1), (x2, y1),
                                                 -horizontal_edges[i - 1, j]))

            # Right edge
            if j == cols - 1:
                points.extend([(x2, y1), (x2, y2)])
            else:
                right_edge = create_puzzle_edge((x2, y1), (x2, y2),
                                                vertical_edges[i, j])
                points.extend(right_edge[1:])

            # Bottom edge
            if i == rows - 1:
                points.extend([(x2, y2), (x1, y2)])
            else:
                bottom_edge = create_puzzle_edge((x2, y2), (x1, y2),
                                                 horizontal_edges[i, j])
                points.extend(bottom_edge[1:])

            # Left edge
            if j == 0:
                points.extend([(x1, y2), (x1, y1)])
            else:
                left_edge = create_puzzle_edge((x1, y2), (x1, y1),
                                               -vertical_edges[i, j - 1])
                points.extend(left_edge[1:])

            # Create and fill mask with gap
            points = np.array(points)
            # Create slightly smaller mask for the gap effect
            kernel = np.ones((gap, gap), np.uint8)
            cv2.fillPoly(mask, [points], 255)
            mask = cv2.erode(mask, kernel, iterations=1)

            # Apply mask to piece
            piece = image.copy()
            piece[mask == 0] = 255  # White background

            # Add piece to output
            output = cv2.bitwise_and(output, piece)

    return output


if __name__ == "__main__":
    image = cv2.imread('ekko.jpg')
    if image is None:
        raise ValueError("Could not load image")

    target_width = 900
    target_height = 600
    image = cv2.resize(image, (target_width, target_height))

    result = generate_puzzle_pieces(image, gap=6)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()