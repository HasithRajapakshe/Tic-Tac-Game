import cv2
import numpy as np

def find_grid(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for square contours (approximate the grid)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            return approx

    return None

def update_game_state(frame, grid, game_state):
    # Get the bounding box of the grid
    x_min = min(point[0][0] for point in grid)
    x_max = max(point[0][0] for point in grid)
    y_min = min(point[0][1] for point in grid)
    y_max = max(point[0][1] for point in grid)

    cell_width = (x_max - x_min) // 3
    cell_height = (y_max - y_min) // 3

    for i in range(3):
        for j in range(3):
            # Define the region
            x_start = x_min + i * cell_width
            x_end = x_start + cell_width
            y_start = y_min + j * cell_height
            y_end = y_start + cell_height

            if x_start < 0 or y_start < 0 or x_end > frame.shape[1] or y_end > frame.shape[0]:
                continue

            region = frame[y_start:y_end, x_start:x_end]

            if region.size == 0:
                print(f"Skipping empty region at ({x_start}, {y_start}, {x_end}, {y_end})")
                continue

            # Simple thresholding to detect X or O
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

            if np.sum(thresh) > 10000:  # Simple threshold to detect a mark
                game_state[j][i] = 'X'  # Assume X for simplicity

    return game_state

def draw_game_state(frame, grid, game_state):
    x_min = min(point[0][0] for point in grid)
    x_max = max(point[0][0] for point in grid)
    y_min = min(point[0][1] for point in grid)
    y_max = max(point[0][1] for point in grid)

    cell_width = (x_max - x_min) // 3
    cell_height = (y_max - y_min) // 3

    for i in range(3):
        for j in range(3):
            x_center = x_min + i * cell_width + cell_width // 2
            y_center = y_min + j * cell_height + cell_height // 2

            if game_state[j][i] == 'X':
                cv2.putText(frame, 'X', (x_center - 10, y_center + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif game_state[j][i] == 'O':
                cv2.putText(frame, 'O', (x_center - 10, y_center + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

game_state = [['' for _ in range(3)] for _ in range(3)]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    grid = find_grid(frame)
    if grid is not None:
        cv2.drawContours(frame, [grid], -1, (0, 255, 0), 2)
        game_state = update_game_state(frame, grid, game_state)
        draw_game_state(frame, grid, game_state)

    cv2.imshow('Tic Tac Toe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
