import cv2
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

def detect_face(image_path):
    # Load the image
    image = cv2.imread(str(image_path))  # Convert Path object to string
    
    # Check if the image is empty
    if image is None:
        print("Error: Unable to read image:", image_path)
        return None, None
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No faces found in:", image_path)
        return None, None
    
    # Return only the first face found
    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], (x, y, w, h)  # Corrected the order of dimensions

def find_similar_image(target_face, image_folder):
    max_similarity = -1
    similar_image = None
    similar_face_coords = None
    
    # Iterate through images in the folder
    for image_path in Path(image_folder).glob('*'):
        if image_path.suffix.lower() in ['.jpg', '.png']:
            # Detect face in the current image
            face, face_coords = detect_face(image_path)
            if face is not None:
                # Ensure both faces have the same dimensions
                if target_face.shape == face.shape:
                    # Calculate Structural Similarity Index (SSIM) between the target face and the current face
                    similarity = ssim(target_face, face)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        similar_image = str(image_path)  # Convert Path object to string
                        similar_face_coords = face_coords
                    
    return similar_image, similar_face_coords, max_similarity

if __name__ == "__main__":
    # Path to the folder containing images
    image_folder = r"your folder path here!!!!"  # Update this with the correct path to your image folder
    
    # Path to the target image provided by the user
    target_image_path = input("Enter the path to the target image: ")
    
    # Detect face in the target image
    target_face, target_face_coords = detect_face(target_image_path)
    if target_face is None:
        print("No face detected in the target image.")
    else:
        # Show the detected face from the target image
        target_image = cv2.imread(target_image_path)
        x, y, w, h = target_face_coords
        cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Target Face", target_image)
        
        # Find similar image in the folder
        similar_image, similar_face_coords, similarity = find_similar_image(target_face, image_folder)
        if similar_image is None:
            print("No similar image found in the folder.")
        else:
            # Show the detected face from the similar image
            similar_image = cv2.imread(similar_image)
            x, y, w, h = similar_face_coords
            cv2.rectangle(similar_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Similar Face", similar_image)
            print(f"Similar image found: {similar_image}")
            print(f"Similarity: {similarity}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
