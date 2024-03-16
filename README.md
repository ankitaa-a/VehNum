# VehNum : License Plate Recognition Web Application

This is a web application for license plate recognition. Users can upload images containing license plates and get the recognized plate text along with the processed image.

## Features

- Upload images containing license plates.
- Recognition of license plate text.
- Display of processed image with bounding box and recognized text.
- Support for both regular and mirror images.

## Technologies Used

- Flask (Python)
- OpenCV (Python)
- Pytesseract (Python)
- HTML/CSS
- JavaScript

## Installation

1. Clone the repository:


2. Install the required dependencies:


3. Run the Flask application:


4. Access the application in your web browser at `http://localhost:5000`.

## Usage

1. Upload an image containing a license plate.
2. Click on the "Submit" button to process the image.
3. View the recognized plate text and processed image with the bounding box.
4. Optionally, upload a mirror image using the "Upload Mirror Image" section and click on the "Submit Mirror" button.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Haarcascade for license plate detection: [source](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml)

