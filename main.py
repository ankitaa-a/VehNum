from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image

app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

cascade = cv2.CascadeClassifier("templates/haarcascade_russian_plate_number.xml")

states = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TG": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "LD": "Lakshadweep",
    "PY": "Puducherry"
}

# Initialize counter
counter = 1


def extract_num(imgname):
    global counter
    img = cv2.imread(imgname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in nplate:
        a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
        plate = img[y + a: y + h - a, x + b: x + w - b, :]
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate_gray = cv2.erode(plate, kernel, iterations=1)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]

        try:
            result_text = "Car belongs to " + states[stat]
        except:
            result_text = "Plate recognition failed"

        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save the processed image with counter
        result_image_path = f'static/result{counter}.jpg'
        cv2.imwrite(result_image_path, img)

        # Increment counter
        counter += 1

        return result_text, result_image_path


def is_mirror_image(img_path):
    img = cv2.imread(img_path)
    flipped = cv2.flip(img, 1)
    diff = cv2.subtract(img, flipped)
    b, g, r = cv2.split(diff)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        return True
    return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    global counter

    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = f'static/uploaded_image{counter}.jpg'

    # Check if the image is a mirror image
    clicked_btn_value = request.form.get('clickedBtnValue', '')
    if clicked_btn_value == 'Submit Mirror':
        image = Image.open(file)
        mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        filename = f'static/uploaded_image{counter}_mirror.jpg'
        mirrored_image.save(filename)
        image.close()
    else:
        file.save(filename)

    result_text, result_image_path = extract_num(filename)

    return jsonify({'plate_text': result_text, 'image_path': result_image_path})


if __name__ == '__main__':
    app.run(debug=True)


#------------------------------------------------------------------------------------------
'''
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

cascade= cv2.CascadeClassifier("templates/haarcascade_russian_plate_number.xml")

states = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TG": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "LD": "Lakshadweep",
    "PY": "Puducherry"
}


def extract_num(imgname):
    global read
    img = cv2.imread(imgname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in nplate:
        a,b= (int(0.02*img.shape[0]),int(0.025*img.shape[1]))
        plate = img[y+a : y+h-a , x+b : x+w-b, : ]
        kernel = np.ones((1,1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate_gray = cv2.erode(plate, kernel, iterations=1)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255,cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat= read[0:2]

        try:
            print("Car belong to", states[stat])
        except:
            print("Not recognised")
        print(read)

        cv2.rectangle(img, (x,y), (x+w,y+h), (51,51,255),2)
        #cv2.rectangle(img, (x,y-40),cv2.FONT_HERSHEY_SIMPLEX)
        cv2.rectangle(img, (x, y - 40), (x + w, y), (51, 51, 255), 2)

        cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Plate",plate)

    cv2.imshow("Result", img)
    cv2.imwrite('result.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

extract_num("static/testtt.jpg")



app = flask.Flask(__name__)

harcascade = "templates/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500
count = 0

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x:x + w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)


        cv2.waitKey(500)
        count += 1

@app.route("/")
def home():
    return flask.render_template("index.html")

if __name__ == '__main__':


    app.run()

'''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
