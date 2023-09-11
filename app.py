from flask import Flask, request, render_template, Response
import os, sys, json, datetime
import uuid as uuid
import tensorflow as tf
from inference import StoolToolCNN


# Host address
HOST = '127.0.0.1'
# Path of models to use
MODEL_PATH = 'output/models'
# Log file path
LOG_PATH = 'log.txt'
# predefined different labels in the classifier
LABELS = ['1', '2', '3', '4', '5', '6', '7', 'weird']
# Arrays to set up class to call model inference function
stool_class = []
model_names = []
# Routine Flask setup
app = Flask(__name__)

# Location of downloaded files and files to upload/rate
app.config['DOWNLOAD_FOLDER'] = "saved_images"
app.config['UPLOAD_FOLDER'] = "saved_images"

# Prepares models to be used
def setup_models():
    """
    Looks for models in the models folder and load them into the stool_class array to be used later for inference
 
    Input:
        None

    Outpute:
        None

    """
    # Make saved images direcotry
    os.makedirs('saved_images', exist_ok=True)

    # Load models
    isdir = os.path.isdir(MODEL_PATH)
    if isdir:
        print("Model path exist")
        subfolders = [f.path for f in os.scandir(MODEL_PATH) if f.is_dir()]
        for s in subfolders:
            stool_class.append(StoolToolCNN(s))
            model_names.append(s)
		
        return True
		
    else:
        print("Model path does not exist")
        return False

if not setup_models():
    sys.exit(0)
	
@app.route('/')
def index():
    return render_template('index.html')
# Web endpoint that accepts GET and POST requests
@app.route("/get_rating_web", methods=["GET", "POST"])
def process_image():
    """
    Web interface endpoint that allows the following:

    POST request: returns a HTML page that allows for users to upload an image to be classified and the predicted rating
    of the stool in the image.

    GET request: returns a HTML page that displays the rating probabilities of the previously uploaded image on all the
    models that were found.

    Any requests and relevant information are logged to log.txt

    Input:
        None

    Output:
        HTML pages or error message strings
    """
    out_file = open("log.txt", "a")
    # POST request, so get file, save it if its an image, and return json of rating probabilities
    ct = str(datetime.datetime.now())
    if request.method == "POST":
        # check if an image was selected to upload or an prediction was given
        if not (request.files['image']):
            out_file.write(
                "POST | " + ct + " | Error, no image was selected to upload\n")
            out_file.close()
            return "Please choose an image to upload"
        if 'options' not in request.form:
            out_file.write(
                "POST | " + ct + " | Error, no Bristol Scale prediction was entered\n")
            out_file.close()
            return "Please enter a predicted Bristol scale category"

        # Get predicted rating from user
        given_rating = request.form['options']
        # Get file and file name from request
        file = request.files["image"]
        file_name = file.filename
        uuid_str = str(uuid.uuid4())

        # Check if the file exists or is a .jpg file
        if request.files and file_name.lower().endswith(('.jpeg', '.jpg')):

            # Save image to specified folder with new UUID
            file.save(os.path.join(app.config['DOWNLOAD_FOLDER'], file.filename))
            os.rename(os.path.join(app.config['DOWNLOAD_FOLDER'], file_name),
                      os.path.join(app.config['DOWNLOAD_FOLDER'], uuid_str + ".jpg"))

            # Perform rating with model and img file path
            out_file.write("POST | " + ct + " | " + uuid_str + "\nRatings per model: \n")
            all_ratings = [] 
            for model in stool_class:
                ratings = (tf.gather(model.inference(os.path.join(app.config['DOWNLOAD_FOLDER'], uuid_str + ".jpg")), 0)).numpy()
                # Update log file
                out_file.write(model_names[stool_class.index(model)] + ": ")
                for i in range(0, len(LABELS)):
                    out_file.write(str(ratings[i]) + " ")
                out_file.write("\n")
                all_ratings.append(ratings)
            out_file.write("Given rating: " + given_rating + " | Success\n")
            out_file.close()
            print(len(all_ratings))
            # Return html template of each label and their respective probability scores
            return render_template("output.html", ratings=all_ratings, labels=LABELS, model_names = model_names)

        else:
            # Update log file and save the incorrect file with UUID
            file.save(os.path.join(app.config['DOWNLOAD_FOLDER'], file.filename))
            os.rename(os.path.join(app.config['DOWNLOAD_FOLDER'], file_name), os.path.join(app.config['DOWNLOAD_FOLDER'], uuid_str))
            out_file.write("POST | " + ct + " | " + uuid_str + " | Given rating: " + given_rating + " | Error, invalid file format\n")
            out_file.close()
            # Return error message
            return "Please select a valid file to upload and rate. Only JPG or JPEG currently supported"

    # GET request, so render form to upload file to rate and store
    else:
        # See if any models are loaded
        if len(model_names) > 0 :
            # Update log file
            out_file.write("GET web | " + ct + "\n")
            out_file.close()
            # Return starting html page with names of the models
            return render_template("stooltool.html", models=model_names)
        else:
            # Update log file
            out_file.write("GET web | " + ct + " | Error, no models detected\n")
            out_file.close()
            # Return error because no models were loaded
            return "Error, no models detected"



# Endpoint only works if files are in local directories
@app.route("/healthz", methods=["GET","POST"])
def healthz():
    return Response("OK", status=200)

# Endpoint only works if files are in local directories
@app.route("/get_rating_json/<file_name>&<model_index>", methods=["GET"])
def get_json(file_name, model_index):
    """
    Returns JSON array of probabilities values for each label when given file name of image in the specified upload
    folder and the index of the model that wanted to be used

    Input:
        file_name: File name of image in the specified upload folder
        model_index: Index of model in the models folder (starting from 0)

    Output:
        A error message string
        or
        json: JSON with array of probability values from the desired model
    """
    out_file = open("log.txt", "a")
    ct = str(datetime.datetime.now())

    # Check if file is a valid image file, if not, update log and return error message
    if not file_name.lower().endswith(('.jpg', '.jpeg')):
        out_file.write(
            "POST | " + ct + "| Error, invalid file format\n")
        out_file.close()

        return "Please select a valid file to rate"

    index = int(model_index)
    # Check if given index is valid, if not, update log and return error message
    if index >= len(stool_class):
        out_file.write(
            "POST | " + ct + "| Error, invalid model\n")
        out_file.close()

        return "Please select a valid model to use"

    # Perform rating with model and img file path
    ratings = (tf.gather(stool_class[index].inference(os.path.join(app.config['UPLOAD_FOLDER'], file_name)), 0)).numpy()

    # Format JSON to output
    json_str = '{"probabilities": []}'
    data = json.loads(json_str)
    for i in range(0, len(LABELS)):
        data["probabilities"].append({LABELS[i]: str(ratings[i])})

    # Update log file
    out_file.write("GET json | " + ct + " | ratings: ")
    for i in range(0, len(LABELS)):
        out_file.write(str(ratings[i]) + " ")
    out_file.write("Success\n")
    out_file.close()

# Generic Flask setup
if __name__ == '__main__':
    app.run(host=HOST)

