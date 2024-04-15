## 50.021 AI FNC-1
### **Prerequisites:**

- Python 3.x
- Access to a terminal or command-line interface

### **Initial Setup:**

1. **Clone the Repository:**

   - Begin by cloning the project repository from GitHub to your local machine:

   - ```shell
     git clone https://github.com/liangjunyi010/50.021_AI.git
     ```

   - Or, unzip our code folder in our submitted compressed file.


2. **Install Required Packages:**

   - Navigate to the root directory of the project and install the required Python packages using the following command:

   - ```shell
     pip install -r requirements.txt
     ```


3. **Download Additional Resources:**

   - Execute the following command to download the VADER lexicon for sentiment analysis:

   - ```python
     python -m nltk.downloader vader_lexicon
     ```

   - Download the pre-trained Google Word2Vec model from the following link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?pli=1&resourcekey=0-wjGZdNAUop6WykTtMip30g After downloading, extract the `.bin` file and place it in the `google_model` directory within your project.

### **Configuring Paths:**

**Update Model Path in** **predict.py**

Modify the path in `predict.py` to point to the `best_model.joblib` file within your project's model directory. Replace the existing path with your absolute path.

```
model = load('path_to_project_root/model/best_model.joblib')
```

**Update Word2Vec Model Path:**

In the file `word_to_vec_feature_generator.py` within the feature_extractor directory, change the `model_path` parameter in the __init__ method to the absolute path of the `GoogleNews-vectors-negative300.bin` file.

```python
def __init__(self, model_path='path_to_project_root/google_model/GoogleNews-vectors-negative300.bin'):
```

### **Running the Application:**

**Database Migrations:**

Make sure you are in the djangoproject directory and execute the following commands to prepare the database:

```python
python manage.py makemigrations
python manage.py migrate
```

**Start the Server:**

Run the Django development server using the command:

```python
python manage.py runserver 8000
```

**Access the Application:**

Open a web browser and go to http://127.0.0.1:8000/ to view and interact with the GUI.

### **Updating the Model:**

**Re-train the Model (Optional):**

If you wish to re-train the model with new data or tweaks, navigate back to the project root directory (50.021_AI) and execute the following command:

```python
python fnc_kfold.py
```

Wait for the process to complete before attempting to use the updated model.

After following these steps, your News Relationship Classifier should be up and running, ready for you to analyze the relationship between news headlines and their corresponding body text.