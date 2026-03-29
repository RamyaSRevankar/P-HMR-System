from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import random
import os




#create flask
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback-secret")  # Change this in production

#create database
DB_NAME = 'database.db'

#create database function and create table
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    # DOCTORS LOGIN
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors_login (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        doctor_name TEXT NOT NULL,
        specialization TEXT
    )
    """)

    # CHAT
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender TEXT,
        receiver TEXT,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # DOCTORS
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        specialization TEXT,
        disease TEXT,
        email TEXT
    )
    """)

    # APPOINTMENTS
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctor_appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        disease TEXT,
        doctor TEXT,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ADMIN
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    # INSERT DEFAULT ADMIN
    cursor.execute("SELECT * FROM admin WHERE id=1")
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO admin (id, username, password) VALUES (1, 'admin', 'admin123')")

    # PREDICTIONS
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symptoms TEXT,
        predicted_disease TEXT,
        stage TEXT,
        confidence REAL,
        date TEXT
    )
    """)

    # CHATBOT LOGS
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chatbot_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        message TEXT,
        reply TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

# Connect to database
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)  # ✅ Correct usage
    conn.row_factory = sqlite3.Row
    return conn


# doctor_seeder.py (run once)

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

doctors = [
    ('Dr. Ravi Kumar', 'Dermatologist', 'Fungal infection', 'ravi@example.com'),
    ('Dr. Nisha Reddy', 'Gastroenterologist', 'GERD', 'nisha@example.com'),
    ('Dr. Sameer Khan', 'General Physician', 'Common Cold', 'sameer@example.com'),
]



conn.commit()
conn.close()



# load databasedataset===================================
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")
doctors = pd.read_csv("datasets/doctors.csv")


# load model===========================================
svc = pickle.load(open('models/svc.pkl','rb'))

symptom_severity={
'itching':1,'skin_rash':3,'nodal_skin_eruptions':4,'continuous_sneezing':4,'shivering':5,'chills':3,'joint_pain':3,'stomach_pain':5,'acidit':3,'ulcers_on_tongue':4,'muscle_wasting':3,'vomiting':5,'burning_micturition':6,'spotting_urination':6,'fatigue':4,'weight_gain':3,'anxiety':4,'cold_hands_and_feets':5,'mood_swings':3,'weight_loss':3,'restlessness':5,'lethargy':2,'patches_in_throat':6,'irregular_sugar_level':5,'cough':4,'high_fever':7,'sunken_eyes':3,'breathlessness':4,'sweating':3,'dehydration':4,'indigestion':5,'headache':3,'yellowish_skin':3,'dark_urine':4,'nausea':5,'loss_of_appetite':4,'pain_behind_the_eyes':4,'back_pain':3,'constipation':4,'abdominal_pain':4,'diarrhoea':6,'mild_fever':5,'yellow_urine':4,'yellowing_of_eyes':4,'acute_liver_failure':6,'fluid_overload':6,'swelling_of_stomach':7,'swelled_lymph_nodes':6,'malaise':6,'blurred_and_distorted_vision':5,'phlegm':5,'throat_irritation':4,'redness_of_eyes':5,'sinus_pressure':4,'runny_nose':5,'congestion':5,'chest_pain':7,'weakness_in_limbs':7,'fast_heart_rate':5,'pain_during_bowel_movements':5,'pain_in_anal_region':6,'bloody_stool':5,'irritation_in_anus':6,'neck_pain':5,'dizziness':4,'cramps':4,'bruising':4,'obesity':4,'swollen_legs':5,'swollen_blood_vessels':5,'puffy_face_and_eyes':5,'enlarged_thyroid':6,'brittle_nails':5,'swollen_extremeties':5,'excessive_hunger':4,'extra_marital_contacts':5,'drying_and_tingling_lips':4,'slurred_speech':4,'knee_pain':3,'hip_joint_pain':2,'muscle_weakness':2,'stiff_neck':4,'swelling_joints':5,'movement_stiffness':5,'spinning_movements':6,'loss_of_balance':4,'unsteadiness':4,'weakness_of_one_body_side':4,'loss_of_smell':3,'bladder_discomfort':4,'foul_smell_ofurine':5,'continuous_feel_of_urine':6,'passage_of_gases':5,'internal_itching':4,'toxic_look_(typhos)':5,'depression':3,'irritability':2,'muscle_pain':2,'altered_sensorium':2,'red_spots_over_body':3,'belly_pain':4,'abnormal_menstruation':6,'dischromic_patches':6,'watering_from_eyes':4,'increased_appetite':5,'polyuria':4,'family_history':5,'mucoid_sputum':4,'rusty_sputum':4,'lack_of_concentration':3,'visual_disturbances':3,'receiving_blood_transfusion':5,'receiving_unsterile_injections':2,'coma':7,'stomach_bleeding':6,'distention_of_abdomen':4,'history_of_alcohol_consumption':5,'blood_in_sputum':5,'prominent_veins_on_calf':6,'palpitations':4,'painful_walking':2,'pus_filled_pimples':2,'blackheads':2,'scurring':2,'skin_peeling':3,'silver_like_dusting':2,'small_dents_in_nails':2,'inflammatory_nails':2,'blister':4,'red_sore_around_nose':2,'yellow_crust_ooze':3,'prognosis':5,
}


#helper funtions================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]
    wrkout = workout[workout['disease'] == dis] ['workout']
    return desc,pre,med,die,wrkout

#list of symptoms and sisease
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
            prediction=svc.predict([input_vector])[0]

            predicted_disease=diseases_list[prediction]
            symptom_count=len(patient_symptoms)
            if symptom_count <= 3:
                stage="stage 1 - mild"
                progression="May recover with minimal treatment. Monitor symptoms."
            elif 4<=symptom_count<=6:
                stage="stage 2- moderate"
                progression = "Seek medical consultation. Risk of complications."
            else:
                stage="stage 3- severe"
                progression = "High risk. Immediate medical attention advised."
            try:
                probabilities = svc.predict_proba([input_vector])[0]
                confidence = round(np.max(probabilities) * 100,2)
            except AttributeError:
                confidence = None
            return predicted_disease,stage,confidence,progression

        return None
    return None

#home page
@app.route('/')
def home():
    return render_template('index.html')

#prediction page
@app.route("/prediction")
def index():
    return render_template("prediction.html", symptoms_dict=symptoms_dict)

# predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        symptoms = request.form.get('symptoms')

        if not symptoms or symptoms == "Symptoms":
            return render_template('prediction.html', message="Please enter valid symptoms")

        try:
            # Convert input into list
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

            # 🔹 Prediction (SAFE)
            result = get_predicted_value(user_symptoms)

            if result is None:
                return render_template('prediction.html', message="Invalid symptoms entered")

            predicted_disease, stage, confidence, progression = result

            # 🔹 Helper (SAFE)
            try:
                dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            except Exception as e:
                print("Helper Error:", e)
                dis_des, precautions, medications, rec_diet, workout = "", [], [], [], []

            # 🔹 Precautions safe handling
            my_precautions = []
            if precautions and len(precautions) > 0:
                for i in precautions[0]:
                    my_precautions.append(i)

            # 🔹 Date
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 🔹 User ID safe
            user_id = session.get('user_id', None)

            # 🔹 DB Insert (SAFE)
            try:
                conn = sqlite3.connect('database.db')
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO predictions 
                    (user_id, symptoms, predicted_disease, stage, date) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, symptoms, predicted_disease, stage, current_date))

                conn.commit()
                conn.close()

            except Exception as db_error:
                print("DB Error:", db_error)

            # 🔹 Return result
            return render_template(
                'prediction.html',
                predicted_disease=predicted_disease,
                stage=stage,
                confidence=confidence,
                dis_des=dis_des,
                progression=progression,
                my_precautions=my_precautions,
                medications=medications,
                my_diet=rec_diet,
                workout=workout
            )

        except Exception as e:
            print("Prediction Error:", e)
            return render_template('prediction.html', message=f"Error occurred: {str(e)}")

    return render_template('prediction.html')


# about view funtion
@app.route('/about')
def about():
    return render_template("about.html")

# contact view funtion and path
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        # Save or send message logic here
        flash("Thank you for contacting us! We'll get back to you soon.")
        return redirect('/contact')
    return render_template('contact.html')


# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("blog.html")

#chat view function
@app.route("/chat", methods=["POST"])
def chatbot():
    user_msg = request.json.get("message", "").lower()
    # Simple rule-based replies
    if "fever" in user_msg:
        reply = "For fever, rest, hydration, and paracetamol are usually recommended."
    elif "headache" in user_msg:
        reply = "Try resting in a quiet room and drink water."
    else:
        reply = "I'm a simple medical bot. Please ask about symptoms or health tips."

    # Get user ID if logged in
    user_id = session.get('user_id', None)

    # Log to DB
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO chatbot_logs (user_id, message, reply, timestamp) VALUES (?, ?, ?, ?)",
                   (user_id, user_msg, reply, timestamp))
    conn.commit()
    conn.close()

    return jsonify({"reply": reply})


#admin chatlog view function
@app.route('/admin_chatlogs')
def admin_chatlogs():
    if not session.get('admin_logged_in'):
        return redirect('/admin_login')

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.username, c.message, c.reply, c.timestamp 
        FROM chatbot_logs c 
        LEFT JOIN users u ON c.user_id = u.id 
        ORDER BY c.timestamp DESC
    ''')
    logs = cursor.fetchall()
    conn.close()

    return render_template('admin_chatlogs.html', logs=logs)



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                           (username, email, password))
            conn.commit()
            flash("Registration successful. You can now log in.")
            return redirect(url_for('user_login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.")
        finally:
            conn.close()

    return render_template('register.html')





#user login
@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]  # ✅ store user ID
            session['user'] = user[1]  # (optional) store username
            return redirect(url_for('dashboard_user'))

        else:
            flash("Invalid credentials for user.")
    return render_template('user_login.html')




#dashboard user
@app.route('/dashboard_user')
def dashboard_user():
    if 'user' in session:
        return render_template('dashboard_user.html', username=session['user'])
    return redirect(url_for('user_login'))

#my prediction function in user
@app.route('/my_predictions')
def my_predictions():
    if 'user_id' not in session:
        return redirect('/user_login')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY date DESC", (session['user_id'],))
    predictions = cursor.fetchall()
    conn.close()

    return render_template('my_predictions.html', predictions=predictions)


#doctor book function
@app.route('/book_doctor/<disease>', methods=['GET', 'POST'])
def book_doctor(disease):
    doctors_df = pd.read_csv("datasets/doctors.csv")
    filtered_doctors=doctors_df[doctors_df['Specialization'].str.lower().str.contains(disease.lower())]
    

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        doctor = request.form['doctor']

        appointment_data=pd.DataFrame([{
            'patient name':name,
            'patient email':email,
            'doctor':doctor,
            'disease':disease
        }])
        appointment_data.to_csv('appointments.csv',mode='a',index=False,header=False)
        success=f"Appointments booked successfully with Dr.{doctor}!"

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO doctor_appointments (name, email, disease, doctor) VALUES (?, ?, ?, ?)",
                       (name, email, disease, doctor))
        conn.commit()
        conn.close()

        return render_template("book_doctor.html", disease=disease, doctors=filtered_doctors, success=success)

    return render_template("book_doctor.html", disease=disease, doctors=doctors_df)


#my appointments of user side
@app.route('/my_appointments')
def my_appointments():
    if 'user' not in session:
        return redirect(url_for('user_login'))

    username = session['user']

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT disease, doctor_name, booked_at
        FROM doctor_appointments
        WHERE user_name = ?
        ORDER BY booked_at DESC
    """, (username,))
    appointments = cursor.fetchall()
    conn.close()

    return render_template('my_appointments.html', appointments=appointments)


#admin login
ADMIN_CREDENTIALS = {
    "username": "admin",
    "password": "admin123"
}

#admin login
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect('/dashboard_admin')
        else:
            return render_template('admin_login.html', error="Invalid admin credentials")

    return render_template('admin_login.html')



#admin dashboard
@app.route('/dashboard_admin')
def dashboard_admin():
    if 'admin_logged_in' not in session:
        return redirect('/admin_login')

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Total users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    # Total predictions
    c.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = c.fetchone()[0]

    # High-risk predictions: stage 3
    c.execute("SELECT COUNT(*) FROM predictions WHERE stage LIKE '%severe%'")
    high_risk_count = c.fetchone()[0]

    conn.close()

    return render_template('dashboard_admin.html',
                           total_users=total_users,
                           total_predictions=total_predictions,
                           high_risk_count=high_risk_count)



#admin predictions
@app.route('/admin_predictions')
def admin_predictions():
    if not session.get('admin_logged_in'):
        return redirect('/admin_login')

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "SELECT u.username, p.symptoms, p.predicted_disease, p.stage, p.date FROM predictions p JOIN users u ON p.user_id = u.id ORDER BY p.date DESC")
    rows = c.fetchall()
    conn.close()

    return render_template('admin_predictions.html', predictions=rows)


#admin panel function
@app.route('/admin_panel')
def admin_panel():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Get all user details
    cursor.execute("SELECT id, username, email FROM users")
    users = cursor.fetchall()

    conn.close()
    return render_template('admin_panel.html', users=users)




#admin charts function
@app.route('/admin_charts')
def admin_charts():
    if not session.get('admin_logged_in'):
        return redirect('/admin_login')

    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT predicted_disease, COUNT(*) as count FROM predictions GROUP BY predicted_disease")
    disease_data = c.fetchall()

    c.execute("SELECT stage, COUNT(*) as count FROM predictions GROUP BY stage")
    stage_data = c.fetchall()

    c.execute("SELECT substr(date, 1, 10) as day, COUNT(*) as count FROM predictions GROUP BY day ORDER BY day DESC LIMIT 7")
    date_data = c.fetchall()

    conn.close()

    return render_template('admin_charts.html',
                           disease_data=disease_data,
                           stage_data=stage_data,
                           date_data=date_data)


#admin appointments
@app.route('/admin_appointments')
def admin_appointments():
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, email, doctor, disease, date
        FROM doctor_appointments
        ORDER BY date DESC
    """)

    appointments = cursor.fetchall()
    conn.close()

    return render_template('admin_appointments.html', appointments=appointments)



#profile dunction
@app.route('/profile')
def profile():
    user = get_db_connection()  # Replace with session or db logic
    return render_template('profile.html', user=user)


#edit profile function
@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    user = get_db_connection()
    if request.method == 'POST':
        # Update user logic
        # Save to DB
        return redirect('/profile')
    return render_template('edit_profile.html', user=user)




#logout page
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/doctor_login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM doctors_login WHERE username = ? AND password = ?", (username, password))
        doctor = cursor.fetchone()
        conn.close()

        if doctor:
            session['doctor_logged_in'] = True
            session['doctor_name'] = doctor['doctor_name']  # doctor must be a dict-like object
            return redirect(url_for('doctor_dashboard', doctor_name=doctor['doctor_name']))  # from loop or data

        else:
            flash("Invalid doctor credentials.")
            return render_template('doctor_login.html')

    return render_template('doctor_login.html')


@app.route('/doctor_dashboard/<doctor_name>')
def doctor_dashboard(doctor_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, email, disease, date FROM doctor_appointments WHERE doctor = ?", (doctor_name,))

    appointments = cursor.fetchall()
    conn.close()
    return render_template("doctor_dashboard.html", doctor_name=doctor_name, appointments=appointments)





@app.route('/doctor_logout')
def doctor_logout():
    session.pop('doctor_logged_in', None)
    session.pop('doctor_name', None)
    return redirect(url_for('doctor_login'))







@app.route('/doctor_register', methods=['GET', 'POST'])
def doctor_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        doctor_name = request.form['doctor_name']
        specialization = request.form['specialization']

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO doctors_login (username, password, doctor_name, specialization) VALUES (?, ?, ?, ?)",
                           (username, password, doctor_name, specialization))
            conn.commit()
            flash("Doctor registered successfully. Please login.")
            return redirect(url_for('doctor_login'))
        except sqlite3.IntegrityError:
            flash("Username already exists. Try another.")
        finally:
            conn.close()

    return render_template('doctor_register.html')


@app.route('/logout_doctor', methods=['POST'])
def logout_doctor():
    session.pop('doctor_logged_in', None)
    session.pop('doctor_name', None)
    return redirect(url_for('doctor_login'))




# --- Route: Send message (POST) ---
@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.json
    sender = data.get("sender")
    receiver = data.get("receiver")
    message = data.get("message")

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO chat_messages (sender, receiver, message) VALUES (?,?,?)",
              (sender, receiver, message))
    conn.commit()
    conn.close()

    return jsonify({"status": "Message saved"})

# --- Route: Get messages (GET) ---
@app.route("/get_messages", methods=["GET"])
def get_messages():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT sender, receiver, message, timestamp FROM chat_messages ORDER BY timestamp ASC")
    rows = c.fetchall()
    conn.close()

    messages = [
        {"sender": r[0], "receiver": r[1], "message": r[2], "timestamp": r[3]}
        for r in rows
    ]
    return jsonify(messages)



@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        # Here you can implement:
        # 1. Check if email exists in DB
        # 2. Send reset link / temporary password to email
        flash("If this email exists, a password reset link has been sent.", "info")
        return redirect(url_for("user_login"))  # redirect back to login
    return render_template("forgot_password.html")

with app.app_context():
    init_db()

if __name__ == '__main__':
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)







