import os
import csv
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///docify.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class Consultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('consultations', lazy=True))


# Initialize Database
with app.app_context():
    db.create_all()


# Export User Details to CSV
def export_users_to_csv():
    users = User.query.all()
    with open('users.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'name', 'phone', 'email']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for user in users:
            writer.writerow({
                'id': user.id,
                'name': user.name,
                'phone': user.phone,
                'email': user.email
            })


# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))

        new_user = User(name=name, phone=phone, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        export_users_to_csv()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'error')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        consultation = Consultation(user_id=user.id, symptoms=symptoms)
        db.session.add(consultation)
        db.session.commit()
        flash('Consultation form submitted successfully!', 'success')
        return redirect(url_for('dashboard'))

    consultations = Consultation.query.filter_by(user_id=user.id).all()
    return render_template('dash.html', user=user, consultations=consultations)


@app.route('/update_consultation/<int:id>', methods=['GET', 'POST'])
def update_consultation(id):
    if 'user_id' not in session:
        flash('Please log in to update consultations.', 'error')
        return redirect(url_for('login'))

    consultation = Consultation.query.get_or_404(id)
    if consultation.user_id != session['user_id']:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        consultation.symptoms = request.form['symptoms']
        consultation.created_at = datetime.utcnow()
        db.session.commit()
        flash('Consultation updated successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('update_consultation.html', consultation=consultation)


@app.route('/faq')
def faq():
    return render_template('faq.html')


# Updated Chatbot Route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get('message')
    if not user_message:
        return jsonify({"reply": "Please provide a message."}), 400

    # Get latest symptoms from user's consultations
    if 'user_id' in session:
        latest_consultation = Consultation.query.filter_by(user_id=session['user_id']).order_by(
            Consultation.created_at.desc()).first()
        symptoms = latest_consultation.symptoms if latest_consultation else None
    else:
        symptoms = None

    try:
        # Forward request to chatbot service
        response = requests.post(
            'http://127.0.0.1:5003/chatbot',
            json={"message": user_message, "symptoms": symptoms}
        )
        print(response)
        response_data = response.json()
        return jsonify(response_data)
    except requests.RequestException:
        return jsonify({"reply": "Error connecting to chatbot service."}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)