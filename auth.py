from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required
from models import db, User

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        if not username or not email or not password:
            flash("All fields are required.")
            return redirect(url_for("auth.register"))

        if User.query.filter_by(username=username).first():
            flash("Username already taken.")
            return redirect(url_for("auth.register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.")
            return redirect(url_for("auth.register"))

        new_user = User(username=username, email=email, role="user")
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful. Please log in.")
        return redirect(url_for("auth.login"))

    return render_template("register.html")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password.")

    return render_template("login.html")

@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not new_password or not confirm_password:
            flash("All fields are required.")
            return redirect(url_for("auth.forgot_password"))

        if new_password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("auth.forgot_password"))

        user = User.query.filter_by(username=username).first()
        if user:
            user.set_password(new_password)
            db.session.commit()
            flash("Password reset successful. Please log in.")
            return redirect(url_for("auth.login"))
        else:
            flash("Username not found.")
            return redirect(url_for("auth.forgot_password"))

    return render_template("forgot_password.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))