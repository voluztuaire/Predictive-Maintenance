from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import bcrypt

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")  # "user" or "admin"

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

    def check_password(self, password):
        return bcrypt.checkpw(
            password.encode("utf-8"), self.password_hash.encode("utf-8")
        )

    @property
    def is_admin(self):
        return self.role == "admin"


class Threshold(db.Model):
    """Store user-configurable alert thresholds"""
    id = db.Column(db.Integer, primary_key=True)
    temperature = db.Column(db.Float, nullable=False, default=75.0)
    vibration = db.Column(db.Float, nullable=False, default=3.5)
    current_deviation = db.Column(db.Float, nullable=False, default=15.0)
    pressure = db.Column(db.Float, nullable=False, default=5.5)

    def to_dict(self):
        """Convert threshold object to dictionary"""
        return {
            "temperature": self.temperature,
            "vibration": self.vibration,
            "current_deviation": self.current_deviation,
            "pressure": self.pressure
        }

    @classmethod
    def get_defaults(cls):
        """Get default threshold values"""
        return {
            "temperature": 75.0,
            "vibration": 3.5,
            "current_deviation": 15.0,
            "pressure": 5.5
        }

    @classmethod
    def get_or_create(cls):
        """Get existing thresholds or create default if none exist"""
        threshold = cls.query.first()
        if threshold is None:
            threshold = cls()
            db.session.add(threshold)
            db.session.commit()
        return threshold

    def update_from_dict(self, data):
        """Update threshold values from dictionary"""
        if "temperature" in data:
            self.temperature = float(data["temperature"])
        if "vibration" in data:
            self.vibration = float(data["vibration"])
        if "current_deviation" in data:
            self.current_deviation = float(data["current_deviation"])
        if "pressure" in data:
            self.pressure = float(data["pressure"])


# Keep for backward compatibility
DEFAULT_THRESHOLDS = {
    "temperature": 75.0,
    "vibration": 3.5,
    "current_deviation": 15.0,
    "pressure": 5.5
}