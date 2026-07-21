from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import bcrypt

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")

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

class AlarmRule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    parameter = db.Column(db.String(50), nullable=False)
    tier = db.Column(db.String(20), nullable=False, default="warning")   # <-- BARU
    device = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(200), nullable=False)
    value = db.Column(db.Float, nullable=False)
    condition = db.Column(db.String(20), nullable=False)
    enabled = db.Column(db.Boolean, nullable=False, default=True)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "parameter": self.parameter,
            "tier": self.tier,          
            "device": self.device,
            "message": self.message,
            "value": self.value,
            "condition": self.condition,
            "enabled": self.enabled,
        }