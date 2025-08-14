from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo, Email

class UserCreateForm(FlaskForm):
    username = StringField('아이디', validators=[DataRequired(), Length(min=3, max=30)])
    email = StringField('이메일', validators=[DataRequired(), Email(), Length(max=100)])
    password1 = PasswordField('비밀번호', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('비밀번호 확인', validators=[DataRequired(), EqualTo('password1')])
    submit = SubmitField('회원가입')

class UserLoginForm(FlaskForm):
    username = StringField('아이디', validators=[DataRequired(), Length(min=3, max=30)])
    password = PasswordField('비밀번호', validators=[DataRequired()])
    submit = SubmitField('로그인')