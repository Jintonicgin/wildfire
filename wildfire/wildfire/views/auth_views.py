from flask import Blueprint, render_template, request, redirect, url_for, flash, session, g
from werkzeug.security import generate_password_hash, check_password_hash

from wildfire import db
from ..forms import UserCreateForm, UserLoginForm
from ..models import Member

bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    form = UserLoginForm()
    if form.validate_on_submit():
        user = Member.query.filter_by(username=form.username.data).first()
        if not user:
            flash('존재하지 않는 아이디입니다. 회원가입이 필요합니다.')
        elif not check_password_hash(user.password, form.password.data):
            flash('아이디 또는 비밀번호가 올바르지 않습니다.')
        else:
            session.clear()
            session['user_username'] = user.username
            return redirect(url_for('main.index'))
    return render_template('auth/login.html', form=form)


@bp.route('/signup', methods=['GET', 'POST'])
def signup():
    form = UserCreateForm()
    if form.validate_on_submit():
        if Member.query.filter_by(username=form.username.data).first():
            flash('이미 존재하는 아이디입니다.')
        elif Member.query.filter_by(email=form.email.data).first():
            flash('이미 사용 중인 이메일입니다.')
        else:
            new_user = Member(
                username=form.username.data,
                password=generate_password_hash(form.password1.data),
                email=form.email.data
            )
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('auth.signup_success'))
    return render_template('auth/signup.html', form=form)

@bp.route('/signup/success')
def signup_success():
    return render_template('auth/signup_success.html')


@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('main.index'))


@bp.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        new_password = request.form.get('newPassword')
        confirm_password = request.form.get('confirmPassword')

        if new_password != confirm_password:
            flash('비밀번호가 일치하지 않습니다.', 'danger')
            return redirect(url_for('auth.reset_password'))

        user = Member.query.filter_by(username=username, email=email).first()
        if not user:
            flash('해당 사용자 정보를 찾을 수 없습니다.', 'danger')
            return redirect(url_for('auth.reset_password'))

        user.password = generate_password_hash(new_password)
        db.session.commit()

        flash('비밀번호가 재설정되었습니다.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/reset_pw.html')


@bp.route('/find-id', methods=['GET', 'POST'])
def find_id():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')

        user = Member.query.filter_by(name=name, email=email).first()

        if user:
            flash(f'회원님의 아이디는: {user.username} 입니다.', 'info')
        else:
            flash('입력한 정보와 일치하는 아이디가 없습니다.', 'danger')

        return redirect(url_for('auth.find_id'))

    return render_template('auth/find_id.html')


@bp.before_app_request
def load_logged_in_user():
    user_username = session.get('user_username')
    print(f"DEBUG: load_logged_in_user - user_username from session: {user_username}")
    if user_username is None:
        g.user = None
        print("DEBUG: g.user set to None (no user_username in session)")
    else:
        user = Member.query.get(user_username)
        g.user = user
        print(f"DEBUG: load_logged_in_user - user object from DB: {user}")
        if user:
            print(f"DEBUG: User {user.username} loaded into g.user")
        else:
            print(f"DEBUG: User {user_username} NOT found in DB, g.user set to None")

